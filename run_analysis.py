#!/usr/bin/env python3
"""RAWRB — Unified analysis, visualization, and metrics pipeline.

Reads results CSVs, deduplicates, computes all metrics (aggregate,
per-benchmark, conditional, prompt sensitivity, derived statistics),
generates publication figures, and runs statistical tests.

This is the single authoritative analysis script for the paper.

Usage:
    python run_analysis.py              # full pipeline
    python run_analysis.py --tables     # tables only (no figures)
    python run_analysis.py --figures    # figures only
    python run_analysis.py --metrics    # derived metrics only (Spearman, etc.)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

from src.config import FIGURES_DIR, MODELS, RESULTS_DIR
from src.config import MODELS_SMALL, MODELS_MEDIUM
from src.metrics import (
    bootstrap_ci,
    bootstrap_gap_test,
    effect_size_cohens_h,
    mcnemar_test,
)

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Constants ──────────────────────────────────────────────────────

MODEL_TAGS = ["qwen2.5_1.5b", "llama3.2_latest", "phi3_mini",
              "llama3.1_8b", "qwen2.5_7b"]
MODEL_DISPLAY = {
    "qwen2.5_1.5b": "Qwen-2.5-1.5B",
    "llama3.2_latest": "Llama-3.2-3B",
    "phi3_mini": "Phi-3-Mini",
    "llama3.1_8b": "Llama-3.1-8B",
    "qwen2.5_7b": "Qwen-2.5-7B",
    "qwen2.5:1.5b": "Qwen-2.5-1.5B",
    "llama3.2:latest": "Llama-3.2-3B",
    "phi3:mini": "Phi-3-Mini",
    "llama3.1:8b": "Llama-3.1-8B",
    "qwen2.5:7b": "Qwen-2.5-7B",
}
BENCHMARKS = ["gsm8k", "math", "folio"]
PROBE_TYPES = ["knockout", "corruption", "counterfactual",
               "counterfactual_unbiased", "question_only"]
SMALL_TAGS = ["qwen2.5_1.5b", "llama3.2_latest", "phi3_mini"]
MEDIUM_TAGS = ["llama3.1_8b", "qwen2.5_7b"]


def safe_model_name(full_name: str) -> str:
    return MODEL_DISPLAY.get(full_name, full_name)


# ── Data Loading (shared by all sections) ──────────────────────────

def _dedup_rows(rows: list[dict]) -> list[dict]:
    """Deduplicate by (model, benchmark, problem_id), keeping last.

    Handles resumed/re-run experiments that appended duplicate rows.
    At T=0 (deterministic), duplicates have identical outcomes.
    """
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("model", ""), r.get("benchmark", ""),
               r.get("problem_id", ""))
        seen[key] = r
    return list(seen.values())


def load_csv_deduped(path: str | Path) -> list[dict]:
    """Load CSV, deduplicate, return list of row dicts (including errors)."""
    rows: list[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return _dedup_rows(rows)


def load_csv_clean(path: str | Path) -> list[dict]:
    """Load CSV, exclude error rows, THEN deduplicate.

    Order matters: if a problem has Run 1 = success and Run 2 = error
    (from a resumed experiment), removing errors first preserves the
    successful run; deduplicating first would keep Run 2 and lose it.
    """
    rows: list[dict] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error", "").strip():
                continue  # remove errors first
            rows.append(r)
    # then deduplicate
    seen: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("model", ""), r.get("benchmark", ""),
               r.get("problem_id", ""))
        seen[key] = r
    return list(seen.values())


def csv_path(model_tag: str, probe: str) -> Path:
    return RESULTS_DIR / f"results_{model_tag}_{probe}.csv"


def load_all_results_pd() -> dict[str, pd.DataFrame]:
    """Load all CSV results into DataFrames (for figure generation)."""
    KNOWN_PROBES = {
        "baseline", "knockout", "corruption",
        "counterfactual", "counterfactual_unbiased",
        "question_only", "paraphrase",
    }
    dfs: dict[str, pd.DataFrame] = {}
    for csv_file in RESULTS_DIR.glob("results_*.csv"):
        stem = csv_file.stem
        probe = None
        for p in sorted(KNOWN_PROBES, key=len, reverse=True):
            if stem.endswith(f"_{p}"):
                probe = p
                break
        if probe is None:
            continue
        df = pd.read_csv(csv_file)
        if "error" in df.columns:
            errors = df["error"].notna() & (df["error"] != "")
            n_err = errors.sum()
            if n_err > 0:
                logger.warning("Excluding %d error rows from %s (%s)",
                               n_err, probe, csv_file.name)
                df = df[~errors].copy()
        if probe in dfs:
            dfs[probe] = pd.concat([dfs[probe], df], ignore_index=True)
        else:
            dfs[probe] = df

    # Deduplicate
    dedup_key = ["model", "benchmark", "problem_id"]
    for probe, df in dfs.items():
        if all(c in df.columns for c in dedup_key):
            before = len(df)
            df = df.drop_duplicates(subset=dedup_key, keep="last")
            after = len(df)
            if before != after:
                logger.warning("Deduplicated %s: %d -> %d rows (%d removed)",
                               probe, before, after, before - after)
            dfs[probe] = df
    return dfs


# ── Helper ─────────────────────────────────────────────────────────

def pct(num: int, denom: int) -> float:
    return 100.0 * num / denom if denom else 0.0


def fmt(val: float) -> str:
    return f"{val:.1f}"


# ── Multiple-comparison correction ─────────────────────────────────

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * n
    max_k = -1
    for k, (_, p) in enumerate(indexed, 1):
        if p <= (k / n) * alpha:
            max_k = k
    if max_k > 0:
        for k in range(max_k):
            significant[indexed[k][0]] = True
    return significant


# ====================================================================
#  PART 1 — TABLES  (textual summaries printed to stdout)
# ====================================================================

def print_baseline_accuracy() -> dict:
    """Section 1: Baseline accuracy table. Returns acc dict."""
    print("=" * 80)
    print("1. BASELINE ACCURACY (% correct, errors excluded, macro-averaged)")
    print("=" * 80)

    acc: dict[tuple, float] = {}
    n_info: dict[tuple, tuple] = {}

    for tag in MODEL_TAGS:
        clean = load_csv_clean(csv_path(tag, "baseline"))
        for bench in BENCHMARKS:
            br = [r for r in clean if r["benchmark"] == bench]
            correct = sum(1 for r in br if r["correct"] == "True")
            acc[(tag, bench)] = pct(correct, len(br))
            n_info[(tag, bench)] = (correct, len(br))

    header = f"{'Model':<20}" + "".join(f"  {b:<10}" for b in BENCHMARKS) + "  Overall"
    print(header)
    print("-" * len(header))
    for tag in MODEL_TAGS:
        vals = [fmt(acc[(tag, b)]) for b in BENCHMARKS]
        overall = sum(acc[(tag, b)] for b in BENCHMARKS) / len(BENCHMARKS)
        acc[(tag, "overall")] = overall
        print(f"{safe_model_name(tag):<20}" + "".join(f"  {v:<10}" for v in vals) + f"  {fmt(overall)}")
    print()
    return acc


def print_faithfulness_scores() -> tuple[dict, dict]:
    """Section 2: Per-probe faithfulness. Returns (overall, per-bench) dicts."""
    print("=" * 80)
    print("2. FAITHFULNESS SCORES (% faithful, errors excluded)")
    print("=" * 80)

    overall: dict[tuple, float] = {}
    by_bench: dict[tuple, float] = {}

    for tag in MODEL_TAGS:
        for probe in PROBE_TYPES:
            clean = load_csv_clean(csv_path(tag, probe))
            total_f, total_n = 0, 0
            for bench in BENCHMARKS:
                br = [r for r in clean if r["benchmark"] == bench]
                faithful = sum(1 for r in br if r["faithful"] == "True")
                by_bench[(tag, probe, bench)] = pct(faithful, len(br))
                total_f += faithful
                total_n += len(br)
            overall[(tag, probe)] = pct(total_f, total_n)

    for probe in PROBE_TYPES:
        print(f"\n--- {probe.upper()} ---")
        header = f"{'Model':<20}" + "".join(f"  {b:<10}" for b in BENCHMARKS) + "  Overall"
        print(header)
        for tag in MODEL_TAGS:
            vals = [fmt(by_bench[(tag, probe, b)]) for b in BENCHMARKS]
            print(f"{safe_model_name(tag):<20}" + "".join(f"  {v:<10}" for v in vals)
                  + f"  {fmt(overall[(tag, probe)])}")
    print()
    return overall, by_bench


def print_tier_comparison(acc: dict, faith: dict) -> None:
    """Section 3: Small vs medium tier comparison."""
    print("=" * 80)
    print("3. TIER COMPARISON (Small 1.5-3.8B vs Medium 7-8B)")
    print("=" * 80)

    print("\nAccuracy by tier:")
    for bench in BENCHMARKS + ["overall"]:
        s = sum(acc[(t, bench)] for t in SMALL_TAGS) / len(SMALL_TAGS)
        m = sum(acc[(t, bench)] for t in MEDIUM_TAGS) / len(MEDIUM_TAGS)
        print(f"  {bench:<10}  Small: {fmt(s)}%  Medium: {fmt(m)}%  Delta: {fmt(m - s)}pp")

    print("\nFaithfulness by tier:")
    for probe in PROBE_TYPES:
        s = sum(faith[(t, probe)] for t in SMALL_TAGS) / len(SMALL_TAGS)
        m = sum(faith[(t, probe)] for t in MEDIUM_TAGS) / len(MEDIUM_TAGS)
        print(f"  {probe:<25}  Small: {fmt(s)}%  Medium: {fmt(m)}%  Delta: {fmt(m - s)}pp")
    print()


def print_qwen_comparison(acc: dict, faith: dict, by_bench: dict) -> None:
    """Section 4: Within-family Qwen 1.5B vs 7B."""
    print("=" * 80)
    print("4. QWEN WITHIN-FAMILY COMPARISON (1.5B vs 7B)")
    print("=" * 80)

    qs, ql = "qwen2.5_1.5b", "qwen2.5_7b"
    for bench in BENCHMARKS + ["overall"]:
        s, l = acc[(qs, bench)], acc[(ql, bench)]
        print(f"  Accuracy {bench:<10}  1.5B: {fmt(s)}%  7B: {fmt(l)}%  Delta: {fmt(l - s)}pp")
    for probe in ["knockout", "corruption"]:
        for bench in BENCHMARKS:
            s, l = by_bench[(qs, probe, bench)], by_bench[(ql, probe, bench)]
            print(f"  {probe} {bench:<10}  1.5B: {fmt(s)}%  7B: {fmt(l)}%  Delta: {fmt(l - s)}pp")
    print()


def print_prompt_sensitivity(by_bench: dict) -> None:
    """Section 5: CF vs CF_U prompt sensitivity."""
    print("=" * 80)
    print("5. PROMPT SENSITIVITY (CF with hint vs CF without hint)")
    print("=" * 80)

    for tag in MODEL_TAGS:
        for bench in BENCHMARKS:
            cf = by_bench[(tag, "counterfactual", bench)]
            cu = by_bench[(tag, "counterfactual_unbiased", bench)]
            print(f"  {safe_model_name(tag):>15} / {bench.upper():<6}  "
                  f"CF={fmt(cf)}%  CF_U={fmt(cu)}%  Delta={fmt(cf - cu)}pp")
    print()


def print_error_rates() -> None:
    """Section 6: Error rates per model per benchmark."""
    print("=" * 80)
    print("6. ERROR RATES (baseline, per model x benchmark)")
    print("=" * 80)

    for tag in MODEL_TAGS:
        rows = load_csv_deduped(csv_path(tag, "baseline"))
        for bench in BENCHMARKS:
            br = [r for r in rows if r["benchmark"] == bench]
            total = len(br)
            errors = sum(1 for r in br if r.get("error", "").strip())
            print(f"  {safe_model_name(tag):>15} / {bench:<6}  "
                  f"errors={errors}/{total}  ({fmt(pct(errors, total))}%)")
    print()


# ====================================================================
#  PART 2 — FIGURES  (saved to figures/)
# ====================================================================

def generate_figures(dfs: dict[str, pd.DataFrame]) -> None:
    """Generate all 4 publication figures."""

    # -- Figure 1: Accuracy vs Faithfulness scatter --
    baseline = dfs.get("baseline")
    if baseline is not None:
        probes = ["knockout", "corruption", "counterfactual"]
        benchmarks = baseline["benchmark"].unique()
        fig, axes = plt.subplots(1, len(benchmarks),
                                 figsize=(5 * len(benchmarks), 5))
        if len(benchmarks) == 1:
            axes = [axes]
        for ax, bench in zip(axes, benchmarks):
            for model in baseline["model"].unique():
                has_data = any(
                    dfs.get(p) is not None and (dfs[p]["model"] == model).sum() > 0
                    for p in probes)
                if not has_data:
                    continue
                mask = (baseline["model"] == model) & (baseline["benchmark"] == bench)
                acc = baseline.loc[mask, "correct"].mean() if mask.sum() > 0 else 0
                fs = []
                for p in probes:
                    df = dfs.get(p)
                    if df is None: continue
                    pm = (df["model"] == model) & (df["benchmark"] == bench)
                    if pm.sum() > 0: fs.append(df.loc[pm, "faithful"].mean())
                mf = np.mean(fs) if fs else 0.0
                ax.scatter(acc, mf, s=120, zorder=5)
                ax.annotate(safe_model_name(model), (acc, mf), fontsize=8,
                            ha="left", va="bottom", xytext=(5, 5),
                            textcoords="offset points")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax.set_xlabel("Task Accuracy"); ax.set_ylabel("Mean Faithfulness Score")
            ax.set_title(bench.upper()); ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
        fig.suptitle("Accuracy vs. Faithfulness: The Faithfulness Gap",
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "accuracy_vs_faithfulness.png")
        plt.savefig(FIGURES_DIR / "accuracy_vs_faithfulness.pdf")
        plt.close()
        logger.info("Saved accuracy_vs_faithfulness plot")

    # -- Figure 2: Probe heatmap --
    probe_types = ["knockout", "corruption", "counterfactual",
                   "counterfactual_unbiased", "question_only"]
    models = set()
    for p in probe_types:
        if p in dfs: models.update(dfs[p]["model"].unique())
    models = sorted(models)
    if models:
        matrix = np.zeros((len(models), len(probe_types)))
        for j, p in enumerate(probe_types):
            df = dfs.get(p)
            if df is None: continue
            for i, m in enumerate(models):
                mask = df["model"] == m
                if mask.sum() > 0: matrix[i, j] = df.loc[mask, "faithful"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                    xticklabels=[p.replace("_", " ").title() for p in probe_types],
                    yticklabels=[safe_model_name(m) for m in models],
                    vmin=0, vmax=1, ax=ax,
                    cbar_kws={"label": "Faithfulness Score"})
        ax.set_title("Faithfulness Scores by Model x Probe Type")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "probe_heatmap.png")
        plt.savefig(FIGURES_DIR / "probe_heatmap.pdf")
        plt.close()
        logger.info("Saved probe_heatmap plot")

    # -- Figure 3: Faithfulness gap bar chart --
    if baseline is not None:
        probes = ["knockout", "corruption", "counterfactual"]
        all_models = baseline["model"].unique()
        models = [m for m in all_models if any(
            dfs.get(p) is not None and (dfs[p]["model"] == m).sum() > 0
            for p in probes)]
        acc_list, faith_per_probe, model_labels = [], {p: [] for p in probes}, []
        for model in models:
            mask = baseline["model"] == model
            acc_list.append(baseline.loc[mask, "correct"].mean() if mask.sum() > 0 else 0)
            model_labels.append(safe_model_name(model))
            for p in probes:
                df = dfs.get(p)
                if df is None: faith_per_probe[p].append(0); continue
                pm = df["model"] == model
                faith_per_probe[p].append(
                    df.loc[pm, "faithful"].mean() if pm.sum() > 0 else 0)
        n_groups = len(model_labels); n_bars = 1 + len(probes)
        width = 0.8 / n_bars; x = np.arange(n_groups)
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#2196F3", "#FF5722", "#FF9800", "#9C27B0"]
        labels = ["Task Accuracy"] + [f"{p.title()} Faith." for p in probes]
        ax.bar(x - width * 1.5, acc_list, width, label=labels[0], color=colors[0])
        for j, p in enumerate(probes):
            ax.bar(x - width * 0.5 + j * width, faith_per_probe[p], width,
                   label=labels[j + 1], color=colors[j + 1])
        ax.set_ylabel("Score")
        ax.set_title("The Faithfulness Gap: Accuracy vs. Per-Probe Faithfulness")
        ax.set_xticks(x); ax.set_xticklabels(model_labels, rotation=15, ha="right")
        ax.legend(fontsize=9); ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
        plt.savefig(FIGURES_DIR / "faithfulness_gap.png")
        plt.savefig(FIGURES_DIR / "faithfulness_gap.pdf")
        plt.close()
        logger.info("Saved faithfulness_gap plot")

    # -- Figure 4: Paraphrase consistency boxplot --
    df = dfs.get("paraphrase")
    if df is not None:
        df = df.copy()
        df["model_short"] = df["model"].apply(safe_model_name)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="model_short", y="consistency_score",
                    palette="Set2", ax=ax)
        ax.set_xlabel("Model"); ax.set_ylabel("CoT Consistency Score (Jaccard)")
        ax.set_title("Paraphrase Stability: How Consistent Are CoT Explanations?")
        ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=15, ha="right"); plt.tight_layout()
        plt.savefig(FIGURES_DIR / "paraphrase_consistency.png")
        plt.savefig(FIGURES_DIR / "paraphrase_consistency.pdf")
        plt.close()
        logger.info("Saved paraphrase_consistency plot")


# ====================================================================
#  PART 3 — STATISTICAL TESTS  (stats_summary.csv, prompt_sensitivity.csv)
# ====================================================================

def generate_stats_tables(dfs: dict[str, pd.DataFrame]) -> None:
    """BH-corrected stats table + prompt sensitivity table."""
    baseline = dfs.get("baseline")
    if baseline is None:
        return

    probe_types = ["knockout", "corruption", "counterfactual"]
    rows_out: list[dict] = []
    all_p: list[float] = []
    p_idx: list[int] = []

    for model in baseline["model"].unique():
        for bench in baseline["benchmark"].unique():
            row: dict = {"Model": safe_model_name(model),
                         "Benchmark": bench.upper()}
            mask = (baseline["model"] == model) & (baseline["benchmark"] == bench)
            cv = baseline.loc[mask, "correct"].tolist()
            if not cv: continue
            ma, la, ha = bootstrap_ci(cv)
            row["Accuracy"] = f"{ma:.1%}"
            row["Acc. CI"] = f"[{la:.1%}, {ha:.1%}]"

            for probe in probe_types:
                df = dfs.get(probe)
                if df is None:
                    row[f"{probe.title()} Faith."] = "-"
                    row[f"{probe.title()} h"] = "-"
                    continue
                pm = (df["model"] == model) & (df["benchmark"] == bench)
                vals = df.loc[pm, "faithful"].tolist()
                if vals:
                    mf, lf, hf = bootstrap_ci(vals)
                    row[f"{probe.title()} Faith."] = f"{mf:.1%} [{lf:.1%}, {hf:.1%}]"
                    row[f"{probe.title()} h"] = f"{effect_size_cohens_h(ma, mf):.2f}"
                else:
                    row[f"{probe.title()} Faith."] = "-"
                    row[f"{probe.title()} h"] = "-"

            # Conditional faithfulness
            fc, fi = [], []
            for probe in probe_types:
                df = dfs.get(probe)
                if df is None: continue
                pm = (df["model"] == model) & (df["benchmark"] == bench)
                for _, pr in df.loc[pm].iterrows():
                    bm = ((baseline["model"] == model)
                          & (baseline["problem_id"] == pr["problem_id"])
                          & (baseline["benchmark"] == bench))
                    if bm.sum() > 0:
                        if bool(baseline.loc[bm, "correct"].iloc[0]):
                            fc.append(bool(pr["faithful"]))
                        else:
                            fi.append(bool(pr["faithful"]))
            row["Faith.|Correct"] = f"{np.mean(fc):.1%}" if fc else "-"
            row["Faith.|Incorrect"] = f"{np.mean(fi):.1%}" if fi else "-"

            # Gap test
            best_p, best_n = None, 0
            for probe in probe_types:
                df = dfs.get(probe)
                if df is None: continue
                pm = (df["model"] == model) & (df["benchmark"] == bench)
                if pm.sum() > best_n: best_n = pm.sum(); best_p = probe
            if best_p and best_n > 0:
                dfp = dfs[best_p]
                pm = (dfp["model"] == model) & (dfp["benchmark"] == bench)
                pa, pf = [], []
                for _, pr in dfp.loc[pm].iterrows():
                    bm = ((baseline["model"] == model)
                          & (baseline["problem_id"] == pr["problem_id"])
                          & (baseline["benchmark"] == bench))
                    if bm.sum() > 0:
                        pa.append(bool(baseline.loc[bm, "correct"].iloc[0]))
                        pf.append(bool(pr["faithful"]))
                if pa:
                    gap, p = bootstrap_gap_test(pa, pf)
                    all_p.append(p); p_idx.append(len(rows_out))
                    row["Gap Test p"] = f"{p:.4f}"; row["Gap"] = f"{gap:+.1%}"
                else:
                    row["Gap Test p"] = "-"; row["Gap"] = "-"
            else:
                row["Gap Test p"] = "-"; row["Gap"] = "-"
            rows_out.append(row)

    if all_p:
        sig = benjamini_hochberg(all_p)
        for i, (s, ri) in enumerate(zip(sig, p_idx)):
            ps = rows_out[ri]["Gap Test p"]
            if s and ps != "-": rows_out[ri]["Gap Test p"] = ps + "*"
            rows_out[ri]["BH Sig."] = "Yes" if s else "No"

    df_table = pd.DataFrame(rows_out)
    print("\n" + "=" * 100)
    print("STATISTICAL SUMMARY (per-model x per-benchmark, BH-corrected)")
    print("=" * 100)
    print(df_table.to_string(index=False))
    print("\n* = significant after BH correction (FDR = 0.05)\n")
    df_table.to_csv(RESULTS_DIR / "stats_summary.csv", index=False)
    df_table.to_latex(FIGURES_DIR / "stats_table.tex", index=False, escape=False)
    logger.info("Saved stats_summary.csv and stats_table.tex")

    # Prompt sensitivity (CF vs CF_U paired McNemar)
    cf = dfs.get("counterfactual")
    cf_ub = dfs.get("counterfactual_unbiased")
    if cf is None or cf_ub is None:
        return
    print("\n" + "=" * 70)
    print("PROMPT SENSITIVITY: CF (hint) vs CF_U (no hint)")
    print("=" * 70)
    sens_rows = []
    for model in sorted(set(cf["model"]) & set(cf_ub["model"])):
        for bench in sorted(set(cf["benchmark"]) & set(cf_ub["benchmark"])):
            cfi = cf.loc[(cf["model"] == model) & (cf["benchmark"] == bench)
                         ].drop_duplicates("problem_id", keep="last").set_index("problem_id")
            ubi = cf_ub.loc[(cf_ub["model"] == model) & (cf_ub["benchmark"] == bench)
                            ].drop_duplicates("problem_id", keep="last").set_index("problem_id")
            shared = sorted(set(cfi.index) & set(ubi.index))
            if len(shared) < 5: continue
            pcf = [bool(cfi.loc[p, "faithful"]) for p in shared]
            pub = [bool(ubi.loc[p, "faithful"]) for p in shared]
            cr, ur = np.mean(pcf), np.mean(pub)
            chi2, pv = mcnemar_test(pcf, pub)
            h = effect_size_cohens_h(cr, ur)
            d = cr - ur
            print(f"  {safe_model_name(model):>15} / {bench.upper():<6}  "
                  f"CF={cr:.1%}  CF_U={ur:.1%}  D={d:+.1%}  p={pv:.4f}  h={h:.2f}")
            sens_rows.append({"Model": safe_model_name(model),
                              "Benchmark": bench.upper(),
                              "CF (hint)": f"{cr:.1%}", "CF (no hint)": f"{ur:.1%}",
                              "Delta": f"{d:+.1%}", "p-value": f"{pv:.4f}",
                              "Cohen's h": f"{h:.2f}"})
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(RESULTS_DIR / "prompt_sensitivity.csv",
                                       index=False)
        logger.info("Saved prompt_sensitivity.csv")
    print()


# ====================================================================
#  PART 4 — DERIVED METRICS  (cited in paper Sections 4.2-4.5)
# ====================================================================

def compute_derived_metrics() -> None:
    """Compute the 5 additional metrics cited in the paper."""

    # -- Metric 1: Spearman rho --
    print("=" * 70)
    print("DERIVED METRIC 1: Spearman rho (accuracy vs faithfulness)")
    print("=" * 70)
    accs, ko_f, cor_f = [], [], []
    for tag in MODEL_TAGS:
        base = load_csv_clean(csv_path(tag, "baseline"))
        accs.append(sum(1 for r in base if r["correct"] == "True") / len(base))
        ko = load_csv_clean(csv_path(tag, "knockout"))
        ko_f.append(sum(1 for r in ko if r["faithful"] == "True") / len(ko))
        cor = load_csv_clean(csv_path(tag, "corruption"))
        cor_f.append(sum(1 for r in cor if r["faithful"] == "True") / len(cor))
    rho_ko, p_ko = scipy_stats.spearmanr(accs, ko_f)
    rho_cor, p_cor = scipy_stats.spearmanr(accs, cor_f)
    print(f"  Accuracy vs Knockout:   rho={rho_ko:.3f}, p={p_ko:.4f}")
    print(f"  Accuracy vs Corruption: rho={rho_cor:.3f}, p={p_cor:.4f}")
    for i, tag in enumerate(MODEL_TAGS):
        print(f"    {safe_model_name(tag):>15}  acc={accs[i]:.3f}  "
              f"KO={ko_f[i]:.3f}  COR={cor_f[i]:.3f}")
    print()

    # -- Metric 2: Concordance rate --
    print("=" * 70)
    print("DERIVED METRIC 2: Concordance rate (correct AND faithful)")
    print("=" * 70)
    for tag in MODEL_TAGS:
        base = load_csv_clean(csv_path(tag, "baseline"))
        ko = load_csv_clean(csv_path(tag, "knockout"))
        bc = {r["problem_id"]: r["correct"] == "True" for r in base}
        conc = sum(1 for r in ko if r["problem_id"] in bc
                   and bc[r["problem_id"]] and r["faithful"] == "True")
        total = sum(1 for r in ko if r["problem_id"] in bc)
        print(f"  {safe_model_name(tag):>15}  concordance={pct(conc, total):.1f}%  "
              f"({conc}/{total})")
    print()

    # -- Metric 3: Per-difficulty MATH --
    print("=" * 70)
    print("DERIVED METRIC 3: Per-difficulty MATH faithfulness")
    print("=" * 70)
    try:
        from src.benchmarks import load_math
        math_probs = load_math()
        diff_map = {p.id: p.difficulty for p in math_probs if p.difficulty}
        for probe in ["knockout", "corruption"]:
            print(f"\n  {probe.upper()}:")
            level_data: dict[str, list[int]] = {}
            for tag in MODEL_TAGS:
                rows = load_csv_clean(csv_path(tag, probe))
                for r in rows:
                    if r["benchmark"] != "math": continue
                    if r["problem_id"] in diff_map:
                        lv = diff_map[r["problem_id"]]
                        level_data.setdefault(lv, []).append(
                            1 if r["faithful"] == "True" else 0)
            for lv in sorted(level_data):
                v = level_data[lv]
                print(f"    {lv}: faith={pct(sum(v), len(v)):.1f}%  n={len(v)}")
    except Exception as e:
        print(f"  Could not load MATH difficulty: {e}")
    print()

    # -- Metric 4: Probe agreement --
    print("=" * 70)
    print("DERIVED METRIC 4: Probe agreement (knockout vs corruption)")
    print("=" * 70)
    for tag in MODEL_TAGS:
        ko = load_csv_clean(csv_path(tag, "knockout"))
        cor = load_csv_clean(csv_path(tag, "corruption"))
        ko_d = {r["problem_id"]: r["faithful"] == "True" for r in ko}
        cor_d = {r["problem_id"]: r["faithful"] == "True" for r in cor}
        shared = set(ko_d) & set(cor_d)
        if not shared: continue
        n = len(shared)
        bf = sum(1 for p in shared if ko_d[p] and cor_d[p])
        bu = sum(1 for p in shared if not ko_d[p] and not cor_d[p])
        agree = 100 * (bf + bu) / n
        kp = sum(1 for p in shared if ko_d[p]) / n
        cp = sum(1 for p in shared if cor_d[p]) / n
        pe = kp * cp + (1 - kp) * (1 - cp)
        po = (bf + bu) / n
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0
        print(f"  {safe_model_name(tag):>15}  agree={agree:.1f}%  "
              f"both_F={bf}  both_UF={bu}  n={n}  kappa={kappa:.3f}")
    print()

    # -- Metric 5: Bootstrap tier permutation test --
    print("=" * 70)
    print("DERIVED METRIC 5: Bootstrap tier comparison (permutation test)")
    print("=" * 70)
    rng = np.random.RandomState(7919)
    n_perm = 10_000
    for probe in ["knockout", "corruption"]:
        sv: list[int] = []
        mv: list[int] = []
        for tag in SMALL_TAGS:
            rows = load_csv_clean(csv_path(tag, probe))
            sv.extend(1 if r["faithful"] == "True" else 0 for r in rows)
        for tag in MEDIUM_TAGS:
            rows = load_csv_clean(csv_path(tag, probe))
            mv.extend(1 if r["faithful"] == "True" else 0 for r in rows)
        sa, ma = np.array(sv, dtype=float), np.array(mv, dtype=float)
        obs = float(sa.mean() - ma.mean())
        combined = np.concatenate([sa, ma])
        ns = len(sa)
        pd_arr = np.empty(n_perm)
        for i in range(n_perm):
            rng.shuffle(combined)
            pd_arr[i] = combined[:ns].mean() - combined[ns:].mean()
        pv = float(np.mean(np.abs(pd_arr) >= abs(obs)))
        sb = [rng.choice(sa, size=len(sa), replace=True).mean() for _ in range(n_perm)]
        mb = [rng.choice(ma, size=len(ma), replace=True).mean() for _ in range(n_perm)]
        slo, shi = np.percentile(sb, [2.5, 97.5])
        mlo, mhi = np.percentile(mb, [2.5, 97.5])
        overlap = bool(slo < mhi and mlo < shi)
        print(f"  {probe:>15}  small={sa.mean():.3f} [{slo:.3f},{shi:.3f}]  "
              f"medium={ma.mean():.3f} [{mlo:.3f},{mhi:.3f}]")
        print(f"                  diff={obs:.3f}  perm_p={pv:.4f}  "
              f"CIs_overlap={overlap}")
    print()


# ====================================================================
#  PART 5 — SCALE TIER & PER-DIFFICULTY (from original run_analysis)
# ====================================================================

def analyze_scale_tiers(dfs: dict[str, pd.DataFrame]) -> None:
    probe_types = ["knockout", "corruption", "counterfactual"]
    small_set, medium_set = set(MODELS_SMALL), set(MODELS_MEDIUM)
    print("\n" + "=" * 70)
    print("SCALE TIER COMPARISON (bootstrap CIs + Cohen's h)")
    print("=" * 70)
    for probe in probe_types:
        df = dfs.get(probe)
        if df is None: continue
        sv = df[df["model"].isin(small_set)]["faithful"].tolist()
        mv = df[df["model"].isin(medium_set)]["faithful"].tolist()
        if not sv or not mv: continue
        sm, sl, sh = bootstrap_ci(sv)
        mm, ml, mh = bootstrap_ci(mv)
        h = effect_size_cohens_h(mm, sm)
        print(f"  {probe:>15}  Small={sm:.1%} [{sl:.1%},{sh:.1%}]  "
              f"Medium={mm:.1%} [{ml:.1%},{mh:.1%}]  h={h:.2f}")
    baseline = dfs.get("baseline")
    if baseline is not None:
        sa = baseline[baseline["model"].isin(small_set)]["correct"].tolist()
        ma = baseline[baseline["model"].isin(medium_set)]["correct"].tolist()
        if sa and ma:
            sm, sl, sh = bootstrap_ci(sa)
            mm, ml, mh = bootstrap_ci(ma)
            print(f"  {'accuracy':>15}  Small={sm:.1%} [{sl:.1%},{sh:.1%}]  "
                  f"Medium={mm:.1%} [{ml:.1%},{mh:.1%}]  "
                  f"h={effect_size_cohens_h(mm, sm):.2f}")
    print()


def analyze_per_difficulty(dfs: dict[str, pd.DataFrame]) -> None:
    baseline = dfs.get("baseline")
    if baseline is None: return
    if baseline[baseline["benchmark"] == "math"].empty: return
    try:
        from src.benchmarks import load_math
        math_probs = load_math()
        diff_map = {p.id: p.difficulty for p in math_probs if p.difficulty}
    except Exception:
        return
    if not diff_map: return
    print("\n" + "=" * 70)
    print("PER-DIFFICULTY ANALYSIS (MATH) — with bootstrap CIs")
    print("=" * 70)
    for probe in ["knockout", "corruption", "counterfactual"]:
        df = dfs.get(probe)
        if df is None: continue
        mp = df[df["benchmark"] == "math"]
        if mp.empty: continue
        print(f"\n  {probe.upper()}:")
        for level in sorted(set(diff_map.values())):
            pids = {pid for pid, d in diff_map.items() if d == level}
            lr = mp[mp["problem_id"].isin(pids)]
            if lr.empty: continue
            vals = lr["faithful"].tolist()
            mf, lf, hf = bootstrap_ci(vals)
            print(f"    {level:>10}: faith={mf:.1%} [{lf:.1%},{hf:.1%}]  n={len(vals)}")
    print()


# ====================================================================
#  MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="RAWRB unified analysis")
    parser.add_argument("--tables", action="store_true", help="Tables only")
    parser.add_argument("--figures", action="store_true", help="Figures only")
    parser.add_argument("--metrics", action="store_true",
                        help="Derived metrics only")
    args = parser.parse_args()
    run_all = not (args.tables or args.figures or args.metrics)

    if run_all or args.tables:
        acc = print_baseline_accuracy()
        faith, by_bench = print_faithfulness_scores()
        print_tier_comparison(acc, faith)
        print_qwen_comparison(acc, faith, by_bench)
        print_prompt_sensitivity(by_bench)
        print_error_rates()

    if run_all or args.figures or args.tables:
        logger.info("Loading results for figures/stats ...")
        dfs = load_all_results_pd()
        logger.info("Probes loaded: %s", list(dfs.keys()))
        for name, df in dfs.items():
            logger.info("  %s: %d rows", name, len(df))

    if run_all or args.figures:
        generate_figures(dfs)

    if run_all or args.tables:
        generate_stats_tables(dfs)
        analyze_scale_tiers(dfs)
        analyze_per_difficulty(dfs)

    if run_all or args.metrics:
        compute_derived_metrics()

    logger.info("All analysis complete. Figures in %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
