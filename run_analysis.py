#!/usr/bin/env python3
"""RAWRB — Analysis and visualization pipeline.

Reads results CSVs, computes aggregate metrics, generates publication
figures, and runs statistical tests.

Usage:
    python run_analysis.py
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, MODELS, RESULTS_DIR
from src.metrics import (
    bootstrap_ci,
    effect_size_cohens_h,
    faithfulness_gap,
    faithfulness_score,
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


# ── Data Loading ────────────────────────────────────────────────────

def load_all_results() -> dict[str, pd.DataFrame]:
    """Load all CSV results into a dict of DataFrames keyed by probe type."""
    # Match against known probe names to handle multi-word names (e.g. question_only)
    KNOWN_PROBES = {
        "baseline", "knockout", "corruption",
        "counterfactual", "question_only", "paraphrase",
    }
    dfs = {}
    for csv_file in RESULTS_DIR.glob("results_*.csv"):
        stem = csv_file.stem  # e.g. "results_qwen2_5_7b_question_only"
        # Find which known probe name the stem ends with (longest match first)
        probe = None
        for p in sorted(KNOWN_PROBES, key=len, reverse=True):
            if stem.endswith(f"_{p}"):
                probe = p
                break
        if probe is None:
            logger.warning("Unknown probe in filename: %s — skipping", csv_file.name)
            continue
        df = pd.read_csv(csv_file)
        # Exclude error rows to prevent biasing faithfulness scores
        if "error" in df.columns:
            errors = df["error"].notna() & (df["error"] != "")
            n_errors = errors.sum()
            if n_errors > 0:
                logger.warning(
                    "Excluding %d error rows from %s (%s)",
                    n_errors, probe, csv_file.name,
                )
                df = df[~errors].copy()
        if probe in dfs:
            dfs[probe] = pd.concat([dfs[probe], df], ignore_index=True)
        else:
            dfs[probe] = df
    return dfs


def safe_model_name(full_name: str) -> str:
    """Shorten model names for display."""
    short = {
        "llama3.1:8b": "Llama-3.1-8B",
        "mistral:7b": "Mistral-7B",
        "phi3:mini": "Phi-3-Mini",
        "gemma2:9b": "Gemma-2-9B",
        "qwen2.5:7b": "Qwen-2.5-7B",
    }
    return short.get(full_name, full_name)


# ── Figure 1: Accuracy vs Faithfulness Scatter ──────────────────────

def plot_accuracy_vs_faithfulness(dfs: dict[str, pd.DataFrame]) -> None:
    """Scatter plot: task accuracy vs mean faithfulness per model."""
    baseline = dfs.get("baseline")
    if baseline is None:
        logger.warning("No baseline data found, skipping accuracy plot")
        return

    probe_types = ["knockout", "corruption", "counterfactual"]
    models = baseline["model"].unique()
    benchmarks = baseline["benchmark"].unique()

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 5))
    if len(benchmarks) == 1:
        axes = [axes]

    for ax, bench in zip(axes, benchmarks):
        for model in models:
            # Accuracy
            mask = (baseline["model"] == model) & (baseline["benchmark"] == bench)
            acc = baseline.loc[mask, "correct"].mean() if mask.sum() > 0 else 0

            # Mean faithfulness across probes
            faith_scores = []
            for probe in probe_types:
                df = dfs.get(probe)
                if df is None:
                    continue
                pmask = (df["model"] == model) & (df["benchmark"] == bench)
                if pmask.sum() > 0:
                    faith_scores.append(df.loc[pmask, "faithful"].mean())

            mean_faith = np.mean(faith_scores) if faith_scores else 0.0

            ax.scatter(
                acc, mean_faith,
                s=120, zorder=5, label=safe_model_name(model),
            )
            ax.annotate(
                safe_model_name(model),
                (acc, mean_faith),
                fontsize=8, ha="left", va="bottom",
                xytext=(5, 5), textcoords="offset points",
            )

        # Diagonal line (perfect faithfulness = accuracy)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect faithfulness")
        ax.set_xlabel("Task Accuracy")
        ax.set_ylabel("Mean Faithfulness Score")
        ax.set_title(bench.upper())
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Accuracy vs. Faithfulness: The Faithfulness Gap", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_vs_faithfulness.png")
    plt.savefig(FIGURES_DIR / "accuracy_vs_faithfulness.pdf")
    plt.close()
    logger.info("Saved accuracy_vs_faithfulness plot")


# ── Figure 2: Probe Heatmap ─────────────────────────────────────────

def plot_probe_heatmap(dfs: dict[str, pd.DataFrame]) -> None:
    """Heatmap: faithfulness by model × probe type."""
    probe_types = ["knockout", "corruption", "counterfactual", "question_only"]
    models = set()
    for probe in probe_types:
        if probe in dfs:
            models.update(dfs[probe]["model"].unique())
    models = sorted(models)

    if not models:
        logger.warning("No probe data found, skipping heatmap")
        return

    matrix = np.zeros((len(models), len(probe_types)))
    for j, probe in enumerate(probe_types):
        df = dfs.get(probe)
        if df is None:
            continue
        for i, model in enumerate(models):
            mask = df["model"] == model
            if mask.sum() > 0:
                matrix[i, j] = df.loc[mask, "faithful"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=[p.title() for p in probe_types],
        yticklabels=[safe_model_name(m) for m in models],
        vmin=0, vmax=1, ax=ax,
        cbar_kws={"label": "Faithfulness Score"},
    )
    ax.set_title("Faithfulness Scores by Model × Probe Type")
    ax.set_xlabel("Probe Type")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probe_heatmap.png")
    plt.savefig(FIGURES_DIR / "probe_heatmap.pdf")
    plt.close()
    logger.info("Saved probe_heatmap plot")


# ── Figure 3: Faithfulness Gap Bar Chart ────────────────────────────

def plot_faithfulness_gap(dfs: dict[str, pd.DataFrame]) -> None:
    """Grouped bar chart: accuracy and faithfulness side by side."""
    baseline = dfs.get("baseline")
    if baseline is None:
        return

    probe_types = ["knockout", "corruption", "counterfactual"]
    models = baseline["model"].unique()

    acc_list = []
    faith_list = []
    model_labels = []

    for model in models:
        mask = baseline["model"] == model
        acc = baseline.loc[mask, "correct"].mean() if mask.sum() > 0 else 0
        acc_list.append(acc)
        model_labels.append(safe_model_name(model))

        faith_scores = []
        for probe in probe_types:
            df = dfs.get(probe)
            if df is None:
                continue
            pmask = df["model"] == model
            if pmask.sum() > 0:
                faith_scores.append(df.loc[pmask, "faithful"].mean())

        faith_list.append(np.mean(faith_scores) if faith_scores else 0)

    x = np.arange(len(model_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, acc_list, width, label="Task Accuracy", color="#2196F3")
    bars2 = ax.bar(x + width / 2, faith_list, width, label="Faithfulness", color="#FF5722")

    # Gap annotations
    for i, (a, f) in enumerate(zip(acc_list, faith_list)):
        gap = a - f
        if gap > 0:
            ax.annotate(
                f"Δ={gap:.0%}",
                xy=(i, max(a, f) + 0.02),
                ha="center", fontsize=9, fontweight="bold", color="#D32F2F",
            )

    ax.set_ylabel("Score")
    ax.set_title("The Faithfulness Gap: Accuracy vs. Faithfulness")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "faithfulness_gap.png")
    plt.savefig(FIGURES_DIR / "faithfulness_gap.pdf")
    plt.close()
    logger.info("Saved faithfulness_gap plot")


# ── Figure 4: Paraphrase Consistency ────────────────────────────────

def plot_paraphrase_consistency(dfs: dict[str, pd.DataFrame]) -> None:
    """Box plot of CoT consistency scores across models."""
    df = dfs.get("paraphrase")
    if df is None:
        logger.warning("No paraphrase data, skipping")
        return

    df = df.copy()
    df["model_short"] = df["model"].apply(safe_model_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x="model_short", y="consistency_score",
        palette="Set2", ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("CoT Consistency Score (Jaccard)")
    ax.set_title("Paraphrase Stability: How Consistent Are CoT Explanations?")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paraphrase_consistency.png")
    plt.savefig(FIGURES_DIR / "paraphrase_consistency.pdf")
    plt.close()
    logger.info("Saved paraphrase_consistency plot")


# ── Table: Statistical Summary ──────────────────────────────────────

def generate_stats_table(dfs: dict[str, pd.DataFrame]) -> None:
    """Generate a LaTeX-ready stats table with bootstrap CIs."""
    baseline = dfs.get("baseline")
    if baseline is None:
        return

    probe_types = ["knockout", "corruption", "counterfactual"]  # question_only is a control, shown separately
    models = baseline["model"].unique()
    rows = []

    for model in models:
        row = {"Model": safe_model_name(model)}

        # Accuracy with CI
        mask = baseline["model"] == model
        correct_vals = baseline.loc[mask, "correct"].tolist()
        mean, lo, hi = bootstrap_ci(correct_vals)
        row["Accuracy"] = f"{mean:.1%}"
        row["Acc. CI"] = f"[{lo:.1%}, {hi:.1%}]"

        # Faithfulness per probe
        faith_all = []
        for probe in probe_types:
            df = dfs.get(probe)
            if df is None:
                row[f"{probe.title()} Faith."] = "—"
                continue
            pmask = df["model"] == model
            vals = df.loc[pmask, "faithful"].tolist()
            if vals:
                mean_f, lo_f, hi_f = bootstrap_ci(vals)
                row[f"{probe.title()} Faith."] = f"{mean_f:.1%} [{lo_f:.1%}, {hi_f:.1%}]"
                faith_all.extend(vals)
            else:
                row[f"{probe.title()} Faith."] = "—"

        # Faithfulness gap
        if faith_all and correct_vals:
            gap = np.mean(correct_vals) - np.mean(faith_all)
            row["Gap"] = f"{gap:+.1%}"

            # McNemar's test — per-probe on matched items
            # Pick the probe with the most data for the primary test
            best_probe = None
            best_count = 0
            for probe in probe_types:
                df = dfs.get(probe)
                if df is None:
                    continue
                pmask = df["model"] == model
                if pmask.sum() > best_count:
                    best_count = pmask.sum()
                    best_probe = probe

            if best_probe and best_count > 0:
                df_p = dfs[best_probe]
                pmask = df_p["model"] == model
                probe_items = df_p.loc[pmask]
                # Build paired arrays: accuracy vs faithfulness for same problems
                paired_acc = []
                paired_faith = []
                for _, prow in probe_items.iterrows():
                    pid = prow["problem_id"]
                    bmask = (baseline["model"] == model) & (baseline["problem_id"] == pid)
                    if bmask.sum() > 0:
                        paired_acc.append(bool(baseline.loc[bmask, "correct"].iloc[0]))
                        paired_faith.append(bool(prow["faithful"]))
                if len(paired_acc) > 0:
                    chi2, p = mcnemar_test(paired_acc, paired_faith)
                    row["McNemar p"] = f"{p:.4f}"
                else:
                    row["McNemar p"] = "—"
            else:
                row["McNemar p"] = "—"
        else:
            row["Gap"] = "—"
            row["McNemar p"] = "—"

        rows.append(row)

    df_table = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print(df_table.to_string(index=False))
    print()

    # Save as CSV
    df_table.to_csv(RESULTS_DIR / "stats_summary.csv", index=False)

    # Save as LaTeX
    latex = df_table.to_latex(index=False, escape=False)
    with open(FIGURES_DIR / "stats_table.tex", "w") as f:
        f.write(latex)
    logger.info("Saved stats_summary.csv and stats_table.tex")


# ── Counterfactual vs Question-Only Comparison ─────────────────────

def compare_counterfactual_vs_question_only(dfs: dict[str, pd.DataFrame]) -> None:
    """Compare answer-change rates: counterfactual vs question-only control."""
    cf = dfs.get("counterfactual")
    qo = dfs.get("question_only")
    if cf is None or qo is None:
        logger.info("Skipping counterfactual vs question_only (missing data)")
        return

    print("\n" + "=" * 60)
    print("COUNTERFACTUAL vs QUESTION-ONLY CONTROL")
    print("=" * 60)

    models = set(cf["model"].unique()) & set(qo["model"].unique())
    for model in sorted(models):
        cf_mask = cf["model"] == model
        qo_mask = qo["model"] == model
        cf_rate = cf.loc[cf_mask, "answer_changed"].mean() if cf_mask.sum() > 0 else 0
        qo_rate = qo.loc[qo_mask, "answer_changed"].mean() if qo_mask.sum() > 0 else 0
        delta = cf_rate - qo_rate
        interp = (
            "stale CoT anchors answer (unfaithful)" if delta < -0.05
            else "CoT helps detection (faithful)" if delta > 0.05
            else "inconclusive"
        )
        print(
            f"  {safe_model_name(model):>15s}  "
            f"CF={cf_rate:.1%}  QO={qo_rate:.1%}  "
            f"Δ={delta:+.1%}  → {interp}"
        )
    print()


# ── Main ────────────────────────────────────────────────────────────

def main():
    logger.info("Loading results from %s …", RESULTS_DIR)
    dfs = load_all_results()

    if not dfs:
        logger.error("No results found in %s", RESULTS_DIR)
        return

    logger.info("Found data for probes: %s", list(dfs.keys()))
    for name, df in dfs.items():
        logger.info("  %s: %d rows", name, len(df))

    # Generate all figures
    plot_accuracy_vs_faithfulness(dfs)
    plot_probe_heatmap(dfs)
    plot_faithfulness_gap(dfs)
    plot_paraphrase_consistency(dfs)
    generate_stats_table(dfs)
    compare_counterfactual_vs_question_only(dfs)

    logger.info("All analysis complete. Figures in %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
