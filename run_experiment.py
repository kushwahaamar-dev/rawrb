#!/usr/bin/env python3
"""RAWRB — Main experiment orchestrator.

Runs all probe types across all models and benchmarks.

Usage:
    # Smoke test (5 problems, 1 model)
    python run_experiment.py --smoke

    # Single model, all probes
    python run_experiment.py --models qwen2.5:7b

    # Full run (all 5 models × 3 benchmarks × all probes)
    python run_experiment.py

    # Specific benchmark and probe
    python run_experiment.py --benchmarks gsm8k --probes baseline knockout
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

from tqdm import tqdm

from src.config import (
    BENCHMARK_SIZES,
    DEFAULT_MODEL,
    FLUSH_EVERY,
    MODELS,
    NUM_PARAPHRASES,
    PROBE_NAMES,
    RESULTS_DIR,
    SEED,
)
from src.benchmarks import LOADERS
from src.llm_client import make_client
from src.probes import (
    run_baseline,
    run_corruption,
    run_counterfactual,
    run_knockout,
    run_paraphrase,
)
from src.models import BaselineRow, CoTResponse, ProbeRow, ParaphraseRow

# ── Logging ─────────────────────────────────────────────────────────

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
))
root_logger.addHandler(console)

fh = logging.FileHandler("experiment.log", mode="a")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
))
root_logger.addHandler(fh)

logger = logging.getLogger("rawrb")


# ── CSV Helpers ─────────────────────────────────────────────────────

def _csv_path(model_name: str, probe: str) -> Path:
    safe = model_name.replace("/", "_").replace(":", "_")
    return RESULTS_DIR / f"results_{safe}_{probe}.csv"


def _load_done(path: Path) -> set[str]:
    """Return set of problem_ids already completed."""
    done = set()
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add(row["problem_id"])
    return done


def _init_csv(path: Path, fields: list[str]) -> None:
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _append_row(path: Path, row_dict: dict, fields: list[str]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(row_dict)


# ── Baseline Phase ──────────────────────────────────────────────────

def run_baseline_phase(
    model_name: str,
    benchmarks: list[str],
    smoke: bool = False,
    limit: int | None = None,
) -> dict[str, dict[str, tuple[BaselineRow, CoTResponse | None]]]:
    """Run baseline CoT on all problems. Returns cached CoTs for probes."""
    llm = make_client(model_name)
    fields = list(BaselineRow.model_fields.keys())
    csv_file = _csv_path(model_name, "baseline")
    _init_csv(csv_file, fields)
    done = _load_done(csv_file)

    n = 5 if smoke else None
    cache: dict[str, dict[str, tuple[BaselineRow, CoTResponse | None]]] = {}

    for bench_name in benchmarks:
        loader = LOADERS[bench_name]
        problems = loader(n=n)
        if limit:
            problems = problems[:limit]

        cache[bench_name] = {}
        logger.info("[%s] Baseline: %s — %d problems", model_name, bench_name, len(problems))

        for problem in tqdm(problems, desc=f"{model_name}/{bench_name}/baseline"):
            if problem.id in done:
                # Still need CoT for probes — re-generate quietly
                pass

            row, cot = run_baseline(problem, llm)
            cache[bench_name][problem.id] = (row, cot)

            if problem.id not in done:
                _append_row(csv_file, row.model_dump(), fields)

    logger.info("[%s] Baseline complete. Usage: %s", model_name, llm.usage_summary)
    return cache


# ── Probe Phase ─────────────────────────────────────────────────────

def run_probe_phase(
    model_name: str,
    probe_name: str,
    cache: dict[str, dict[str, tuple[BaselineRow, CoTResponse | None]]],
):
    """Run a single probe type using cached baseline CoTs."""
    llm = make_client(model_name)

    if probe_name == "paraphrase":
        fields = list(ParaphraseRow.model_fields.keys())
    else:
        fields = list(ProbeRow.model_fields.keys())

    csv_file = _csv_path(model_name, probe_name)
    _init_csv(csv_file, fields)
    done = _load_done(csv_file)

    probe_fn = {
        "knockout": run_knockout,
        "corruption": run_corruption,
        "counterfactual": run_counterfactual,
        "paraphrase": run_paraphrase,
    }[probe_name]

    for bench_name, problems in cache.items():
        skipped = 0
        for problem_id, (baseline_row, cot) in tqdm(
            problems.items(),
            desc=f"{model_name}/{bench_name}/{probe_name}",
        ):
            if problem_id in done:
                skipped += 1
                continue
            if cot is None:
                logger.warning("No CoT for %s, skipping probe", problem_id)
                continue

            # Reconstruct problem from baseline row
            from src.models import BenchmarkProblem, BenchmarkName
            problem = BenchmarkProblem(
                id=problem_id,
                benchmark=BenchmarkName(baseline_row.benchmark),
                text=baseline_row.question,
                answer=baseline_row.ground_truth,
            )

            if probe_name == "paraphrase":
                result = probe_fn(problem, cot, llm, num_paraphrases=NUM_PARAPHRASES)
            else:
                result = probe_fn(problem, cot, llm, seed=SEED)

            row_dict = result.model_dump()
            # Convert lists to JSON strings for CSV
            for k, v in row_dict.items():
                if isinstance(v, list):
                    row_dict[k] = json.dumps(v)
            _append_row(csv_file, row_dict, fields)

        logger.info(
            "[%s] %s/%s complete (skipped %d). Usage: %s",
            model_name, bench_name, probe_name, skipped, llm.usage_summary,
        )


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAWRB: Right Answer Wrong Reason Benchmark"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to test (default: all {len(MODELS)})",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["gsm8k", "math", "folio"],
        choices=["gsm8k", "math", "folio"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--probes", nargs="+", default=None,
        choices=PROBE_NAMES,
        help="Probes to run (default: all)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 5 problems per benchmark, 1 model",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max problems per benchmark",
    )
    args = parser.parse_args()

    models = args.models or ([DEFAULT_MODEL] if args.smoke else MODELS)
    probes = args.probes or PROBE_NAMES

    logger.info("=" * 60)
    logger.info("RAWRB Experiment")
    logger.info("Models: %s", models)
    logger.info("Benchmarks: %s", args.benchmarks)
    logger.info("Probes: %s", probes)
    logger.info("Smoke: %s | Limit: %s", args.smoke, args.limit)
    logger.info("=" * 60)

    for model_name in models:
        logger.info("━━━ Model: %s ━━━", model_name)

        # Phase 1: Baseline CoT (always needed)
        if "baseline" in probes:
            cache = run_baseline_phase(
                model_name, args.benchmarks,
                smoke=args.smoke, limit=args.limit,
            )
        else:
            # Still need baselines for other probes
            cache = run_baseline_phase(
                model_name, args.benchmarks,
                smoke=args.smoke, limit=args.limit,
            )

        # Phase 2: Run each probe
        active_probes = [p for p in probes if p != "baseline"]
        for probe_name in active_probes:
            run_probe_phase(model_name, probe_name, cache)

    logger.info("All experiments complete. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
