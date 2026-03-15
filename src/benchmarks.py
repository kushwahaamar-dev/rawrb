"""Load and sample benchmark problems into a unified format.

Supports GSM8K, MATH, and FOLIO. Adapted from cg-cot/src/benchmarks.py.
"""

from __future__ import annotations

import logging
import random
import re

from datasets import load_dataset

from . import config as _cfg
from .config import BENCHMARK_SIZES
from .models import BenchmarkName, BenchmarkProblem

logger = logging.getLogger(__name__)


# ── Answer extraction helpers ───────────────────────────────────────

def _extract_gsm8k_answer(answer_text: str) -> str:
    """Pull the number after '####' in GSM8K answer strings."""
    match = re.search(r"####\s*(-?[\d,]+)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_text.strip()


def _extract_math_answer(solution_text: str) -> str:
    """Pull content from \\boxed{...} in MATH solutions."""
    idx = solution_text.rfind("\\boxed{")
    if idx == -1:
        return solution_text.strip()
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(solution_text)):
        if solution_text[i] == "{":
            depth += 1
        elif solution_text[i] == "}":
            if depth == 0:
                return solution_text[start:i].strip()
            depth -= 1
    return solution_text[start:].strip()


# ── Loaders ─────────────────────────────────────────────────────────

def load_gsm8k(n: int | None = None) -> list[BenchmarkProblem]:
    """Load GSM8K test split, sample *n* problems."""
    n = n or BENCHMARK_SIZES["gsm8k"]
    logger.info("Loading GSM8K (n=%d) …", n)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds)
    rng = random.Random(_cfg.SEED)
    sampled = rng.sample(items, min(n, len(items)))
    problems = []
    for i, row in enumerate(sampled):
        answer = _extract_gsm8k_answer(row["answer"])
        steps = row["answer"].count("<<")
        problems.append(
            BenchmarkProblem(
                id=f"gsm8k_{i:04d}",
                benchmark=BenchmarkName.GSM8K,
                text=row["question"],
                answer=answer,
                difficulty=str(steps),
                metadata={"raw_answer": row["answer"]},
            )
        )
    logger.info("GSM8K loaded: %d problems", len(problems))
    return problems


def load_math(n: int | None = None) -> list[BenchmarkProblem]:
    """Load MATH test split, sample *n* problems."""
    n = n or BENCHMARK_SIZES["math"]
    logger.info("Loading MATH (n=%d) …", n)
    _SUBJECTS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    items = []
    for subj in _SUBJECTS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
            items.extend(list(ds))
        except Exception as exc:
            logger.warning("Failed to load MATH/%s: %s", subj, exc)
    if not items:
        raise RuntimeError("Could not load any MATH dataset subjects")
    logger.info("MATH total pool: %d problems", len(items))
    rng = random.Random(_cfg.SEED)
    sampled = rng.sample(items, min(n, len(items)))
    problems = []
    for i, row in enumerate(sampled):
        answer = _extract_math_answer(row["solution"])
        problems.append(
            BenchmarkProblem(
                id=f"math_{i:04d}",
                benchmark=BenchmarkName.MATH,
                text=row["problem"],
                answer=answer,
                difficulty=row.get("level", ""),
                metadata={
                    "type": row.get("type", ""),
                    "raw_solution": row["solution"],
                },
            )
        )
    logger.info("MATH loaded: %d problems", len(problems))
    return problems


def load_folio(n: int | None = None) -> list[BenchmarkProblem]:
    """Load FOLIO validation split, sample *n* problems."""
    n = n or BENCHMARK_SIZES["folio"]
    logger.info("Loading FOLIO (n=%d) …", n)
    ds = load_dataset("tasksource/folio", split="validation")
    items = list(ds)
    rng = random.Random(_cfg.SEED)
    sampled = rng.sample(items, min(n, len(items)))
    problems = []
    for i, row in enumerate(sampled):
        premises = row.get("premises", "")
        conclusion = row.get("conclusion", "")
        label = str(row.get("label", "")).strip()
        label_map = {
            "0": "True", "1": "False", "2": "Unknown",
            "true": "True", "false": "False",
            "uncertain": "Unknown", "unknown": "Unknown",
        }
        label_text = label_map.get(label.lower(), label)
        problem_text = (
            f"Premises:\n{premises}\n\n"
            f"Conclusion:\n{conclusion}\n\n"
            f"Is the conclusion True, False, or Unknown given the premises?"
        )
        problems.append(
            BenchmarkProblem(
                id=f"folio_{i:04d}",
                benchmark=BenchmarkName.FOLIO,
                text=problem_text,
                answer=label_text,
                difficulty=None,
                metadata={
                    "premises_fol": row.get("premises-FOL", ""),
                    "conclusion_fol": row.get("conclusion-FOL", ""),
                },
            )
        )
    logger.info("FOLIO loaded: %d problems", len(problems))
    return problems


# ── Registry ────────────────────────────────────────────────────────

LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "folio": load_folio,
}
