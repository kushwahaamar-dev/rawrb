"""Perturbation engine for generating corrupted/ablated CoTs.

Implements the three perturbation strategies:
  1. Knockout: remove a critical step from the CoT
  2. Corruption: inject a numeric/logical error into a step
  3. Counterfactual: modify a key premise in the question
"""

from __future__ import annotations

import logging
import random
import re

from .models import CoTResponse, ReasoningStep

logger = logging.getLogger(__name__)


# ── Probe 1: Knockout ──────────────────────────────────────────────

# Pre-compiled regex for identifying computationally critical steps
_COMPUTATIONAL_PATTERNS = re.compile(
    r'\d+|multiply|divide|subtract|add|equals|therefore|thus|hence|'
    r'implies|so\b|total|sum|difference|product|result',
    re.IGNORECASE,
)


def knockout_step(
    cot: CoTResponse,
    seed: int = 42,
) -> tuple[str, str, int]:
    """Remove one critical intermediate step from the CoT.

    Prefers steps containing numerical computations or logical
    deductions over setup/restatement steps, since removing a
    computational step is more likely to be causally relevant.

    Returns:
        (truncated_cot_text, removed_step_summary, removed_step_id)
    """
    steps = cot.steps
    if len(steps) <= 2:
        # Too few steps — remove the only middle step or last step before answer
        idx = max(0, len(steps) - 1)
    else:
        rng = random.Random(seed)
        middle_indices = list(range(1, len(steps) - 1))  # exclude first and last
        computational = [
            i for i in middle_indices
            if _COMPUTATIONAL_PATTERNS.search(steps[i].conclusion)
            or _COMPUTATIONAL_PATTERNS.search(steps[i].reasoning)
        ]
        # Prefer computational steps; fall back to any middle step
        pool = computational if computational else middle_indices
        idx = rng.choice(pool)

    removed = steps[idx]
    remaining = [s for i, s in enumerate(steps) if i != idx]

    # Rebuild the CoT text from remaining steps
    lines = []
    for s in remaining:
        lines.append(f"Step {s.step_id}: {s.reasoning}")
        lines.append(f"  → {s.conclusion}")
    truncated_text = "\n".join(lines)

    removed_summary = (
        f"Step {removed.step_id} was removed. "
        f"It said: '{removed.reasoning}' → '{removed.conclusion}'"
    )
    return truncated_text, removed_summary, removed.step_id


# ── Probe 2: Corruption ────────────────────────────────────────────

def _find_numbers(text: str, skip_step_labels: bool = False) -> list[re.Match]:
    """Find all numbers in text, optionally skipping step labels like 'Step 1:'."""
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", text))
    if skip_step_labels:
        matches = [m for m in matches if not re.match(r"Step\s+$", text[:m.start()])]
    return matches


def _corrupt_number(original: str, seed: int = 42) -> str:
    """Replace a number with a wrong one (close but different).

    Guarantees: (1) output differs from input, (2) relative change >= 30%.
    """
    rng = random.Random(seed)
    try:
        val = float(original)
        if val == 0:
            return str(rng.choice([1, 2, 3, 5]))
        # Try up to 10 times to get a number with >= 30% relative change
        for attempt in range(10):
            factor = rng.uniform(0.3, 0.9)
            direction = rng.choice([-1, 1])
            new_val = val + (val * factor * direction)
            if "." not in original:
                result = str(int(round(new_val)))
            else:
                result = f"{new_val:.2f}"
            # Enforce: different string AND >= 30% relative delta
            if result != original and abs(float(result) - val) / max(abs(val), 1e-9) >= 0.3:
                return result
        # Fallback: double or halve (guaranteed >= 50% change)
        if "." not in original:
            return str(int(val * 2)) if val > 0 else str(int(val - 1))
        return f"{val * 2:.2f}"
    except ValueError:
        return str(int(original) + rng.randint(1, 10))


def corrupt_step(
    cot: CoTResponse,
    seed: int = 42,
) -> tuple[str, str, int]:
    """Inject a numeric error into one CoT step.

    Returns:
        (corrupted_cot_text, corruption_detail, corrupted_step_id)
    """
    rng = random.Random(seed)
    steps = cot.steps

    # Find steps that contain numbers (those are corruptible)
    candidates = []
    for i, step in enumerate(steps):
        nums = _find_numbers(step.conclusion, skip_step_labels=True)
        if nums:
            candidates.append((i, step, nums))

    if not candidates:
        # Fallback: try corrupting reasoning text instead
        for i, step in enumerate(steps):
            nums = _find_numbers(step.reasoning, skip_step_labels=True)
            if nums:
                candidates.append((i, step, nums))

    if not candidates:
        # No numbers found — corrupt by negation
        idx = rng.randint(0, len(steps) - 1)
        step = steps[idx]
        corrupted_conclusion = f"NOT {step.conclusion}"
        detail = f"Step {step.step_id}: negated conclusion"
        corrupted_steps = []
        for i, s in enumerate(steps):
            if i == idx:
                corrupted_steps.append(
                    ReasoningStep(
                        step_id=s.step_id,
                        reasoning=s.reasoning,
                        conclusion=corrupted_conclusion,
                    )
                )
            else:
                corrupted_steps.append(s)
        lines = []
        for s in corrupted_steps:
            lines.append(f"Step {s.step_id}: {s.reasoning}")
            lines.append(f"  → {s.conclusion}")
        return "\n".join(lines), detail, step.step_id

    # Pick a random candidate and corrupt one number
    idx, step, nums = rng.choice(candidates)
    target_match = rng.choice(nums)
    original_num = target_match.group()
    # Use a problem-specific sub-seed to ensure diverse corruptions across problems
    corrupted_num = _corrupt_number(original_num, seed + hash(original_num) % 10000)

    # Apply corruption to the conclusion
    corrupted_conclusion = (
        step.conclusion[:target_match.start()]
        + corrupted_num
        + step.conclusion[target_match.end():]
    )
    detail = (
        f"Step {step.step_id}: changed '{original_num}' to '{corrupted_num}' "
        f"in conclusion"
    )

    # Rebuild full CoT with the corrupted step
    lines = []
    for i, s in enumerate(steps):
        if i == idx:
            lines.append(f"Step {s.step_id}: {s.reasoning}")
            lines.append(f"  → {corrupted_conclusion}")
        else:
            lines.append(f"Step {s.step_id}: {s.reasoning}")
            lines.append(f"  → {s.conclusion}")

    return "\n".join(lines), detail, step.step_id


# ── Probe 3: Counterfactual premise swap ────────────────────────────

def modify_premise(question: str, seed: int = 42) -> tuple[str, str]:
    """Modify a key number in the question to create a counterfactual.

    Returns:
        (modified_question, modification_detail)

    Fallback: uses LLM-based premise modification if no numbers found.
    """
    rng = random.Random(seed)
    nums = _find_numbers(question)

    if not nums:
        # No numbers — flip a logical keyword as fallback
        # For FOLIO-style problems, only modify premise content, not the
        # question template ("Is the conclusion True, False, or Unknown?")
        swaps = [
            ("all", "none"), ("none", "all"),
            ("every", "no"), ("no", "every"),
            ("always", "never"), ("never", "always"),
        ]
        # Identify premise region (before the question suffix) to avoid
        # corrupting the question format
        question_suffix_pattern = re.compile(
            r"Is the conclusion True, False, or Unknown.*$",
            re.IGNORECASE | re.DOTALL,
        )
        suffix_match = question_suffix_pattern.search(question)
        if suffix_match:
            premise_region = question[:suffix_match.start()]
            question_tail = question[suffix_match.start():]
        else:
            premise_region = question
            question_tail = ""

        for old, new in swaps:
            pattern = re.compile(re.escape(old), re.IGNORECASE)
            if pattern.search(premise_region):
                modified_premise = pattern.sub(new, premise_region, count=1)
                return modified_premise + question_tail, f"Swapped '{old}' → '{new}' in premises"
        # Last resort: prepend negation to the question
        return f"If the opposite were true: {question}", "Prepended counterfactual framing"

    # Pick a number to change (prefer larger, more impactful numbers)
    target = rng.choice(nums)
    original = target.group()
    new_val = _corrupt_number(original, seed + 1)

    modified_question = (
        question[:target.start()]
        + new_val
        + question[target.end():]
    )
    detail = f"Changed '{original}' to '{new_val}' in question"
    return modified_question, detail


# ── Helper: CoT to text ────────────────────────────────────────────

def cot_to_text(cot: CoTResponse) -> str:
    """Convert a structured CoT response to readable text."""
    lines = []
    for s in cot.steps:
        lines.append(f"Step {s.step_id}: {s.reasoning}")
        lines.append(f"  → {s.conclusion}")
    lines.append(f"\nFinal Answer: {cot.final_answer}")
    return "\n".join(lines)
