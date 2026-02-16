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

def knockout_step(
    cot: CoTResponse,
    seed: int = 42,
) -> tuple[str, str, int]:
    """Remove one critical intermediate step from the CoT.

    Returns:
        (truncated_cot_text, removed_step_summary, removed_step_id)
    """
    steps = cot.steps
    if len(steps) <= 2:
        # Too few steps — remove the only middle step or last step before answer
        idx = max(0, len(steps) - 1)
    else:
        # Remove a middle step (not first or last) for maximum disruption
        rng = random.Random(seed)
        idx = rng.randint(1, len(steps) - 2)

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

def _find_numbers(text: str) -> list[re.Match]:
    """Find all numbers in text."""
    return list(re.finditer(r"-?\d+(?:\.\d+)?", text))


def _corrupt_number(original: str, seed: int = 42) -> str:
    """Replace a number with a wrong one (close but different).

    Guarantees the output is always different from the input.
    """
    rng = random.Random(seed)
    try:
        val = float(original)
        if val == 0:
            return str(rng.choice([1, 2, 3, 5]))
        # Try up to 5 times to get a different number
        for attempt in range(5):
            factor = rng.uniform(0.3, 0.9)
            direction = rng.choice([-1, 1])
            new_val = val + (val * factor * direction)
            if "." not in original:
                result = str(int(round(new_val)))
            else:
                result = f"{new_val:.2f}"
            if result != original:
                return result
        # Fallback: just double or halve
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
        nums = _find_numbers(step.conclusion)
        if nums:
            candidates.append((i, step, nums))

    if not candidates:
        # Fallback: try corrupting reasoning text instead
        for i, step in enumerate(steps):
            nums = _find_numbers(step.reasoning)
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
    corrupted_num = _corrupt_number(original_num, seed)

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
        swaps = [
            ("True", "False"), ("False", "True"),
            ("all", "none"), ("none", "all"),
            ("every", "no"), ("no", "every"),
            ("always", "never"), ("never", "always"),
        ]
        for old, new in swaps:
            if old in question:
                modified = question.replace(old, new, 1)
                return modified, f"Swapped '{old}' → '{new}'"
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
