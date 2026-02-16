"""Faithfulness probes — the core experimental logic.

Implements the 4 probe types:
  1. Baseline:       Standard CoT → measure accuracy
  2. Knockout:       Remove a step → see if answer changes
  3. Corruption:     Inject error → see if model propagates it
  4. Counterfactual: Swap premise → see if model detects mismatch
  5. Paraphrase:     Rephrase question → measure CoT consistency
"""

from __future__ import annotations

import logging
import time

from .llm_client import OllamaClient
from .models import (
    BaselineRow,
    BenchmarkProblem,
    CoTResponse,
    ContinuationResponse,
    ParaphraseResponse,
    ParaphraseRow,
    ProbeRow,
)
from .perturbations import (
    cot_to_text,
    corrupt_step,
    knockout_step,
    modify_premise,
)
from .prompts import (
    baseline_cot_prompt,
    corruption_continuation_prompt,
    counterfactual_prompt,
    counterfactual_question_prompt,
    knockout_continuation_prompt,
    paraphrase_prompt,
)
from .metrics import normalize_answer, answers_match

logger = logging.getLogger(__name__)


# ── Baseline CoT ────────────────────────────────────────────────────

def run_baseline(
    problem: BenchmarkProblem,
    llm: OllamaClient,
) -> tuple[BaselineRow, CoTResponse | None]:
    """Run standard CoT on a problem. Returns (row, cot_response)."""
    start = time.time()
    tokens_before = llm.total_tokens_used
    try:
        sys_prompt, usr_prompt = baseline_cot_prompt(problem.text)
        cot = llm.call(sys_prompt, usr_prompt, CoTResponse)
        predicted = normalize_answer(cot.final_answer)
        correct = answers_match(predicted, problem.answer)
        cot_text = cot_to_text(cot)

        row = BaselineRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="baseline",
            question=problem.text,
            cot_text=cot_text,
            cot_steps=len(cot.steps),
            predicted_answer=predicted,
            ground_truth=problem.answer,
            correct=correct,
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )
        return row, cot
    except Exception as exc:
        logger.error("Baseline failed for %s: %s", problem.id, exc)
        row = BaselineRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="baseline",
            question=problem.text,
            ground_truth=problem.answer,
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
            error=str(exc),
        )
        return row, None


# ── Probe 1: Knockout ──────────────────────────────────────────────

def run_knockout(
    problem: BenchmarkProblem,
    cot: CoTResponse,
    llm: OllamaClient,
    seed: int = 42,
) -> ProbeRow:
    """Remove a step and test if the answer changes."""
    start = time.time()
    tokens_before = llm.total_tokens_used
    try:
        truncated_text, removed_summary, step_id = knockout_step(cot, seed)
        sys_prompt, usr_prompt = knockout_continuation_prompt(
            problem.text, truncated_text, removed_summary,
        )
        response = llm.call(sys_prompt, usr_prompt, ContinuationResponse)
        new_answer = normalize_answer(response.final_answer)
        original_answer = normalize_answer(cot.final_answer)
        answer_changed = not answers_match(new_answer, original_answer)
        original_correct = answers_match(original_answer, problem.answer)

        return ProbeRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="knockout",
            original_answer=original_answer,
            original_correct=original_correct,
            perturbed_answer=new_answer,
            answer_changed=answer_changed,
            perturbation_detail=f"Removed step {step_id}: {removed_summary}",
            faithful=answer_changed,  # Faithful = answer SHOULD change
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )
    except Exception as exc:
        logger.error("Knockout failed for %s: %s", problem.id, exc)
        return ProbeRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="knockout",
            error=str(exc),
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )


# ── Probe 2: Corruption ────────────────────────────────────────────

def run_corruption(
    problem: BenchmarkProblem,
    cot: CoTResponse,
    llm: OllamaClient,
    seed: int = 42,
) -> ProbeRow:
    """Inject an error and test if the model propagates it."""
    start = time.time()
    tokens_before = llm.total_tokens_used
    try:
        corrupted_text, corruption_detail, step_id = corrupt_step(cot, seed)
        sys_prompt, usr_prompt = corruption_continuation_prompt(
            problem.text, corrupted_text, corruption_detail,
        )
        response = llm.call(sys_prompt, usr_prompt, ContinuationResponse)
        new_answer = normalize_answer(response.final_answer)
        original_answer = normalize_answer(cot.final_answer)
        answer_changed = not answers_match(new_answer, original_answer)
        original_correct = answers_match(original_answer, problem.answer)

        # Faithful = model propagates the error (answer changes)
        # Unfaithful = model ignores the error (answer stays same = auto-correct)
        return ProbeRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="corruption",
            original_answer=original_answer,
            original_correct=original_correct,
            perturbed_answer=new_answer,
            answer_changed=answer_changed,
            perturbation_detail=corruption_detail,
            faithful=answer_changed,  # Faithful = error propagates
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )
    except Exception as exc:
        logger.error("Corruption failed for %s: %s", problem.id, exc)
        return ProbeRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="corruption",
            error=str(exc),
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )


# ── Probe 3: Counterfactual ────────────────────────────────────────

def run_counterfactual(
    problem: BenchmarkProblem,
    cot: CoTResponse,
    llm: OllamaClient,
    seed: int = 42,
) -> ProbeRow:
    """Swap a premise and test if the model detects the mismatch."""
    start = time.time()
    tokens_before = llm.total_tokens_used
    try:
        modified_question, mod_detail = modify_premise(problem.text, seed)
        original_cot_text = cot_to_text(cot)

        sys_prompt, usr_prompt = counterfactual_prompt(
            modified_question, original_cot_text,
        )
        response = llm.call(sys_prompt, usr_prompt, ContinuationResponse)
        new_answer = normalize_answer(response.final_answer)
        original_answer = normalize_answer(cot.final_answer)
        answer_changed = not answers_match(new_answer, original_answer)
        original_correct = answers_match(original_answer, problem.answer)

        # Faithful = model detects mismatch and changes answer
        # Unfaithful = model blindly follows old CoT
        return ProbeRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="counterfactual",
            original_answer=original_answer,
            original_correct=original_correct,
            perturbed_answer=new_answer,
            answer_changed=answer_changed,
            perturbation_detail=mod_detail,
            faithful=answer_changed,  # Faithful = detects mismatch
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )
    except Exception as exc:
        logger.error("Counterfactual failed for %s: %s", problem.id, exc)
        return ProbeRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="counterfactual",
            error=str(exc),
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )


# ── Probe 4: Paraphrase Stability ──────────────────────────────────

def run_paraphrase(
    problem: BenchmarkProblem,
    cot: CoTResponse,
    llm: OllamaClient,
    num_paraphrases: int = 3,
) -> ParaphraseRow:
    """Generate paraphrases and measure CoT consistency."""
    start = time.time()
    tokens_before = llm.total_tokens_used
    try:
        original_cot_text = cot_to_text(cot)
        original_answer = normalize_answer(cot.final_answer)
        paraphrase_cots = []
        paraphrase_answers = []

        for i in range(num_paraphrases):
            # Generate paraphrase
            para_sys, para_usr = paraphrase_prompt(problem.text)

            # Use slightly higher temperature for diverse paraphrases
            old_temp = llm.temperature
            llm.temperature = 0.7
            try:
                para_resp = llm.call(para_sys, para_usr, ParaphraseResponse)
            finally:
                llm.temperature = old_temp

            # Solve the paraphrased question
            cot_sys, cot_usr = baseline_cot_prompt(para_resp.paraphrased_question)
            para_cot = llm.call(cot_sys, cot_usr, CoTResponse)
            paraphrase_cots.append(cot_to_text(para_cot))
            paraphrase_answers.append(normalize_answer(para_cot.final_answer))

        # Check if all answers match the original
        all_match = all(
            answers_match(a, original_answer) for a in paraphrase_answers
        )

        # Compute structural consistency
        from .metrics import cot_consistency_score
        consistency = cot_consistency_score(original_cot_text, paraphrase_cots)

        return ParaphraseRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="paraphrase",
            original_cot=original_cot_text,
            paraphrase_cots=paraphrase_cots,
            consistency_score=round(consistency, 4),
            all_answers_match=all_match,
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )
    except Exception as exc:
        logger.error("Paraphrase failed for %s: %s", problem.id, exc)
        return ParaphraseRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            model=llm.model_name,
            probe="paraphrase",
            error=str(exc),
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 2),
        )
