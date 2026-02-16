"""Pydantic schemas for the RAWRB faithfulness benchmark pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ───────────────────────────────────────────────────────────

class BenchmarkName(str, Enum):
    GSM8K = "gsm8k"
    MATH = "math"
    FOLIO = "folio"


class ProbeName(str, Enum):
    BASELINE = "baseline"
    KNOCKOUT = "knockout"
    CORRUPTION = "corruption"
    COUNTERFACTUAL = "counterfactual"
    PARAPHRASE = "paraphrase"


# ── Benchmark ───────────────────────────────────────────────────────

class BenchmarkProblem(BaseModel):
    """Unified problem representation across all benchmarks."""
    id: str
    benchmark: BenchmarkName
    text: str                              # full problem statement
    answer: str                            # ground-truth answer (string)
    difficulty: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


# ── LLM Structured Output Schemas ───────────────────────────────────

class ReasoningStep(BaseModel):
    """A single step in a chain-of-thought."""
    step_id: int
    reasoning: str                         # natural-language explanation
    conclusion: str                        # what this step concludes

    @field_validator("conclusion", "reasoning", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        return str(v) if not isinstance(v, str) else v


class CoTResponse(BaseModel):
    """Full chain-of-thought response from the model."""
    steps: list[ReasoningStep]
    final_answer: str

    @field_validator("final_answer", mode="before")
    @classmethod
    def coerce_final(cls, v):
        return str(v) if not isinstance(v, str) else v


class ContinuationResponse(BaseModel):
    """Model's answer when given a partial/corrupted CoT to continue from."""
    continued_reasoning: str = ""
    final_answer: str

    @field_validator("final_answer", mode="before")
    @classmethod
    def coerce_final(cls, v):
        return str(v) if not isinstance(v, str) else v


class ParaphraseResponse(BaseModel):
    """Paraphrased version of a question."""
    paraphrased_question: str


# ── Experiment Rows ─────────────────────────────────────────────────

class BaselineRow(BaseModel):
    """One row of baseline CoT results."""
    benchmark: str
    problem_id: str
    model: str
    probe: str = "baseline"
    question: str = ""
    cot_text: str = ""                     # full CoT as text
    cot_steps: int = 0
    predicted_answer: str = ""
    ground_truth: str = ""
    correct: bool = False
    tokens_used: int = 0
    latency_s: float = 0.0
    error: str = ""


class ProbeRow(BaseModel):
    """One row of probe results (knockout, corruption, counterfactual)."""
    benchmark: str
    problem_id: str
    model: str
    probe: str                             # which probe type
    original_answer: str = ""              # from baseline CoT
    original_correct: bool = False
    perturbed_answer: str = ""             # answer after perturbation
    answer_changed: bool = False           # did perturbation change the answer?
    perturbation_detail: str = ""          # what was changed
    faithful: bool = False                 # interpretation: was the model faithful?
    tokens_used: int = 0
    latency_s: float = 0.0
    error: str = ""


class ParaphraseRow(BaseModel):
    """One row of paraphrase stability results (Probe 4)."""
    benchmark: str
    problem_id: str
    model: str
    probe: str = "paraphrase"
    original_cot: str = ""
    paraphrase_cots: list[str] = Field(default_factory=list)
    consistency_score: float = 0.0         # Jaccard similarity of step structures
    all_answers_match: bool = False
    tokens_used: int = 0
    latency_s: float = 0.0
    error: str = ""
