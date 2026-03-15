"""Global configuration: paths, model names, experiment constants."""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
PAPER_DIR = ROOT_DIR / "paper"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── Models (Ollama) ─────────────────────────────────────────────────
# We study faithfulness across 5 model families
# NOTE: Ollama uses quantized models (typically Q4_K_M). The exact
# quantization depends on the Ollama model tag. Pin Ollama version
# and model digests for exact reproducibility.
# Two tiers for scale analysis:
#   Small tier (1.5-3.8B): tests whether faithfulness emerges with scale
#   Medium tier (7-9B): primary evaluation targets
MODELS_SMALL = [
    "qwen2.5:1.5b",       # 1.5B Q4_K_M
    "llama3.2:latest",     # 3.2B Q4_K_M
    "phi3:mini",           # 3.8B Q4_K_M
]
MODELS_MEDIUM = [
    "llama3.1:8b",         # 8.0B Q4_K_M
    "qwen2.5:7b",          # 7.6B Q4_K_M
]
MODELS = MODELS_SMALL + MODELS_MEDIUM
DEFAULT_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Experiment settings ─────────────────────────────────────────────
TEMPERATURE = 0.0           # deterministic for main probes
PARAPHRASE_TEMPERATURE = 0.7  # used for paraphrase generation
MAX_TOKENS = 2048
NUM_PARAPHRASES = 3         # Probe 4: paraphrase stability

# ── Benchmark sample sizes ──────────────────────────────────────────
BENCHMARK_SIZES = {
    "gsm8k": 200,
    "math": 200,
    "folio": 200,
}

# ── Incremental save ────────────────────────────────────────────────
FLUSH_EVERY = 1             # flush CSV after every row (crash safety)
SEED = 42                   # reproducibility

# ── Probes ──────────────────────────────────────────────────────────
PROBE_NAMES = [
    "baseline",             # Standard CoT (no perturbation)
    "knockout",             # Probe 1: remove a critical step
    "corruption",           # Probe 2: inject numeric/logical error
    "counterfactual",       # Probe 3: swap premise, keep old CoT
    "counterfactual_unbiased",  # Probe 3a: same as 3 but no consistency hint
    "question_only",        # Probe 3b: control — swap premise, no CoT
    "paraphrase",           # Probe 4: rephrase question, compare CoTs
]
