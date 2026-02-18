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
MODELS = [
    "llama3.1:8b",
    "mistral:7b",
    "phi3:mini",
    "gemma2:9b",
    "qwen2.5:7b",
]
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
    "question_only",        # Probe 3b: control — swap premise, no CoT
    "paraphrase",           # Probe 4: rephrase question, compare CoTs
]
