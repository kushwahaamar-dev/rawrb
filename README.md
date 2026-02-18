# RAWRB: Right Answer, Wrong Reason Benchmark

**A Diagnostic Framework for Chain-of-Thought Faithfulness in Small Language Models**

> 🎯 Target: ACL 2026 Student Research Workshop — Submission Deadline March 18, 2026  
> 🏷️ Theme: *Explainability of NLP Models* (ACL 2026 Special Theme)

---

## Overview

Small language models (≤9B parameters) increasingly use Chain-of-Thought (CoT) prompting to solve multi-step reasoning tasks. But **how faithful are their explanations?** When a model produces a correct answer alongside a well-structured CoT, is the CoT *causally responsible* for the answer — or is the model just generating a plausible post-hoc rationalization?

RAWRB answers this by introducing **4 causal faithfulness probes** plus a **question-only control** that systematically perturb CoT reasoning and measure whether models respond appropriately.

### The Faithfulness Gap

```
Faithfulness Gap = Task Accuracy − Mean Faithfulness Score
```

A large gap means the model *gets the right answer for the wrong reason* — it produces correct outputs while its explanations bear little causal relationship to those outputs. RAWRB quantifies this gap across models, benchmarks, and probe types.

---

## Faithfulness Probes

| # | Probe | What It Tests | Faithful Model Should… |
|---|-------|---------------|------------------------|
| 1 | **Knockout** | Remove a computationally critical step | …change its answer |
| 2 | **Corruption** | Inject a numeric error (≥30% change) | …propagate the error |
| 3 | **Counterfactual** | Swap a premise, keep old CoT | …detect the mismatch |
| 3b | **Question-Only** | Modified question without CoT (control) | — (control condition) |
| 4 | **Paraphrase** | Rephrase the question | …produce consistent CoTs |

### Probe Design Details

- **Knockout** preferentially targets steps containing numerical operations or logical deductions (not setup/restatement steps), maximizing causal relevance.
- **Corruption** guarantees a minimum 30% relative change in the corrupted value, ensuring the perturbation is non-trivial. The continuation prompt is *neutral* — no bias toward accepting or rejecting the reasoning.
- **Counterfactual** modifies numeric values in math problems and swaps logical quantifiers (all↔none, every↔no) for logic problems (FOLIO), with case-insensitive matching.
- **Question-Only** serves as a control for the counterfactual probe: it provides the modified question *without* any CoT. Comparing counterfactual vs. question-only rates reveals whether answer changes stem from genuine CoT engagement or simple question comprehension.
- **Paraphrase** generates 3 semantically equivalent rephrasings per question and measures CoT structural consistency via Jaccard similarity of step-level keyword sets.

---

## Models Under Study

Five open-source model families spanning four organizations, all runnable locally via [Ollama](https://ollama.com):

| Model | Organization | Parameters |
|-------|-------------|------------|
| Llama 3.1 8B | Meta | 8B |
| Mistral 7B | Mistral AI | 7B |
| Phi-3 Mini | Microsoft | 3.8B |
| Gemma 2 9B | Google | 9B |
| Qwen 2.5 7B | Alibaba | 7B |

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull phi3:mini
ollama pull gemma2:9b
ollama pull qwen2.5:7b
```

---

## Benchmarks

| Benchmark | Domain | Sample Size | Answer Type | Source |
|-----------|--------|-------------|-------------|--------|
| **GSM8K** | Grade-school arithmetic | 200 | Numeric | [Cobbe et al., 2021](https://arxiv.org/abs/2110.14168) |
| **MATH** | Competition mathematics | 200 | Symbolic/LaTeX | [Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874) |
| **FOLIO** | First-order logic | 200 | True/False/Unknown | [Han et al., 2022](https://arxiv.org/abs/2209.00840) |

All samples drawn with fixed seed (`SEED=42`) using instance-level `random.Random(SEED)` for full reproducibility.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- At least one model pulled (see above)

### Installation

```bash
git clone https://github.com/kushwahaamar-dev/rarwb.git
cd rarwb
pip install -r requirements.txt
```

### Running Experiments

```bash
# Smoke test — 5 problems, 1 model, all probes (~5 min)
python run_experiment.py --smoke

# Single model, all probes
python run_experiment.py --models qwen2.5:7b

# Specific benchmark and probes
python run_experiment.py --benchmarks gsm8k --probes baseline knockout corruption

# Full run — all 5 models × 3 benchmarks × 6 probes (~15 hours)
python run_experiment.py

# Generate figures, statistical tables, and analysis
python run_analysis.py
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--models` | Space-separated model names | All 5 models (or 1 for `--smoke`) |
| `--benchmarks` | `gsm8k`, `math`, `folio` | All 3 |
| `--probes` | `baseline`, `knockout`, `corruption`, `counterfactual`, `question_only`, `paraphrase` | All 6 |
| `--smoke` | Quick test: 5 problems, 1 model | Off |
| `--limit N` | Max problems per benchmark | None (uses config) |

---

## Project Structure

```
rawrb/
├── run_experiment.py          # Main experiment orchestrator
│                              #   - Baseline CoT generation with JSON caching
│                              #   - Probe dispatch for all 5 non-baseline probes
│                              #   - Resumable (skips completed problem_ids)
│                              #   - ETA logging for runtime estimates
│
├── run_analysis.py            # Analysis and visualization pipeline
│                              #   - Accuracy vs faithfulness scatter plots
│                              #   - Probe × model heatmaps
│                              #   - Faithfulness gap bar charts
│                              #   - Paraphrase consistency box plots
│                              #   - LaTeX-ready stats tables with bootstrap CIs
│                              #   - Counterfactual vs question-only comparison
│
├── requirements.txt           # Python dependencies
├── SMOKE_TEST_REPORT.md       # Proof-of-concept report from initial testing
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Global config: models, paths, seeds, probe names
│   ├── models.py              # Pydantic schemas: BenchmarkProblem, CoTResponse,
│   │                          #   BaselineRow, ProbeRow, ParaphraseRow, enums
│   ├── llm_client.py          # Ollama REST API wrapper with structured JSON output,
│   │                          #   retry logic (tenacity), and usage tracking
│   ├── benchmarks.py          # Dataset loaders for GSM8K, MATH, FOLIO
│   │                          #   - Deterministic sampling: random.Random(SEED)
│   │                          #   - Answer extraction: #### for GSM8K, \boxed{} for MATH
│   ├── prompts.py             # All 5 prompt templates (system + user pairs)
│   │                          #   - Neutral framing (no bias in corruption prompt)
│   ├── perturbations.py       # Perturbation engine:
│   │                          #   - knockout_step: remove computational step
│   │                          #   - corrupt_step: inject ≥30% numeric error
│   │                          #   - modify_premise: numeric or keyword swap
│   ├── probes.py              # 6 probe implementations:
│   │                          #   - run_baseline, run_knockout, run_corruption,
│   │                          #   - run_counterfactual, run_question_only, run_paraphrase
│   │                          #   - Paraphrase validates overlap against questions
│   └── metrics.py             # Scoring and statistics:
│                              #   - normalize_answer: whitespace, prefix, LaTeX
│                              #   - answers_match: numeric, LaTeX, word-boundary
│                              #   - faithfulness_score, faithfulness_gap
│                              #   - bootstrap_ci (n=10,000), McNemar's test
│                              #   - cohen's h effect size
│
├── paper/
│   ├── acl2026_srw.tex        # ACL SRW LaTeX source (9 sections)
│   └── acl2026_srw.bib        # Bibliography (12 entries)
│
├── results/                   # CSV outputs (auto-created)
│   └── cot_cache_*.json       # CoT cache for deterministic resume
│
└── figures/                   # Generated plots (auto-created)
    ├── accuracy_vs_faithfulness.{png,pdf}
    ├── probe_heatmap.{png,pdf}
    ├── faithfulness_gap.{png,pdf}
    ├── paraphrase_consistency.{png,pdf}
    ├── stats_table.tex
    └── stats_summary.csv
```

---

## Statistical Methodology

- **Bootstrap confidence intervals**: 95% CIs with 10,000 resamples for all reported metrics
- **McNemar's test**: Paired comparison of accuracy vs. faithfulness on matched problem pairs, applied per-probe
- **Cohen's h effect size**: Quantifies practical significance of faithfulness differences
- **Error exclusion**: Rows with LLM errors are excluded from analysis to prevent biasing scores

---

## Reproducibility

RAWRB is designed for full reproducibility:

| Feature | Implementation |
|---------|---------------|
| **Deterministic sampling** | Instance-level `random.Random(42)` in all benchmark loaders |
| **Temperature** | `T=0.0` for all probes (except paraphrase generation: `T=0.7`) |
| **CoT caching** | Baseline CoT responses serialized to JSON; resume runs use identical CoTs |
| **Crash recovery** | CSV results flushed after every row; completed problem_ids are skipped |
| **Local inference** | All models run via Ollama — no API keys, no cloud dependency |

> ⚠️ **Note**: GPU non-determinism may introduce minor variation even at `T=0.0`. We document this limitation in the paper.

---

## Key Design Decisions

1. **Causal faithfulness definition**: Following [Jacovi & Goldberg (2020)](https://aclanthology.org/2020.acl-main.386/), an explanation is faithful if it is *causally responsible* for the output — not merely plausible.

2. **Question-only control**: The counterfactual probe alone cannot distinguish between "model detects CoT mismatch" and "model just answers the new question." The question-only control disentangles these.

3. **Neutral corruption prompt**: The corruption continuation prompt does not instruct the model to accept or reject the reasoning — it simply asks "what is the final answer?" This avoids biasing toward artificial faithfulness.

4. **Computational step targeting**: Knockout preferentially removes steps containing numeric operations or logical keywords, avoiding trivial setup steps that wouldn't change the answer regardless.

5. **Minimum corruption delta**: Number corruptions enforce a ≥30% relative change, ensuring the perturbation is large enough that propagation (or lack thereof) is meaningful.

---

## Paper

The LaTeX source for the ACL 2026 SRW submission is in `paper/`:

- **9 sections**: Introduction, Related Work, Methodology, Experiments, Analysis, Discussion, Ethical Considerations, Limitations, Conclusion
- **Pre-registered methodology**: All probe designs, metrics, and statistical tests documented before experiment execution
- **12 bibliography entries**: Wei et al. (CoT), Turpin et al. (unfaithfulness), Lanham et al. (measuring faithfulness), Jacovi & Goldberg (definitions), and benchmark citations

---

## Output Files

After running experiments and analysis:

| File | Description |
|------|-------------|
| `results/results_{model}_{probe}.csv` | Raw per-problem results |
| `results/cot_cache_{model}.json` | Cached CoT objects for resume |
| `results/stats_summary.csv` | Aggregate statistics table |
| `figures/accuracy_vs_faithfulness.png` | Accuracy vs faithfulness scatter |
| `figures/probe_heatmap.png` | Model × probe faithfulness heatmap |
| `figures/faithfulness_gap.png` | Gap bar chart with Δ annotations |
| `figures/paraphrase_consistency.png` | CoT consistency box plots |
| `figures/stats_table.tex` | LaTeX-ready statistics table |

---

## Dependencies

```
requests>=2.31.0      # Ollama REST API
pydantic>=2.0.0       # Structured schemas & validation
tenacity>=8.0.0       # Retry with exponential backoff
datasets>=2.14.0      # HuggingFace dataset loaders
tqdm>=4.65.0          # Progress bars
numpy>=1.24.0         # Numerical operations
scipy>=1.11.0         # Statistical tests (McNemar)
matplotlib>=3.7.0     # Publication-quality plots
seaborn>=0.12.0       # Statistical visualizations
pandas>=2.0.0         # Data manipulation
```

---

## Citation

```bibtex
@inproceedings{rawrb2026,
  title={Right Answer, Wrong Reason: A Diagnostic Benchmark for
         Chain-of-Thought Faithfulness in Small Language Models},
  author={Kushwaha, Amar},
  booktitle={Proceedings of the ACL 2026 Student Research Workshop},
  year={2026}
}
```

---

## License

This project is for academic research purposes. Please cite appropriately if you use RAWRB in your work.
