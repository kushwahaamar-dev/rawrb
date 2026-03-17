# RAWRB: Right Answer, Wrong Reason Benchmark

**A Diagnostic Framework for Chain-of-Thought Faithfulness in Small Language Models**

> ACL 2026 Student Research Workshop (Long Paper, 8 pages)

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| **Faithfulness--capability tradeoff** | Medium-tier models: +20pp accuracy but -11 to -18pp faithfulness (Spearman rho = -0.90, p = 0.037; permutation test p < 0.0001, non-overlapping CIs) |
| **CoT anchoring effect** | Stale CoT suppresses answer-change rates by 50--70pp vs question-only baseline |
| **Prompt sensitivity confound** | Consistency-checking instruction inflates counterfactual scores by 42--67pp (Cohen's h up to 1.85) |
| **Right answer, wrong reason** | Only 4--14% of problems are simultaneously correct AND faithful |
| **Harder = more faithful** | MATH Level 5 (hardest): 42% faithful vs Level 1 (easiest): 34% |

---

## Overview

Small language models (<=9B parameters) increasingly use Chain-of-Thought (CoT) prompting to solve multi-step reasoning tasks. But **how faithful are their explanations?** When a model produces a correct answer alongside a well-structured CoT, is the CoT *causally responsible* for the answer -- or is the model generating a plausible post-hoc rationalization?

RAWRB answers this with **4 causal faithfulness probes**, a **question-only control**, and an **unbiased prompt ablation** that systematically perturb CoT reasoning and measure whether models respond appropriately.

---

## Faithfulness Probes

| # | Probe | What It Tests | Faithful Model Should... |
|---|-------|---------------|--------------------------|
| 1 | **Knockout** | Remove a computationally critical step | ...change its answer |
| 2 | **Corruption** | Inject a numeric error (>=30% change) | ...propagate the error |
| 3 | **Counterfactual** | Swap a premise, keep old CoT (with consistency hint) | ...detect the mismatch |
| 3a | **Counterfactual (Unbiased)** | Same but WITHOUT consistency hint | ...detect independently |
| 3b | **Question-Only** | Modified question without CoT (control) | -- (baseline control) |
| 4 | **Paraphrase** | Rephrase the question | ...produce consistent CoTs |

---

## Models

Five open-source models across two size tiers, all run locally via [Ollama](https://ollama.com):

| Tier | Model | Parameters | Organization |
|------|-------|------------|--------------|
| Small | Qwen-2.5 | 1.5B | Alibaba |
| Small | Llama-3.2 | 3.2B | Meta |
| Small | Phi-3 Mini | 3.8B | Microsoft |
| Medium | Llama-3.1 | 8.0B | Meta |
| Medium | Qwen-2.5 | 7.6B | Alibaba |

Qwen-2.5 appears in both tiers, enabling a controlled within-family comparison (all 6 benchmark x probe deltas are negative -- scaling consistently reduces faithfulness).

```bash
ollama pull qwen2.5:1.5b
ollama pull llama3.2:latest
ollama pull phi3:mini
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

---

## Benchmarks

| Benchmark | Domain | Sample Size | Contamination Risk | Source |
|-----------|--------|-------------|-------------------|--------|
| **GSM8K** | Grade-school arithmetic | 200 | High | [Cobbe et al., 2021](https://arxiv.org/abs/2110.14168) |
| **MATH** | Competition mathematics (7 subjects, Levels 1-5) | 200 | Moderate | [Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874) |
| **FOLIO** | First-order logic | 200 | Low | [Han et al., 2022](https://arxiv.org/abs/2209.00840) |

The three-benchmark design enables contamination-sensitivity analysis. All samples drawn with `random.Random(42)`.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- At least one model pulled (see above)

### Installation

```bash
git clone https://github.com/<your-username>/rawrb.git
cd rawrb
pip install -r requirements.txt
```

### Running Experiments

```bash
# Smoke test -- 5 problems, 1 model, all probes (~5 min)
python run_experiment.py --smoke

# Single model, all probes
python run_experiment.py --models qwen2.5:7b

# Full run -- all 5 models x 3 benchmarks x 7 probes (~15 hours)
python run_experiment.py
```

### Running Analysis

```bash
# Full pipeline: tables + figures + stats + derived metrics
python run_analysis.py

# Individual components
python run_analysis.py --tables     # textual tables to stdout
python run_analysis.py --figures    # generate 4 PDF/PNG figures
python run_analysis.py --metrics    # Spearman, concordance, permutation test, etc.
```

---

## Project Structure

```
rawrb/
├── run_experiment.py          # Experiment orchestrator (baseline + 6 probes)
├── run_analysis.py            # Unified analysis: tables, figures, stats, derived metrics
├── requirements.txt           # Python dependencies
│
├── src/
│   ├── config.py              # Models, paths, seeds, probe names
│   ├── models.py              # Pydantic schemas (CoTResponse, ProbeRow, etc.)
│   ├── llm_client.py          # Ollama API client with retry + JSON parsing
│   ├── benchmarks.py          # GSM8K, MATH, FOLIO dataset loaders
│   ├── prompts.py             # Prompt templates for all probe types
│   ├── perturbations.py       # Knockout, corruption, counterfactual logic
│   ├── probes.py              # 7 probe implementations
│   └── metrics.py             # Answer matching, bootstrap CI, McNemar, Cohen's h
│
├── results/                   # Experiment outputs
│   ├── results_{model}_{probe}.csv   # Per-problem results (36 files)
│   ├── cot_cache_{model}.json        # CoT cache for deterministic resume
│   ├── stats_summary.csv             # BH-corrected statistical summary
│   ├── prompt_sensitivity.csv        # CF vs CF_U paired comparison
│   └── qualitative_exemplars.json    # Auto-captured interesting examples
│
└── figures/                   # Generated visualizations
    ├── accuracy_vs_faithfulness.{png,pdf}
    ├── probe_heatmap.{png,pdf}
    ├── faithfulness_gap.{png,pdf}
    ├── paraphrase_consistency.{png,pdf}
    └── stats_table.tex
```

---

## Results Summary

### Aggregate Faithfulness (%)

| Model | KO | COR | CF | CF_U | QO | PAR |
|-------|-----|------|-----|-------|-----|------|
| Qwen-2.5-1.5B | 35.9 | 29.6 | 5.0 | 2.6 | 71.2 | 58.8 |
| Llama-3.2 | 26.2 | 20.9 | 79.8 | 16.6 | 73.9 | 47.2 |
| Phi-3 Mini | 41.1 | 28.8 | 65.8 | 15.0 | 71.3 | 54.4 |
| Llama-3.1 | 23.9 | 19.9 | 78.4 | 11.8 | 76.5 | 48.4 |
| Qwen-2.5-7B | 9.7 | 11.5 | 57.7 | 15.6 | 68.4 | 59.8 |
| *Small tier* | *34.4* | *26.5* | *50.2* | *11.4* | *72.2* | *53.5* |
| *Medium tier* | *16.8* | *15.7* | *68.1* | *13.7* | *72.4* | *54.1* |

KO=knockout, COR=corruption, CF=counterfactual (with hint), CF_U=unbiased, QO=question-only, PAR=paraphrase.

### Derived Metrics

| Metric | Value | Significance |
|--------|-------|-------------|
| Spearman rho (accuracy vs KO faithfulness) | -0.900 | p = 0.037 |
| Spearman rho (accuracy vs COR faithfulness) | -1.000 | p < 0.001 |
| Tier difference (knockout, permutation test) | 17.7pp | p < 0.0001, non-overlapping CIs |
| Tier difference (corruption, permutation test) | 10.8pp | p < 0.0001, non-overlapping CIs |
| Concordance rate (correct AND faithful) | 4--14% | Qwen-2.5-7B lowest at 4.4% |
| Probe agreement (KO vs COR) | 68--89% | Cohen's kappa 0.29--0.39 |

---

## Statistical Methodology

| Method | Purpose |
|--------|---------|
| Bootstrap CIs (n=10,000) | 95% confidence intervals for all rates |
| Bootstrap gap test (n=10,000) | Tests H0: accuracy - faithfulness = 0 |
| Benjamini-Hochberg (FDR=0.05) | Multiple comparison correction across 15 tests |
| Paired McNemar's test | CF vs QO and CF vs CF_U comparisons |
| Cohen's h | Effect sizes for proportion comparisons |
| Spearman rho | Accuracy-faithfulness correlation across models |
| Permutation test (n=10,000) | Tier difference significance |
| Stratified analysis | Faithfulness conditional on baseline correctness |

---

## Reproducibility

| Feature | Implementation |
|---------|---------------|
| Deterministic sampling | `random.Random(42)` in all loaders |
| Temperature | T=0.0 for all probes (T=0.7 for paraphrase generation) |
| CoT caching | Baseline CoTs serialized to JSON for deterministic resume |
| Crash recovery | CSV flushed after every row; completed problem_ids skipped |
| Local inference | All models via Ollama -- no API keys, no cloud |
| Deduplication | Error-first, then dedup by (model, benchmark, problem_id) |

> **Note**: GPU non-determinism may cause minor variation at T=0.0. Llama-3.1-8B was run twice (MATH/FOLIO); duplicate rows are handled by the deduplication pipeline.

---

## Citation

If you use RAWRB in your work, please cite the associated paper (details to follow upon publication).

---

## License

This project is for academic research purposes. Please cite appropriately if you use RAWRB in your work.
