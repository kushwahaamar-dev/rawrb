# RAWRB: Right Answer, Wrong Reason Benchmark

**A Diagnostic Framework for Chain-of-Thought Faithfulness in Small Language Models**

> Target: ACL 2026 Student Research Workshop — Submission Deadline March 18, 2026
> Theme Alignment: ✅ *Explainability of NLP Models* (ACL 2026 Special Theme)

## Overview

Small language models (≤9B parameters) increasingly use Chain-of-Thought (CoT) prompting for reasoning tasks. But **how faithful are their explanations?** RAWRB probes whether CoT steps are *causally connected* to the final answer, or whether models are merely generating plausible rationalizations.

We introduce **4 faithfulness probes** that test causal reliance on reasoning steps:

| Probe | What It Tests | Faithful Model Should… |
|-------|-------------|----------------------|
| **Knockout** | Remove a critical step | …change its answer |
| **Corruption** | Inject a numeric error | …propagate the error |
| **Counterfactual** | Swap a premise, keep old CoT | …detect the mismatch |
| **Paraphrase** | Rephrase the question | …produce consistent CoTs |

## Models Under Study

All models run locally via [Ollama](https://ollama.com):

```bash
ollama pull llama3.1:8b      # Meta
ollama pull mistral:7b        # Mistral AI
ollama pull phi3:mini          # Microsoft (3.8B)
ollama pull gemma2:9b          # Google (9B)
ollama pull qwen2.5:7b         # Alibaba
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Smoke test (5 problems, 1 model, all probes)
python run_experiment.py --smoke

# Full run — single model
python run_experiment.py --models qwen2.5:7b

# Full run — all 5 models (takes ~24-48 hours)
python run_experiment.py

# Generate figures and statistical tables
python run_analysis.py
```

## Benchmarks

| Benchmark | Domain | Size | Answer Type |
|-----------|--------|------|-------------|
| GSM8K | Arithmetic reasoning | 200 | Numeric |
| MATH | Competition math | 200 | Symbolic |
| FOLIO | First-order logic | 200 | True/False/Unknown |

## Project Structure

```
rawrb/
├── run_experiment.py       # Main experiment orchestrator
├── run_analysis.py         # Generate figures + stats
├── requirements.txt
├── src/
│   ├── config.py           # Models, paths, experiment settings
│   ├── models.py           # Pydantic schemas
│   ├── llm_client.py       # Ollama API wrapper
│   ├── benchmarks.py       # GSM8K, MATH, FOLIO loaders
│   ├── prompts.py          # All prompt templates
│   ├── perturbations.py    # Perturbation engine (knockout, corrupt, counterfactual)
│   ├── probes.py           # The 4 faithfulness probes
│   └── metrics.py          # Scoring, bootstrap CI, McNemar's test
├── paper/                  # ACL SRW LaTeX source
├── results/                # Raw CSV outputs
└── figures/                # Generated plots
```

## Key Metric: The Faithfulness Gap

```
Faithfulness Gap = Task Accuracy − Mean Faithfulness Score
```

A large gap means the model "gets the right answer for the wrong reason" — it produces correct outputs while its explanations bear little causal relationship to those outputs.

## Citation

```bibtex
@inproceedings{rawrb2026,
  title={Right Answer, Wrong Reason: A Diagnostic Benchmark for Chain-of-Thought Faithfulness in Small Language Models},
  author={[Your Name]},
  booktitle={Proceedings of the ACL 2026 Student Research Workshop},
  year={2026}
}
```
