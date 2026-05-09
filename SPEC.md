# Project Specification

> **Status: Stub.** This document fills out as the project progresses.

## Problem statement

Modern instruction-tuned LLMs answer questions in a single default register. Real users — students, professionals, kids — need answers calibrated to their level. This project investigates whether small open LLMs (under 400M parameters) can be **parameter-efficiently fine-tuned** to control answer register conditioned on an explicit audience tag.

## Goals

1. Demonstrate audience-conditioned generation on two small base models with QLoRA.
2. Compare decoder-only (SmolLM2-360M) vs encoder-decoder (flan-t5-small) under the same training data and hyperparameters.
3. Build a reusable evaluation harness comparing base vs fine-tuned outputs.
4. Document attention mechanics with a from-scratch implementation and visualizations.

## Non-goals

- Beating frontier models on accuracy.
- Production deployment, API serving, or scaling beyond a single GPU.
- Novel architectural research.

## Datasets

- **Source**: [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k), filtered to `open_qa` items whose instructions are recognizable questions (contain "why", "how", "what is", "explain").
- **Augmentation**: a teacher LLM (Nebius Token Factory, model TBD at Task 1.1) rewrites each filtered question's answer at three levels: child, student, expert.
- **Splits**: ~200 source questions × 3 levels = ~600 training rows. 20-row evaluation set hand-curated separately.

## Models

| | Architecture | Parameters | HuggingFace ID |
|---|---|---|---|
| Causal | decoder-only | 362M | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| Seq2seq | encoder-decoder | 77M | `google/flan-t5-small` |

## Method

- **Quantization**: 4-bit NF4 via `bitsandbytes`, double quantization enabled.
- **LoRA** (initial config — to be tuned): rank 16, alpha 32, dropout 0.05.
- **Target modules**: TBD per architecture (attention projections at minimum).
- **Training**: 1–3 epochs, learning rate 2e-4, effective batch size 16.
- **Trainer**: `trl.SFTTrainer` for the causal model; `transformers.Seq2SeqTrainer` for flan-t5.

## Evaluation protocol

- 20-example held-out test set, evenly distributed across child/student/expert levels.
- For each test item: generate from base model and fine-tuned model on both architectures (= 80 generations).
- Manual scoring on three axes:
  - **Level adherence** — does the register match the requested audience?
  - **Clarity** — is the answer well-formed?
  - **Faithfulness** — does it answer the question without drifting?

## Reproducibility

- Random seeds fixed (training + generation).
- `requirements.txt` pinned to working versions.
- Trained adapters published to HuggingFace Hub; revision hash recorded in `RESULTS.md`.

## Risks & known limitations

- Teacher-LLM-generated training data inherits the teacher's biases and stylistic quirks.
- 20-example test set is too small for statistical claims — results are illustrative, not benchmark-grade.
- flan-t5-small is borderline too small for nuanced instruction following; results may underwhelm regardless of training quality.
