# Technical Specification

This document captures the design and the as-built configuration of the QLoRA fine-tuning and attention-implementation project. Headline results live in [README.md](README.md); this file is for the *how* and *why*.

## Problem statement

Instruction-tuned LLMs answer questions in a single default register. Users (students, professionals, kids) often need answers calibrated to their level. This project tests whether **small open LLMs (<400M parameters)** can be parameter-efficiently fine-tuned to control answer register, conditioned on an explicit audience tag, on a modest budget (one consumer / Colab GPU, ~10 MB of adapter weights, no full fine-tune).

## Goals

1. Demonstrate audience-conditioned generation on two small base models with QLoRA.
2. Compare decoder-only (SmolLM2-360M-Instruct) vs encoder-decoder (flan-t5-small) under matched training data and hyperparameters.
3. Build a transparent evaluation harness that compares base vs fine-tuned outputs side-by-side and persists per-row manual judgments.
4. Implement scaled dot-product attention from scratch in PyTorch and use it to visualize how attention patterns shift with input changes and across heads.

## Non-goals

- Beating frontier models on accuracy.
- Production deployment, API serving, or scaling beyond a single GPU.
- Novel architectural research.

## Dataset

| | |
|---|---|
| Source | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k), `open_qa` split |
| Filter | Question-shaped instructions (contain "why", "how", "what is", "explain"), length bounds on both question and reference answer (length-filtered to keep the teacher LLM's job tractable and answers comparable). |
| Teacher LLM | **Qwen3-30B-A3B-Instruct-2507** via [Nebius Token Factory](https://api.studio.nebius.com/v1) — JSON-schema-enforced output for reliability, one call per question returns all three levels (Option A in [Task 1.1 decisions](.claude/projects/-Users-mghanayim-Dev-Claude-qlora-finetuning-and-attention/memory/task_1_1_decisions.md)). |
| Augmentation | Teacher rewrites each filtered answer at three levels: child, student, expert. |
| Train split | 495 source questions × 3 levels = **1,485 training rows** |
| Test split | 7 hand-curated source questions × 3 levels = **21 evaluation rows**, matched-pair design across levels |
| Persisted as | [data/train.jsonl](data/train.jsonl), [data/test.jsonl](data/test.jsonl), [data/pilot_raw.jsonl](data/pilot_raw.jsonl) (teacher reliability validation) |

The test set is constructed *matched-pair*: every test question appears at all three levels, so per-question across-level comparisons isolate the model's level-adaptation behavior from question difficulty.

## Models

| Role | Architecture | Parameters | HuggingFace ID | Trainable (LoRA) |
|---|---|---|---|---|
| Causal | decoder-only | 362M | [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 0.8% |
| Seq2seq | encoder-decoder | 77M | [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small) | 0.9% |

## Training configuration

### SmolLM2-360M-Instruct (causal QLoRA)

| Setting | Value | Why |
|---|---|---|
| Quantization | 4-bit NF4 + double quant | bitsandbytes standard QLoRA recipe |
| Compute dtype | bf16 end-to-end (`bnb_4bit_compute_dtype=torch.bfloat16`, `bf16=True` in trainer) | SmolLM2 was pretrained in bf16; PEFT casts LoRA adapters to match. Using fp16 GradScaler on bf16 LoRA grads crashes on T4 (`_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`). |
| Preparation | `prepare_model_for_kbit_training` | enables input grads on quantized base + gradient checkpointing |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` | attention projections only (SPEC minimum) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | |
| EOS handling | `tokenizer.eos_token` (`<|im_end|>`) appended in `_render_smollm2` | Llama-style tokenizers don't auto-append; without it the model never learns to stop |
| Loss masking | none (full-text loss) | Fixed-format prompts make prompt-loss roughly constant; dropping completion-only masking avoids BPE-boundary footguns and unlocks packing |
| Packing | `packing=True`, `max_length=384` | Longest real example ~200 tokens; 384 has safety margin |
| Schedule | 2 epochs, lr 2e-4 cosine, warmup 3%, effective batch 16 (per-device 4 × grad-accum 4) | |
| Seed | 42 | |
| Checkpointing | `save_strategy="epoch"` | cheap crash recovery |

Adapter on disk: 9.7 MB at [adapters/smollm2-lora/](adapters/smollm2-lora/).

### flan-t5-small (seq2seq LoRA, no quantization)

Initial attempt used the same 4-bit NF4 recipe as SmolLM2. It failed reproducibly:
- Training loss stuck at ~18-19 across 186 steps (random baseline `ln(32000) ≈ 10.4`).
- Outputs were gibberish loops: `"locationlocationlocationssss"`.
- Load-time warning: `lm_head` was not tied to `shared.weight` because both were present in the checkpoint.

Hypothesis: 4-bit NF4 quantization broke the tied-embeddings invariant flan-t5 relies on. Verified by changing exactly one variable — loading in bf16 instead of 4-bit, all other hyperparameters fixed:

| Setting | Value | Why |
|---|---|---|
| Precision | bf16 (no quantization) | 4-bit broke this checkpoint; SmolLM2 was unaffected by the same NF4 path, so the sensitivity is specific to flan-t5 |
| LoRA target modules | `q`, `v` | seq2seq attention naming differs from SmolLM2 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | matched to SmolLM2 |
| Learning rate | 1e-4 | half of SmolLM2's, empirically more stable on the smaller model |
| Schedule | 2 epochs, effective batch 16, seed 42 | matched |

Adapter on disk: 4.7 MB at [adapters/flant5-lora/](adapters/flant5-lora/).

## Evaluation protocol

- 21-example held-out test set, 7 questions × 3 levels, matched-pair across levels.
- For each (model, test row), generate twice on the *same* loaded model:
  - **base** output: adapter disabled via `model.disable_adapter()`.
  - **fine-tuned** output: adapter enabled (default state).
- Greedy decoding (`do_sample=False`) + `model.eval()` for reproducibility.
- `max_new_tokens=250` (> longest expected expert-level answer); `min_new_tokens=50` for SmolLM2 (forces past its early-EOS tendency so we see real content).
- Manual scoring per row on three axes (Level adherence / Clarity / Faithfulness) collapsed into a single `Better after FT?` verdict: `yes`, `partial`, or `no`.
- Per-row verdicts and notes persisted to [data/eval_smollm2_final.csv](data/eval_smollm2_final.csv) and [data/eval_flant5_final.csv](data/eval_flant5_final.csv); the un-judged scratch CSVs (`data/eval_*.csv`, generated on every training run) are kept separate so reruns can't clobber the judgments.

### Results

| Model | yes | partial | no | Any improvement |
|---|---|---|---|---|
| **SmolLM2-360M** | **13** | 7 | 1 | **20 / 21 (95%)** |
| **flan-t5-small** | 0 | 5 | 16 | 5 / 21 (24%, partial only) |

Differences attributable to architecture and scale (matched LoRA hyperparameters; learning rate is the only architecture-mandated diverging knob). See [README → Headline results](README.md#headline-results) for the qualitative summary and [Notebook cell 61](From_Finetuning_to_Attention_Inside_LLMs.ipynb) for the row-by-row analysis.

## Part 2 — attention from scratch

Pure PyTorch, no model loads, runs on CPU. Three deliverables in the same notebook:

1. **Implementation** ([cell 69](From_Finetuning_to_Attention_Inside_LLMs.ipynb)): `scaled_dot_product_attention(Q, K, V, mask=None)` supporting both 2-D `(seq, d)` and 4-D `(batch, heads, seq, d)` tensor shapes. Mask convention: `mask == 0` positions are set to `-inf` before softmax (Vaswani / Harvard NLP annotated transformer).
2. **Single-sentence heatmap** ([cell 72](From_Finetuning_to_Attention_Inside_LLMs.ipynb)): random 16-dim embeddings used as Q = K = V for `"the cat sat on the mat"`. Diagonal-dominant softmax confirms random vectors are most similar to themselves.
3. **Experiments** (cells [75-79](From_Finetuning_to_Attention_Inside_LLMs.ipynb)):
   - *One-word change* (cat → dog): heatmaps are **bit-identical** under shared seed because random embeddings ignore token identity entirely.
   - *Multi-head with separate `W_Q/W_K/W_V`*: three heads on the same input, each with independently-sampled projection matrices into `d_k=8`. Patterns differ visibly per head, demonstrating the precondition for head specialization (which in trained transformers comes from *learned*, not random, W matrices).

Saved figures: [results/attention_heatmaps/](results/attention_heatmaps/).

## Reproducibility

| Knob | Setting |
|---|---|
| Random seeds | 42 (training, generation, Part 2 plots) |
| Decoding | Greedy (`do_sample=False`) |
| Dependencies | [requirements.txt](requirements.txt), pinned with `>=` floors that match what worked on Colab L4 (May 2026) |
| Trained adapters | Committed in-repo at [adapters/](adapters/) so eval is reproducible without retraining |
| Cached data | [data/train.jsonl](data/train.jsonl), [data/test.jsonl](data/test.jsonl), [data/pilot_raw.jsonl](data/pilot_raw.jsonl) committed so the Nebius teacher LLM doesn't need to be re-run |

## Risks and known limitations

- **Teacher inheritance**: training data inherits Qwen3-30B's stylistic and factual quirks. The fine-tuned student inherits them in turn.
- **Test-set size**: 21 examples is too small for statistical claims. Results are illustrative, not benchmark-grade.
- **Single seed**: training loss and eval verdicts come from one seed; we did not measure variance across seeds.
- **Manual evaluation**: verdicts reflect a single reviewer's judgment. A real benchmark would use blind multi-rater agreement.
- **No ablation on LoRA rank**: `r=16` was chosen up front; `r=8` would likely be safer on a 1,485-example dataset (SmolLM2 already shows mild overfitting symptoms) but was not tested.
- **flan-t5-small ceiling**: the model is borderline too small for nuanced instruction following at this scale, independent of training quality. Its fine-tuned outputs were sentence fragments at best.
- **Adapter portability**: adapter file format depends on PEFT version. Re-loading on a future PEFT may require minor config tweaks.
