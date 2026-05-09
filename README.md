# Fine-Tuning Small LLMs for Audience-Adaptive Q&A
*A study in QLoRA fine-tuning across decoder-only and encoder-decoder architectures, plus a from-scratch implementation of scaled dot-product attention.*

> **Status: Work in progress.** This README is a stub — headline results, plots, and trained adapter links land here as the project completes.

## Project goals

1. Fine-tune two small open LLMs — **SmolLM2-360M-Instruct** (decoder-only) and **flan-t5-small** (encoder-decoder) — to answer the same question at three different audience levels: *child*, *student*, *expert*.
2. Use **QLoRA** (4-bit quantization + LoRA adapters) to train efficiently on a single GPU.
3. Compare the two architectures on the same task.
4. Implement **scaled dot-product attention** from scratch in PyTorch and visualize how attention patterns shift with input changes and across heads.

## Repository structure

```
qlora-finetuning-and-attention/
├── README.md                   # this file
├── SPEC.md                     # technical specification
├── requirements.txt            # pinned dependencies
├── From_Finetuning_to_Attention_Inside_LLMs.ipynb   # main notebook (deliverable)
├── src/
│   └── attention/              # from-scratch attention implementation
├── data/                       # generated train/test JSONL (Part 1)
└── results/
    └── attention_heatmaps/     # Part 2 plots
```

## How to run

### Part 2 — attention (local, CPU is fine)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook From_Finetuning_to_Attention_Inside_LLMs.ipynb
```
Run the cells under **Part 2**.

### Part 1 — QLoRA fine-tuning (Google Colab required)
QLoRA depends on `bitsandbytes`, which needs CUDA. macOS has no stable build, so Part 1 must run on Colab (or another CUDA host).

1. Push this repo to GitHub.
2. In Colab: **File → Open notebook → GitHub** → select this repo's notebook.
3. Run the cells under **Part 1**.

## Tech stack

PyTorch · transformers · peft · trl · bitsandbytes · datasets · matplotlib · seaborn

## License

MIT
