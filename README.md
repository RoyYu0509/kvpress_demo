# KVPress Demo

Minimal end-to-end KV cache compression + local conformal calibration demo for an Apple Silicon MacBook Pro.

## Environment

- Recommended Python: `3.12`
- Device policy: prefer `MPS`, otherwise `CPU`
- No CUDA-only dependencies are required
- Outputs are written under `./outputs`

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Smoke Run

This validates the full pipeline quickly with 40 non-empty samples. A smaller value can leave Task 2 with too few long sequences for Task 3.

```bash
.venv/bin/python scripts/task1_baseline.py --num-samples 40
.venv/bin/python scripts/task2_collect_scores.py --num-samples 40
.venv/bin/python scripts/task3_local_cp.py
```

## Default Run

```bash
.venv/bin/python scripts/task1_baseline.py
.venv/bin/python scripts/task2_collect_scores.py
.venv/bin/python scripts/task3_local_cp.py
```

Or run the whole pipeline:

```bash
./run_all.sh
```

## Outputs

Task 1:
- `outputs/results.csv`
- `outputs/plot_perplexity.png`
- `outputs/plot_memory.png`

Task 2:
- `outputs/scores.csv`
- `outputs/histogram_scores.png`
- `outputs/scores_vs_seqlen.png`

Task 3:
- `outputs/model_g_phi.pt`
- `outputs/calibration_results.json`
- `outputs/comparison_uniform_vs_adaptive.png`

## Notes

- The scripts first try `Qwen/Qwen2.5-0.5B` and fall back to `HuggingFaceTB/SmolLM2-360M` if needed.
- If `ExpectedAttentionPress` fails on `MPS`, Task 1 and Task 2 automatically switch to a labeled CPU fallback instead of stalling.
- Task 1 computes perplexity on the final `eval_length` tokens of each truncated sample after prefilling the preceding context.
- Task 3 splits by `input_id`, not by individual rows, to avoid leaking the same source sample across train/calibration/test.
