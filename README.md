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

This validates the full pipeline quickly with smaller settings. A smaller value can leave Task 2 with too few grouped inputs for Task 3.

```bash
.venv/bin/python scripts/task1_baseline.py --num-samples 8 --seq-len-cap 512 --eval-length 32
.venv/bin/python scripts/task2_collect_scores.py --num-samples 20 --target-lengths 256,512 --eval-length 32
.venv/bin/python scripts/task3_local_cp.py
```

## Default Run

```bash
.venv/bin/python scripts/task1_baseline.py
.venv/bin/python scripts/task2_collect_scores.py
.venv/bin/python scripts/task3_local_cp.py
```

Long-context Task 1 run used for the improved demo:

```bash
.venv/bin/python scripts/task1_baseline.py --seq-len-cap 2048 --num-samples 32 --eval-length 64
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
- `outputs/compression_debug.csv`
- `outputs/compression_summary.json`

Task 2:
- `outputs/scores.csv`
- `outputs/histogram_scores.png`
- `outputs/scores_vs_seqlen.png`
- `outputs/scores_summary.csv`
- `outputs/scores_diagnostics.json`

Task 3:
- `outputs/model_g_phi.pt`
- `outputs/calibration_results.json`
- `outputs/comparison_uniform_vs_adaptive.png`

Validation:
- `outputs/pre_improvement_audit.json`
- `outputs/validation_report.json`

## Notes

- The scripts first try `Qwen/Qwen2.5-0.5B` and fall back to `HuggingFaceTB/SmolLM2-360M` if needed.
- If `ExpectedAttentionPress` fails on `MPS`, Task 1 and Task 2 automatically switch to a labeled CPU fallback instead of stalling.
- Task 1 now uses contiguous long token windows from WikiText-2 and logs actual KV tokens kept plus a KV-cache memory proxy.
- Task 2 now uses balanced variable-length token windows instead of short raw lines, which greatly increases grouped input count and sequence-length diversity.
- Task 3 splits by `input_id`, not by individual rows, to avoid leaking the same source sample across train/calibration/test.
