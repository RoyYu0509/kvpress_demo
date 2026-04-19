# Recommended prompt to give Codex
Build and run a minimal end-to-end prototype for KV cache compression + local conformal calibration on an Apple Silicon MacBook Pro.

Project constraints:
- Machine: Apple Silicon MacBook Pro
- Available unified GPU memory for MPS: about 10 GB
- OS: macOS
- Python: 3.10+
- Device policy: use PyTorch MPS if available; otherwise fall back to CPU
- Do NOT assume CUDA
- Do NOT install or use flash-attn, bitsandbytes, vllm, xformers, triton, or any CUDA-only dependency
- Prioritize a working demo tonight over scale
- Use batch size 1 unless there is a strong reason not to
- Cap sequence length to 256 tokens by default
- Prefer a small causal LM that fits comfortably on MPS:
  1) first choice: Qwen/Qwen2.5-0.5B
  2) fallback: HuggingFaceTB/SmolLM2-360M
- Use WikiText-2 via the datasets library
- Save all outputs under ./outputs
- Create clean, runnable scripts and actually execute them locally
- If native KVPress on MPS fails because of an unsupported op, do not get stuck silently; clearly report the blocker and implement the closest working fallback that preserves the experiment pipeline, while labeling fallback outputs clearly

Deliverables:
- requirements.txt
- README.md with exact run commands
- scripts/task1_baseline.py
- scripts/task2_collect_scores.py
- scripts/task3_local_cp.py
- optionally run_all.sh
- all requested CSV / JSON / PNG outputs

General implementation rules:
- Use deterministic seeds where possible
- Print progress and intermediate summaries to the terminal
- Fail loudly with clear error messages
- Keep code simple and readable
- Use matplotlib, not seaborn
- Do not ask the user follow-up questions; make reasonable decisions and proceed


Task 1 prompt for Codex
## Task 1: Environment setup + baseline compression experiments on Apple Silicon MacBook

Goal:
Set up a local KVPress-compatible experiment on macOS and run a small but complete baseline sweep for KV cache compression.

Environment assumptions:
- Apple Silicon MacBook Pro
- ~10 GB available unified GPU memory
- Python 3.10+
- Use torch with MPS if available, else CPU
- Do not use CUDA-only packages
- Keep runtime practical for a same-day demo

Implementation requirements:
1. Create a Python environment and install the minimum required packages:
   - torch
   - transformers
   - datasets
   - pandas
   - matplotlib
   - psutil
   - kvpress
2. Write the code so it chooses device automatically:
   - "mps" if torch.backends.mps.is_available()
   - otherwise "cpu"
3. Use model:
   - first try: Qwen/Qwen2.5-0.5B
   - if that fails to load cleanly, fall back to HuggingFaceTB/SmolLM2-360M
4. Use WikiText-2 from the datasets library.
5. For speed, use the first 200 non-empty validation samples by default.
6. Truncate or chunk each sample to a maximum sequence length of 256 tokens.
7. Run a no-compression baseline plus Expected Attention compression at ratios:
   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
8. For each ratio, record:
   - perplexity on the selected WikiText-2 subset
   - peak MPS memory if available
   - tokens/sec latency
   - memory saving relative to the no-compression baseline
9. Save results to CSV.
10. Generate two PNG plots:
   - compression_ratio vs perplexity
   - compression_ratio vs memory_saving_pct

Measurement details:
- Perplexity:
  - use standard teacher-forced causal LM loss
  - compute mean negative log-likelihood over the selected subset
  - report exp(mean_nll) as perplexity
- Peak GPU memory:
  - if running on MPS, use torch.mps.current_allocated_memory() and/or torch.mps.driver_allocated_memory()
  - if MPS memory APIs are not usable, use process RSS from psutil as a proxy and label it clearly
- Latency:
  - measure processed tokens/sec during forward evaluation on the same subset
  - report average tokens/sec for each ratio
- Memory saving:
  - compute as 100 * (1 - peak_memory_ratio / peak_memory_no_compression)

Output files:
- outputs/results.csv
- outputs/plot_perplexity.png
- outputs/plot_memory.png

CSV columns:
- ratio
- perplexity
- peak_memory_mb
- memory_metric_name
- memory_saving_pct
- tokens_per_sec
- device
- model_name
- num_samples
- seq_len_cap

Acceptance criteria:
- The script runs end to end on the MacBook
- A CSV is produced
- Both plots are produced
- A short terminal summary prints the best compression ratio under the smallest perplexity increase


Task 2 prompt for Codex
## Task 2: Collect nonconformity scores for conformal prediction on multiple compression ratios

Goal:
Collect sequence-level degradation scores comparing uncompressed vs compressed KV cache behavior, so that Task 3 can learn an adaptive compression policy.

Important note:
Do NOT collect scores only at ratio=0.5. To support adaptive ratio selection later, collect scores at multiple ratios.

Setup:
- Reuse the same model/device loading code from Task 1
- Reuse WikiText-2
- Use the first 200 non-empty samples from the train split by default for calibration-score collection
- Cap total sequence length to 256
- Use batch size 1
- Candidate compression ratios:
  [0.2, 0.4, 0.6, 0.8]

For each input sample:
1. Tokenize and truncate to 256 tokens max.
2. Split into:
   - context prefix: first 192 tokens
   - evaluation window: next 32 tokens
   - skip samples that are too short
3. Run the uncompressed model on the prefix and evaluate next-token distributions across the 32-token evaluation window:
   - logits_full
4. Run the compressed model at each candidate ratio on the same prefix and evaluation window:
   - logits_compressed
5. For each token position in the evaluation window, compute:
   KL(softmax(logits_full) || softmax(logits_compressed))
6. For each (sample, ratio), compute sequence-level score:
   - mean KL over the 32-token window
7. Also record:
   - input_id
   - ratio
   - sequence_length
   - context_length
   - eval_length
   - mean_hidden_state_norm
   - attention_entropy if available
8. If output_attentions=True is too slow or unsupported, store attention_entropy as NaN and continue.

Definitions:
- mean_hidden_state_norm:
  - compute from the final hidden states of the uncompressed run
  - average L2 norm across tokens
- attention_entropy:
  - if attention weights are available, compute mean entropy of the final-layer attention distribution across heads/tokens
  - otherwise NaN

Save outputs:
- outputs/scores.csv
- outputs/histogram_scores.png
- outputs/scores_vs_seqlen.png

CSV columns:
- input_id
- ratio
- score_mean_kl
- sequence_length
- context_length
- eval_length
- mean_hidden_state_norm
- attention_entropy
- device
- model_name

Plot requirements:
1. Histogram of score_mean_kl over all rows
2. Scatter plot of score_mean_kl vs sequence_length, colored or grouped by ratio if easy; otherwise plain scatter is fine

Acceptance criteria:
- scores.csv contains rows for multiple ratios per input
- plots are saved
- script prints summary statistics by ratio:
  - count
  - mean score
  - median score
  - 90th percentile score



Task 3 prompt for Codex
## Task 3: Implement a practical Local CP calibration pipeline for adaptive KV cache compression

Goal:
Train a small ratio-aware predictor of compression degradation and use split conformal calibration to choose compression ratios adaptively while targeting 90% coverage.

Important modeling choice:
Because we want to choose a ratio r for each new input, the predictor must depend on both:
- input features
- compression ratio

So use a ratio-aware model:
g_phi(x, r)

Inputs:
- Load outputs/scores.csv from Task 2

Feature definition:
- Required features:
  - ratio
  - sequence_length
  - mean_hidden_state_norm
- Optional feature:
  - attention_entropy
- If attention_entropy is missing or NaN, impute with the training-set median and add a binary missingness indicator column

Data split:
1. Randomly split rows from scores.csv into:
   - train: 60%
   - calibration: 20%
   - test: 20%
2. Use a fixed random seed.

Model:
- Use a simple MLP with 2 hidden layers of width 64
- ReLU activations
- Target: score_mean_kl
- Train the model as quantile regression at level 0.9 (alpha = 0.1)

Calibration method:
1. On the calibration set, compute:
   residual_j = score_j / max(g_phi(x_j, r_j), 1e-8)
2. Compute:
   ell_star = empirical quantile of residual_j at level ceil((N_c + 1)(1 - alpha)) / N_c
3. Define the conformal upper bound for a new sample and candidate ratio r:
   q_hat(x, r) = ell_star * g_phi(x, r)

Define the degradation budget tau:
- Let tau be the empirical 90th percentile of score_mean_kl at ratio = 0.5 on the calibration set
- If ratio 0.5 is not present, use the nearest available ratio and state it clearly

Adaptive policy:
- For each test input, evaluate candidate ratios [0.2, 0.4, 0.6, 0.8]
- Choose the largest ratio r such that:
  q_hat(x, r) <= tau
- If no ratio satisfies the budget, choose the smallest ratio

Uniform policy:
- On the calibration set, find the single largest fixed ratio whose empirical coverage satisfies:
  P(score_mean_kl <= tau) >= 0.9
- Use that fixed ratio for all test inputs

Evaluation on the test set:
Report:
- adaptive average selected ratio
- adaptive empirical coverage
- uniform selected ratio
- uniform empirical coverage
- number of test inputs
- alpha
- tau
- ell_star

Save outputs:
- outputs/model_g_phi.pt
- outputs/calibration_results.json
- outputs/comparison_uniform_vs_adaptive.png

Plot:
- A simple bar chart comparing:
  - average compression ratio: uniform vs adaptive
  - and/or empirical coverage: uniform vs adaptive

JSON keys:
- alpha
- tau
- ell_star
- candidate_ratios
- num_train
- num_calibration
- num_test
- uniform_ratio
- uniform_coverage
- adaptive_avg_ratio
- adaptive_coverage
- model_features

Acceptance criteria:
- The model trains successfully
- calibration_results.json is created
- the comparison plot is created
- the terminal output clearly states whether adaptive selection achieved higher average compression ratio than uniform under similar coverage
A slightly shorter version you can tell Codex before the 3 tasks
This project must run on an Apple Silicon MacBook Pro with MPS, not CUDA. Keep everything small and practical for a same-day demo. Use a small model, batch size 1, seq_len <= 256, no flash-attn, and generate runnable scripts plus outputs. If a CUDA-only path fails, do not get stuck—use the closest working MPS-safe implementation and label it clearly.

One more thing: your original Task 3 also left sigma(x) undefined. I replaced it with a simpler conformal scaling that Codex can actually implement tonight. That makes the pipeline much more likely to run end to end.
