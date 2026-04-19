from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    DEFAULT_TASK1_RATIOS,
    build_expected_attention_press,
    detect_device,
    ensure_output_dir,
    evaluate_window,
    load_model_bundle,
    load_wikitext_samples,
    logits_to_token_nll,
    maybe_fallback_bundle_for_press,
    parse_ratio_list,
    set_seed,
    split_tail_window,
    summarize_notes,
    MemoryTracker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 baseline KV cache compression sweep.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seq-len-cap", type=int, default=256)
    parser.add_argument("--eval-length", type=int, default=32)
    parser.add_argument("--ratios", type=str, default=",".join(str(value) for value in DEFAULT_TASK1_RATIOS))
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-future-positions", type=int, default=256)
    return parser.parse_args()


def make_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["ratio"], results_df["perplexity"], marker="o", linewidth=2, color="#114B5F")
    plt.xlabel("Compression Ratio")
    plt.ylabel("Perplexity")
    plt.title("Compression Ratio vs Perplexity")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_perplexity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["ratio"], results_df["memory_saving_pct"], marker="o", linewidth=2, color="#D95D39")
    plt.xlabel("Compression Ratio")
    plt.ylabel("Memory Saving (%)")
    plt.title("Compression Ratio vs Memory Saving")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_memory.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_output_dir()
    ratios = [0.0] + parse_ratio_list(args.ratios)
    min_prefix_tokens = 5

    bundle = load_model_bundle(preferred_device=detect_device())
    samples = load_wikitext_samples(
        tokenizer=bundle.tokenizer,
        split=args.split,
        num_samples=args.num_samples,
        seq_len_cap=args.seq_len_cap,
    )
    usable_samples = []
    for sample in samples:
        split = split_tail_window(sample.input_ids, args.eval_length)
        if split is not None and split[0].shape[1] >= min_prefix_tokens:
            usable_samples.append(sample)

    if not usable_samples:
        raise RuntimeError("No usable samples were found after applying the evaluation window.")

    probe_prefix, probe_eval = split_tail_window(usable_samples[0].input_ids, args.eval_length)
    assert probe_prefix is not None and probe_eval is not None
    bundle = maybe_fallback_bundle_for_press(
        bundle=bundle,
        prefix_ids=probe_prefix.to(bundle.device),
        eval_ids=probe_eval.to(bundle.device),
        compression_ratio=0.2,
        n_future_positions=args.n_future_positions,
    )
    summarize_notes(bundle.notes)

    print(
        f"Task 1: evaluating {len(usable_samples)} usable samples from {args.split} "
        f"with model={bundle.model_name} on device={bundle.device}."
    )

    rows: list[dict[str, float | int | str]] = []
    for ratio in ratios:
        memory_tracker = MemoryTracker(bundle.device)
        total_nll = 0.0
        total_eval_tokens = 0
        total_processed_tokens = 0
        total_elapsed = 0.0

        print(f"Running ratio={ratio:.1f}")
        press = None
        if ratio > 0:
            press = build_expected_attention_press(
                compression_ratio=ratio,
                n_future_positions=args.n_future_positions,
            )

        for sample in usable_samples:
            split = split_tail_window(sample.input_ids, args.eval_length)
            if split is None:
                continue
            prefix_ids, eval_ids = split
            prefix_ids = prefix_ids.to(bundle.device)
            eval_ids = eval_ids.to(bundle.device)
            result = evaluate_window(
                model=bundle.model,
                prefix_ids=prefix_ids,
                eval_ids=eval_ids,
                device=bundle.device,
                press=press,
                memory_tracker=memory_tracker,
            )
            losses = logits_to_token_nll(result["logits"], eval_ids)
            total_nll += float(losses.sum().item())
            total_eval_tokens += int(eval_ids.shape[1])
            total_processed_tokens += int(result["processed_tokens"])
            total_elapsed += float(result["elapsed_sec"])

        mean_nll = total_nll / max(total_eval_tokens, 1)
        perplexity = math.exp(mean_nll)
        tokens_per_sec = total_processed_tokens / max(total_elapsed, 1e-8)
        rows.append(
            {
                "ratio": ratio,
                "perplexity": perplexity,
                "peak_memory_mb": memory_tracker.peak_mb(),
                "memory_metric_name": memory_tracker.metric_name,
                "memory_saving_pct": 0.0,
                "tokens_per_sec": tokens_per_sec,
                "device": bundle.device,
                "model_name": bundle.model_name,
                "num_samples": len(usable_samples),
                "seq_len_cap": args.seq_len_cap,
            }
        )

    results_df = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
    baseline_peak = float(results_df.loc[results_df["ratio"] == 0.0, "peak_memory_mb"].iloc[0])
    if baseline_peak > 0:
        results_df["memory_saving_pct"] = 100.0 * (1.0 - results_df["peak_memory_mb"] / baseline_peak)
    results_df.loc[results_df["ratio"] == 0.0, "memory_saving_pct"] = 0.0

    results_df.to_csv(output_dir / "results.csv", index=False)
    make_plots(results_df, output_dir)

    baseline_perplexity = float(results_df.loc[results_df["ratio"] == 0.0, "perplexity"].iloc[0])
    compressed_df = results_df[results_df["ratio"] > 0.0].copy()
    compressed_df["perplexity_delta"] = compressed_df["perplexity"] - baseline_perplexity
    compressed_df = compressed_df.sort_values(["perplexity_delta", "memory_saving_pct"], ascending=[True, False])
    best_row = compressed_df.iloc[0]

    print("\nTask 1 summary")
    print(results_df.to_string(index=False))
    print(
        "Best compression ratio under the smallest perplexity increase: "
        f"ratio={best_row['ratio']:.1f}, "
        f"perplexity_delta={best_row['perplexity_delta']:.4f}, "
        f"memory_saving_pct={best_row['memory_saving_pct']:.2f}"
    )
    print(f"Saved CSV to {output_dir / 'results.csv'}")
    print(f"Saved plot to {output_dir / 'plot_perplexity.png'}")
    print(f"Saved plot to {output_dir / 'plot_memory.png'}")


if __name__ == "__main__":
    main()
