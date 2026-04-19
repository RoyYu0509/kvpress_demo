from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DEFAULT_TASK1_RATIOS,
    ResourceTracker,
    build_expected_attention_press,
    build_non_overlapping_windows,
    detect_device,
    ensure_output_dir,
    evaluate_window,
    load_model_bundle,
    load_wikitext_token_stream,
    logits_to_token_nll,
    maybe_fallback_bundle_for_press,
    parse_ratio_list,
    save_json,
    set_seed,
    split_tail_window,
    summarize_notes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 baseline KV cache compression sweep.")
    parser.add_argument("--num-samples", type=int, default=48)
    parser.add_argument("--seq-len-cap", type=int, default=1024)
    parser.add_argument("--eval-length", type=int, default=64)
    parser.add_argument("--ratios", type=str, default=",".join(str(value) for value in DEFAULT_TASK1_RATIOS))
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-future-positions", type=int, default=512)
    return parser.parse_args()


def safe_saving_pct(current: pd.Series, baseline_value: float) -> pd.Series:
    if baseline_value <= 0:
        return pd.Series(np.zeros(len(current)), index=current.index)
    return 100.0 * (1.0 - current / baseline_value)


def make_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["ratio"], results_df["perplexity"], marker="o", linewidth=2, color="#0F4C5C")
    plt.xlabel("Compression Ratio")
    plt.ylabel("Perplexity")
    plt.title("Compression Ratio vs Perplexity")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_perplexity.png", dpi=180)
    plt.close()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(results_df["ratio"], results_df["avg_estimated_kv_cache_mb"], marker="o", label="KV cache proxy (MB)", color="#1B998B")
    axes[0].plot(results_df["ratio"], results_df["peak_process_rss_mb_delta"], marker="o", label="RSS delta (MB)", color="#BC4749")
    if results_df["peak_mps_driver_allocated_mb_delta"].notna().any():
        axes[0].plot(
            results_df["ratio"],
            results_df["peak_mps_driver_allocated_mb_delta"],
            marker="o",
            label="MPS driver delta (MB)",
            color="#6A4C93",
        )
    axes[0].set_xlabel("Compression Ratio")
    axes[0].set_ylabel("MB")
    axes[0].set_title("Absolute Memory Metrics")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(results_df["ratio"], results_df["kv_cache_saving_pct"], marker="o", label="KV cache proxy saving", color="#1B998B")
    axes[1].plot(results_df["ratio"], results_df["rss_delta_saving_pct"], marker="o", label="RSS delta saving", color="#BC4749")
    if results_df["mps_driver_delta_saving_pct"].notna().any():
        axes[1].plot(
            results_df["ratio"],
            results_df["mps_driver_delta_saving_pct"],
            marker="o",
            label="MPS driver delta saving",
            color="#6A4C93",
        )
    axes[1].set_xlabel("Compression Ratio")
    axes[1].set_ylabel("Saving (%)")
    axes[1].set_title("Relative Memory Savings")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "plot_memory.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.seq_len_cap <= args.eval_length:
        raise ValueError("--seq-len-cap must be larger than --eval-length.")

    set_seed(args.seed)
    output_dir = ensure_output_dir()
    ratios = [0.0] + parse_ratio_list(args.ratios)

    bundle = load_model_bundle(preferred_device=detect_device())
    token_stream = load_wikitext_token_stream(bundle.tokenizer, split=args.split)
    samples = build_non_overlapping_windows(
        token_stream=token_stream,
        window_length=args.seq_len_cap,
        num_samples=args.num_samples,
        split_name=args.split,
    )
    if not samples:
        raise RuntimeError("No full-length validation windows were available for Task 1.")

    probe_prefix, probe_eval = split_tail_window(samples[0].input_ids, args.eval_length)
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
        f"Task 1: evaluating {len(samples)} long windows from {args.split} "
        f"with seq_len_cap={args.seq_len_cap}, eval_length={args.eval_length}, "
        f"model={bundle.model_name}, device={bundle.device}."
    )

    rows: list[dict[str, float | int | str | bool]] = []
    debug_rows: list[dict[str, float | int | str]] = []
    prefix_length = args.seq_len_cap - args.eval_length

    for ratio in ratios:
        resource_tracker = ResourceTracker(bundle.device)
        total_nll = 0.0
        total_eval_tokens = 0
        total_processed_tokens = 0
        total_elapsed = 0.0
        cache_mb_values = []
        kept_token_values = []
        keep_fraction_values = []
        print(f"Running ratio={ratio:.2f}")

        press = None
        if ratio > 0:
            press = build_expected_attention_press(
                compression_ratio=ratio,
                n_future_positions=args.n_future_positions,
            )

        for sample in samples:
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
                resource_tracker=resource_tracker,
                collect_cache_stats=True,
            )
            losses = logits_to_token_nll(result["logits"], eval_ids)
            cache_stats = result["cache_stats"]
            avg_kept_tokens = float(cache_stats["avg_kept_tokens_per_layer"])
            keep_fraction = avg_kept_tokens / prefix_ids.shape[1]
            total_nll += float(losses.sum().item())
            total_eval_tokens += int(eval_ids.shape[1])
            total_processed_tokens += int(result["processed_tokens"])
            total_elapsed += float(result["elapsed_sec"])
            cache_mb_values.append(float(cache_stats["estimated_kv_cache_mb"]))
            kept_token_values.append(avg_kept_tokens)
            keep_fraction_values.append(keep_fraction)
            expected_kept = prefix_ids.shape[1] if ratio == 0 else int(prefix_ids.shape[1] * (1.0 - ratio))

            debug_rows.append(
                {
                    "ratio": ratio,
                    "input_id": sample.input_id,
                    "start_position": sample.source_index,
                    "sequence_length": sample.sequence_length,
                    "prefix_length": int(prefix_ids.shape[1]),
                    "eval_length": int(eval_ids.shape[1]),
                    "expected_kept_tokens_per_layer": expected_kept,
                    "avg_actual_kept_tokens_per_layer": avg_kept_tokens,
                    "min_actual_kept_tokens_per_layer": cache_stats["min_kept_tokens_per_layer"],
                    "max_actual_kept_tokens_per_layer": cache_stats["max_kept_tokens_per_layer"],
                    "actual_keep_fraction": keep_fraction,
                    "estimated_kv_cache_mb": cache_stats["estimated_kv_cache_mb"],
                    "baseline_token_nll_mean": result["baseline_token_nll_mean"],
                    "baseline_logit_entropy_mean": result["baseline_logit_entropy_mean"],
                    "peak_process_rss_mb_delta": result["resource_peaks"].get("peak_process_rss_mb_delta", math.nan),
                    "peak_mps_driver_allocated_mb_delta": result["resource_peaks"].get("peak_mps_driver_allocated_mb_delta", math.nan),
                }
            )

        mean_nll = total_nll / max(total_eval_tokens, 1)
        perplexity = math.exp(mean_nll)
        tokens_per_sec = total_processed_tokens / max(total_elapsed, 1e-8)
        resource_summary = resource_tracker.summary()
        rows.append(
            {
                "ratio": ratio,
                "perplexity": perplexity,
                "peak_memory_mb": float(np.mean(cache_mb_values)),
                "memory_metric_name": "estimated_kv_cache_mb",
                "memory_saving_pct": 0.0,
                "tokens_per_sec": tokens_per_sec,
                "device": bundle.device,
                "model_name": bundle.model_name,
                "num_samples": len(samples),
                "seq_len_cap": args.seq_len_cap,
                "eval_length": args.eval_length,
                "prefix_length": prefix_length,
                "expected_kept_tokens_per_layer": prefix_length if ratio == 0 else int(prefix_length * (1.0 - ratio)),
                "avg_actual_kept_tokens_per_layer": float(np.mean(kept_token_values)),
                "min_actual_kept_tokens_per_layer": float(np.min(kept_token_values)),
                "max_actual_kept_tokens_per_layer": float(np.max(kept_token_values)),
                "avg_keep_fraction": float(np.mean(keep_fraction_values)),
                "avg_estimated_kv_cache_mb": float(np.mean(cache_mb_values)),
                "peak_process_rss_mb": resource_summary.get("peak_process_rss_mb", math.nan),
                "peak_process_rss_mb_delta": resource_summary.get("peak_process_rss_mb_delta", math.nan),
                "peak_mps_current_allocated_mb": resource_summary.get("peak_mps_current_allocated_mb", math.nan),
                "peak_mps_current_allocated_mb_delta": resource_summary.get("peak_mps_current_allocated_mb_delta", math.nan),
                "peak_mps_driver_allocated_mb": resource_summary.get("peak_mps_driver_allocated_mb", math.nan),
                "peak_mps_driver_allocated_mb_delta": resource_summary.get("peak_mps_driver_allocated_mb_delta", math.nan),
                "compression_applied": ratio == 0 or float(np.mean(kept_token_values)) < prefix_length,
            }
        )

    results_df = pd.DataFrame(rows).sort_values("ratio").reset_index(drop=True)
    debug_df = pd.DataFrame(debug_rows).sort_values(["ratio", "input_id"]).reset_index(drop=True)

    baseline_row = results_df.loc[results_df["ratio"] == 0.0].iloc[0]
    kv_baseline = float(baseline_row["avg_estimated_kv_cache_mb"])
    rss_baseline = float(baseline_row["peak_process_rss_mb_delta"])
    mps_driver_baseline = float(baseline_row["peak_mps_driver_allocated_mb_delta"])

    results_df["kv_cache_saving_pct"] = safe_saving_pct(results_df["avg_estimated_kv_cache_mb"], kv_baseline)
    results_df["rss_delta_saving_pct"] = safe_saving_pct(results_df["peak_process_rss_mb_delta"], rss_baseline)
    results_df["mps_driver_delta_saving_pct"] = safe_saving_pct(results_df["peak_mps_driver_allocated_mb_delta"], mps_driver_baseline)
    results_df["memory_saving_pct"] = results_df["kv_cache_saving_pct"]
    results_df.loc[results_df["ratio"] == 0.0, ["memory_saving_pct", "kv_cache_saving_pct", "rss_delta_saving_pct", "mps_driver_delta_saving_pct"]] = 0.0

    kept_tokens = results_df["avg_actual_kept_tokens_per_layer"].tolist()
    monotonic_keep_counts = all(
        later <= earlier + 1e-6 for earlier, later in zip(kept_tokens, kept_tokens[1:])
    )
    expected_close = bool(
        np.allclose(
            results_df["avg_actual_kept_tokens_per_layer"].to_numpy(),
            results_df["expected_kept_tokens_per_layer"].to_numpy(),
            atol=1.0,
        )
    )
    unique_keep_counts = int(results_df["avg_actual_kept_tokens_per_layer"].nunique())

    results_df.to_csv(output_dir / "results.csv", index=False)
    debug_df.to_csv(output_dir / "compression_debug.csv", index=False)

    compression_summary = {
        "seq_len_cap": args.seq_len_cap,
        "eval_length": args.eval_length,
        "num_samples": len(samples),
        "compression_keep_counts_monotonic": monotonic_keep_counts,
        "compression_keep_counts_match_expected": expected_close,
        "unique_keep_count_levels": unique_keep_counts,
    }
    save_json(output_dir / "compression_summary.json", compression_summary)

    make_plots(results_df, output_dir)

    baseline_perplexity = float(baseline_row["perplexity"])
    compressed_df = results_df[results_df["ratio"] > 0.0].copy()
    compressed_df["perplexity_delta"] = compressed_df["perplexity"] - baseline_perplexity
    compressed_df["perplexity_increase_pct"] = 100.0 * compressed_df["perplexity_delta"] / baseline_perplexity
    compressed_df = compressed_df.sort_values(["perplexity_delta", "kv_cache_saving_pct"], ascending=[True, False])
    best_row = compressed_df.iloc[0]

    print("\nTask 1 summary")
    print(results_df.to_string(index=False))
    print(
        "Best compression ratio under the smallest perplexity increase: "
        f"ratio={best_row['ratio']:.2f}, "
        f"perplexity_delta={best_row['perplexity_delta']:.4f}, "
        f"kv_cache_saving_pct={best_row['kv_cache_saving_pct']:.2f}"
    )
    print(
        "Compression sanity checks: "
        f"monotonic_keep_counts={monotonic_keep_counts}, "
        f"expected_close={expected_close}, "
        f"unique_keep_count_levels={unique_keep_counts}"
    )
    print(f"Saved CSV to {output_dir / 'results.csv'}")
    print(f"Saved debug CSV to {output_dir / 'compression_debug.csv'}")
    print(f"Saved JSON to {output_dir / 'compression_summary.json'}")
    print(f"Saved plot to {output_dir / 'plot_perplexity.png'}")
    print(f"Saved plot to {output_dir / 'plot_memory.png'}")


if __name__ == "__main__":
    main()
