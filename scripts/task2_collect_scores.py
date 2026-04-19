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
    DEFAULT_TASK23_RATIOS,
    build_balanced_variable_length_windows,
    build_expected_attention_press,
    detect_device,
    ensure_output_dir,
    evaluate_window,
    load_model_bundle,
    load_wikitext_token_stream,
    mean_kl_divergence,
    maybe_fallback_bundle_for_press,
    parse_int_list,
    parse_ratio_list,
    save_json,
    set_seed,
    split_tail_window,
    summarize_notes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 multi-ratio nonconformity score collection.")
    parser.add_argument("--num-samples", type=int, default=120, help="Total number of unique inputs to build across all target lengths.")
    parser.add_argument("--target-lengths", type=str, default="256,512,768,1024")
    parser.add_argument("--eval-length", type=int, default=64)
    parser.add_argument("--ratios", type=str, default=",".join(str(value) for value in DEFAULT_TASK23_RATIOS))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-future-positions", type=int, default=512)
    return parser.parse_args()


def make_plots(scores_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(scores_df["score_mean_kl"], bins=30, color="#2D6A4F", edgecolor="white")
    plt.xlabel("Mean KL")
    plt.ylabel("Count")
    plt.title("Distribution of Nonconformity Scores")
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_scores.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    palette = ["#0F4C5C", "#E36414", "#6A4C93", "#1B998B", "#C1121F"]
    for color, ratio in zip(palette, sorted(scores_df["ratio"].unique())):
        subset = scores_df[scores_df["ratio"] == ratio]
        plt.scatter(
            subset["sequence_length"],
            subset["score_mean_kl"],
            label=f"ratio={ratio:.1f}",
            alpha=0.65,
            s=26,
            color=color,
        )
    plt.xlabel("Sequence Length")
    plt.ylabel("Mean KL")
    plt.title("Nonconformity Score vs Sequence Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "scores_vs_seqlen.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    target_lengths = parse_int_list(args.target_lengths)
    if any(length <= args.eval_length for length in target_lengths):
        raise ValueError("Every target length must be larger than --eval-length.")

    set_seed(args.seed)
    output_dir = ensure_output_dir()
    ratios = parse_ratio_list(args.ratios)

    bundle = load_model_bundle(preferred_device=detect_device())
    token_stream = load_wikitext_token_stream(bundle.tokenizer, split=args.split)
    samples = build_balanced_variable_length_windows(
        token_stream=token_stream,
        target_lengths=target_lengths,
        total_samples=args.num_samples,
        seed=args.seed,
        split_name=args.split,
    )
    if not samples:
        raise RuntimeError("No train windows were available for Task 2.")

    probe_prefix, probe_eval = split_tail_window(samples[0].input_ids, args.eval_length)
    assert probe_prefix is not None and probe_eval is not None
    bundle = maybe_fallback_bundle_for_press(
        bundle=bundle,
        prefix_ids=probe_prefix.to(bundle.device),
        eval_ids=probe_eval.to(bundle.device),
        compression_ratio=ratios[0],
        n_future_positions=args.n_future_positions,
    )
    summarize_notes(bundle.notes)

    print(
        f"Task 2: collecting scores for {len(samples)} token windows from {args.split} "
        f"across lengths={target_lengths}, model={bundle.model_name}, device={bundle.device}."
    )

    attention_supported = True
    rows: list[dict[str, float | int | str]] = []
    for sample in samples:
        split = split_tail_window(sample.input_ids, args.eval_length)
        if split is None:
            continue
        prefix_ids, eval_ids = split
        prefix_ids = prefix_ids.to(bundle.device)
        eval_ids = eval_ids.to(bundle.device)

        use_attention_entropy = attention_supported and sample.sequence_length <= 512
        try:
            full_result = evaluate_window(
                model=bundle.model,
                prefix_ids=prefix_ids,
                eval_ids=eval_ids,
                device=bundle.device,
                output_hidden_states=True,
                output_attentions=use_attention_entropy,
                collect_cache_stats=True,
            )
            if not use_attention_entropy:
                full_result["attention_entropy"] = math.nan
        except Exception as exc:
            if use_attention_entropy:
                print(
                    "Attention entropy was too expensive or unsupported. "
                    f"Falling back to NaN attention entropy. Root cause: {type(exc).__name__}: {exc}"
                )
                attention_supported = False
                full_result = evaluate_window(
                    model=bundle.model,
                    prefix_ids=prefix_ids,
                    eval_ids=eval_ids,
                    device=bundle.device,
                    output_hidden_states=True,
                    output_attentions=False,
                    collect_cache_stats=True,
                )
                full_result["attention_entropy"] = math.nan
            else:
                raise

        for ratio in ratios:
            press = build_expected_attention_press(
                compression_ratio=ratio,
                n_future_positions=args.n_future_positions,
            )
            compressed_result = evaluate_window(
                model=bundle.model,
                prefix_ids=prefix_ids,
                eval_ids=eval_ids,
                device=bundle.device,
                press=press,
                collect_cache_stats=True,
            )
            compressed_cache = compressed_result["cache_stats"]
            rows.append(
                {
                    "input_id": sample.input_id,
                    "ratio": ratio,
                    "score_mean_kl": mean_kl_divergence(full_result["logits"], compressed_result["logits"]),
                    "sequence_length": sample.sequence_length,
                    "context_length": int(prefix_ids.shape[1]),
                    "eval_length": int(eval_ids.shape[1]),
                    "target_length": sample.target_length or sample.sequence_length,
                    "start_position": sample.source_index,
                    "mean_hidden_state_norm": full_result["mean_hidden_state_norm"],
                    "attention_entropy": full_result["attention_entropy"],
                    "baseline_token_nll_mean": full_result["baseline_token_nll_mean"],
                    "baseline_logit_entropy_mean": full_result["baseline_logit_entropy_mean"],
                    "baseline_max_probability_mean": full_result["baseline_max_probability_mean"],
                    "actual_kept_tokens_per_layer": compressed_cache["avg_kept_tokens_per_layer"],
                    "actual_keep_fraction": compressed_cache["avg_kept_tokens_per_layer"] / prefix_ids.shape[1],
                    "estimated_kv_cache_mb": compressed_cache["estimated_kv_cache_mb"],
                    "device": bundle.device,
                    "model_name": bundle.model_name,
                }
            )

    scores_df = pd.DataFrame(rows).sort_values(["input_id", "ratio"]).reset_index(drop=True)
    scores_df.to_csv(output_dir / "scores.csv", index=False)
    make_plots(scores_df, output_dir)

    summary_df = (
        scores_df.groupby("ratio")
        .agg(
            count=("score_mean_kl", "count"),
            mean_score=("score_mean_kl", "mean"),
            median_score=("score_mean_kl", "median"),
            std_score=("score_mean_kl", "std"),
            p10_score=("score_mean_kl", lambda values: float(np.quantile(values, 0.1))),
            p90_score=("score_mean_kl", lambda values: float(np.quantile(values, 0.9))),
            mean_keep_fraction=("actual_keep_fraction", "mean"),
            mean_kv_cache_mb=("estimated_kv_cache_mb", "mean"),
        )
        .reset_index()
    )
    summary_df.to_csv(output_dir / "scores_summary.csv", index=False)

    ratio_means = summary_df.sort_values("ratio")["mean_score"].to_numpy()
    keep_means = summary_df.sort_values("ratio")["mean_keep_fraction"].to_numpy()
    score_semantics_monotonic = bool(np.all(np.diff(ratio_means) >= -1e-6))
    keep_fraction_monotonic = bool(np.all(np.diff(keep_means) <= 1e-6))

    diagnostics = {
        "num_rows": int(len(scores_df)),
        "num_unique_inputs": int(scores_df["input_id"].nunique()),
        "target_lengths": target_lengths,
        "observed_sequence_lengths": sorted(int(value) for value in scores_df["sequence_length"].unique()),
        "score_semantics_monotonic": score_semantics_monotonic,
        "keep_fraction_monotonic": keep_fraction_monotonic,
    }
    save_json(output_dir / "scores_diagnostics.json", diagnostics)

    print("\nTask 2 summary by ratio")
    print(summary_df.to_string(index=False))
    print(
        "Task 2 diagnostics: "
        f"unique_inputs={diagnostics['num_unique_inputs']}, "
        f"observed_lengths={diagnostics['observed_sequence_lengths']}, "
        f"score_semantics_monotonic={score_semantics_monotonic}, "
        f"keep_fraction_monotonic={keep_fraction_monotonic}"
    )
    print(f"Saved CSV to {output_dir / 'scores.csv'}")
    print(f"Saved summary CSV to {output_dir / 'scores_summary.csv'}")
    print(f"Saved diagnostics JSON to {output_dir / 'scores_diagnostics.json'}")
    print(f"Saved plot to {output_dir / 'histogram_scores.png'}")
    print(f"Saved plot to {output_dir / 'scores_vs_seqlen.png'}")


if __name__ == "__main__":
    main()
