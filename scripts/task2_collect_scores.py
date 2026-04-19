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
    build_expected_attention_press,
    detect_device,
    ensure_output_dir,
    evaluate_window,
    load_model_bundle,
    load_wikitext_samples,
    mean_kl_divergence,
    maybe_fallback_bundle_for_press,
    parse_ratio_list,
    set_seed,
    split_prefix_eval_window,
    summarize_notes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 multi-ratio nonconformity score collection.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seq-len-cap", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--eval-length", type=int, default=32)
    parser.add_argument("--ratios", type=str, default=",".join(str(value) for value in DEFAULT_TASK23_RATIOS))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-future-positions", type=int, default=256)
    return parser.parse_args()


def make_plots(scores_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(scores_df["score_mean_kl"], bins=24, color="#2D6A4F", edgecolor="white")
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
            alpha=0.7,
            s=24,
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
    set_seed(args.seed)
    output_dir = ensure_output_dir()
    ratios = parse_ratio_list(args.ratios)

    bundle = load_model_bundle(preferred_device=detect_device())
    samples = load_wikitext_samples(
        tokenizer=bundle.tokenizer,
        split=args.split,
        num_samples=args.num_samples,
        seq_len_cap=args.seq_len_cap,
    )
    usable_samples = []
    for sample in samples:
        if split_prefix_eval_window(sample.input_ids, args.context_length, args.eval_length) is not None:
            usable_samples.append(sample)

    if not usable_samples:
        raise RuntimeError("No usable samples were found after applying the prefix/eval split.")

    probe_prefix, probe_eval = split_prefix_eval_window(
        usable_samples[0].input_ids,
        args.context_length,
        args.eval_length,
    )
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
        f"Task 2: collecting scores for {len(usable_samples)} usable samples from {args.split} "
        f"with model={bundle.model_name} on device={bundle.device}."
    )

    attention_supported = True
    rows: list[dict[str, float | int | str]] = []
    for sample in usable_samples:
        split = split_prefix_eval_window(sample.input_ids, args.context_length, args.eval_length)
        if split is None:
            continue
        prefix_ids, eval_ids = split
        prefix_ids = prefix_ids.to(bundle.device)
        eval_ids = eval_ids.to(bundle.device)

        try:
            full_result = evaluate_window(
                model=bundle.model,
                prefix_ids=prefix_ids,
                eval_ids=eval_ids,
                device=bundle.device,
                output_hidden_states=True,
                output_attentions=attention_supported,
            )
        except Exception as exc:
            if attention_supported:
                print(
                    "Attention entropy is unavailable for this runtime. "
                    f"Continuing with NaN attention entropy. Root cause: {type(exc).__name__}: {exc}"
                )
                attention_supported = False
                full_result = evaluate_window(
                    model=bundle.model,
                    prefix_ids=prefix_ids,
                    eval_ids=eval_ids,
                    device=bundle.device,
                    output_hidden_states=True,
                    output_attentions=False,
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
            )
            rows.append(
                {
                    "input_id": sample.input_id,
                    "ratio": ratio,
                    "score_mean_kl": mean_kl_divergence(full_result["logits"], compressed_result["logits"]),
                    "sequence_length": sample.sequence_length,
                    "context_length": args.context_length,
                    "eval_length": args.eval_length,
                    "mean_hidden_state_norm": full_result["mean_hidden_state_norm"],
                    "attention_entropy": full_result["attention_entropy"],
                    "device": bundle.device,
                    "model_name": bundle.model_name,
                }
            )

    scores_df = pd.DataFrame(rows).sort_values(["input_id", "ratio"]).reset_index(drop=True)
    scores_df.to_csv(output_dir / "scores.csv", index=False)
    make_plots(scores_df, output_dir)

    summary_df = (
        scores_df.groupby("ratio")["score_mean_kl"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p90=lambda values: float(np.quantile(values, 0.9)),
        )
        .reset_index()
    )

    print("\nTask 2 summary by ratio")
    print(summary_df.to_string(index=False))
    print(f"Saved CSV to {output_dir / 'scores.csv'}")
    print(f"Saved plot to {output_dir / 'histogram_scores.png'}")
    print(f"Saved plot to {output_dir / 'scores_vs_seqlen.png'}")


if __name__ == "__main__":
    main()
