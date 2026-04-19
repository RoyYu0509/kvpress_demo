from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_output_dir, save_json, set_seed


@dataclass
class SplitFrames:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    train_ids: list[int]
    calibration_ids: list[int]
    test_ids: list[int]


class ScoreTransform:
    def __init__(self, name: str):
        self.name = name

    def fit(self, train_scores: np.ndarray) -> None:
        self.train_scores = np.asarray(train_scores, dtype=float)

    def transform(self, values: np.ndarray | float) -> np.ndarray:
        raise NotImplementedError

    def transform_scalar(self, value: float) -> float:
        return float(self.transform(np.asarray([value], dtype=float))[0])


class RawTransform(ScoreTransform):
    def transform(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=float)


class LogTransform(ScoreTransform):
    def transform(self, values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))


class WinsorizedTransform(ScoreTransform):
    def fit(self, train_scores: np.ndarray) -> None:
        super().fit(train_scores)
        self.lower = float(np.quantile(self.train_scores, 0.01))
        self.upper = float(np.quantile(self.train_scores, 0.95))

    def transform(self, values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.clip(values, self.lower, self.upper)


class RankNormalizedTransform(ScoreTransform):
    def fit(self, train_scores: np.ndarray) -> None:
        super().fit(train_scores)
        self.sorted_scores = np.sort(self.train_scores)
        self.normal = NormalDist()

    def transform(self, values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        ranks = np.searchsorted(self.sorted_scores, values, side="right")
        percentiles = np.clip((ranks + 0.5) / (len(self.sorted_scores) + 1.0), 1e-4, 1.0 - 1e-4)
        return np.asarray([self.normal.inv_cdf(float(p)) for p in percentiles], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simpler conformal calibration baselines.")
    parser.add_argument("--scores-path", type=Path, default=Path("outputs/scores.csv"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--reference-ratio", type=float, default=0.4)
    parser.add_argument("--min-bucket-calibration", type=int, default=4)
    return parser.parse_args()


def split_input_ids(input_ids: list[int]) -> tuple[list[int], list[int], list[int]]:
    total = len(input_ids)
    if total < 15:
        raise RuntimeError("Task 4 needs at least 15 grouped inputs.")
    n_train = max(1, int(total * 0.6))
    n_calibration = max(1, int(total * 0.2))
    n_test = total - n_train - n_calibration
    if n_test < 1:
        n_test = 1
        if n_train >= n_calibration and n_train > 1:
            n_train -= 1
        elif n_calibration > 1:
            n_calibration -= 1
    train_ids = input_ids[:n_train]
    calibration_ids = input_ids[n_train : n_train + n_calibration]
    test_ids = input_ids[n_train + n_calibration :]
    return train_ids, calibration_ids, test_ids


def build_split_frames(df: pd.DataFrame, seed: int) -> SplitFrames:
    rng = np.random.default_rng(seed)
    input_ids = sorted(int(value) for value in df["input_id"].unique())
    rng.shuffle(input_ids)
    train_ids, calibration_ids, test_ids = split_input_ids(input_ids)
    return SplitFrames(
        train=df[df["input_id"].isin(train_ids)].copy(),
        calibration=df[df["input_id"].isin(calibration_ids)].copy(),
        test=df[df["input_id"].isin(test_ids)].copy(),
        train_ids=train_ids,
        calibration_ids=calibration_ids,
        test_ids=test_ids,
    )


def choose_reference_ratio(candidate_ratios: list[float], requested: float) -> float:
    return min(candidate_ratios, key=lambda ratio: (abs(ratio - requested), 0 if ratio <= requested else 1, ratio))


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    n = len(sorted_values)
    index = math.ceil((n + 1) * (1.0 - alpha)) - 1
    index = min(max(index, 0), n - 1)
    return float(sorted_values[index])


def ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    values = np.sort(np.unique(np.concatenate([x, y])))
    x_cdf = np.searchsorted(x, values, side="right") / len(x)
    y_cdf = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(x_cdf - y_cdf)))


def skewness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    centered = values - mean
    std = values.std(ddof=0)
    if std <= 1e-12:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def excess_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    centered = values - mean
    std = values.std(ddof=0)
    if std <= 1e-12:
        return 0.0
    return float(np.mean((centered / std) ** 4) - 3.0)


def jarque_bera_pvalue(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    s = skewness(values)
    k = excess_kurtosis(values)
    jb = n / 6.0 * (s**2 + 0.25 * k**2)
    pvalue = math.exp(-jb / 2.0)
    return float(jb), float(pvalue)


def total_variation_from_counts(a: dict[Any, int], b: dict[Any, int]) -> float:
    keys = sorted(set(a) | set(b))
    a_total = sum(a.values())
    b_total = sum(b.values())
    distance = 0.0
    for key in keys:
        pa = a.get(key, 0) / a_total if a_total else 0.0
        pb = b.get(key, 0) / b_total if b_total else 0.0
        distance += abs(pa - pb)
    return 0.5 * distance


def compute_uniform_coverage(frame: pd.DataFrame, ratio: float, tau_raw: float) -> float:
    subset = frame[np.isclose(frame["ratio"], ratio)]
    return float((subset["score_mean_kl"] <= tau_raw).mean())


def best_uniform_ratio(frame: pd.DataFrame, candidate_ratios: list[float], tau_raw: float, alpha: float) -> float:
    target = 1.0 - alpha
    eligible = []
    for ratio in candidate_ratios:
        coverage = compute_uniform_coverage(frame, ratio, tau_raw)
        if coverage >= target:
            eligible.append(ratio)
    return max(eligible) if eligible else min(candidate_ratios)


def best_uniform_ratio_oracle(frame: pd.DataFrame, candidate_ratios: list[float], tau_raw: float, alpha: float) -> float:
    return best_uniform_ratio(frame, candidate_ratios, tau_raw, alpha)


def select_ratio_from_quantiles(
    feasible_ratios: list[float],
    qhat_by_ratio: dict[float, float],
) -> tuple[float, str]:
    if feasible_ratios:
        return max(feasible_ratios), "largest_certified_ratio"
    best_ratio = min(qhat_by_ratio.items(), key=lambda item: (item[1], -item[0]))[0]
    return float(best_ratio), "fallback_min_qhat"


def run_global_baseline(
    transform: ScoreTransform,
    split_frames: SplitFrames,
    candidate_ratios: list[float],
    tau_raw: float,
    alpha: float,
) -> dict[str, Any]:
    transform.fit(split_frames.train["score_mean_kl"].to_numpy())
    tau_transformed = transform.transform_scalar(tau_raw)

    per_ratio = []
    feasible_ratios = []
    qhat_by_ratio: dict[float, float] = {}
    for ratio in candidate_ratios:
        calibration_subset = split_frames.calibration[np.isclose(split_frames.calibration["ratio"], ratio)]
        test_subset = split_frames.test[np.isclose(split_frames.test["ratio"], ratio)]

        calibration_scores_t = transform.transform(calibration_subset["score_mean_kl"].to_numpy())
        q_conf = conformal_quantile(calibration_scores_t, alpha)
        feasible = q_conf <= tau_transformed
        qhat_by_ratio[ratio] = q_conf
        if feasible:
            feasible_ratios.append(ratio)
        per_ratio.append(
            {
                "ratio": ratio,
                "calibration_quantile_transformed": q_conf,
                "tau_transformed": tau_transformed,
                "feasible": feasible,
                "calibration_sample_count": int(len(calibration_subset)),
                "test_sample_count": int(len(test_subset)),
                "calibration_qhat_coverage": float((calibration_scores_t <= q_conf).mean()),
                "test_qhat_coverage": float((transform.transform(test_subset["score_mean_kl"].to_numpy()) <= q_conf).mean()),
                "test_budget_coverage": float((test_subset["score_mean_kl"] <= tau_raw).mean()),
            }
        )

    selected_ratio, selection_mode = select_ratio_from_quantiles(feasible_ratios, qhat_by_ratio)
    selected_test = split_frames.test[np.isclose(split_frames.test["ratio"], selected_ratio)]
    return {
        "baseline_family": "plain_split_cp",
        "variant": transform.name,
        "conditioning": "global",
        "transform": transform.name,
        "selected_ratio": selected_ratio,
        "avg_selected_ratio": selected_ratio,
        "empirical_coverage": float((selected_test["score_mean_kl"] <= tau_raw).mean()),
        "selection_mode": selection_mode,
        "certified_ratios": feasible_ratios,
        "per_ratio": per_ratio,
    }


def assign_bucket_metadata(split_frames: SplitFrames) -> tuple[SplitFrames, dict[str, list[float]]]:
    input_level_train = split_frames.train.drop_duplicates("input_id")[["input_id", "sequence_length", "mean_hidden_state_norm"]].copy()
    norm_quantiles = input_level_train["mean_hidden_state_norm"].quantile([0.33, 0.67]).to_list()

    def norm_bucket(value: float) -> str:
        if value <= norm_quantiles[0]:
            return "low"
        if value <= norm_quantiles[1]:
            return "mid"
        return "high"

    def add_bucket_columns(frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        enriched["length_bucket"] = enriched["sequence_length"].astype(int).astype(str)
        enriched["difficulty_bucket"] = enriched["mean_hidden_state_norm"].apply(norm_bucket)
        enriched["length_difficulty_bucket"] = enriched["length_bucket"] + "|norm=" + enriched["difficulty_bucket"]
        return enriched

    return (
        SplitFrames(
            train=add_bucket_columns(split_frames.train),
            calibration=add_bucket_columns(split_frames.calibration),
            test=add_bucket_columns(split_frames.test),
            train_ids=split_frames.train_ids,
            calibration_ids=split_frames.calibration_ids,
            test_ids=split_frames.test_ids,
        ),
        {"norm_quantiles": norm_quantiles},
    )


def lookup_bucket_quantile(
    calibration_frame: pd.DataFrame,
    ratio: float,
    bucket_column: str,
    bucket_value: str,
    alpha: float,
    min_bucket_calibration: int,
) -> tuple[float, str, int]:
    cal_ratio = calibration_frame[np.isclose(calibration_frame["ratio"], ratio)]
    cal_bucket = cal_ratio[cal_ratio[bucket_column] == bucket_value]

    if len(cal_bucket) < min_bucket_calibration:
        q_conf = conformal_quantile(cal_ratio["score_mean_kl"].to_numpy(), alpha)
        source = "global_fallback"
    else:
        q_conf = conformal_quantile(cal_bucket["score_mean_kl"].to_numpy(), alpha)
        source = "bucket"

    return float(q_conf), source, int(len(cal_bucket))


def run_bucketed_baseline(
    split_frames: SplitFrames,
    candidate_ratios: list[float],
    tau_raw: float,
    alpha: float,
    bucket_column: str,
    min_bucket_calibration: int,
) -> dict[str, Any]:
    selected_rows = []
    bucket_summaries = []
    for input_id, group in split_frames.test.groupby("input_id", sort=True):
        ordered = group.sort_values("ratio").copy()
        bucket_value = str(ordered[bucket_column].iloc[0])
        qhat_by_ratio = {}
        quantile_source = {}
        bucket_counts = {}
        feasible = []
        for ratio in candidate_ratios:
            qhat, source, bucket_count = lookup_bucket_quantile(
                split_frames.calibration,
                ratio=ratio,
                bucket_column=bucket_column,
                bucket_value=bucket_value,
                alpha=alpha,
                min_bucket_calibration=min_bucket_calibration,
            )
            qhat_by_ratio[ratio] = qhat
            quantile_source[str(ratio)] = source
            bucket_counts[str(ratio)] = bucket_count
            if qhat <= tau_raw:
                feasible.append(ratio)
        selected_ratio, selection_mode = select_ratio_from_quantiles(feasible, qhat_by_ratio)
        selected_row = ordered[np.isclose(ordered["ratio"], selected_ratio)].iloc[0].copy()
        selected_row["selected_ratio"] = selected_ratio
        selected_row["selected_bucket"] = bucket_value
        selected_row["selection_mode"] = selection_mode
        selected_rows.append(selected_row)
        bucket_summaries.append(
            {
                "input_id": int(input_id),
                "bucket_value": bucket_value,
                "selected_ratio": float(selected_ratio),
                "selection_mode": selection_mode,
                "quantiles": {str(ratio): qhat for ratio, qhat in qhat_by_ratio.items()},
                "quantile_source": quantile_source,
                "bucket_calibration_counts": bucket_counts,
            }
        )

    selected_df = pd.DataFrame(selected_rows)
    coverage = float((selected_df["score_mean_kl"] <= tau_raw).mean())
    return {
        "baseline_family": "conditional_split_cp",
        "variant": bucket_column,
        "conditioning": bucket_column,
        "transform": "raw",
        "selected_ratio": float(selected_df["selected_ratio"].mode().iloc[0]),
        "avg_selected_ratio": float(selected_df["selected_ratio"].mean()),
        "empirical_coverage": coverage,
        "bucket_trace": bucket_summaries,
        "selection_mode_counts": selected_df["selection_mode"].value_counts().to_dict(),
        "coverage_by_bucket": selected_df.groupby("selected_bucket")["score_mean_kl"].apply(lambda values: float((values <= tau_raw).mean())).to_dict(),
    }


def build_distribution_plot(
    output_path: Path,
    pooled_calibration_scores: np.ndarray,
    pooled_test_scores: np.ndarray,
    slice_label: str,
    slice_scores: np.ndarray,
    calibration_means_by_ratio: pd.Series,
    test_means_by_ratio: pd.Series,
    calibration_q90_by_ratio: pd.Series,
    test_q90_by_ratio: pd.Series,
) -> dict[str, Any]:
    mean = float(np.mean(pooled_calibration_scores))
    std = float(np.std(pooled_calibration_scores, ddof=0))
    jb_stat, jb_pvalue = jarque_bera_pvalue(pooled_calibration_scores)
    normal = NormalDist(mu=mean, sigma=std if std > 1e-8 else 1.0)
    slice_mean = float(np.mean(slice_scores))
    slice_std = float(np.std(slice_scores, ddof=0))
    slice_normal = NormalDist(mu=slice_mean, sigma=slice_std if slice_std > 1e-8 else 1.0)
    slice_jb_stat, slice_jb_pvalue = jarque_bera_pvalue(slice_scores)

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].hist(pooled_calibration_scores, bins=18, color="#1B998B", alpha=0.75, density=True)
    xs = np.linspace(pooled_calibration_scores.min(), pooled_calibration_scores.max(), 200)
    ys = np.array([normal.pdf(float(x)) for x in xs])
    axes[0, 0].plot(xs, ys, color="#BC4749", linewidth=2)
    axes[0, 0].set_title(f"Pooled Calibration Scores (JB p={jb_pvalue:.2e})")
    axes[0, 0].set_xlabel("score_mean_kl")

    sorted_scores = np.sort(slice_scores)
    probs = (np.arange(1, len(sorted_scores) + 1) - 0.5) / len(sorted_scores)
    normal_quantiles = np.array([slice_normal.inv_cdf(float(p)) for p in probs])
    axes[0, 1].scatter(normal_quantiles, sorted_scores, s=20, alpha=0.7, color="#0F4C5C")
    line_min = min(normal_quantiles.min(), sorted_scores.min())
    line_max = max(normal_quantiles.max(), sorted_scores.max())
    axes[0, 1].plot([line_min, line_max], [line_min, line_max], color="#BC4749", linestyle="--")
    axes[0, 1].set_title(f"QQ Plot: {slice_label} (JB p={slice_jb_pvalue:.2e})")
    axes[0, 1].set_xlabel("Normal quantiles")
    axes[0, 1].set_ylabel("Observed quantiles")

    axes[1, 0].plot(calibration_means_by_ratio.index, calibration_means_by_ratio.values, marker="o", label="Calibration", color="#1B998B")
    axes[1, 0].plot(test_means_by_ratio.index, test_means_by_ratio.values, marker="o", label="Test", color="#BC4749")
    axes[1, 0].set_title("Mean Score by Ratio")
    axes[1, 0].set_xlabel("Ratio")
    axes[1, 0].set_ylabel("Mean score_mean_kl")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(calibration_q90_by_ratio.index, calibration_q90_by_ratio.values, marker="o", label="Calibration q90", color="#0F4C5C")
    axes[1, 1].plot(test_q90_by_ratio.index, test_q90_by_ratio.values, marker="o", label="Test q90", color="#F4A259")
    axes[1, 1].set_title("Tail Drift by Ratio")
    axes[1, 1].set_xlabel("Ratio")
    axes[1, 1].set_ylabel("q90 score_mean_kl")
    axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return {
        "pooled_score_skewness": skewness(pooled_calibration_scores),
        "pooled_score_excess_kurtosis": excess_kurtosis(pooled_calibration_scores),
        "pooled_score_jarque_bera": jb_stat,
        "pooled_score_jarque_bera_pvalue": jb_pvalue,
        "slice_label": slice_label,
        "slice_score_jarque_bera": slice_jb_stat,
        "slice_score_jarque_bera_pvalue": slice_jb_pvalue,
    }


def build_transform_ablation_plot(output_path: Path, comparison_df: pd.DataFrame) -> None:
    baseline_a = comparison_df[comparison_df["baseline_family"] == "plain_split_cp"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(baseline_a["variant"], baseline_a["empirical_coverage"], color="#2B9348")
    axes[0].axhline(0.9, color="#BC4749", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Coverage by Score Transform")
    axes[0].set_ylabel("Empirical coverage")

    axes[1].bar(baseline_a["variant"], baseline_a["avg_selected_ratio"], color="#0F4C5C")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Selected Ratio by Score Transform")
    axes[1].set_ylabel("Average selected ratio")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def normality_summary(values: np.ndarray, label: str) -> dict[str, Any]:
    jb_stat, jb_pvalue = jarque_bera_pvalue(values)
    return {
        "label": label,
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "skewness": skewness(values),
        "excess_kurtosis": excess_kurtosis(values),
        "jarque_bera": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
    }


def collect_normality_summaries(
    split_frames: SplitFrames,
    split_frames_bucketed: SplitFrames,
) -> list[dict[str, Any]]:
    summaries = [normality_summary(split_frames.calibration["score_mean_kl"].to_numpy(), "pooled")]
    for ratio in sorted(split_frames.calibration["ratio"].unique()):
        subset = split_frames.calibration[np.isclose(split_frames.calibration["ratio"], ratio)]["score_mean_kl"].to_numpy()
        summaries.append(normality_summary(subset, f"ratio={ratio:.1f}"))
    for bucket_value, subset in split_frames_bucketed.calibration.groupby("difficulty_bucket"):
        values = subset["score_mean_kl"].to_numpy()
        if len(values) >= 8:
            summaries.append(normality_summary(values, f"difficulty_bucket={bucket_value}"))
    for bucket_value, subset in split_frames_bucketed.calibration.groupby("length_bucket"):
        values = subset["score_mean_kl"].to_numpy()
        if len(values) >= 8:
            summaries.append(normality_summary(values, f"length_bucket={bucket_value}"))
    return sorted(summaries, key=lambda item: item["jarque_bera_pvalue"])


def flatten_results_for_csv(
    baseline_a_results: list[dict[str, Any]],
    baseline_b_results: list[dict[str, Any]],
    uniform_summary: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for result in baseline_a_results + baseline_b_results:
        rows.append(
            {
                "baseline_family": result["baseline_family"],
                "variant": result["variant"],
                "conditioning": result["conditioning"],
                "transform": result["transform"],
                "selected_ratio": result["selected_ratio"],
                "avg_selected_ratio": result["avg_selected_ratio"],
                "empirical_coverage": result["empirical_coverage"],
                "selection_mode": result.get("selection_mode", "per_input"),
                "uniform_ratio_calibration": uniform_summary["uniform_ratio_calibration"],
                "uniform_coverage_calibration_selected": uniform_summary["uniform_coverage_calibration_selected"],
                "uniform_ratio_oracle_test": uniform_summary["uniform_ratio_oracle_test"],
                "uniform_coverage_oracle_test": uniform_summary["uniform_coverage_oracle_test"],
                "coverage_gap_vs_uniform": result["empirical_coverage"] - uniform_summary["uniform_coverage_calibration_selected"],
                "avg_ratio_gap_vs_uniform": result["avg_selected_ratio"] - uniform_summary["uniform_ratio_calibration"],
            }
        )
    return pd.DataFrame(rows)


def write_recommendations(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CP Baseline Recommendations",
        "",
        "## Direction 1: Monotone Global Ratio Calibration",
        "Why feasible here: the score-vs-ratio relation is globally stable and mostly monotone in the current data.",
        "Bottleneck addressed: the local model is unnecessary when ratio semantics already determine the safest action globally.",
        "Success would look like: a simple calibration rule that selects the same high ratio as the best uniform baseline while preserving >=90% coverage across held-out inputs.",
        "Next experiment: repeat the plain split CP baseline on another long-context corpus or a shifted WikiText slice to test whether the same monotone safe ratio generalizes.",
        "",
        "## Direction 2: Observable-Feature Bucketed Calibration",
        "Why feasible here: sequence length and prefix difficulty proxies are available without training a deep predictor and can define interpretable strata.",
        "Bottleneck addressed: per-input heterogeneity is likely real, but the current learned local model does not exploit it robustly.",
        "Success would look like: a bucketed policy that exceeds the uniform average ratio by at least 0.05 while keeping coverage within 0.03 of the target.",
        "Next experiment: increase bucket support with more grouped inputs and test a length x hidden-state-norm Mondrian baseline using only prefix-observable features.",
        "",
        "## Notes",
        f"- Plain split CP best coverage: {summary['plain_best_coverage']:.4f}",
        f"- Best bucketed coverage: {summary['bucketed_best_coverage']:.4f}",
        f"- Strongest observed non-normality slice: {summary['worst_slice_label']} (JB p={summary['worst_slice_pvalue']:.2e})",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_output_dir()

    if not args.scores_path.exists():
        raise FileNotFoundError(f"Missing scores file: {args.scores_path}")

    df = pd.read_csv(args.scores_path)
    split_frames = build_split_frames(df, args.seed)
    split_frames_bucketed, bucket_metadata = assign_bucket_metadata(split_frames)

    candidate_ratios = sorted(float(value) for value in df["ratio"].unique())
    reference_ratio = choose_reference_ratio(candidate_ratios, args.reference_ratio)
    tau_raw = float(
        np.quantile(
            split_frames.calibration[np.isclose(split_frames.calibration["ratio"], reference_ratio)]["score_mean_kl"].to_numpy(),
            0.9,
        )
    )

    uniform_ratio_calibration = best_uniform_ratio(split_frames.calibration, candidate_ratios, tau_raw, args.alpha)
    uniform_coverage_calibration_selected = compute_uniform_coverage(split_frames.test, uniform_ratio_calibration, tau_raw)
    uniform_ratio_oracle_test = best_uniform_ratio_oracle(split_frames.test, candidate_ratios, tau_raw, args.alpha)
    uniform_coverage_oracle_test = compute_uniform_coverage(split_frames.test, uniform_ratio_oracle_test, tau_raw)

    transforms: list[ScoreTransform] = [
        RawTransform("raw"),
        LogTransform("log"),
        RankNormalizedTransform("rank_normalized"),
        WinsorizedTransform("winsorized"),
    ]
    baseline_a_results = [
        run_global_baseline(transform, split_frames, candidate_ratios, tau_raw, args.alpha) for transform in transforms
    ]

    baseline_b_results = [
        run_bucketed_baseline(
            split_frames_bucketed,
            candidate_ratios,
            tau_raw,
            args.alpha,
            bucket_column="length_bucket",
            min_bucket_calibration=args.min_bucket_calibration,
        ),
        run_bucketed_baseline(
            split_frames_bucketed,
            candidate_ratios,
            tau_raw,
            args.alpha,
            bucket_column="difficulty_bucket",
            min_bucket_calibration=args.min_bucket_calibration,
        ),
        run_bucketed_baseline(
            split_frames_bucketed,
            candidate_ratios,
            tau_raw,
            args.alpha,
            bucket_column="length_difficulty_bucket",
            min_bucket_calibration=args.min_bucket_calibration,
        ),
    ]

    comparison_df = flatten_results_for_csv(
        baseline_a_results,
        baseline_b_results,
        uniform_summary={
            "uniform_ratio_calibration": uniform_ratio_calibration,
            "uniform_coverage_calibration_selected": uniform_coverage_calibration_selected,
            "uniform_ratio_oracle_test": uniform_ratio_oracle_test,
            "uniform_coverage_oracle_test": uniform_coverage_oracle_test,
        },
    )
    comparison_df.to_csv(output_dir / "cp_baseline_comparison.csv", index=False)
    build_transform_ablation_plot(output_dir / "cp_transform_ablation.png", comparison_df)

    calibration_ref_scores = split_frames.calibration[np.isclose(split_frames.calibration["ratio"], reference_ratio)]["score_mean_kl"].to_numpy()
    test_ref_scores = split_frames.test[np.isclose(split_frames.test["ratio"], reference_ratio)]["score_mean_kl"].to_numpy()
    normality_summaries = collect_normality_summaries(split_frames, split_frames_bucketed)
    worst_normality_slice = normality_summaries[0]
    distribution_summary = build_distribution_plot(
        output_path=output_dir / "cp_distribution_diagnostics.png",
        pooled_calibration_scores=split_frames.calibration["score_mean_kl"].to_numpy(),
        pooled_test_scores=split_frames.test["score_mean_kl"].to_numpy(),
        slice_label=worst_normality_slice["label"],
        slice_scores=(
            split_frames_bucketed.calibration[split_frames_bucketed.calibration["difficulty_bucket"] == worst_normality_slice["label"].split("=", 1)[1]]["score_mean_kl"].to_numpy()
            if worst_normality_slice["label"].startswith("difficulty_bucket=")
            else split_frames_bucketed.calibration[split_frames_bucketed.calibration["length_bucket"] == worst_normality_slice["label"].split("=", 1)[1]]["score_mean_kl"].to_numpy()
            if worst_normality_slice["label"].startswith("length_bucket=")
            else split_frames.calibration[np.isclose(split_frames.calibration["ratio"], float(worst_normality_slice["label"].split("=", 1)[1]))]["score_mean_kl"].to_numpy()
            if worst_normality_slice["label"].startswith("ratio=")
            else split_frames.calibration["score_mean_kl"].to_numpy()
        ),
        calibration_means_by_ratio=split_frames.calibration.groupby("ratio")["score_mean_kl"].mean().sort_index(),
        test_means_by_ratio=split_frames.test.groupby("ratio")["score_mean_kl"].mean().sort_index(),
        calibration_q90_by_ratio=split_frames.calibration.groupby("ratio")["score_mean_kl"].quantile(0.9).sort_index(),
        test_q90_by_ratio=split_frames.test.groupby("ratio")["score_mean_kl"].quantile(0.9).sort_index(),
    )

    calibration_ratio_means = split_frames.calibration.groupby("ratio")["score_mean_kl"].mean().sort_index()
    test_ratio_means = split_frames.test.groupby("ratio")["score_mean_kl"].mean().sort_index()
    ratio_semantics_direction = (
        "decreasing"
        if np.all(np.diff(calibration_ratio_means.to_numpy()) <= 1e-6)
        else "increasing"
        if np.all(np.diff(calibration_ratio_means.to_numpy()) >= -1e-6)
        else "mixed"
    )

    input_trends = []
    input_corrs = []
    input_ranges = []
    for _, group in df.groupby("input_id"):
        ordered = group.sort_values("ratio")
        diffs = np.diff(ordered["score_mean_kl"].to_numpy())
        input_trends.append(np.all(diffs <= 1e-6))
        input_corrs.append(float(np.corrcoef(ordered["ratio"].to_numpy(), ordered["score_mean_kl"].to_numpy())[0, 1]))
        input_ranges.append(float(ordered["score_mean_kl"].max() - ordered["score_mean_kl"].min()))

    cal_lengths = split_frames.calibration.drop_duplicates("input_id")["sequence_length"].astype(int).value_counts().to_dict()
    test_lengths = split_frames.test.drop_duplicates("input_id")["sequence_length"].astype(int).value_counts().to_dict()
    cal_difficulty = split_frames_bucketed.calibration.drop_duplicates("input_id")["difficulty_bucket"].value_counts().to_dict()
    test_difficulty = split_frames_bucketed.test.drop_duplicates("input_id")["difficulty_bucket"].value_counts().to_dict()

    tail_summary = {}
    for ratio in candidate_ratios:
        cal_subset = split_frames.calibration[np.isclose(split_frames.calibration["ratio"], ratio)]["score_mean_kl"].to_numpy()
        test_subset = split_frames.test[np.isclose(split_frames.test["ratio"], ratio)]["score_mean_kl"].to_numpy()
        tail_summary[str(ratio)] = {
            "calibration_q90": float(np.quantile(cal_subset, 0.9)),
            "test_q90": float(np.quantile(test_subset, 0.9)),
            "calibration_q95": float(np.quantile(cal_subset, 0.95)),
            "test_q95": float(np.quantile(test_subset, 0.95)),
            "ks_distance": ks_distance(cal_subset, test_subset),
        }

    diagnostics = {
        "alpha": args.alpha,
        "reference_ratio": reference_ratio,
        "tau_raw": tau_raw,
        "uniform_ratio_calibration": uniform_ratio_calibration,
        "uniform_coverage_calibration_selected": uniform_coverage_calibration_selected,
        "uniform_ratio_oracle_test": uniform_ratio_oracle_test,
        "uniform_coverage_oracle_test": uniform_coverage_oracle_test,
        "ratio_semantics_direction": ratio_semantics_direction,
        "fraction_inputs_monotone_decreasing": float(np.mean(input_trends)),
        "fraction_inputs_negative_ratio_score_correlation": float(np.mean(np.asarray(input_corrs) < 0.0)),
        "fraction_inputs_positive_ratio_score_correlation": float(np.mean(np.asarray(input_corrs) > 0.0)),
        "mean_per_input_score_range": float(np.mean(input_ranges)),
        "median_per_input_score_range": float(np.median(input_ranges)),
        "length_distribution_tv_distance_calibration_vs_test": total_variation_from_counts(cal_lengths, test_lengths),
        "difficulty_distribution_tv_distance_calibration_vs_test": total_variation_from_counts(cal_difficulty, test_difficulty),
        "tail_summary_by_ratio": tail_summary,
        "distribution_summary": distribution_summary,
        "exchangeability_proxy_ks_ref_ratio": ks_distance(calibration_ref_scores, test_ref_scores),
        "exchangeability_proxy_ks_by_ratio": {
            str(ratio): ks_distance(
                split_frames.calibration[np.isclose(split_frames.calibration["ratio"], ratio)]["score_mean_kl"].to_numpy(),
                split_frames.test[np.isclose(split_frames.test["ratio"], ratio)]["score_mean_kl"].to_numpy(),
            )
            for ratio in candidate_ratios
        },
        "normality_summaries": normality_summaries,
        "norm_quantiles": bucket_metadata["norm_quantiles"],
    }

    save_json(
        output_dir / "cp_baseline_results.json",
        {
            "baseline_a_results": baseline_a_results,
            "diagnostics": diagnostics,
        },
    )
    save_json(
        output_dir / "cp_bucketed_results.json",
        {
            "baseline_b_results": baseline_b_results,
            "diagnostics": diagnostics,
        },
    )

    best_plain = max(baseline_a_results, key=lambda item: (item["empirical_coverage"], item["avg_selected_ratio"]))
    best_bucketed = max(baseline_b_results, key=lambda item: (item["empirical_coverage"], item["avg_selected_ratio"]))
    write_recommendations(
        output_dir / "cp_recommendations.md",
        summary={
            "plain_best_coverage": best_plain["empirical_coverage"],
            "bucketed_best_coverage": best_bucketed["empirical_coverage"],
            "worst_slice_label": worst_normality_slice["label"],
            "worst_slice_pvalue": worst_normality_slice["jarque_bera_pvalue"],
        },
    )

    print("Task 4 summary")
    print(f"reference_ratio={reference_ratio:.1f} tau_raw={tau_raw:.6f}")
    print("Baseline A")
    for result in baseline_a_results:
        print(
            f"  {result['variant']}: selected_ratio={result['selected_ratio']:.2f} "
            f"coverage={result['empirical_coverage']:.4f}"
        )
    print("Baseline B")
    for result in baseline_b_results:
        print(
            f"  {result['variant']}: avg_selected_ratio={result['avg_selected_ratio']:.4f} "
            f"coverage={result['empirical_coverage']:.4f}"
        )
    print(
        "Diagnostics: "
        f"ratio_semantics_direction={ratio_semantics_direction}, "
        f"worst_slice_JB_pvalue={worst_normality_slice['jarque_bera_pvalue']:.2e}, "
        f"fraction_inputs_monotone_decreasing={diagnostics['fraction_inputs_monotone_decreasing']:.4f}"
    )
    print(f"Saved JSON to {output_dir / 'cp_baseline_results.json'}")
    print(f"Saved CSV to {output_dir / 'cp_baseline_comparison.csv'}")
    print(f"Saved plot to {output_dir / 'cp_distribution_diagnostics.png'}")
    print(f"Saved plot to {output_dir / 'cp_transform_ablation.png'}")
    print(f"Saved JSON to {output_dir / 'cp_bucketed_results.json'}")
    print(f"Saved recommendations to {output_dir / 'cp_recommendations.md'}")


if __name__ == "__main__":
    main()
