from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from common import detect_device, ensure_output_dir, save_json, set_seed


@dataclass
class SplitFrames:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    train_ids: list[int]
    calibration_ids: list[int]
    test_ids: list[int]


class QuantileMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(inputs))
        hidden = F.relu(self.fc2(hidden))
        output = F.softplus(self.fc3(hidden)).squeeze(-1) + 1e-8
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 3 local conformal calibration.")
    parser.add_argument("--scores-path", type=Path, default=Path("outputs/scores.csv"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def split_input_ids(input_ids: list[int]) -> tuple[list[int], list[int], list[int]]:
    total = len(input_ids)
    if total < 15:
        raise RuntimeError(
            "Task 3 needs substantially more grouped inputs for meaningful calibration. "
            "Rerun Task 2 with a larger --num-samples value."
        )

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


def choose_tau_ratio(candidate_ratios: list[float]) -> float:
    return min(candidate_ratios, key=lambda ratio: (abs(ratio - 0.5), 0 if ratio <= 0.5 else 1, ratio))


def prepare_feature_frames(
    split_frames: SplitFrames,
) -> tuple[SplitFrames, list[str], dict[str, float], dict[str, float], float]:
    feature_frames = SplitFrames(
        train=split_frames.train.copy(),
        calibration=split_frames.calibration.copy(),
        test=split_frames.test.copy(),
        train_ids=split_frames.train_ids,
        calibration_ids=split_frames.calibration_ids,
        test_ids=split_frames.test_ids,
    )

    for frame in (feature_frames.train, feature_frames.calibration, feature_frames.test):
        if "attention_entropy" not in frame.columns:
            frame["attention_entropy"] = math.nan
        frame["attention_entropy_missing"] = frame["attention_entropy"].isna().astype(float)
        frame["budget_tokens"] = frame["context_length"] * (1.0 - frame["ratio"])

    train_attention = feature_frames.train["attention_entropy"].dropna()
    attention_median = float(train_attention.median()) if not train_attention.empty else 0.0
    for frame in (feature_frames.train, feature_frames.calibration, feature_frames.test):
        frame["attention_entropy_filled"] = frame["attention_entropy"].fillna(attention_median)

    feature_names = [
        "ratio",
        "budget_tokens",
        "context_length",
        "sequence_length",
        "target_length",
        "mean_hidden_state_norm",
        "baseline_token_nll_mean",
        "baseline_logit_entropy_mean",
        "baseline_max_probability_mean",
        "actual_keep_fraction",
        "estimated_kv_cache_mb",
        "attention_entropy_filled",
        "attention_entropy_missing",
    ]

    feature_means = {name: float(feature_frames.train[name].mean()) for name in feature_names}
    feature_stds: dict[str, float] = {}
    for name in feature_names:
        value = float(feature_frames.train[name].std(ddof=0))
        feature_stds[name] = value if value > 0 else 1.0

    for frame in (feature_frames.train, feature_frames.calibration, feature_frames.test):
        for name in feature_names:
            frame[f"{name}_scaled"] = (frame[name] - feature_means[name]) / feature_stds[name]

    return feature_frames, feature_names, feature_means, feature_stds, attention_median


def frame_to_tensors(frame: pd.DataFrame, feature_names: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_names = [f"{name}_scaled" for name in feature_names]
    features = torch.tensor(frame[scaled_names].to_numpy(), dtype=torch.float32, device=device)
    targets = torch.tensor(frame["score_mean_kl"].to_numpy(), dtype=torch.float32, device=device)
    return features, targets


def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor, quantile: float) -> torch.Tensor:
    errors = targets - predictions
    return torch.maximum(quantile * errors, (quantile - 1.0) * errors).mean()


def train_model(
    train_frame: pd.DataFrame,
    feature_names: list[str],
    alpha: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> tuple[QuantileMLP, float]:
    quantile = 1.0 - alpha
    features, targets = frame_to_tensors(train_frame, feature_names, device)
    model = QuantileMLP(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    num_rows = features.shape[0]
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(num_rows, device=device)
        epoch_losses = []
        for start in range(0, num_rows, batch_size):
            index = permutation[start : start + batch_size]
            batch_features = features[index]
            batch_targets = targets[index]
            predictions = model(batch_features)
            loss = quantile_loss(predictions, batch_targets, quantile)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} quantile_loss={mean_loss:.6f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_loss


def predict_frame(model: QuantileMLP, frame: pd.DataFrame, feature_names: list[str], device: str) -> np.ndarray:
    features, _ = frame_to_tensors(frame, feature_names, device)
    with torch.no_grad():
        predictions = model(features).detach().cpu().numpy()
    return predictions


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = math.ceil((n + 1) * (1.0 - alpha)) - 1
    index = min(max(index, 0), n - 1)
    return float(sorted_values[index])


def infer_ratio_semantics_direction(calibration_df: pd.DataFrame) -> str:
    grouped = calibration_df.groupby("ratio")["score_mean_kl"].mean().sort_index()
    diffs = np.diff(grouped.to_numpy())
    if np.all(diffs >= -1e-6):
        return "increasing"
    if np.all(diffs <= 1e-6):
        return "decreasing"
    return "none"


def enforce_monotone_qhat(frame: pd.DataFrame, direction: str) -> pd.DataFrame:
    adjusted_groups = []
    for _, group in frame.groupby("input_id", sort=True):
        ordered = group.sort_values("ratio").copy()
        qhat = ordered["q_hat"].to_numpy()
        if direction == "increasing":
            qhat = np.maximum.accumulate(qhat)
        elif direction == "decreasing":
            qhat = np.minimum.accumulate(qhat)
        ordered["q_hat_policy"] = qhat
        adjusted_groups.append(ordered)
    return pd.concat(adjusted_groups, ignore_index=True)


def select_uniform_ratio(calibration_df: pd.DataFrame, candidate_ratios: list[float], tau: float, alpha: float) -> float:
    target_coverage = 1.0 - alpha
    eligible = []
    for ratio in candidate_ratios:
        subset = calibration_df[np.isclose(calibration_df["ratio"], ratio)]
        coverage = float((subset["score_mean_kl"] <= tau).mean())
        if coverage >= target_coverage:
            eligible.append(ratio)
    return max(eligible) if eligible else min(candidate_ratios)


def select_adaptive_rows(test_df: pd.DataFrame, tau: float) -> pd.DataFrame:
    selected_rows = []
    for _, group in test_df.groupby("input_id", sort=True):
        acceptable = group[group["q_hat_policy"] <= tau].sort_values("ratio")
        if acceptable.empty:
            selected_rows.append(group.sort_values("ratio").iloc[0])
        else:
            selected_rows.append(acceptable.iloc[-1])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def make_plot(uniform_ratio: float, adaptive_avg_ratio: float, uniform_coverage: float, adaptive_coverage: float, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Uniform", "Adaptive"], [uniform_ratio, adaptive_avg_ratio], color=["#8D99AE", "#2B9348"])
    axes[0].set_title("Average Compression Ratio")
    axes[0].set_ylim(0, 1.0)

    axes[1].bar(["Uniform", "Adaptive"], [uniform_coverage, adaptive_coverage], color=["#8D99AE", "#2B9348"])
    axes[1].set_title("Empirical Coverage")
    axes[1].set_ylim(0, 1.0)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_output_dir()

    if not args.scores_path.exists():
        raise FileNotFoundError(f"Missing scores file: {args.scores_path}")

    raw_df = pd.read_csv(args.scores_path)
    split_frames = build_split_frames(raw_df, args.seed)
    feature_frames, feature_names, feature_means, feature_stds, attention_median = prepare_feature_frames(split_frames)

    train_device = detect_device()
    print(f"Task 3: training quantile model on device={train_device}")
    try:
        model, best_loss = train_model(
            train_frame=feature_frames.train,
            feature_names=feature_names,
            alpha=args.alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=train_device,
        )
    except Exception as exc:
        if train_device != "mps":
            raise
        print(f"MPS training failed; retrying on CPU. Root cause: {type(exc).__name__}: {exc}")
        train_device = "cpu"
        model, best_loss = train_model(
            train_frame=feature_frames.train,
            feature_names=feature_names,
            alpha=args.alpha,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=train_device,
        )

    calibration_with_preds = feature_frames.calibration.copy()
    test_with_preds = feature_frames.test.copy()
    calibration_with_preds["pred_upper"] = predict_frame(model, calibration_with_preds, feature_names, train_device)
    test_with_preds["pred_upper"] = predict_frame(model, test_with_preds, feature_names, train_device)

    calibration_with_preds["nonconformity"] = calibration_with_preds["score_mean_kl"] - calibration_with_preds["pred_upper"]
    q_conformal = conformal_quantile(calibration_with_preds["nonconformity"].to_numpy(), args.alpha)
    calibration_with_preds["q_hat"] = np.maximum(calibration_with_preds["pred_upper"] + q_conformal, 0.0)
    test_with_preds["q_hat"] = np.maximum(test_with_preds["pred_upper"] + q_conformal, 0.0)
    ratio_semantics_direction = infer_ratio_semantics_direction(calibration_with_preds)
    calibration_with_preds = enforce_monotone_qhat(calibration_with_preds, ratio_semantics_direction)
    test_with_preds = enforce_monotone_qhat(test_with_preds, ratio_semantics_direction)

    candidate_ratios = sorted(float(value) for value in raw_df["ratio"].unique())
    tau_ratio_used = choose_tau_ratio(candidate_ratios)
    tau_subset = calibration_with_preds[np.isclose(calibration_with_preds["ratio"], tau_ratio_used)]
    tau = float(np.quantile(tau_subset["score_mean_kl"], 0.9))

    uniform_ratio = select_uniform_ratio(calibration_with_preds, candidate_ratios, tau, args.alpha)
    uniform_test = test_with_preds[np.isclose(test_with_preds["ratio"], uniform_ratio)].copy()
    uniform_coverage = float((uniform_test["score_mean_kl"] <= tau).mean())

    adaptive_policy_mode = "qhat_threshold"
    if ratio_semantics_direction == "decreasing" and np.isclose(uniform_ratio, max(candidate_ratios)):
        adaptive_policy_mode = "decreasing_semantics_fallback"
        adaptive_selected = uniform_test.copy()
    else:
        adaptive_selected = select_adaptive_rows(test_with_preds, tau)
    adaptive_avg_ratio = float(adaptive_selected["ratio"].mean())
    adaptive_coverage = float((adaptive_selected["score_mean_kl"] <= tau).mean())

    qhat_test_coverage = float((test_with_preds["score_mean_kl"] <= test_with_preds["q_hat"]).mean())
    coverage_gap = abs(adaptive_coverage - uniform_coverage)
    under_similar_coverage = coverage_gap <= 0.03
    adaptive_beats_uniform = adaptive_avg_ratio > (uniform_ratio + 1e-6) and under_similar_coverage
    adaptive_competitive = abs(adaptive_avg_ratio - uniform_ratio) <= 0.05 and under_similar_coverage

    make_plot(
        uniform_ratio=uniform_ratio,
        adaptive_avg_ratio=adaptive_avg_ratio,
        uniform_coverage=uniform_coverage,
        adaptive_coverage=adaptive_coverage,
        output_path=output_dir / "comparison_uniform_vs_adaptive.png",
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": feature_names,
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "attention_entropy_median": attention_median,
            "alpha": args.alpha,
            "best_quantile_loss": best_loss,
            "conformal_additive_quantile": q_conformal,
        },
        output_dir / "model_g_phi.pt",
    )

    results_payload = {
        "alpha": args.alpha,
        "tau": tau,
        "tau_ratio_used": tau_ratio_used,
        "ell_star": q_conformal,
        "candidate_ratios": candidate_ratios,
        "num_train": len(split_frames.train_ids),
        "num_calibration": len(split_frames.calibration_ids),
        "num_test": len(split_frames.test_ids),
        "uniform_ratio": uniform_ratio,
        "uniform_coverage": uniform_coverage,
        "adaptive_avg_ratio": adaptive_avg_ratio,
        "adaptive_coverage": adaptive_coverage,
        "model_features": feature_names,
        "best_quantile_loss": best_loss,
        "coverage_gap": coverage_gap,
        "qhat_test_coverage": qhat_test_coverage,
        "ratio_semantics_direction": ratio_semantics_direction,
        "adaptive_policy_mode": adaptive_policy_mode,
    }
    save_json(output_dir / "calibration_results.json", results_payload)

    print("\nTask 3 summary")
    print(f"train_inputs={len(split_frames.train_ids)} calibration_inputs={len(split_frames.calibration_ids)} test_inputs={len(split_frames.test_ids)}")
    print(f"tau={tau:.6f} using ratio={tau_ratio_used:.1f}")
    print(f"ell_star={q_conformal:.6f}")
    print(f"uniform_ratio={uniform_ratio:.1f} uniform_coverage={uniform_coverage:.4f}")
    print(f"adaptive_avg_ratio={adaptive_avg_ratio:.4f} adaptive_coverage={adaptive_coverage:.4f}")
    print(f"qhat_test_coverage={qhat_test_coverage:.4f}")
    print(f"ratio_semantics_direction={ratio_semantics_direction}")
    print(f"adaptive_policy_mode={adaptive_policy_mode}")
    if adaptive_beats_uniform:
        print("Adaptive selection achieved a higher average compression ratio than uniform under similar coverage.")
    elif adaptive_competitive:
        print("Adaptive selection matched the uniform policy closely under similar coverage.")
    elif adaptive_avg_ratio > uniform_ratio and not under_similar_coverage:
        print("Adaptive selection increased the average compression ratio, but coverage was not similar enough to claim a win.")
    else:
        print("Adaptive selection did not beat the uniform policy under the current calibration outcome.")
    print(f"Saved model to {output_dir / 'model_g_phi.pt'}")
    print(f"Saved JSON to {output_dir / 'calibration_results.json'}")
    print(f"Saved plot to {output_dir / 'comparison_uniform_vs_adaptive.png'}")


if __name__ == "__main__":
    main()
