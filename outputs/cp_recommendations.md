# CP Baseline Recommendations

## Direction 1: Monotone Global Ratio Calibration
Why feasible here: the score-vs-ratio relation is globally stable and mostly monotone in the current data.
Bottleneck addressed: the local model is unnecessary when ratio semantics already determine the safest action globally.
Success would look like: a simple calibration rule that selects the same high ratio as the best uniform baseline while preserving >=90% coverage across held-out inputs.
Next experiment: repeat the plain split CP baseline on another long-context corpus or a shifted WikiText slice to test whether the same monotone safe ratio generalizes.

## Direction 2: Observable-Feature Bucketed Calibration
Why feasible here: sequence length and prefix difficulty proxies are available without training a deep predictor and can define interpretable strata.
Bottleneck addressed: per-input heterogeneity is likely real, but the current learned local model does not exploit it robustly.
Success would look like: a bucketed policy that exceeds the uniform average ratio by at least 0.05 while keeping coverage within 0.03 of the target.
Next experiment: increase bucket support with more grouped inputs and test a length x hidden-state-norm Mondrian baseline using only prefix-observable features.

## Notes
- Plain split CP best coverage: 0.9583
- Best bucketed coverage: 0.9583
- Strongest observed non-normality slice: difficulty_bucket=high (JB p=3.10e-02)
