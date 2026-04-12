"""Statistical utilities for SSH-Hybrid experiments.

Provides bootstrap confidence intervals, sample size calculations,
and significance tests needed for rigorous empirical evaluation.
"""

from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    statistic: str = "mean",
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations (e.g., binary 0/1 for ASR).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        statistic: "mean" or "median".
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    stat_fn = np.mean if statistic == "mean" else np.median

    point = float(stat_fn(data))

    # Handle degenerate cases
    if np.all(data == data[0]):
        return point, point, point

    bootstrap_stats = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        resample = data[rng.randint(0, n, size=n)]
        bootstrap_stats[i] = stat_fn(resample)

    alpha = 1.0 - ci
    lo = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    hi = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point, lo, hi


def min_sample_size_proportion(
    p1: float = 0.10,
    p2: float = 0.20,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Compute minimum sample size per group for two-proportion z-test.

    Uses the formula:
        n = (z_alpha/2 + z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p2 - p1)^2

    Args:
        p1: Expected proportion in control group.
        p2: Expected proportion in treatment group.
        alpha: Significance level.
        power: Statistical power (1 - beta).

    Returns:
        Minimum sample size per group (rounded up).
    """
    from scipy import stats as sp_stats

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)

    effect = p2 - p1
    if abs(effect) < 1e-10:
        return 10000  # infinite for zero effect

    pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
    n = (z_alpha + z_beta) ** 2 * pooled_var / effect ** 2

    return int(math.ceil(n))


def compute_asr_with_ci(
    is_unsafe: list[bool] | np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> dict[str, float]:
    """Compute ASR with bootstrap confidence interval.

    Args:
        is_unsafe: Boolean array where True = model produced unsafe output.
        n_bootstrap: Bootstrap resamples.
        ci: Confidence level.

    Returns:
        Dict with asr, ci_lower, ci_upper, n_total, n_unsafe.
    """
    data = np.array(is_unsafe, dtype=float)
    asr, lo, hi = bootstrap_ci(data, n_bootstrap=n_bootstrap, ci=ci)

    return {
        "asr": asr,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_total": len(data),
        "n_unsafe": int(data.sum()),
        "ci_width": hi - lo,
    }


def required_samples_for_experiment(
    baseline_asr: float = 0.10,
    min_detectable_increase: float = 0.10,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, int]:
    """Compute required sample sizes for the distance-decay experiment.

    Args:
        baseline_asr: Expected ASR at distance=0.
        min_detectable_increase: Minimum ASR increase to detect.
        alpha: Significance level.
        power: Statistical power.

    Returns:
        Dict with per_distance and total sample requirements.
    """
    n_per_group = min_sample_size_proportion(
        p1=baseline_asr,
        p2=baseline_asr + min_detectable_increase,
        alpha=alpha,
        power=power,
    )

    return {
        "n_per_distance": n_per_group,
        "n_distances_recommended": 7,
        "n_models": 4,
        "total_generations": n_per_group * 7 * 4,
        "alpha": alpha,
        "power": power,
        "min_detectable_increase": min_detectable_increase,
        "baseline_asr": baseline_asr,
    }
