"""Experiment 1: Spectral Radius Measurement and H(rho) Computation.

Objective: Measure rho for all four models. Compute H(rho) and predicted Delta*(pi_hybrid).

Expected ranking: H(rho, Pythia) = inf > H(rho, Zamba) > H(rho, Jamba) > H(rho, Mamba)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch

from ssh_hybrid.models.loader import load_model, get_model_config
from ssh_hybrid.models.config import MODEL_REGISTRY
from ssh_hybrid.spectral.radius import compute_layer_spectral_radii, compute_mean_spectral_radius
from ssh_hybrid.spectral.horizon import safety_memory_horizon, attenuation_factor
from ssh_hybrid.spectral.margin import spectral_safety_margin_bound

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["jamba-1.5-mini", "zamba-7b", "pythia-2.8b", "mamba-2.8b"]
DEFAULT_L = 512
DEFAULT_EPSILON = 0.01
DEFAULT_DELTA_STAR_TRANSFORMER = 1.0


def run_experiment1(
    model_names: list[str] | None = None,
    L: float = DEFAULT_L,
    epsilon: float = DEFAULT_EPSILON,
    delta_star_transformer: float = DEFAULT_DELTA_STAR_TRANSFORMER,
    device: str = "auto",
    output_dir: str = "results/exp1",
) -> pd.DataFrame:
    """Run Experiment 1: Spectral Radius Measurement.

    Args:
        model_names: Models to evaluate. None = all 4.
        L: Mean oversight interaction length.
        epsilon: Safety signal threshold.
        delta_star_transformer: Transformer baseline safety margin.
        device: Device for model loading.
        output_dir: Directory for result output.

    Returns:
        DataFrame with per-model results.
    """
    model_names = model_names or DEFAULT_MODELS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for model_name in model_names:
        config = get_model_config(model_name)
        logger.info("=" * 60)
        logger.info("Processing model: %s (r_SSM=%.3f)", model_name, config.r_ssm)
        logger.info("=" * 60)

        t0 = time.perf_counter()

        # Load model
        loaded = load_model(model_name, device=device, load_tokenizer=False)
        model = loaded["model"]

        # Compute spectral radii
        layer_results = compute_layer_spectral_radii(model, config.model_type)
        mean_rho = compute_mean_spectral_radius(layer_results)

        # Compute derived quantities
        H_rho = safety_memory_horizon(mean_rho, epsilon)
        g = attenuation_factor(mean_rho, config.r_ssm, L, epsilon)
        delta_star_hybrid = spectral_safety_margin_bound(
            delta_star_transformer, mean_rho, config.r_ssm, L, epsilon,
        )

        elapsed_s = time.perf_counter() - t0

        result = {
            "model": model_name,
            "model_type": config.model_type,
            "r_ssm": config.r_ssm,
            "n_ssm_layers": len(layer_results),
            "mean_rho": mean_rho,
            "H_rho": H_rho,
            "attenuation_factor_g": g,
            "delta_star_hybrid": delta_star_hybrid,
            "delta_star_transformer": delta_star_transformer,
            "L": L,
            "epsilon": epsilon,
            "computation_time_s": elapsed_s,
        }

        # Per-layer details
        for lr in layer_results:
            result[f"rho_layer_{lr.layer_index}"] = lr.spectral_radius
            result[f"time_ms_layer_{lr.layer_index}"] = lr.computation_time_ms

        results.append(result)

        logger.info(
            "  rho=%.6f, H(rho)=%.1f tokens, g=%.4f, Delta*(hybrid)=%.4f, time=%.1fs",
            mean_rho, H_rho, g, delta_star_hybrid, elapsed_s,
        )

        # Free model memory
        del model, loaded
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    df = pd.DataFrame(results)

    # Save results
    df.to_csv(output_path / "exp1_spectral_radius.csv", index=False)
    with open(output_path / "exp1_spectral_radius.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 1 RESULTS: Spectral Radius and Safety Horizon")
    logger.info("=" * 80)
    summary_cols = ["model", "r_ssm", "mean_rho", "H_rho", "attenuation_factor_g", "delta_star_hybrid"]
    logger.info("\n%s", df[summary_cols].to_string(index=False))

    # Verify expected ranking
    ranked = df.sort_values("H_rho", ascending=False)
    logger.info("\nSafety horizon ranking (highest = safest):")
    for i, row in ranked.iterrows():
        logger.info("  %d. %s: H(rho) = %.1f tokens", i + 1, row["model"], row["H_rho"])

    return df


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Spectral Radius Measurement")
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate")
    parser.add_argument("--L", type=float, default=DEFAULT_L, help="Mean interaction length")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Signal threshold")
    parser.add_argument("--delta-star", type=float, default=DEFAULT_DELTA_STAR_TRANSFORMER)
    parser.add_argument("--device", default="auto", help="Device for model loading")
    parser.add_argument("--output-dir", default="results/exp1", help="Output directory")
    args = parser.parse_args()

    run_experiment1(
        model_names=args.models,
        L=args.L,
        epsilon=args.epsilon,
        delta_star_transformer=args.delta_star,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
