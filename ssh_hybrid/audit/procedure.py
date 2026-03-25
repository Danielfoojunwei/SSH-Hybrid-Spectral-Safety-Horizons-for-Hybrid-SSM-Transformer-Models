"""Architectural Safety Audit Procedure.

Implements the full 3-phase audit: spectral radius measurement,
safety margin computation, and MBCA implementation with coverage measurement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ssh_hybrid.spectral.radius import (
    compute_layer_spectral_radii,
    compute_mean_spectral_radius,
    SpectralRadiusResult,
)
from ssh_hybrid.spectral.horizon import safety_memory_horizon, attenuation_factor
from ssh_hybrid.spectral.margin import spectral_safety_margin_bound, mbca_compensated_margin
from ssh_hybrid.mbca.register import MBCARegister
from ssh_hybrid.mbca.probes import train_safety_probes, extract_attention_hidden_states
from ssh_hybrid.mbca.monitor import MBCAMonitor

logger = logging.getLogger(__name__)


@dataclass
class AuditResult:
    """Complete result from an architectural safety audit."""

    model_name: str
    model_type: str
    r_ssm: float

    # Phase 1: Spectral radius
    spectral_radii: list[SpectralRadiusResult] = field(default_factory=list)
    mean_rho: float = 0.0
    H_rho: float = float("inf")
    phase1_time_ms: float = 0.0

    # Phase 2: Safety margin
    attenuation_factor_g: float = 0.0
    delta_star_hybrid: float = 0.0
    delta_star_transformer: float = 0.0
    L: float = 512.0
    phase2_time_ms: float = 0.0

    # Phase 3: MBCA
    beta_mbca: float = 0.0
    delta_star_compensated: float = 0.0
    K: int = 0
    mbca_benign_degradation: float = 0.0
    phase3_time_ms: float = 0.0

    # Overall
    total_time_s: float = 0.0
    safety_rank: int = 0  # lower = less safe


class ArchitecturalSafetyAudit:
    """Full architectural safety audit for hybrid SSM-Transformer models.

    Runs all three phases:
    1. Spectral radius measurement (SpectralGuard methodology)
    2. Safety margin computation (Theorem 1)
    3. MBCA implementation and coverage measurement (Theorem 2)

    Args:
        delta_star_transformer: Baseline Transformer safety margin.
        L: Mean oversight interaction length in tokens.
        epsilon: Safety signal threshold.
        K: Number of MBCA probes.
        safety_formula: MBCA safety formula type.
    """

    def __init__(
        self,
        delta_star_transformer: float = 1.0,
        L: float = 512.0,
        epsilon: float = 0.01,
        K: int = 8,
        safety_formula: str = "any",
    ):
        self.delta_star_transformer = delta_star_transformer
        self.L = L
        self.epsilon = epsilon
        self.K = K
        self.safety_formula = safety_formula

    def run_phase1(
        self,
        model: torch.nn.Module,
        model_type: str,
    ) -> tuple[list[SpectralRadiusResult], float, float, float]:
        """Phase 1: Spectral Radius Measurement.

        Uses SpectralGuard methodology to compute rho for each SSM layer
        and derive H(rho).

        Returns:
            Tuple of (per_layer_results, mean_rho, H_rho, elapsed_ms).
        """
        t0 = time.perf_counter()

        results = compute_layer_spectral_radii(model, model_type)
        mean_rho = compute_mean_spectral_radius(results)
        H_rho = safety_memory_horizon(mean_rho, self.epsilon)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Phase 1 complete: rho=%.6f, H(rho)=%.1f tokens, time=%.1fms",
            mean_rho, H_rho, elapsed_ms,
        )

        return results, mean_rho, H_rho, elapsed_ms

    def run_phase2(
        self,
        mean_rho: float,
        r_ssm: float,
    ) -> tuple[float, float, float]:
        """Phase 2: Safety Margin Computation.

        Computes attenuation factor g and safety margin bound.

        Returns:
            Tuple of (g, delta_star_hybrid, elapsed_ms).
        """
        t0 = time.perf_counter()

        g = attenuation_factor(mean_rho, r_ssm, self.L, self.epsilon)
        delta_star_hybrid = spectral_safety_margin_bound(
            self.delta_star_transformer, mean_rho, r_ssm, self.L, self.epsilon,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Phase 2 complete: g=%.4f, Delta*(hybrid)=%.4f, time=%.1fms",
            g, delta_star_hybrid, elapsed_ms,
        )

        return g, delta_star_hybrid, elapsed_ms

    def run_phase3(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        model_type: str,
        mean_rho: float,
        r_ssm: float,
        training_texts: list[str],
        training_labels: torch.Tensor,
        attack_inputs: list[torch.Tensor],
        attack_masks: list[torch.Tensor] | None = None,
        hidden_dim: int | None = None,
        device: str = "cuda",
    ) -> tuple[float, float, float]:
        """Phase 3: MBCA Implementation and Coverage Measurement.

        Trains safety probes, creates MBCA register, and measures beta_MBCA.

        Returns:
            Tuple of (beta_mbca, delta_star_compensated, elapsed_ms).
        """
        t0 = time.perf_counter()

        # Extract attention hidden states for probe training
        logger.info("Extracting attention hidden states for probe training...")
        hidden_states = extract_attention_hidden_states(
            model=model,
            tokenizer=tokenizer,
            texts=training_texts,
            model_type=model_type,
            device=device,
        )

        if hidden_dim is None:
            hidden_dim = hidden_states.shape[1]

        # Train safety probes
        logger.info("Training %d safety probes...", self.K)
        trained_linear, probe_results = train_safety_probes(
            hidden_states=hidden_states,
            labels=training_labels,
            K=self.K,
            hidden_dim=hidden_dim,
            device=device,
        )

        # Create MBCA register with trained probes
        register = MBCARegister(
            K=self.K,
            hidden_dim=hidden_dim,
            safety_formula=self.safety_formula,
        )
        register.probes.weight.data = trained_linear.weight.data.clone()
        register.probes.bias.data = trained_linear.bias.data.clone()
        register = register.to(device)

        # Create monitor and measure beta_MBCA
        monitor = MBCAMonitor(
            model=model,
            register=register,
            model_type=model_type,
        )

        beta_mbca = monitor.measure_beta_mbca(
            attack_inputs=attack_inputs,
            attack_masks=attack_masks,
        )

        # Compute compensated margin
        delta_star_compensated = mbca_compensated_margin(
            self.delta_star_transformer, mean_rho, r_ssm, beta_mbca,
            self.L, self.epsilon,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Phase 3 complete: beta_MBCA=%.3f, Delta*(compensated)=%.4f, time=%.1fms",
            beta_mbca, delta_star_compensated, elapsed_ms,
        )

        return beta_mbca, delta_star_compensated, elapsed_ms

    def run_full_audit(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        model_name: str,
        model_type: str,
        r_ssm: float,
        training_texts: list[str] | None = None,
        training_labels: torch.Tensor | None = None,
        attack_inputs: list[torch.Tensor] | None = None,
        attack_masks: list[torch.Tensor] | None = None,
        hidden_dim: int | None = None,
        device: str = "cuda",
        skip_phase3: bool = False,
    ) -> AuditResult:
        """Run complete 3-phase architectural safety audit.

        Args:
            model: The model to audit.
            tokenizer: Associated tokenizer.
            model_name: Model identifier.
            model_type: Architecture type.
            r_ssm: SSM layer fraction.
            training_texts: Texts for probe training (Phase 3).
            training_labels: Labels for probe training (Phase 3).
            attack_inputs: HiSPA attack inputs for beta measurement (Phase 3).
            attack_masks: Attention masks for attack inputs (Phase 3).
            hidden_dim: Model hidden dimension (auto-detected if None).
            device: Computation device.
            skip_phase3: Skip MBCA phase (for models without SSM layers).

        Returns:
            Complete AuditResult.
        """
        t_start = time.perf_counter()

        result = AuditResult(
            model_name=model_name,
            model_type=model_type,
            r_ssm=r_ssm,
            delta_star_transformer=self.delta_star_transformer,
            L=self.L,
            K=self.K,
        )

        # Phase 1
        spectral_results, mean_rho, H_rho, p1_time = self.run_phase1(model, model_type)
        result.spectral_radii = spectral_results
        result.mean_rho = mean_rho
        result.H_rho = H_rho
        result.phase1_time_ms = p1_time

        # Phase 2
        g, delta_hybrid, p2_time = self.run_phase2(mean_rho, r_ssm)
        result.attenuation_factor_g = g
        result.delta_star_hybrid = delta_hybrid
        result.phase2_time_ms = p2_time

        # Phase 3 (skip for pure Transformer or if no training data)
        if not skip_phase3 and r_ssm > 0 and training_texts and training_labels is not None:
            beta, delta_comp, p3_time = self.run_phase3(
                model, tokenizer, model_type, mean_rho, r_ssm,
                training_texts, training_labels,
                attack_inputs or [], attack_masks,
                hidden_dim, device,
            )
            result.beta_mbca = beta
            result.delta_star_compensated = delta_comp
            result.phase3_time_ms = p3_time
        else:
            result.delta_star_compensated = delta_hybrid

        result.total_time_s = time.perf_counter() - t_start

        logger.info(
            "Audit complete for %s: rho=%.4f, H=%.1f, g=%.4f, "
            "Delta*(hybrid)=%.4f, beta=%.3f, total=%.1fs",
            model_name, mean_rho, H_rho, g, delta_hybrid,
            result.beta_mbca, result.total_time_s,
        )

        return result


def rank_models_by_safety(audit_results: list[AuditResult]) -> list[AuditResult]:
    """Rank models by safety margin (highest = safest).

    Expected ranking: Pythia (r_SSM=0) > Zamba (0.85) > Jamba (0.875) > Mamba (1.0).

    Args:
        audit_results: List of audit results for different models.

    Returns:
        Sorted list (safest first), with safety_rank populated.
    """
    sorted_results = sorted(
        audit_results,
        key=lambda r: r.delta_star_hybrid,
        reverse=True,
    )

    for rank, result in enumerate(sorted_results, start=1):
        result.safety_rank = rank

    return sorted_results
