"""Tests for the architectural safety audit procedure.

Tests Phase 1 and Phase 2 computations with real math — no mocks.
Phase 3 requires a loaded model so is tested at integration level.
"""

import pytest

from ssh_hybrid.audit.procedure import (
    ArchitecturalSafetyAudit,
    AuditResult,
    rank_models_by_safety,
)
from ssh_hybrid.spectral.radius import SpectralRadiusResult


class TestAuditPhase2:
    """Test Phase 2 safety margin computation."""

    def test_pure_transformer_no_attenuation(self):
        audit = ArchitecturalSafetyAudit(delta_star_transformer=1.0)
        g, delta, _ = audit.run_phase2(mean_rho=0.99, r_ssm=0.0)
        assert g == 0.0
        assert delta == pytest.approx(1.0)

    def test_hybrid_model_reduced_margin(self):
        audit = ArchitecturalSafetyAudit(delta_star_transformer=1.0)
        g, delta, _ = audit.run_phase2(mean_rho=0.99, r_ssm=0.875)
        assert 0 < g < 1
        assert 0 < delta < 1

    def test_pure_ssm_maximum_reduction(self):
        audit = ArchitecturalSafetyAudit(delta_star_transformer=1.0, L=10000)
        g, delta, _ = audit.run_phase2(mean_rho=0.5, r_ssm=1.0)
        assert g == pytest.approx(1.0)
        assert delta == pytest.approx(0.0)


class TestModelRanking:
    """Test model ranking by safety margin."""

    def test_expected_ranking_order(self):
        results = [
            AuditResult(model_name="mamba", model_type="mamba", r_ssm=1.0,
                        delta_star_hybrid=0.1),
            AuditResult(model_name="jamba", model_type="jamba", r_ssm=0.875,
                        delta_star_hybrid=0.3),
            AuditResult(model_name="pythia", model_type="pythia", r_ssm=0.0,
                        delta_star_hybrid=1.0),
            AuditResult(model_name="zamba", model_type="zamba", r_ssm=0.85,
                        delta_star_hybrid=0.35),
        ]

        ranked = rank_models_by_safety(results)

        assert ranked[0].model_name == "pythia"   # safest
        assert ranked[1].model_name == "zamba"
        assert ranked[2].model_name == "jamba"
        assert ranked[3].model_name == "mamba"     # least safe

    def test_rank_numbers_assigned(self):
        results = [
            AuditResult(model_name="a", model_type="t", r_ssm=0.0, delta_star_hybrid=0.5),
            AuditResult(model_name="b", model_type="t", r_ssm=0.5, delta_star_hybrid=0.8),
        ]

        ranked = rank_models_by_safety(results)
        assert ranked[0].safety_rank == 1
        assert ranked[1].safety_rank == 2

    def test_jamba_less_safe_than_zamba(self):
        """Jamba (r_SSM=0.875) should rank below Zamba (r_SSM=0.85)."""
        audit = ArchitecturalSafetyAudit(delta_star_transformer=1.0)
        _, d_jamba, _ = audit.run_phase2(mean_rho=0.995, r_ssm=0.875)
        _, d_zamba, _ = audit.run_phase2(mean_rho=0.995, r_ssm=0.850)

        results = [
            AuditResult(model_name="jamba", model_type="jamba", r_ssm=0.875,
                        delta_star_hybrid=d_jamba),
            AuditResult(model_name="zamba", model_type="zamba", r_ssm=0.85,
                        delta_star_hybrid=d_zamba),
        ]

        ranked = rank_models_by_safety(results)
        assert ranked[0].model_name == "zamba"  # safer
        assert ranked[1].model_name == "jamba"  # less safe
