"""Tests for spectral analysis module.

Tests core mathematical computations: spectral radius, safety memory horizon,
attenuation factor, and safety margin theorems. Uses real numpy/scipy
computations — no mocks.
"""

import math

import numpy as np
import pytest
import torch
from scipy import linalg as la

from ssh_hybrid.spectral.radius import (
    compute_spectral_radius,
    compute_spectral_radius_diagonal,
    discretize_A,
)
from ssh_hybrid.spectral.horizon import (
    safety_memory_horizon,
    attenuation_factor,
    signal_strength_at_distance,
)
from ssh_hybrid.spectral.margin import (
    spectral_safety_margin_bound,
    mbca_compensated_margin,
    safety_margin_deficit,
)


class TestSpectralRadius:
    """Tests for spectral radius computation on real matrices."""

    def test_identity_matrix_has_radius_one(self):
        A = np.eye(4)
        assert compute_spectral_radius(A) == pytest.approx(1.0)

    def test_zero_matrix_has_radius_zero(self):
        A = np.zeros((4, 4))
        assert compute_spectral_radius(A) == pytest.approx(0.0)

    def test_diagonal_matrix(self):
        diag = np.array([0.95, 0.90, 0.85, 0.80])
        rho = compute_spectral_radius_diagonal(diag)
        assert rho == pytest.approx(0.95)

    def test_complex_diagonal(self):
        diag = np.array([0.9 + 0.1j, 0.8 - 0.2j, 0.7 + 0.3j])
        rho = compute_spectral_radius_diagonal(diag)
        expected = max(abs(0.9 + 0.1j), abs(0.8 - 0.2j), abs(0.7 + 0.3j))
        assert rho == pytest.approx(expected)

    def test_known_eigenvalue_matrix(self):
        # Matrix with known eigenvalues 2, 3
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        assert compute_spectral_radius(A) == pytest.approx(3.0)

    def test_rotation_matrix_has_radius_one(self):
        theta = np.pi / 4
        A = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        assert compute_spectral_radius(A) == pytest.approx(1.0, abs=1e-10)

    def test_contraction_matrix(self):
        # Diagonal with values < 1 (stable SSM)
        A = np.diag([0.99, 0.98, 0.97, 0.96])
        rho = compute_spectral_radius(A)
        assert rho == pytest.approx(0.99)
        assert rho < 1.0

    def test_torch_tensor_input(self):
        A = torch.tensor([0.95, 0.90, 0.85])
        rho = compute_spectral_radius(A)
        assert rho == pytest.approx(0.95)

    def test_torch_matrix_input(self):
        A = torch.eye(3) * 0.9
        rho = compute_spectral_radius(A)
        assert rho == pytest.approx(0.9)

    def test_discretize_diagonal(self):
        A = torch.tensor([-1.0, -2.0, -3.0])
        delta = torch.tensor([0.1, 0.1, 0.1])
        A_bar = discretize_A(A, delta)
        expected = torch.exp(delta * A)
        assert torch.allclose(A_bar, expected)
        # All values should be < 1 (stable)
        assert (A_bar.abs() < 1.0).all()


class TestSafetyMemoryHorizon:
    """Tests for H(rho) computation."""

    def test_high_rho_short_horizon(self):
        # rho close to 1 => long horizon
        H = safety_memory_horizon(0.999)
        assert H > 1000

    def test_low_rho_long_horizon(self):
        # rho far from 1 => short horizon
        H = safety_memory_horizon(0.5)
        assert H < 20

    def test_exact_computation(self):
        # H(0.99) = log(100) / log(1/0.99) = log(100) / log(100/99)
        H = safety_memory_horizon(0.99, epsilon=0.01)
        expected = math.log(100) / math.log(1 / 0.99)
        assert H == pytest.approx(expected)

    def test_rho_zero_infinite_horizon(self):
        assert safety_memory_horizon(0.0) == float("inf")

    def test_rho_one_infinite_horizon(self):
        assert safety_memory_horizon(1.0) == float("inf")

    def test_rho_greater_than_one_infinite(self):
        assert safety_memory_horizon(1.1) == float("inf")

    def test_typical_ssm_rho(self):
        # Typical Mamba rho ~ 0.995
        H = safety_memory_horizon(0.995)
        # Should be on the order of hundreds of tokens
        assert 100 < H < 2000

    def test_different_epsilon(self):
        H1 = safety_memory_horizon(0.99, epsilon=0.01)
        H2 = safety_memory_horizon(0.99, epsilon=0.001)
        # Stricter threshold => longer horizon
        assert H2 > H1


class TestAttenuationFactor:
    """Tests for g(rho, r_SSM, L) computation."""

    def test_pure_transformer_zero_attenuation(self):
        # r_SSM = 0 => g = 0
        g = attenuation_factor(rho=0.99, r_ssm=0.0)
        assert g == 0.0

    def test_pure_ssm_maximum_attenuation(self):
        # r_SSM = 1, L >> H(rho) => g = 1.0
        g = attenuation_factor(rho=0.5, r_ssm=1.0, L=10000)
        assert g == pytest.approx(1.0)

    def test_hybrid_intermediate(self):
        g = attenuation_factor(rho=0.99, r_ssm=0.875, L=512)
        assert 0.0 < g < 1.0

    def test_short_interaction_low_attenuation(self):
        # L << H(rho) => g is small
        g = attenuation_factor(rho=0.999, r_ssm=0.875, L=10)
        assert g < 0.1

    def test_attenuation_increases_with_r_ssm(self):
        g_low = attenuation_factor(rho=0.99, r_ssm=0.3, L=512)
        g_high = attenuation_factor(rho=0.99, r_ssm=0.9, L=512)
        assert g_high > g_low

    def test_attenuation_increases_with_L(self):
        g_short = attenuation_factor(rho=0.99, r_ssm=0.875, L=100)
        g_long = attenuation_factor(rho=0.99, r_ssm=0.875, L=1000)
        assert g_long > g_short


class TestSignalStrength:
    """Tests for signal decay computation."""

    def test_no_decay_at_zero_distance(self):
        s = signal_strength_at_distance(0.99, tau=0)
        assert s == pytest.approx(1.0)

    def test_exponential_decay(self):
        rho = 0.9
        tau = 10
        s = signal_strength_at_distance(rho, tau)
        assert s == pytest.approx(0.9 ** 10)

    def test_full_decay_at_large_distance(self):
        s = signal_strength_at_distance(0.5, tau=100)
        assert s < 1e-20


class TestSafetyMarginBound:
    """Tests for Theorem 1 and Theorem 2."""

    def test_theorem1_pure_transformer(self):
        # r_SSM = 0 => Delta*(hybrid) = Delta*(transformer)
        delta = spectral_safety_margin_bound(1.0, rho=0.99, r_ssm=0.0)
        assert delta == pytest.approx(1.0)

    def test_theorem1_reduces_margin(self):
        delta = spectral_safety_margin_bound(1.0, rho=0.99, r_ssm=0.875)
        assert delta < 1.0

    def test_theorem1_higher_r_ssm_lower_margin(self):
        d1 = spectral_safety_margin_bound(1.0, rho=0.99, r_ssm=0.5)
        d2 = spectral_safety_margin_bound(1.0, rho=0.99, r_ssm=0.9)
        assert d2 < d1

    def test_theorem2_full_recovery(self):
        # beta_MBCA = 1 => full recovery
        delta = mbca_compensated_margin(1.0, rho=0.99, r_ssm=0.875, beta_mbca=1.0)
        assert delta == pytest.approx(1.0)

    def test_theorem2_no_recovery(self):
        # beta_MBCA = 0 => no improvement
        delta_hybrid = spectral_safety_margin_bound(1.0, rho=0.99, r_ssm=0.875)
        delta_comp = mbca_compensated_margin(1.0, rho=0.99, r_ssm=0.875, beta_mbca=0.0)
        assert delta_comp == pytest.approx(delta_hybrid)

    def test_theorem2_partial_recovery(self):
        delta_trans = 1.0
        delta_hybrid = spectral_safety_margin_bound(delta_trans, rho=0.99, r_ssm=0.875)
        delta_comp = mbca_compensated_margin(delta_trans, rho=0.99, r_ssm=0.875, beta_mbca=0.5)
        # Should be between hybrid and transformer
        assert delta_hybrid < delta_comp < delta_trans

    def test_deficit_computation(self):
        deficit = safety_margin_deficit(1.0, rho=0.99, r_ssm=0.875)
        delta = spectral_safety_margin_bound(1.0, rho=0.99, r_ssm=0.875)
        assert deficit == pytest.approx(1.0 - delta)

    def test_jamba_lower_margin_than_zamba(self):
        # Jamba has higher r_SSM (0.875) than Zamba (0.85) => lower margin
        d_jamba = spectral_safety_margin_bound(1.0, rho=0.99, r_ssm=0.875)
        d_zamba = spectral_safety_margin_bound(1.0, rho=0.99, r_ssm=0.850)
        assert d_jamba < d_zamba

    def test_expected_model_ranking(self):
        # Verify: Pythia > Zamba > Jamba > Mamba
        rho = 0.995  # typical SSM rho
        d_pythia = spectral_safety_margin_bound(1.0, rho=rho, r_ssm=0.0)
        d_zamba = spectral_safety_margin_bound(1.0, rho=rho, r_ssm=0.85)
        d_jamba = spectral_safety_margin_bound(1.0, rho=rho, r_ssm=0.875)
        d_mamba = spectral_safety_margin_bound(1.0, rho=rho, r_ssm=1.0)
        assert d_pythia > d_zamba > d_jamba > d_mamba
