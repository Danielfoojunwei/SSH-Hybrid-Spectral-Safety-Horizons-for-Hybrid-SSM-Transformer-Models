"""Tests for CHSS computation utilities."""

import pytest

from ssh_hybrid.evaluation.chss import compute_chss_degradation, CHSSResult, _std


class TestCHSSDegradation:
    """Tests for CHSS degradation computation."""

    def test_no_degradation(self):
        clean = CHSSResult("model", "clean", chss_score=0.9, chss_per_layer=[], n_samples=10, std_dev=0.01)
        attack = CHSSResult("model", "attack", chss_score=0.9, chss_per_layer=[], n_samples=10, std_dev=0.01)
        assert compute_chss_degradation(clean, attack) == pytest.approx(0.0)

    def test_full_degradation(self):
        clean = CHSSResult("model", "clean", chss_score=1.0, chss_per_layer=[], n_samples=10, std_dev=0.0)
        attack = CHSSResult("model", "attack", chss_score=0.0, chss_per_layer=[], n_samples=10, std_dev=0.0)
        assert compute_chss_degradation(clean, attack) == pytest.approx(100.0)

    def test_partial_degradation(self):
        clean = CHSSResult("model", "clean", chss_score=0.8, chss_per_layer=[], n_samples=10, std_dev=0.01)
        attack = CHSSResult("model", "attack", chss_score=0.4, chss_per_layer=[], n_samples=10, std_dev=0.01)
        assert compute_chss_degradation(clean, attack) == pytest.approx(50.0)

    def test_improvement_negative_degradation(self):
        clean = CHSSResult("model", "clean", chss_score=0.5, chss_per_layer=[], n_samples=10, std_dev=0.01)
        attack = CHSSResult("model", "attack", chss_score=0.8, chss_per_layer=[], n_samples=10, std_dev=0.01)
        degradation = compute_chss_degradation(clean, attack)
        assert degradation < 0  # improvement, not degradation

    def test_zero_clean_score(self):
        clean = CHSSResult("model", "clean", chss_score=0.0, chss_per_layer=[], n_samples=10, std_dev=0.0)
        attack = CHSSResult("model", "attack", chss_score=0.5, chss_per_layer=[], n_samples=10, std_dev=0.01)
        assert compute_chss_degradation(clean, attack) == 0.0


class TestStd:
    def test_empty(self):
        assert _std([]) == 0.0

    def test_single(self):
        assert _std([5.0]) == 0.0

    def test_known_values(self):
        result = _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        # Sample std dev of this dataset
        assert result == pytest.approx(2.138, abs=0.01)
