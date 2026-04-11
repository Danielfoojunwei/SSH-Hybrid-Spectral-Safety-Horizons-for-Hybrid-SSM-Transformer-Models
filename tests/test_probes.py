"""Tests for safety probe training.

Uses real torch training loops with synthetic data — no mocks.
"""

import torch
import pytest

from ssh_hybrid.mbca.probes import train_safety_probes, SafetyProbe


class TestSafetyProbeTraining:
    """Tests for training K safety probes on real data."""

    def test_train_basic_probes(self):
        """Train probes on linearly separable synthetic data."""
        # Create linearly separable data
        n_samples = 200
        hidden_dim = 32

        # Positive class: mean at +1, negative class: mean at -1
        pos_data = torch.randn(n_samples // 2, hidden_dim) + 1.0
        neg_data = torch.randn(n_samples // 2, hidden_dim) - 1.0
        hidden_states = torch.cat([pos_data, neg_data], dim=0)
        labels = torch.cat([
            torch.ones(n_samples // 2),
            torch.zeros(n_samples // 2),
        ])

        linear, probes = train_safety_probes(
            hidden_states=hidden_states,
            labels=labels,
            K=4,
            hidden_dim=hidden_dim,
            epochs=30,
            lr=1e-2,
            device="cpu",
        )

        assert isinstance(linear, torch.nn.Linear)
        assert linear.weight.shape == (4, hidden_dim)
        assert len(probes) == 4

        # All probes should achieve reasonable accuracy on separable data
        for probe in probes:
            assert probe.accuracy > 0.6

    def test_probe_metadata(self):
        """Verify SafetyProbe metadata fields."""
        n_samples = 100
        hidden_dim = 16
        K = 2

        hidden_states = torch.randn(n_samples, hidden_dim)
        labels = (torch.randn(n_samples) > 0).float()

        _, probes = train_safety_probes(
            hidden_states=hidden_states,
            labels=labels,
            K=K,
            hidden_dim=hidden_dim,
            epochs=5,
            device="cpu",
        )

        for i, probe in enumerate(probes):
            assert probe.probe_index == i
            assert probe.weight.shape == (hidden_dim,)
            assert isinstance(probe.bias, float)
            assert 0.0 <= probe.accuracy <= 1.0
            assert 0.0 <= probe.f1_score <= 1.0

    def test_multi_label_training(self):
        """Train with different labels per probe."""
        n_samples = 200
        hidden_dim = 16
        K = 3

        hidden_states = torch.randn(n_samples, hidden_dim)
        # Different labels for each probe
        labels = torch.zeros(n_samples, K)
        labels[:, 0] = (hidden_states[:, 0] > 0).float()
        labels[:, 1] = (hidden_states[:, 1] > 0).float()
        labels[:, 2] = (hidden_states[:, 2] > 0).float()

        linear, probes = train_safety_probes(
            hidden_states=hidden_states,
            labels=labels,
            K=K,
            hidden_dim=hidden_dim,
            epochs=30,
            lr=1e-2,
            device="cpu",
        )

        assert len(probes) == K

    def test_single_label_broadcast_with_diversity_noise(self):
        """Single label column is broadcast with noise for diversity."""
        n_samples = 100
        hidden_dim = 8
        K = 4

        hidden_states = torch.randn(n_samples, hidden_dim)
        labels = (torch.randn(n_samples) > 0).float()

        # Should emit a warning about redundant probes
        linear, probes = train_safety_probes(
            hidden_states=hidden_states,
            labels=labels,
            K=K,
            hidden_dim=hidden_dim,
            epochs=5,
            device="cpu",
        )

        assert linear.weight.shape == (K, hidden_dim)
        assert len(probes) == K

    def test_mismatched_label_dim_raises(self):
        """Label dimension not matching K should raise ValueError."""
        n_samples = 100
        hidden_dim = 8
        K = 4

        hidden_states = torch.randn(n_samples, hidden_dim)
        # 3 label columns but K=4 probes
        labels = torch.zeros(n_samples, 3)

        with pytest.raises(ValueError, match="Label dimension"):
            train_safety_probes(
                hidden_states=hidden_states,
                labels=labels,
                K=K,
                hidden_dim=hidden_dim,
                epochs=5,
                device="cpu",
            )
