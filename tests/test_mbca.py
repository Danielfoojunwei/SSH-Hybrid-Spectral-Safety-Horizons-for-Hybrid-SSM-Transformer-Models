"""Tests for MBCA register module.

Tests monotone boolean register behavior, probe training, and safety
formula evaluation. Uses real torch computations — no mocks.
"""

import torch
import pytest

from ssh_hybrid.mbca.register import MBCARegister, MBCAState


class TestMBCARegister:
    """Tests for MBCARegister monotone boolean logic."""

    def test_initial_state_all_zeros(self):
        register = MBCARegister(K=8, hidden_dim=64)
        state = register.reset(batch_size=1)
        assert state.bits.shape == (1, 8)
        assert not state.bits.any()

    def test_monotone_or_update(self):
        """Bits can only go 0 -> 1, never 1 -> 0."""
        register = MBCARegister(K=4, hidden_dim=16)
        # Set probes with very large bias so they always trigger regardless of input
        register.probes.weight.data = torch.zeros(4, 16)
        register.probes.bias.data = torch.ones(4) * 100.0

        state = register.reset(batch_size=1)
        hidden = torch.randn(1, 16)

        state = register.update(state, hidden)
        assert state.bits.all()  # all bits should be set

        # Now set probes to never trigger (large negative bias, zero weights)
        register.probes.weight.data = torch.zeros(4, 16)
        register.probes.bias.data = torch.ones(4) * -100.0

        # Bits should STILL be set (monotone — cannot go back to 0)
        state = register.update(state, hidden)
        assert state.bits.all()

    def test_selective_triggering(self):
        """Only specific probes trigger based on input."""
        register = MBCARegister(K=4, hidden_dim=2)
        # Set up so probe 0 triggers on positive inputs, probe 1 on negative
        register.probes.weight.data = torch.tensor([
            [10.0, 0.0],   # probe 0: triggers on dim 0 positive
            [-10.0, 0.0],  # probe 1: triggers on dim 0 negative
            [0.0, 10.0],   # probe 2: triggers on dim 1 positive
            [0.0, -10.0],  # probe 3: triggers on dim 1 negative
        ])
        register.probes.bias.data = torch.zeros(4)

        state = register.reset(batch_size=1)
        hidden = torch.tensor([[5.0, -5.0]])  # positive dim0, negative dim1

        state = register.update(state, hidden)
        assert state.bits[0, 0].item() is True   # probe 0 triggered
        assert state.bits[0, 1].item() is False  # probe 1 not triggered
        assert state.bits[0, 2].item() is False  # probe 2 not triggered
        assert state.bits[0, 3].item() is True   # probe 3 triggered

    def test_batch_processing(self):
        register = MBCARegister(K=4, hidden_dim=8)
        state = register.reset(batch_size=3)
        hidden = torch.randn(3, 8)

        state = register.update(state, hidden)
        assert state.bits.shape == (3, 4)

    def test_safety_formula_any(self):
        register = MBCARegister(K=4, hidden_dim=8, safety_formula="any")
        state = MBCAState(
            bits=torch.tensor([[True, False, False, False]]),
        )
        assert register.should_block(state).item() is True

    def test_safety_formula_all(self):
        register = MBCARegister(K=4, hidden_dim=8, safety_formula="all")
        state_partial = MBCAState(bits=torch.tensor([[True, False, True, False]]))
        state_full = MBCAState(bits=torch.tensor([[True, True, True, True]]))
        assert register.should_block(state_partial).item() is False
        assert register.should_block(state_full).item() is True

    def test_safety_formula_majority(self):
        register = MBCARegister(K=4, hidden_dim=8, safety_formula="majority")
        state_minority = MBCAState(bits=torch.tensor([[True, False, False, False]]))
        state_majority = MBCAState(bits=torch.tensor([[True, True, True, False]]))
        assert register.should_block(state_minority).item() is False
        assert register.should_block(state_majority).item() is True

    def test_safety_formula_threshold(self):
        register = MBCARegister(K=4, hidden_dim=8, safety_formula="3")
        state_2 = MBCAState(bits=torch.tensor([[True, True, False, False]]))
        state_3 = MBCAState(bits=torch.tensor([[True, True, True, False]]))
        assert register.should_block(state_2).item() is False
        assert register.should_block(state_3).item() is True

    def test_forward_returns_state_and_block(self):
        register = MBCARegister(K=4, hidden_dim=8)
        hidden = torch.randn(2, 8)
        state, block = register(hidden)
        assert isinstance(state, MBCAState)
        assert block.shape == (2,)
        assert block.dtype == torch.bool

    def test_token_position_tracking(self):
        register = MBCARegister(K=4, hidden_dim=8)
        state = register.reset(batch_size=1)
        hidden = torch.randn(1, 8)

        for t in range(5):
            state = register.update(state, hidden)
            assert state.token_position == t + 1

    def test_triggered_indices_tracked(self):
        register = MBCARegister(K=4, hidden_dim=8)
        # Force all probes to trigger (zero weights + large bias)
        register.probes.weight.data = torch.zeros(4, 8)
        register.probes.bias.data = torch.ones(4) * 100.0

        state = register.reset(batch_size=1)
        hidden = torch.randn(1, 8)
        state = register.update(state, hidden)

        # All 4 probes should be in triggered list
        assert len(state.triggered_indices) == 4

    def test_no_double_counting_triggers(self):
        register = MBCARegister(K=2, hidden_dim=4)
        register.probes.weight.data = torch.ones(2, 4) * 100.0
        register.probes.bias.data = torch.ones(2) * 100.0

        state = register.reset(batch_size=1)
        hidden = torch.randn(1, 4)

        state = register.update(state, hidden)
        first_triggers = len(state.triggered_indices)

        state = register.update(state, hidden)
        # No new triggers on second update (already all set)
        assert len(state.triggered_indices) == first_triggers
