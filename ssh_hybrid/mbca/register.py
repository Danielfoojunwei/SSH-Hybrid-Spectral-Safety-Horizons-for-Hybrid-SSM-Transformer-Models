"""MBCA Monotone Boolean Register.

Implements Phase 3 of the architectural safety audit procedure.
The register maintains K boolean safety facts that can only transition
from 0 to 1 (monotone), providing non-decaying safety signal persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class MBCAState:
    """State of the MBCA register at a given token position."""

    bits: torch.Tensor  # (K,) boolean tensor
    token_position: int = 0
    triggered_indices: list[int] = field(default_factory=list)


class MBCARegister(nn.Module):
    """Monotone Boolean Composition Accumulator register.

    Maintains K boolean safety facts extracted from attention-layer hidden
    states. Each bit c[k] can only transition from 0 -> 1 (monotone OR update).
    A safety formula phi(c) over the K bits determines whether to block output.

    c[k] = c[k] OR (w_k . a_t + b_k > 0)  -- at each token t

    Args:
        K: Number of boolean safety probes (recommended 8-16).
        hidden_dim: Dimension of attention hidden states.
        safety_formula: Callable that takes (K,) bool tensor and returns True to block.
                        Default: block if ANY bit is set.
    """

    def __init__(
        self,
        K: int,
        hidden_dim: int,
        safety_formula: str = "any",
    ):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.safety_formula_type = safety_formula

        # K linear probes: w_k (hidden_dim,) and b_k (scalar)
        self.probes = nn.Linear(hidden_dim, K, bias=True)

        # Initialize with small random weights
        nn.init.xavier_uniform_(self.probes.weight)
        nn.init.zeros_(self.probes.bias)

    def reset(self, batch_size: int = 1, device: torch.device | None = None) -> MBCAState:
        """Reset the register to all zeros.

        Args:
            batch_size: Number of sequences in the batch.
            device: Device for the state tensor.

        Returns:
            Fresh MBCAState with all bits cleared.
        """
        device = device or self.probes.weight.device
        return MBCAState(
            bits=torch.zeros(batch_size, self.K, dtype=torch.bool, device=device),
            token_position=0,
            triggered_indices=[],
        )

    def update(
        self,
        state: MBCAState,
        attention_hidden: torch.Tensor,
    ) -> MBCAState:
        """Update register state with new attention hidden state.

        Applies monotone OR update: c[k] = c[k] OR (w_k . a_t + b_k > 0)

        Args:
            state: Current register state.
            attention_hidden: Attention layer hidden state (batch, hidden_dim).

        Returns:
            Updated MBCAState.
        """
        # Compute probe activations: (batch, K)
        activations = self.probes(attention_hidden.float())
        new_triggers = activations > 0  # (batch, K) bool

        # Monotone OR update — bits can only go from 0 to 1
        updated_bits = state.bits | new_triggers

        # Track which probes were newly triggered
        newly_triggered = (~state.bits & new_triggers).any(dim=0)
        triggered = [
            i for i in range(self.K) if newly_triggered[i].item()
        ]

        return MBCAState(
            bits=updated_bits,
            token_position=state.token_position + 1,
            triggered_indices=state.triggered_indices + triggered,
        )

    def should_block(self, state: MBCAState) -> torch.Tensor:
        """Evaluate safety formula phi(c) over register bits.

        Args:
            state: Current register state.

        Returns:
            Boolean tensor (batch,) — True means BLOCK output.
        """
        return self._evaluate_formula(state.bits)

    def _evaluate_formula(self, bits: torch.Tensor) -> torch.Tensor:
        """Evaluate the safety formula on the bit vector.

        Args:
            bits: (batch, K) boolean tensor.

        Returns:
            (batch,) boolean tensor.
        """
        if self.safety_formula_type == "any":
            return bits.any(dim=-1)
        elif self.safety_formula_type == "majority":
            return bits.float().mean(dim=-1) > 0.5
        elif self.safety_formula_type == "all":
            return bits.all(dim=-1)
        else:
            # Threshold: block if >= N bits are set
            try:
                threshold = int(self.safety_formula_type)
            except ValueError:
                raise ValueError(
                    f"Unknown safety formula: {self.safety_formula_type}. "
                    f"Use 'any', 'majority', 'all', or an integer threshold."
                )
            return bits.float().sum(dim=-1) >= threshold

    def forward(
        self,
        attention_hidden: torch.Tensor,
        state: MBCAState | None = None,
    ) -> tuple[MBCAState, torch.Tensor]:
        """Process one token's attention hidden state.

        Args:
            attention_hidden: (batch, hidden_dim) from attention layers.
            state: Current state. If None, creates fresh state.

        Returns:
            Tuple of (updated_state, should_block).
        """
        if state is None:
            state = self.reset(
                batch_size=attention_hidden.shape[0],
                device=attention_hidden.device,
            )

        state = self.update(state, attention_hidden)
        block = self.should_block(state)
        return state, block
