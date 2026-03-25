"""MBCA Monitor: runtime safety monitoring wrapper for hybrid models.

Wraps a hybrid model with MBCA register to provide real-time safety
monitoring during generation. Implements the full Phase 3 pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ssh_hybrid.mbca.register import MBCARegister, MBCAState

logger = logging.getLogger(__name__)


@dataclass
class MonitorResult:
    """Result from MBCA-monitored generation."""

    output_tokens: list[int]
    blocked: bool
    block_position: int | None
    register_state: MBCAState
    triggered_probes: list[int]
    total_tokens_processed: int


class MBCAMonitor(nn.Module):
    """Runtime MBCA safety monitor for hybrid SSM-Transformer models.

    Wraps a model and applies MBCA register checks at each generation step.
    When the safety formula triggers, generation is blocked.

    Args:
        model: The hybrid model to monitor.
        register: Trained MBCA register.
        model_type: Model architecture type ('jamba', 'zamba', etc.).
        attention_layer_indices: Which layers to extract attention states from.
    """

    def __init__(
        self,
        model: nn.Module,
        register: MBCARegister,
        model_type: str = "jamba",
        attention_layer_indices: list[int] | None = None,
    ):
        super().__init__()
        self.model = model
        self.register = register
        self.model_type = model_type
        self.attention_layer_indices = attention_layer_indices

    @torch.no_grad()
    def monitored_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> MonitorResult:
        """Generate tokens with MBCA safety monitoring.

        At each step:
        1. Run forward pass with output_hidden_states=True
        2. Extract attention-layer hidden states
        3. Update MBCA register
        4. If safety formula triggers, BLOCK output

        Args:
            input_ids: (batch, seq_len) input token IDs.
            max_new_tokens: Maximum tokens to generate.
            attention_mask: Optional attention mask.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            MonitorResult with generation output and safety info.
        """
        self.model.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        state = self.register.reset(batch_size=batch_size, device=device)
        generated_tokens: list[int] = []
        blocked = False
        block_position = None

        current_ids = input_ids

        for step in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Extract attention hidden state from the last position
            attention_hidden = self._extract_attention_hidden(
                outputs.hidden_states, current_ids.shape[1] - 1,
            )

            # Update MBCA register
            state, should_block = self.register(attention_hidden, state)

            if should_block.any().item():
                blocked = True
                block_position = step
                logger.info(
                    "MBCA BLOCK at token position %d, triggered probes: %s",
                    step, state.triggered_indices,
                )
                break

            # Sample next token
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_p < 1.0:
                    logits = _top_p_filter(logits, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated_tokens.append(next_token[0, 0].item())

            # Prepare for next step
            current_ids = next_token
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1, device=device)],
                    dim=1,
                )

        return MonitorResult(
            output_tokens=generated_tokens,
            blocked=blocked,
            block_position=block_position,
            register_state=state,
            triggered_probes=state.triggered_indices,
            total_tokens_processed=len(generated_tokens),
        )

    def _extract_attention_hidden(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        position: int,
    ) -> torch.Tensor:
        """Extract attention-layer hidden state at a specific position.

        Args:
            hidden_states: Tuple of (batch, seq, hidden) from all layers.
            position: Sequence position to extract.

        Returns:
            (batch, hidden_dim) tensor from attention layers.
        """
        if self.attention_layer_indices is not None:
            selected = [
                hidden_states[i][:, position, :]
                for i in self.attention_layer_indices
                if i < len(hidden_states)
            ]
        else:
            # Use all layers (for pure Transformer or unknown architecture)
            selected = [h[:, position, :] for h in hidden_states[1:]]

        if not selected:
            # Fallback: use last hidden state
            return hidden_states[-1][:, position, :]

        return torch.stack(selected, dim=0).mean(dim=0)

    def measure_beta_mbca(
        self,
        attack_inputs: list[torch.Tensor],
        attack_masks: list[torch.Tensor] | None = None,
        max_new_tokens: int = 256,
    ) -> float:
        """Measure MBCA coverage fraction beta_MBCA on attack samples.

        beta_MBCA = P(BLOCK | HiSPA attack, MBCA active)

        Args:
            attack_inputs: List of (batch, seq) attack input tensors.
            attack_masks: Optional list of attention masks.
            max_new_tokens: Max tokens per generation.

        Returns:
            beta_MBCA coverage fraction.
        """
        n_blocked = 0
        n_total = 0

        for i, input_ids in enumerate(attack_inputs):
            mask = attack_masks[i] if attack_masks else None
            result = self.monitored_generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                attention_mask=mask,
            )
            n_total += input_ids.shape[0]
            if result.blocked:
                n_blocked += input_ids.shape[0]

        return n_blocked / max(1, n_total)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float("-inf")
    return logits
