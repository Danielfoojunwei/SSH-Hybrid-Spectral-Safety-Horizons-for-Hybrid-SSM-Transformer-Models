"""Content-Hidden State Score (CHSS) computation.

Measures the alignment between model hidden states under clean vs.
perturbed conditions via cosine similarity.

IMPORTANT LIMITATION: CHSS measures REPRESENTATION SHIFT, not safety.
A model could have high representation shift yet remain behaviorally safe
(different internal pathway, same output), or have minimal representation
shift yet produce unsafe outputs (small perturbation in a critical
direction). CHSS should be treated as a supplementary analysis metric
and validated against behavioral safety metrics (attack success rate)
before being used as a safety indicator. See evaluation/safety_margin.py
for behavioral safety measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class CHSSResult:
    """Result of CHSS computation for a model under specific conditions."""

    model_name: str
    condition: str  # 'clean' or attack configuration name
    chss_score: float
    chss_per_layer: list[float]
    n_samples: int
    std_dev: float


def compute_chss(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    reference_texts: list[str],
    model_name: str = "",
    condition: str = "clean",
    max_length: int = 512,
    batch_size: int = 8,
    layers: list[int] | None = None,
    device: str = "cuda",
) -> CHSSResult:
    """Compute Content-Hidden State Score between texts and reference content.

    CHSS measures how well the model's hidden states at each layer preserve
    the semantic content of the input. Under HiSPA attacks, SSM layers show
    significantly degraded CHSS (71% degradation) compared to Transformer
    layers (58% improvement).

    Args:
        model: The model to evaluate.
        tokenizer: Associated tokenizer.
        texts: Input texts (possibly with HiSPA triggers).
        reference_texts: Clean reference texts for comparison.
        model_name: Name for logging/result tracking.
        condition: Condition label ('clean', 'z-hispa-1', etc.).
        max_length: Maximum sequence length.
        batch_size: Batch size for processing.
        layers: Specific layer indices. None = all layers.
        device: Computation device.

    Returns:
        CHSSResult with score and per-layer breakdown.
    """
    model.eval()
    all_scores = []
    per_layer_scores: dict[int, list[float]] = {}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_refs = reference_texts[i : i + batch_size]

        # Encode input texts
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Encode reference texts
        ref_inputs = tokenizer(
            batch_refs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            ref_outputs = model(**ref_inputs, output_hidden_states=True)

        hidden = outputs.hidden_states
        ref_hidden = ref_outputs.hidden_states

        target_layers = layers if layers is not None else list(range(len(hidden)))

        for layer_idx in target_layers:
            if layer_idx >= len(hidden):
                continue

            # Get mean-pooled representations
            h = _mean_pool(hidden[layer_idx], inputs.get("attention_mask"))
            h_ref = _mean_pool(ref_hidden[layer_idx], ref_inputs.get("attention_mask"))

            # CHSS = cosine similarity between hidden states
            similarity = F.cosine_similarity(h, h_ref, dim=-1)
            score = similarity.mean().item()

            if layer_idx not in per_layer_scores:
                per_layer_scores[layer_idx] = []
            per_layer_scores[layer_idx].append(score)

        # Overall CHSS: average across layers
        batch_layer_scores = [
            per_layer_scores[l][-1] for l in target_layers if l in per_layer_scores
        ]
        if batch_layer_scores:
            all_scores.append(sum(batch_layer_scores) / len(batch_layer_scores))

    mean_chss = sum(all_scores) / max(1, len(all_scores))
    std_chss = _std(all_scores)

    layer_means = [
        sum(per_layer_scores.get(l, [0])) / max(1, len(per_layer_scores.get(l, [1])))
        for l in sorted(per_layer_scores.keys())
    ]

    return CHSSResult(
        model_name=model_name,
        condition=condition,
        chss_score=mean_chss,
        chss_per_layer=layer_means,
        n_samples=len(texts),
        std_dev=std_chss,
    )


def compute_chss_degradation(
    clean_result: CHSSResult,
    attack_result: CHSSResult,
) -> float:
    """Compute CHSS degradation percentage.

    Degradation = (CHSS_clean - CHSS_attack) / CHSS_clean * 100

    Args:
        clean_result: CHSS under clean conditions.
        attack_result: CHSS under attack conditions.

    Returns:
        Degradation percentage (positive = worse under attack).
    """
    if clean_result.chss_score == 0:
        return 0.0
    return (
        (clean_result.chss_score - attack_result.chss_score)
        / abs(clean_result.chss_score)
        * 100.0
    )


def _mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-pool hidden states over sequence length, respecting mask."""
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return hidden.mean(dim=1)


def _std(values: list[float]) -> float:
    """Compute standard deviation of a list."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5
