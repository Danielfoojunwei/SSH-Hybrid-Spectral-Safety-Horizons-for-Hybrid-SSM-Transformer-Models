"""Safety probe training for MBCA register.

Trains K linear probes on attention-layer hidden states to detect
safety-relevant features. Training data comes from BeaverTails
(Ji et al., 2023) or equivalent safety-labelled datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class SafetyProbe:
    """A trained safety probe with its performance metrics."""

    probe_index: int
    weight: torch.Tensor  # (hidden_dim,)
    bias: float
    accuracy: float
    f1_score: float
    description: str = ""


def train_safety_probes(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    K: int = 8,
    hidden_dim: int | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    val_fraction: float = 0.2,
    device: str = "cuda",
) -> tuple[nn.Linear, list[SafetyProbe]]:
    """Train K safety probes for MBCA register on labelled hidden states.

    Each probe learns to detect a specific safety-relevant pattern in
    attention-layer hidden states. The probes are trained jointly as a
    single K-output linear layer with binary cross-entropy loss.

    Args:
        hidden_states: (N, hidden_dim) attention hidden states from the model.
        labels: (N, K) binary labels for each probe dimension.
                If (N,) or (N, 1), the same label is used for all K probes.
        K: Number of probes.
        hidden_dim: Hidden dimension (inferred from hidden_states if None).
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        val_fraction: Fraction of data for validation.
        device: Device for training.

    Returns:
        Tuple of (trained nn.Linear layer, list of SafetyProbe metadata).
    """
    if hidden_dim is None:
        hidden_dim = hidden_states.shape[1]

    # Handle label shapes
    if labels.ndim == 1:
        labels = labels.unsqueeze(1).expand(-1, K).float()
    elif labels.shape[1] == 1:
        labels = labels.expand(-1, K).float()
    else:
        labels = labels.float()

    # Train/val split
    N = hidden_states.shape[0]
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val

    perm = torch.randperm(N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_hs = hidden_states[train_idx].to(device)
    train_labels = labels[train_idx].to(device)
    val_hs = hidden_states[val_idx].to(device)
    val_labels = labels[val_idx].to(device)

    train_loader = DataLoader(
        TensorDataset(train_hs, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    # Train linear probes
    linear = nn.Linear(hidden_dim, K, bias=True).to(device)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)

    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        linear.train()
        total_loss = 0.0
        n_batches = 0

        for batch_hs, batch_labels in train_loader:
            logits = linear(batch_hs)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation
        linear.eval()
        with torch.no_grad():
            val_logits = linear(val_hs)
            val_loss = criterion(val_logits, val_labels).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in linear.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            logger.info(
                "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f",
                epoch + 1, epochs, total_loss / n_batches, val_loss,
            )

    # Restore best model
    if best_state is not None:
        linear.load_state_dict(best_state)

    # Compute per-probe metrics
    linear.eval()
    probe_results = []
    with torch.no_grad():
        val_logits = linear(val_hs)
        val_preds = (val_logits > 0).float()

        for k in range(K):
            tp = ((val_preds[:, k] == 1) & (val_labels[:, k] == 1)).sum().item()
            fp = ((val_preds[:, k] == 1) & (val_labels[:, k] == 0)).sum().item()
            fn = ((val_preds[:, k] == 0) & (val_labels[:, k] == 1)).sum().item()
            tn = ((val_preds[:, k] == 0) & (val_labels[:, k] == 0)).sum().item()

            accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, precision + recall)

            probe_results.append(SafetyProbe(
                probe_index=k,
                weight=linear.weight[k].detach().cpu(),
                bias=linear.bias[k].item(),
                accuracy=accuracy,
                f1_score=f1,
                description=f"safety_probe_{k}",
            ))

            logger.info(
                "Probe %d: accuracy=%.3f, F1=%.3f, precision=%.3f, recall=%.3f",
                k, accuracy, f1, precision, recall,
            )

    return linear.cpu(), probe_results


def extract_attention_hidden_states(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    model_type: str,
    layer_indices: list[int] | None = None,
    max_length: int = 512,
    batch_size: int = 8,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract attention-layer hidden states from a model for probe training.

    For hybrid models, extracts hidden states specifically from attention layers
    (not SSM layers), as these are the inputs to the MBCA probes.

    Args:
        model: The loaded model.
        tokenizer: Associated tokenizer.
        texts: Input texts to process.
        model_type: One of 'jamba', 'zamba', 'pythia'.
        layer_indices: Specific layer indices to extract from. If None, uses
                      all attention layers.
        max_length: Maximum sequence length for tokenization.
        batch_size: Batch size for processing.
        device: Device for inference.

    Returns:
        (N, hidden_dim) tensor of attention hidden states.
    """
    model.eval()
    all_hidden_states = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states

        if layer_indices is not None:
            selected = [hidden[idx] for idx in layer_indices if idx < len(hidden)]
        else:
            selected = _select_attention_layers(hidden, model_type, model)

        # Average over selected layers and take last token
        if selected:
            stacked = torch.stack(selected, dim=0)  # (n_layers, batch, seq, hidden)
            mean_hidden = stacked.mean(dim=0)  # (batch, seq, hidden)
            # Use mean pooling over sequence positions
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (mean_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = mean_hidden.mean(dim=1)
            all_hidden_states.append(pooled.cpu())

    return torch.cat(all_hidden_states, dim=0)


def _select_attention_layers(
    hidden_states: tuple[torch.Tensor, ...],
    model_type: str,
    model: torch.nn.Module,
) -> list[torch.Tensor]:
    """Select hidden states from attention layers only.

    For hybrid models, identifies which layers are attention (not SSM)
    and returns their hidden states.
    """
    n_layers = len(hidden_states) - 1  # exclude embedding layer

    if model_type == "pythia":
        # Pure transformer: all layers are attention
        return list(hidden_states[1:])

    if model_type == "jamba":
        # Jamba: every 8th layer is attention (1 attention per 7 SSM)
        attention_indices = list(range(7, n_layers, 8))
        return [hidden_states[i + 1] for i in attention_indices if i < n_layers]

    if model_type == "zamba":
        # Zamba: similar pattern, attention layers are less frequent
        attention_indices = list(range(5, n_layers, 6))
        return [hidden_states[i + 1] for i in attention_indices if i < n_layers]

    # Default: use all layers
    return list(hidden_states[1:])
