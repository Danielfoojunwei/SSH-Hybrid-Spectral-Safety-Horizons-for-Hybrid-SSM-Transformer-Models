"""Spectral radius measurement for SSM transition matrices.

Implements Phase 1 of the architectural safety audit procedure.
Extracts learned SSM parameters (A, delta) from model weights and
computes spectral radius via eigendecomposition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy import linalg as la


@dataclass
class SpectralRadiusResult:
    """Result of spectral radius computation for a single SSM layer."""

    layer_index: int
    spectral_radius: float
    eigenvalues: np.ndarray
    is_diagonal: bool
    computation_time_ms: float


def compute_spectral_radius(A_bar: torch.Tensor | np.ndarray) -> float:
    """Compute spectral radius of a discretized SSM transition matrix.

    Args:
        A_bar: Discretized transition matrix of shape (d, d) or (d,) for diagonal.

    Returns:
        Spectral radius rho = max(|eigenvalues(A_bar)|).
    """
    if isinstance(A_bar, torch.Tensor):
        A_bar = A_bar.detach().cpu().numpy()

    if A_bar.ndim == 1:
        return compute_spectral_radius_diagonal(A_bar)

    eigenvalues = la.eigvals(A_bar)
    return float(np.max(np.abs(eigenvalues)))


def compute_spectral_radius_diagonal(diag: torch.Tensor | np.ndarray) -> float:
    """Compute spectral radius for diagonal SSM (Mamba-class models).

    For diagonal A matrices: rho = max(|a_i|) for diagonal elements a_i.

    Args:
        diag: Diagonal elements of shape (d,), possibly complex.

    Returns:
        Spectral radius rho.
    """
    if isinstance(diag, torch.Tensor):
        diag = diag.detach().cpu().numpy()
    return float(np.max(np.abs(diag)))


def discretize_A(A: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Discretize continuous SSM matrix A using ZOH (zero-order hold).

    A_bar = exp(delta * A)

    For diagonal A (Mamba): A_bar_diag = exp(delta * A_diag).

    Args:
        A: Continuous transition matrix (d, d) or diagonal (d,).
        delta: Step size, scalar or (d,).

    Returns:
        Discretized A_bar.
    """
    if A.ndim == 1:
        return torch.exp(delta * A)
    return torch.matrix_exp(delta.unsqueeze(-1).unsqueeze(-1) * A)


def compute_layer_spectral_radii(
    model: torch.nn.Module,
    model_type: str,
) -> list[SpectralRadiusResult]:
    """Extract A matrices from SSM layers and compute spectral radii.

    Args:
        model: The loaded model.
        model_type: One of 'jamba', 'zamba', 'mamba', 'pythia'.

    Returns:
        List of SpectralRadiusResult for each SSM layer.
    """
    results = []

    if model_type == "pythia":
        return results

    ssm_layers = _extract_ssm_layers(model, model_type)

    for idx, layer_params in enumerate(ssm_layers):
        t0 = time.perf_counter()

        A = layer_params["A"]
        delta = layer_params.get("delta", None)

        if delta is not None:
            A_bar = discretize_A(A, delta)
        else:
            A_bar = A

        is_diag = A_bar.ndim == 1
        rho = compute_spectral_radius(A_bar)
        eigenvalues = A_bar.detach().cpu().numpy() if is_diag else la.eigvals(
            A_bar.detach().cpu().numpy()
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append(SpectralRadiusResult(
            layer_index=idx,
            spectral_radius=rho,
            eigenvalues=np.abs(eigenvalues) if is_diag else eigenvalues,
            is_diagonal=is_diag,
            computation_time_ms=elapsed_ms,
        ))

    return results


def compute_mean_spectral_radius(results: list[SpectralRadiusResult]) -> float:
    """Compute mean spectral radius across all SSM layers.

    Args:
        results: List of per-layer spectral radius results.

    Returns:
        Mean spectral radius rho. Returns 0.0 if no SSM layers.
    """
    if not results:
        return 0.0
    return float(np.mean([r.spectral_radius for r in results]))


def _extract_learned_delta(module: torch.nn.Module, state_dim: int) -> torch.Tensor:
    """Extract the learned discretization step size from an SSM layer.

    Mamba-class models learn delta via a dt_proj linear layer. We use the
    bias (or a forward pass with a zero input) to get a representative
    delta value. Falls back to dt_init values or a default if unavailable.

    Args:
        module: An SSM layer module (e.g., a Mamba mixer block).
        state_dim: Dimension of the SSM state (for fallback shape).

    Returns:
        delta tensor of shape (state_dim,) or scalar.
    """
    device = next(module.parameters()).device

    # Try to read dt_proj bias as the learned baseline delta
    if hasattr(module, "dt_proj") and module.dt_proj is not None:
        dt_proj = module.dt_proj
        if hasattr(dt_proj, "bias") and dt_proj.bias is not None:
            # dt_proj.bias gives the baseline step size before softplus
            raw_delta = dt_proj.bias.data.float()
            # Mamba applies softplus to get positive delta
            delta = torch.nn.functional.softplus(raw_delta)
            # dt_proj maps from inner_dim to state_dim; bias is (dt_rank,)
            # Use mean as representative step size
            return delta.mean().expand(state_dim)

    # Try D parameter or dt-related attributes (Mamba-2 variants)
    if hasattr(module, "D") and hasattr(module, "dt_bias"):
        raw_delta = module.dt_bias.data.float()
        delta = torch.nn.functional.softplus(raw_delta)
        return delta.mean().expand(state_dim)

    # Fallback: use a conservative default with a warning
    import logging
    logging.getLogger(__name__).warning(
        "Could not extract learned delta from %s, using default=1.0 "
        "(no discretization). Spectral radius will be computed on "
        "continuous A matrix directly.",
        type(module).__name__,
    )
    return torch.ones(state_dim, device=device)


def _extract_ssm_layers(
    model: torch.nn.Module,
    model_type: str,
) -> list[dict[str, torch.Tensor]]:
    """Extract SSM layer parameters (A, delta) from a model.

    Handles architecture-specific parameter locations:
    - Mamba: A_log in each MambaBlock, delta via dt_proj
    - Jamba: Mamba layers interleaved 7:1 with attention in MoE blocks
    - Zamba: Mamba layers with shared attention block

    Args:
        model: The loaded model.
        model_type: One of 'jamba', 'zamba', 'mamba'.

    Returns:
        List of dicts with 'A' and 'delta' tensors per SSM layer.
    """
    layers = []

    if model_type == "mamba":
        # Pure Mamba: every layer is SSM with A_log and dt_proj
        for name, module in model.named_modules():
            if hasattr(module, "A_log"):
                A = -torch.exp(module.A_log.float())
                A_flat = A.mean(dim=0) if A.ndim > 1 else A
                delta = _extract_learned_delta(module, A_flat.shape[-1])
                layers.append({"A": A_flat, "delta": delta})

    elif model_type == "jamba":
        # Jamba: MoE architecture with interleaved Mamba and attention layers.
        # Mamba layers are in JambaMambaDecoderLayer blocks (7 out of every 8).
        # Navigate the model hierarchy to find SSM-specific layers.
        for name, module in model.named_modules():
            # Jamba's Mamba layers contain A_log in the mamba mixer
            if hasattr(module, "A_log"):
                A = -torch.exp(module.A_log.float())
                A_flat = A.mean(dim=0) if A.ndim > 1 else A
                delta = _extract_learned_delta(module, A_flat.shape[-1])
                layers.append({
                    "A": A_flat,
                    "delta": delta,
                    "layer_name": name,
                })

    elif model_type == "zamba":
        # Zamba: Uses shared attention layers interleaved with Mamba blocks.
        # The Mamba blocks have their own A_log and dt parameters.
        # Zamba's architecture differs from Jamba in using a shared attention
        # block rather than per-layer attention.
        for name, module in model.named_modules():
            if hasattr(module, "A_log"):
                A = -torch.exp(module.A_log.float())
                A_flat = A.mean(dim=0) if A.ndim > 1 else A
                delta = _extract_learned_delta(module, A_flat.shape[-1])
                layers.append({
                    "A": A_flat,
                    "delta": delta,
                    "layer_name": name,
                })

    return layers


def compute_r_ssm_from_model(
    model: torch.nn.Module,
    model_type: str,
) -> float:
    """Compute r_SSM (SSM layer fraction) by introspecting model architecture.

    Counts SSM vs attention layers to validate the hardcoded r_ssm values
    in the model config.

    Args:
        model: The loaded model.
        model_type: Architecture type.

    Returns:
        Fraction of layers that are SSM (0.0 to 1.0).
    """
    if model_type == "pythia":
        return 0.0

    n_ssm = 0
    n_attention = 0

    for name, module in model.named_modules():
        module_type = type(module).__name__.lower()
        # Count SSM layers (look for Mamba-related modules)
        if hasattr(module, "A_log") and ("mamba" in module_type or "ssm" in module_type
                                          or "mixer" in module_type):
            n_ssm += 1
        # Count attention layers
        elif "attention" in module_type and hasattr(module, "q_proj"):
            n_attention += 1

    total = n_ssm + n_attention
    if total == 0:
        return 0.0
    return n_ssm / total
