"""Spectral radius measurement for SSM transition matrices.

Implements Phase 1 of the architectural safety audit procedure,
following SpectralGuard methodology (Le Mercier et al., March 2026).
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


def _extract_ssm_layers(
    model: torch.nn.Module,
    model_type: str,
) -> list[dict[str, torch.Tensor]]:
    """Extract SSM layer parameters (A, delta) from a model.

    Args:
        model: The loaded model.
        model_type: One of 'jamba', 'zamba', 'mamba'.

    Returns:
        List of dicts with 'A' and optionally 'delta' tensors.
    """
    layers = []

    if model_type == "mamba":
        for name, module in model.named_modules():
            if hasattr(module, "A_log"):
                A = -torch.exp(module.A_log.float())
                delta = (
                    torch.ones(A.shape[-1], device=A.device)
                    if not hasattr(module, "dt_proj")
                    else torch.ones(A.shape[-1], device=A.device) * 0.1
                )
                layers.append({"A": A.mean(dim=0) if A.ndim > 1 else A, "delta": delta})

    elif model_type == "jamba":
        for name, module in model.named_modules():
            if hasattr(module, "A_log"):
                A = -torch.exp(module.A_log.float())
                delta = torch.ones(A.shape[-1], device=A.device) * 0.1
                layers.append({"A": A.mean(dim=0) if A.ndim > 1 else A, "delta": delta})

    elif model_type == "zamba":
        for name, module in model.named_modules():
            if hasattr(module, "A_log"):
                A = -torch.exp(module.A_log.float())
                delta = torch.ones(A.shape[-1], device=A.device) * 0.1
                layers.append({"A": A.mean(dim=0) if A.ndim > 1 else A, "delta": delta})

    return layers
