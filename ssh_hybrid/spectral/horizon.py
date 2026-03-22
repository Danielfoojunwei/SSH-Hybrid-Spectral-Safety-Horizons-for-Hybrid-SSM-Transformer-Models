"""Safety memory horizon and attenuation factor computation.

Implements Definition 1 (Safety Memory Horizon) and Definition 2 (Attenuation Factor)
from the SSH-Hybrid theoretical framework.
"""

from __future__ import annotations

import math

# Default safety signal threshold: 1% of original signal strength
DEFAULT_EPSILON = 0.01

# Default mean oversight interaction length in tokens
DEFAULT_L = 512


def safety_memory_horizon(
    rho: float,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Compute the safety memory horizon H(rho).

    Definition 1: H(rho) = log(1/epsilon) / log(1/rho) tokens.

    This is the token distance at which safety signals embedded in SSM
    hidden states decay below epsilon fraction of original strength.

    Args:
        rho: Spectral radius of SSM (mean across layers). Must be in (0, 1).
        epsilon: Safety signal threshold (default 0.01 = 1%).

    Returns:
        H(rho) in tokens. Returns float('inf') if rho == 0 (no decay)
        or rho >= 1 (no convergence — pure Transformer).
    """
    if rho <= 0.0 or rho >= 1.0:
        return float("inf")

    return math.log(1.0 / epsilon) / math.log(1.0 / rho)


def attenuation_factor(
    rho: float,
    r_ssm: float,
    L: float = DEFAULT_L,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Compute the attenuation factor g(rho, r_SSM, L).

    Definition 2: g(rho, r_SSM, L) = r_SSM * min(1, L / H(rho))

    Args:
        rho: Spectral radius of SSM.
        r_ssm: Fraction of model layers that are SSM layers (0 to 1).
        L: Mean oversight interaction length in tokens.
        epsilon: Safety signal threshold.

    Returns:
        Attenuation factor g in [0, 1].
    """
    if r_ssm <= 0.0:
        return 0.0

    H = safety_memory_horizon(rho, epsilon)

    if math.isinf(H):
        return 0.0

    return r_ssm * min(1.0, L / H)


def signal_strength_at_distance(
    rho: float,
    tau: int,
    initial_strength: float = 1.0,
) -> float:
    """Compute the remaining signal strength after tau tokens.

    ||S in h_{t0+tau}||_2 <= ||S in h_{t0}||_2 * rho^tau

    Args:
        rho: Spectral radius.
        tau: Token distance from signal embedding.
        initial_strength: Initial signal strength (default 1.0).

    Returns:
        Upper bound on remaining signal strength.
    """
    return initial_strength * (rho ** tau)
