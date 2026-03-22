"""Safety margin theorems: Spectral Safety Margin Bound and MBCA Compensation.

Implements Theorem 1 (Spectral Safety Margin Bound) and
Theorem 2 (MBCA Compensation) from the SSH-Hybrid framework.
"""

from __future__ import annotations

from ssh_hybrid.spectral.horizon import attenuation_factor


def spectral_safety_margin_bound(
    delta_star_transformer: float,
    rho: float,
    r_ssm: float,
    L: float = 512.0,
    epsilon: float = 0.01,
) -> float:
    """Compute the spectral safety margin bound (Theorem 1).

    Theorem 1: Delta*(pi_hybrid) <= Delta*(pi_transformer) * (1 - g(rho, r_SSM, L))

    A deceptive AI exploiting the safety blind window (tokens beyond H(rho)
    where early constraints have decayed) can pass oversight despite being unsafe.

    Args:
        delta_star_transformer: Safety margin of the pure Transformer baseline.
        rho: Mean spectral radius of SSM layers.
        r_ssm: Fraction of model layers that are SSM layers.
        L: Mean oversight interaction length in tokens.
        epsilon: Safety signal detection threshold.

    Returns:
        Upper bound on hybrid model safety margin Delta*(pi_hybrid).
    """
    g = attenuation_factor(rho, r_ssm, L, epsilon)
    return delta_star_transformer * (1.0 - g)


def mbca_compensated_margin(
    delta_star_transformer: float,
    rho: float,
    r_ssm: float,
    beta_mbca: float,
    L: float = 512.0,
    epsilon: float = 0.01,
) -> float:
    """Compute MBCA-compensated safety margin (Theorem 2).

    Theorem 2: Delta*(pi_hybrid + MBCA) >= Delta*(pi_transformer) * (1 - g * (1 - beta_MBCA))

    When beta_MBCA = 1: full recovery to Transformer margin.
    When beta_MBCA = 0: no improvement over uncompensated hybrid.

    Args:
        delta_star_transformer: Safety margin of the pure Transformer baseline.
        rho: Mean spectral radius of SSM layers.
        r_ssm: Fraction of model layers that are SSM layers.
        beta_mbca: MBCA coverage fraction (0 to 1).
        L: Mean oversight interaction length in tokens.
        epsilon: Safety signal detection threshold.

    Returns:
        Lower bound on compensated hybrid safety margin.
    """
    g = attenuation_factor(rho, r_ssm, L, epsilon)
    return delta_star_transformer * (1.0 - g * (1.0 - beta_mbca))


def safety_margin_deficit(
    delta_star_transformer: float,
    rho: float,
    r_ssm: float,
    L: float = 512.0,
    epsilon: float = 0.01,
) -> float:
    """Compute the safety margin deficit of the hybrid model.

    Deficit = Delta*(pi_transformer) - Delta*(pi_hybrid)
            = Delta*(pi_transformer) * g

    Args:
        delta_star_transformer: Safety margin of the pure Transformer baseline.
        rho: Mean spectral radius of SSM layers.
        r_ssm: Fraction of model layers that are SSM layers.
        L: Mean oversight interaction length in tokens.
        epsilon: Safety signal threshold.

    Returns:
        Safety margin deficit.
    """
    g = attenuation_factor(rho, r_ssm, L, epsilon)
    return delta_star_transformer * g


def mbca_recovery_fraction(
    delta_star_transformer: float,
    rho: float,
    r_ssm: float,
    beta_mbca: float,
    L: float = 512.0,
    epsilon: float = 0.01,
) -> float:
    """Compute what fraction of the deficit MBCA recovers.

    Recovery = (Delta*(hybrid+MBCA) - Delta*(hybrid)) / (Delta*(transformer) - Delta*(hybrid))
             = beta_MBCA

    Args:
        delta_star_transformer: Safety margin of the pure Transformer baseline.
        rho: Mean spectral radius of SSM layers.
        r_ssm: Fraction of model layers that are SSM layers.
        beta_mbca: MBCA coverage fraction.
        L: Mean oversight interaction length in tokens.
        epsilon: Safety signal threshold.

    Returns:
        Recovery fraction (equals beta_mbca by construction of Theorem 2).
    """
    return beta_mbca
