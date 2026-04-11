"""Safety margin propositions: Spectral Safety Margin Bound and MBCA Compensation.

Implements Proposition 1 (Spectral Safety Margin Bound) and
Proposition 2 (MBCA Compensation) from the SSH-Hybrid framework.

IMPORTANT: These propositions rest on the hypothesis that safety margin
degrades proportionally to SSM hidden-state signal attenuation. This is
a testable assumption, not a proven theorem. The key gap is that hidden-state
decay does not necessarily imply proportional output-level safety degradation,
because attention layers and residual connections may compensate. These
formulas should be validated empirically against behavioral safety metrics
(attack success rate) rather than treated as proven bounds.
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
    """Compute the predicted safety margin bound (Proposition 1).

    Proposition 1: Delta*(pi_hybrid) <= Delta*(pi_transformer) * (1 - g(rho, r_SSM, L))

    ASSUMPTION: Safety margin degrades proportionally to the fraction of
    information lost through SSM spectral decay. This assumes SSM and
    attention pathways contribute independently to safety, which is
    approximate — real architectures share a residual stream.

    Args:
        delta_star_transformer: Safety margin of the pure Transformer baseline.
            Should be measured empirically, not assumed to be 1.0.
        rho: Mean spectral radius of SSM layers.
        r_ssm: Fraction of model layers that are SSM layers.
        L: Mean oversight interaction length in tokens.
        epsilon: Safety signal detection threshold.

    Returns:
        Predicted upper bound on hybrid model safety margin.
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
    """Compute MBCA-compensated safety margin (Proposition 2).

    Proposition 2: Delta*(pi_hybrid + MBCA) >= Delta*(pi_transformer) * (1 - g * (1 - beta_MBCA))

    ASSUMPTION: MBCA recovery is linear in beta_MBCA. This holds when
    MBCA detections are independent of the safety violation distribution.
    In practice, beta_MBCA should be measured empirically as the fraction
    of actual safety violations caught, not assumed.

    When beta_MBCA = 1: full recovery to Transformer margin.
    When beta_MBCA = 0: no improvement over uncompensated hybrid.

    Args:
        delta_star_transformer: Safety margin of the pure Transformer baseline.
        rho: Mean spectral radius of SSM layers.
        r_ssm: Fraction of model layers that are SSM layers.
        beta_mbca: MBCA coverage fraction (0 to 1). Must be measured empirically.
        L: Mean oversight interaction length in tokens.
        epsilon: Safety signal detection threshold.

    Returns:
        Predicted lower bound on compensated hybrid safety margin.
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
    n_violations_blocked: int,
    n_violations_total: int,
) -> float:
    """Compute the empirical fraction of safety violations that MBCA catches.

    This replaces the previous tautological implementation that simply
    returned beta_mbca by construction. The recovery fraction must be
    MEASURED, not assumed.

    Args:
        n_violations_blocked: Number of safety violations blocked by MBCA.
        n_violations_total: Total number of safety violations in the test set.

    Returns:
        Empirical recovery fraction (0.0 to 1.0).
    """
    if n_violations_total <= 0:
        return 0.0
    return min(1.0, n_violations_blocked / n_violations_total)


def compute_empirical_safety_margin(
    attack_success_rates: list[float],
) -> float:
    """Compute empirical safety margin from attack success rates.

    Rather than assuming delta_star_transformer=1.0, this measures safety
    as 1 minus the mean attack success rate across a battery of attacks.
    A model that resists all attacks has margin 1.0; one that fails all
    has margin 0.0.

    Args:
        attack_success_rates: List of attack success rates (0.0 to 1.0)
            across different attack configurations.

    Returns:
        Empirical safety margin in [0.0, 1.0].
    """
    if not attack_success_rates:
        return 0.0
    mean_asr = sum(attack_success_rates) / len(attack_success_rates)
    return max(0.0, 1.0 - mean_asr)
