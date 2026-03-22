"""Spectral analysis for SSM safety horizons."""

from ssh_hybrid.spectral.radius import (
    compute_spectral_radius,
    compute_spectral_radius_diagonal,
    compute_layer_spectral_radii,
    compute_mean_spectral_radius,
)
from ssh_hybrid.spectral.horizon import (
    safety_memory_horizon,
    attenuation_factor,
)
from ssh_hybrid.spectral.margin import (
    spectral_safety_margin_bound,
    mbca_compensated_margin,
)
