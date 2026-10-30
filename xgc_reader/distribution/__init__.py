"""Distribution data structures and operations."""

from .core import (
    VelocityGrid,
    XGCDistribution,
    convert_distribution,
    poloidal_smooth_f_init,
    set_sol_maxwellian,
    sol_weight,
    split_open_surface,
    open_segment_weights,
    smooth_open_segment,
    vspace_smooth,
)
from .canonical import (
    average_distribution_emu_pphi,
    average_analytic_maxwellian_emu_pphi,
    average_mc_maxwellian_emu_pphi,
    canonical_coordinates,
    distribution_with_perp_jacobian,
    distribution_without_perp_jacobian,
    interpolate_fmean_to_velocity_grid,
)

__all__ = [
    "VelocityGrid",
    "XGCDistribution",
    "convert_distribution",
    "poloidal_smooth_f_init",
    "set_sol_maxwellian",
    "sol_weight",
    "split_open_surface",
    "open_segment_weights",
    "smooth_open_segment",
    "vspace_smooth",
    "canonical_coordinates",
    "distribution_without_perp_jacobian",
    "distribution_with_perp_jacobian",
    "average_distribution_emu_pphi",
    "average_analytic_maxwellian_emu_pphi",
    "average_mc_maxwellian_emu_pphi",
    "interpolate_fmean_to_velocity_grid",
]
