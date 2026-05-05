"""Poincare plot utilities for XGC magnetic-field data."""

from .field_following import (
    EquilibriumMagneticField,
    FieldLinePoint,
    FieldLinePoints,
    follow_field_to_nearest_midplane,
    nearest_midplane,
)
from .toroidal_interpolation import (
    FineToroidalGrid,
    fine_dB_memory_bytes,
    interpolate_dB_to_fine_toroidal_file,
    interpolate_dB_to_fine_toroidal_grid,
)
from .poincare_map import (
    FineDeltaBField,
    PoincareTraceResult,
    TotalMagneticFieldLineRHS,
    plot_poincare_heatmap,
    plot_poincare_map,
    trace_poincare_map,
)

__all__ = [
    "EquilibriumMagneticField",
    "FineDeltaBField",
    "FieldLinePoint",
    "FieldLinePoints",
    "FineToroidalGrid",
    "PoincareTraceResult",
    "TotalMagneticFieldLineRHS",
    "fine_dB_memory_bytes",
    "interpolate_dB_to_fine_toroidal_file",
    "follow_field_to_nearest_midplane",
    "interpolate_dB_to_fine_toroidal_grid",
    "nearest_midplane",
    "plot_poincare_heatmap",
    "plot_poincare_map",
    "trace_poincare_map",
]
