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

__all__ = [
    "EquilibriumMagneticField",
    "FieldLinePoint",
    "FieldLinePoints",
    "FineToroidalGrid",
    "fine_dB_memory_bytes",
    "interpolate_dB_to_fine_toroidal_file",
    "follow_field_to_nearest_midplane",
    "interpolate_dB_to_fine_toroidal_grid",
    "nearest_midplane",
]
