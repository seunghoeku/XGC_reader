"""Input readers and transforms for XGC setup data.

Lightweight helpers are imported eagerly. Optional readers that depend on
extra packages are imported conditionally so ``import xgc_reader.input`` works
without all optional dependencies installed.
"""

from __future__ import annotations

from .profile_transforms import autoscale, calculate_new_limit, get_xy, merge
from .profiles import display_ion_species, load_prf, load_prf2, plot_profiles, read_kefit_profile, save_prf

_OPTIONAL_IMPORT_ERRORS: dict[str, Exception] = {}

_OPTIONAL_EQD_NAMES = (
    "eqd_class",
    "get_eqd_from_eqdsk",
    "refine_axis_or_x_spline_optimize",
)
_OPTIONAL_GEQDSK_NAMES = (
    "geqdsk_reader",
    "read_wall",
    "add_values",
    "show_geqdsk",
    "find_x_point",
)

__all__ = [
    "load_prf",
    "load_prf2",
    "save_prf",
    "read_kefit_profile",
    "plot_profiles",
    "display_ion_species",
    "merge",
    "autoscale",
    "calculate_new_limit",
    "get_xy",
]

try:
    from .eqd import eqd_class, get_eqd_from_eqdsk, refine_axis_or_x_spline_optimize

    __all__.extend(_OPTIONAL_EQD_NAMES)
except Exception as exc:  # pragma: no cover - optional dependency
    _OPTIONAL_IMPORT_ERRORS["eqd"] = exc

try:
    from .geqdsk import add_values, find_x_point, geqdsk_reader, read_wall, show_geqdsk

    __all__.extend(_OPTIONAL_GEQDSK_NAMES)
except Exception as exc:  # pragma: no cover - optional dependency
    _OPTIONAL_IMPORT_ERRORS["geqdsk"] = exc


def __getattr__(name: str):
    if name in _OPTIONAL_EQD_NAMES and "eqd" in _OPTIONAL_IMPORT_ERRORS:
        raise ImportError(
            "xgc_reader.input optional eqd reader requires extra dependencies "
            "(e.g. scipy/matplotlib)."
        ) from _OPTIONAL_IMPORT_ERRORS["eqd"]
    if name in _OPTIONAL_GEQDSK_NAMES and "geqdsk" in _OPTIONAL_IMPORT_ERRORS:
        raise ImportError(
            "xgc_reader.input optional geqdsk reader requires extra dependencies "
            "(e.g. freeqdsk/matplotlib)."
        ) from _OPTIONAL_IMPORT_ERRORS["geqdsk"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
