"""Compatibility wrapper for moved module: xgc_reader.input.geqdsk."""

import warnings

from xgc_reader.input.geqdsk import *  # noqa: F401,F403

warnings.warn(
    "geqdsk_reader is deprecated; use xgc_reader.input.geqdsk instead.",
    DeprecationWarning,
    stacklevel=2,
)
