"""Compatibility wrapper for moved module: xgc_reader.input.eqd."""

import warnings

from xgc_reader.input.eqd import *  # noqa: F401,F403

warnings.warn(
    "eqd_file_reader is deprecated; use xgc_reader.input.eqd instead.",
    DeprecationWarning,
    stacklevel=2,
)
