"""Compatibility wrapper for moved module: xgc_reader.distribution.core."""

import warnings

from xgc_reader.distribution.core import *  # noqa: F401,F403

warnings.warn(
    "xgc_distribution is deprecated; use xgc_reader.distribution.core instead.",
    DeprecationWarning,
    stacklevel=2,
)
