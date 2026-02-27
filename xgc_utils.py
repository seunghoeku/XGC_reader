"""Compatibility wrapper for moved modules under xgc_reader.input."""

import warnings

from xgc_reader.input.profiles import *  # noqa: F401,F403
from xgc_reader.input.profile_transforms import *  # noqa: F401,F403

warnings.warn(
    "xgc_utils is deprecated; use xgc_reader.input.profiles and xgc_reader.input.profile_transforms.",
    DeprecationWarning,
    stacklevel=2,
)
