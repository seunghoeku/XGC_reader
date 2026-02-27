# XGC_reader
Python package for XGC data analysis.

Requires ADIOS2 2.10 or newer.

## Package layout

The code is now organized under a single package root, `xgc_reader`.

- `xgc_reader/`: core XGC reader package (`xgc1` and analysis modules)
- `xgc_reader/input/`: input-data readers and transforms
- `xgc_reader/distribution/`: distribution-function classes
- `xgc_reader_old.py`: legacy compatibility module (kept as-is)

## New import paths

### Core reader

```python
import xgc_reader
x = xgc_reader.xgc1("/path/to/xgc")
```

### Input modules

```python
from xgc_reader.input.eqd import eqd_class, get_eqd_from_eqdsk
from xgc_reader.input.geqdsk import geqdsk_reader
from xgc_reader.input.profiles import load_prf, save_prf, read_kefit_profile
from xgc_reader.input.profile_transforms import merge
```

### Distribution module

```python
from xgc_reader.distribution.core import VelocityGrid, XGCDistribution
```

## Backward compatibility

Legacy top-level modules are still available as wrappers:

- `eqd_file_reader.py`
- `geqdsk_reader.py`
- `xgc_utils.py`
- `xgc_distribution.py`

These wrappers re-export the moved code and emit deprecation warnings. Existing scripts should keep working, but new code should use `xgc_reader.*` paths.

## Notes

- `xgc_reader_old.py` remains unchanged for compatibility workflows.
- `profile_input_reader.py` is still not ready; use `xgc_reader.input.profiles` instead.



