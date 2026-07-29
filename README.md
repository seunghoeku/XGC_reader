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

### XGC-Analysis backend

The default backend keeps the historical ``xgc1`` API but uses the sibling
[XGC-Analysis](https://github.com/PrincetonUniversity/XGC-Analysis) package for
data access:

```bash
python -m pip install -e ~/Documents/git/XGC-Analysis
```

```python
import xgc_reader

x = xgc_reader.xgc1("/path/to/xgc")
x.load_unitsm()
x.setup_mesh()
x.load_oned()
x.setup_f0mesh()
x.load_volumes()
```

Use ``backend="legacy"`` only when the original direct ADIOS2 readers are
specifically required.

``x.mesh``, ``x.f0``, ``x.vol``, and ``x.bfield`` expose references or NumPy
views of arrays owned by XGC-Analysis wherever the legacy API allows it.
Stacked 1-D diagnostic arrays and converted one-based surface indices are
materialized once and cached.

For new scripts, ``change_cwd=False`` avoids the historical process-wide
directory change:

```python
x = xgc_reader.xgc1(
    "/path/to/xgc",
    backend="analysis",
    change_cwd=False,
)
```

The backend currently covers units/equilibrium aliases, mesh, 1-D diagnostics,
f0 metadata, volume data, magnetic-field arrays, gradient/field-following
matrices, and the legacy heatdiag2 species/derived-analysis interface. For
backward compatibility,
``load_heatdiag()`` reuses xgc_reader's existing direct reader for legacy
``xgc.heatdiag.bp`` directory datasets; this legacy path is not available for
ADIOS Campaign Archives. ``load_bfieldm()`` likewise keeps its small legacy
reader and reuses the catalog's open campaign handle when applicable.

Older directory outputs that predate a complete XGC-Analysis ``Simulation``
are also supported. ``load_unitsm()`` falls back to the ASCII ``units.m``
reader when ``xgc.units.bp`` is absent, and standalone OneD diagnostics load
directly through the XGC-Analysis catalog without requiring mesh/equilibrium
products. If the catalog lacks the products required to construct a
``Simulation``, the compatibility facade uses the existing static mesh, f0,
and volume readers for that old dataset only.

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
