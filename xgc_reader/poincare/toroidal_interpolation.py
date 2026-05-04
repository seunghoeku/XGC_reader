"""Fine toroidal interpolation for XGC perturbation magnetic fields."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import adios2
import numpy as np
from numpy.lib.format import open_memmap
from matplotlib.tri import LinearTriInterpolator

from ..mesh_data import meshdata
from .field_following import EquilibriumMagneticField


_DB_FIELD_NAMES = ("dBphi", "dBpsi", "dBtheta")


@dataclass(frozen=True)
class FineToroidalGrid:
    """Fine toroidal grid metadata."""

    nphi_input: int
    nphi_source: int
    mult: int
    nphi: int
    wedge_angle: float
    delta_phi_source: float
    delta_phi_fine: float


def _resolve_path(path_or_xgc):
    if hasattr(path_or_xgc, "path"):
        return Path(path_or_xgc.path)
    return Path(path_or_xgc)


def _read_dB_fields(filename, dtype=np.float32):
    with adios2.FileReader(str(filename)) as f:
        fields = [np.asarray(f.read(name), dtype=dtype) for name in _DB_FIELD_NAMES]

    shapes = {field.shape for field in fields}
    if len(shapes) != 1:
        raise ValueError(f"dB field shapes do not match: {sorted(shapes)}")

    return fields


def _read_dB_shape(filename):
    with adios2.FileReader(str(filename)) as f:
        shape = np.asarray(f.read(_DB_FIELD_NAMES[0])).shape
    if len(shape) != 2:
        raise ValueError(f"{_DB_FIELD_NAMES[0]} must have shape (nphi, nnode), got {shape}")
    return shape


def _make_grid(mesh, nphi_input, nphi_source):
    if not hasattr(mesh, "wedge_angle") or mesh.wedge_angle is None:
        raise ValueError("mesh.wedge_angle is required")
    if not hasattr(mesh, "delta_phi") or mesh.delta_phi is None:
        raise ValueError("mesh.delta_phi is required")

    mult = int(np.rint(float(nphi_input) / float(nphi_source)))
    if mult < 1:
        mult = 1
    nphi = mult * nphi_source
    delta_phi_fine = float(mesh.delta_phi) / float(mult)
    return FineToroidalGrid(
        nphi_input=int(nphi_input),
        nphi_source=int(nphi_source),
        mult=mult,
        nphi=nphi,
        wedge_angle=float(mesh.wedge_angle),
        delta_phi_source=float(mesh.delta_phi),
        delta_phi_fine=delta_phi_fine,
    )


def _build_component_interpolators(triobj, source_fields):
    return [
        [LinearTriInterpolator(triobj, field[plane_index]) for plane_index in range(field.shape[0])]
        for field in source_fields
    ]


def _interpolate_components(interpolators, plane_index, r, z, fill_value=np.nan):
    out = []
    for component_interpolators in interpolators:
        interpolator = component_interpolators[plane_index]
        values = interpolator(r, z)
        if np.ma.isMaskedArray(values):
            values = values.filled(fill_value)
        out.append(np.asarray(values))
    return out


def interpolate_dB_to_fine_toroidal_grid(
    path_or_xgc,
    xgc3d_filename,
    nphi_input,
    *,
    magnetic_field=None,
    mesh=None,
    dtype=np.float32,
    ff_step=2,
    ff_order=4,
    fill_value=np.nan,
    output=None,
):
    """Interpolate ``dBphi``, ``dBpsi``, ``dBtheta`` to a fine toroidal grid.

    Coarse XGC planes are copied directly to fine planes
    ``0, mult, 2*mult, ...``. Intermediate planes are computed by field
    following each mesh node to the adjacent coarse planes, doing linear
    triangle interpolation there, and linearly weighting by toroidal angle.

    Parameters
    ----------
    path_or_xgc
        XGC output directory or an ``xgc1``-like object with ``path``.
    xgc3d_filename
        Path to one ``xgc.3d.*.bp`` file, or a filename relative to
        ``path_or_xgc``.
    nphi_input
        Requested fine toroidal plane count. The actual count is
        ``round(nphi_input / sml_nphi_total) * sml_nphi_total``.
    magnetic_field
        Optional ``EquilibriumMagneticField`` instance.
    mesh
        Optional ``meshdata`` instance.
    dtype
        Output dtype. Use ``np.float32`` for memory efficiency.
    ff_step, ff_order
        Runge-Kutta settings for field following.
    fill_value
        Value used when linear triangle interpolation lands outside the mesh.
    output
        Optional preallocated array or ``np.memmap`` with shape
        ``(nphi, nnode, 3)``.

    Returns
    -------
    fine_dB, grid
        ``fine_dB[..., 0] = dBphi``, ``fine_dB[..., 1] = dBpsi``,
        ``fine_dB[..., 2] = dBtheta`` and the fine grid metadata.
    """

    path = _resolve_path(path_or_xgc)
    xgc3d_path = Path(xgc3d_filename)
    if not xgc3d_path.is_absolute():
        xgc3d_path = path / xgc3d_path

    if mesh is None:
        mesh = meshdata(str(path) + "/")
    if magnetic_field is None:
        magnetic_field = EquilibriumMagneticField(path)

    source_fields = _read_dB_fields(xgc3d_path, dtype=dtype)
    nphi_source, nnode = source_fields[0].shape
    if nnode != mesh.nnodes:
        raise ValueError(f"dB nnode={nnode} does not match mesh.nnodes={mesh.nnodes}")

    grid = _make_grid(mesh, nphi_input, nphi_source)

    if output is None:
        fine = np.empty((grid.nphi, nnode, len(_DB_FIELD_NAMES)), dtype=dtype)
    else:
        fine = output
        if fine.shape != (grid.nphi, nnode, len(_DB_FIELD_NAMES)):
            raise ValueError(
                f"output shape {fine.shape} does not match "
                f"({grid.nphi}, {nnode}, {len(_DB_FIELD_NAMES)})"
            )

    r0 = mesh.r
    z0 = mesh.z
    interpolators = None if grid.mult == 1 else _build_component_interpolators(mesh.triobj, source_fields)

    for iphi_fine in range(grid.nphi):
        left = iphi_fine // grid.mult
        offset = iphi_fine - left * grid.mult

        if offset == 0:
            for icomp, field in enumerate(source_fields):
                fine[iphi_fine, :, icomp] = field[left]
            continue

        right = (left + 1) % grid.nphi_source
        frac = float(offset) / float(grid.mult)

        phi_target = iphi_fine * grid.delta_phi_fine
        phi_left = left * grid.delta_phi_source
        phi_right = (left + 1) * grid.delta_phi_source

        left_pos = magnetic_field.follow_field(
            r0,
            z0,
            phi_target,
            phi_left,
            ff_step=ff_step,
            ff_order=ff_order,
        )
        right_pos = magnetic_field.follow_field(
            r0,
            z0,
            phi_target,
            phi_right,
            ff_step=ff_step,
            ff_order=ff_order,
        )

        left_values = _interpolate_components(
            interpolators,
            left,
            left_pos.r,
            left_pos.z,
            fill_value=fill_value,
        )
        right_values = _interpolate_components(
            interpolators,
            right,
            right_pos.r,
            right_pos.z,
            fill_value=fill_value,
        )

        for icomp, (left_value, right_value) in enumerate(zip(left_values, right_values)):
            fine[iphi_fine, :, icomp] = (1.0 - frac) * left_value + frac * right_value

    return fine, grid


def interpolate_dB_to_fine_toroidal_file(
    path_or_xgc,
    xgc3d_filename,
    nphi_input,
    output_filename,
    *,
    metadata_filename=None,
    magnetic_field=None,
    mesh=None,
    dtype=np.float32,
    ff_step=2,
    ff_order=4,
    fill_value=np.nan,
):
    """Interpolate fine toroidal ``dB`` and save it as a memory-mapped ``.npy``.

    The saved array has shape ``(nphi, nnode, 3)`` with component order
    ``("dBphi", "dBpsi", "dBtheta")``. A JSON metadata file is written next to
    the array unless ``metadata_filename`` is explicitly provided.
    """

    path = _resolve_path(path_or_xgc)
    xgc3d_path = Path(xgc3d_filename)
    if not xgc3d_path.is_absolute():
        xgc3d_path = path / xgc3d_path

    if mesh is None:
        mesh = meshdata(str(path) + "/")
    if magnetic_field is None:
        magnetic_field = EquilibriumMagneticField(path)

    nphi_source, nnode = _read_dB_shape(xgc3d_path)
    if nnode != mesh.nnodes:
        raise ValueError(f"dB nnode={nnode} does not match mesh.nnodes={mesh.nnodes}")
    grid = _make_grid(mesh, nphi_input, nphi_source)

    output_path = Path(output_filename)
    output = open_memmap(
        output_path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(grid.nphi, nnode, len(_DB_FIELD_NAMES)),
    )

    fine, grid = interpolate_dB_to_fine_toroidal_grid(
        path,
        xgc3d_path,
        nphi_input,
        magnetic_field=magnetic_field,
        mesh=mesh,
        dtype=dtype,
        ff_step=ff_step,
        ff_order=ff_order,
        fill_value=fill_value,
        output=output,
    )
    fine.flush()

    metadata = {
        "array_file": str(output_path),
        "source_file": str(xgc3d_path),
        "field_names": list(_DB_FIELD_NAMES),
        "shape": list(fine.shape),
        "dtype": str(np.dtype(dtype)),
        "nphi_input": grid.nphi_input,
        "nphi_source": grid.nphi_source,
        "mult": grid.mult,
        "nphi": grid.nphi,
        "nnode": int(nnode),
        "wedge_angle": grid.wedge_angle,
        "delta_phi_source": grid.delta_phi_source,
        "delta_phi_fine": grid.delta_phi_fine,
        "ff_step": int(ff_step),
        "ff_order": int(ff_order),
        "fill_value": None if fill_value is None else float(fill_value),
    }

    if metadata_filename is None:
        metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    else:
        metadata_path = Path(metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return output_path, metadata_path, grid


def fine_dB_memory_bytes(nphi, nnode, ncomponents=3, dtype=np.float32):
    """Return memory required for a fine toroidal ``dB`` array."""

    return int(nphi) * int(nnode) * int(ncomponents) * np.dtype(dtype).itemsize
