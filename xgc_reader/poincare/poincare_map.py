"""Vectorized Poincare-map tracing for XGC magnetic fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..mesh_data import meshdata
from .field_following import EquilibriumMagneticField
from .toroidal_interpolation import _apply_triangle_map, _build_triangle_map


_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class PoincareTraceResult:
    """Poincare crossing output."""

    r: np.ndarray
    z: np.ndarray
    crossing_index: np.ndarray
    fieldline_index: np.ndarray
    fieldline_lost: np.ndarray | None = None
    lost_step: np.ndarray | None = None
    lost_crossing: np.ndarray | None = None


class FineDeltaBField:
    """Fine toroidal perturbation field on an XGC triangular R-Z mesh.

    Parameters
    ----------
    fine_dB
        Array with shape ``(nphi, nnode, 3)`` and component order
        ``dBphi, dBpsi, dBtheta``.
    mesh
        XGC mesh object with ``triobj``, ``wedge_angle``, and ``delta_phi``.
    magnetic_field
        Background equilibrium field object, used for ``psi`` gradients.
    """

    def __init__(self, fine_dB, mesh, magnetic_field):
        if fine_dB.ndim != 3 or fine_dB.shape[2] != 3:
            raise ValueError("fine_dB must have shape (nphi, nnode, 3)")
        if fine_dB.shape[1] != mesh.nnodes:
            raise ValueError(f"fine_dB nnode={fine_dB.shape[1]} does not match mesh.nnodes={mesh.nnodes}")

        self.fine_dB = fine_dB
        self.mesh = mesh
        self.magnetic_field = magnetic_field
        self.nphi = fine_dB.shape[0]
        self.wedge_angle = float(mesh.wedge_angle)
        self.delta_phi = self.wedge_angle / float(self.nphi)

    @classmethod
    def from_file(cls, filename, path_or_xgc, *, mesh=None, magnetic_field=None, mmap_mode="r"):
        path = Path(path_or_xgc.path) if hasattr(path_or_xgc, "path") else Path(path_or_xgc)
        if mesh is None:
            mesh = meshdata(str(path) + "/")
        if magnetic_field is None:
            magnetic_field = EquilibriumMagneticField(path)
        fine_dB = np.load(filename, mmap_mode=mmap_mode)
        return cls(fine_dB, mesh, magnetic_field)

    def components(self, r, z, phi, *, fill_value=0.0):
        """Return interpolated ``dBphi, dBpsi, dBtheta`` at vectorized points."""

        r_arr, z_arr, phi_arr = np.broadcast_arrays(
            np.asarray(r, dtype=float),
            np.asarray(z, dtype=float),
            np.asarray(phi, dtype=float),
        )
        out_shape = r_arr.shape
        r_flat = r_arr.reshape(-1)
        z_flat = z_arr.reshape(-1)
        phi_flat = phi_arr.reshape(-1)
        phi_mod = np.mod(phi_flat, self.wedge_angle)
        phi_index = phi_mod / self.delta_phi
        left = np.floor(phi_index).astype(np.int64)
        frac = phi_index - left
        left = np.mod(left, self.nphi)
        right = np.mod(left + 1, self.nphi)

        rz_map = _build_triangle_map(self.mesh.triobj, r_flat, z_flat)
        out = np.empty((r_flat.size, 3), dtype=np.result_type(self.fine_dB.dtype, np.float64))

        for iplane in np.unique(left):
            mask = left == iplane
            values_l = _apply_triangle_map(self.fine_dB[iplane], rz_map, fill_value=fill_value)
            out[mask] = (1.0 - frac[mask, None]) * values_l[mask]

        for iplane in np.unique(right):
            mask = right == iplane
            values_r = _apply_triangle_map(self.fine_dB[iplane], rz_map, fill_value=fill_value)
            out[mask] += frac[mask, None] * values_r[mask]

        if fill_value is not None:
            out[~np.isfinite(out)] = fill_value

        return (
            out[:, 0].reshape(out_shape),
            out[:, 1].reshape(out_shape),
            out[:, 2].reshape(out_shape),
        )

    def bvec(self, r, z, phi, *, fill_value=0.0):
        """Return perturbation field components ``(dBr, dBz, dBphi)``."""

        dBphi, dBpsi, dBtheta = self.components(r, z, phi, fill_value=fill_value)
        _, dpsi_dr, dpsi_dz = self.magnetic_field.psi_and_gradient(r, z)
        gradpsi_mag = np.sqrt(dpsi_dr * dpsi_dr + dpsi_dz * dpsi_dz)
        gradpsi_mag = np.where(gradpsi_mag == 0.0, np.finfo(float).tiny, gradpsi_mag)
        gamma = 1.0 / gradpsi_mag

        dBr = gamma * (dpsi_dr * dBpsi - dpsi_dz * dBtheta)
        dBz = gamma * (dpsi_dz * dBpsi + dpsi_dr * dBtheta)
        return dBr, dBz, dBphi


class TotalMagneticFieldLineRHS:
    """Field-line RHS using ``phi`` as the independent variable."""

    def __init__(self, background_field, delta_b_field=None, *, bphi_floor=1.0e-50):
        self.background_field = background_field
        self.delta_b_field = delta_b_field
        self.bphi_floor = bphi_floor

    def bvec(self, r, z, phi):
        br, bz, bphi = self.background_field.bvec(r, z, phi)
        if self.delta_b_field is not None:
            dbr, dbz, dbphi = self.delta_b_field.bvec(r, z, phi)
            br = br + dbr
            bz = bz + dbz
            bphi = bphi + dbphi
        bphi = np.where(bphi == 0.0, self.bphi_floor, bphi)
        return br, bz, bphi

    def derivs(self, r, z, phi):
        br, bz, bphi = self.bvec(r, z, phi)
        r_arr = np.asarray(r, dtype=float)
        return r_arr * br / bphi, r_arr * bz / bphi


def rk2_step_phi(rhs, r, z, phi, dphi):
    """Advance vectorized field-line positions by one RK2 step in ``phi``."""

    k1_r, k1_z = rhs.derivs(r, z, phi)
    r_mid = r + 0.5 * dphi * k1_r
    z_mid = z + 0.5 * dphi * k1_z
    phi_mid = phi + 0.5 * dphi
    k2_r, k2_z = rhs.derivs(r_mid, z_mid, phi_mid)
    return r + dphi * k2_r, z + dphi * k2_z, phi + dphi


def _inside_mesh(trifinder, r, z):
    return np.asarray(trifinder(r, z)) >= 0


def _mark_lost(active, inside, lost_step, lost_crossing, istep, steps_per_period):
    lost_now = active & ~inside
    if np.any(lost_now):
        active[lost_now] = False
        lost_step[lost_now] = istep + 1
        lost_crossing[lost_now] = (istep + 1) // steps_per_period
    return active


def _trace_steps_rk2(rhs, mesh, r, z, phi, dphi, nsteps, crossings_r, crossings_z, steps_per_period):
    active = _inside_mesh(mesh.triobj.get_trifinder(), r, z)
    lost_step = np.full(r.shape, -1, dtype=np.int64)
    lost_crossing = np.full(r.shape, -1, dtype=np.int64)
    lost_step[~active] = 0
    lost_crossing[~active] = 0
    trifinder = mesh.triobj.get_trifinder()

    for istep in range(nsteps):
        if not np.any(active):
            break

        idx = np.nonzero(active)[0]
        r_next, z_next, phi_next = rk2_step_phi(rhs, r[idx], z[idx], phi[idx], dphi)
        r[idx] = r_next
        z[idx] = z_next
        phi[idx] = phi_next
        active_idx_inside = _inside_mesh(trifinder, r_next, z_next)
        inside = np.zeros_like(active, dtype=bool)
        inside[idx] = active_idx_inside
        active = _mark_lost(active, inside, lost_step, lost_crossing, istep, steps_per_period)

        if (istep + 1) % steps_per_period == 0:
            icross = (istep + 1) // steps_per_period - 1
            crossings_r[icross, active] = r[active]
            crossings_z[icross, active] = z[active]
    return r, z, phi, active, lost_step, lost_crossing


def _trace_steps_abm2(rhs, mesh, r, z, phi, dphi, nsteps, crossings_r, crossings_z, steps_per_period):
    active = _inside_mesh(mesh.triobj.get_trifinder(), r, z)
    lost_step = np.full(r.shape, -1, dtype=np.int64)
    lost_crossing = np.full(r.shape, -1, dtype=np.int64)
    lost_step[~active] = 0
    lost_crossing[~active] = 0
    trifinder = mesh.triobj.get_trifinder()

    if nsteps < 1:
        return r, z, phi, active, lost_step, lost_crossing
    if not np.any(active):
        return r, z, phi, active, lost_step, lost_crossing

    f_prev_r = np.zeros_like(r)
    f_prev_z = np.zeros_like(z)
    f_cur_r = np.zeros_like(r)
    f_cur_z = np.zeros_like(z)

    idx = np.nonzero(active)[0]
    f_prev_r[idx], f_prev_z[idx] = rhs.derivs(r[idx], z[idx], phi[idx])
    r_next, z_next, phi_next = rk2_step_phi(rhs, r[idx], z[idx], phi[idx], dphi)
    r[idx], z[idx], phi[idx] = r_next, z_next, phi_next
    inside = np.zeros_like(active, dtype=bool)
    inside[idx] = _inside_mesh(trifinder, r_next, z_next)
    active = _mark_lost(active, inside, lost_step, lost_crossing, 0, steps_per_period)
    if np.any(active):
        idx = np.nonzero(active)[0]
        f_cur_r[idx], f_cur_z[idx] = rhs.derivs(r[idx], z[idx], phi[idx])
    if steps_per_period == 1:
        crossings_r[0, active] = r[active]
        crossings_z[0, active] = z[active]

    for istep in range(1, nsteps):
        if not np.any(active):
            break

        idx = np.nonzero(active)[0]
        r_pred = r[idx] + dphi * (1.5 * f_cur_r[idx] - 0.5 * f_prev_r[idx])
        z_pred = z[idx] + dphi * (1.5 * f_cur_z[idx] - 0.5 * f_prev_z[idx])
        phi_next = phi[idx] + dphi
        f_pred_r, f_pred_z = rhs.derivs(r_pred, z_pred, phi_next)

        r_next = r[idx] + 0.5 * dphi * (f_cur_r[idx] + f_pred_r)
        z_next = z[idx] + 0.5 * dphi * (f_cur_z[idx] + f_pred_z)

        f_prev_r[idx], f_prev_z[idx] = f_cur_r[idx], f_cur_z[idx]
        r[idx], z[idx], phi[idx] = r_next, z_next, phi_next
        f_cur_r[idx], f_cur_z[idx] = f_pred_r, f_pred_z
        inside = np.zeros_like(active, dtype=bool)
        inside[idx] = _inside_mesh(trifinder, r_next, z_next)
        active = _mark_lost(active, inside, lost_step, lost_crossing, istep, steps_per_period)

        if (istep + 1) % steps_per_period == 0:
            icross = (istep + 1) // steps_per_period - 1
            crossings_r[icross, active] = r[active]
            crossings_z[icross, active] = z[active]

    return r, z, phi, active, lost_step, lost_crossing


def trace_poincare_map(
    r0,
    z0,
    *,
    path_or_xgc,
    fine_dB_filename=None,
    n_crossings=100,
    steps_per_period=1024,
    phi0=0.0,
    direction=1.0,
    integrator="rk2",
    mesh=None,
    background_field=None,
    delta_b_field=None,
    output_filename=None,
):
    """Trace field lines and record intersections with ``phi = 0 mod wedge``.

    The integration uses constant ``dphi`` and vectorized updates. One
    crossing is recorded after each toroidal wedge period. If neither
    ``fine_dB_filename`` nor ``delta_b_field`` is provided, the perturbation is
    treated as zero and only the background equilibrium field is followed.

    ``integrator`` can be ``"rk2"`` or ``"abm2"``. ``"abm2"`` uses an
    Adams-Bashforth predictor and Adams-Moulton corrector after one RK2
    bootstrap step.

    Field lines that leave the triangular R-Z mesh are marked lost. After the
    lost step, their remaining crossing positions stay NaN.
    """

    path = Path(path_or_xgc.path) if hasattr(path_or_xgc, "path") else Path(path_or_xgc)
    if mesh is None:
        mesh = meshdata(str(path) + "/")
    if background_field is None:
        background_field = EquilibriumMagneticField(path)
    if delta_b_field is None and fine_dB_filename is not None:
        delta_b_field = FineDeltaBField.from_file(
            fine_dB_filename,
            path,
            mesh=mesh,
            magnetic_field=background_field,
        )

    rhs = TotalMagneticFieldLineRHS(background_field, delta_b_field)

    r = np.asarray(r0, dtype=float).copy()
    z = np.asarray(z0, dtype=float).copy()
    if r.shape != z.shape:
        raise ValueError("r0 and z0 must have the same shape")

    phi = np.full(r.shape, float(phi0), dtype=float)
    dphi = float(direction) * float(mesh.wedge_angle) / float(steps_per_period)

    crossings_r = np.full((n_crossings, r.size), np.nan, dtype=np.float64)
    crossings_z = np.full((n_crossings, r.size), np.nan, dtype=np.float64)

    nsteps = int(n_crossings) * int(steps_per_period)
    if integrator == "rk2":
        r, z, phi, active, lost_step, lost_crossing = _trace_steps_rk2(
            rhs,
            mesh,
            r,
            z,
            phi,
            dphi,
            nsteps,
            crossings_r,
            crossings_z,
            steps_per_period,
        )
    elif integrator == "abm2":
        r, z, phi, active, lost_step, lost_crossing = _trace_steps_abm2(
            rhs,
            mesh,
            r,
            z,
            phi,
            dphi,
            nsteps,
            crossings_r,
            crossings_z,
            steps_per_period,
        )
    else:
        raise ValueError("integrator must be 'rk2' or 'abm2'")

    fieldline_index = np.broadcast_to(np.arange(r.size), crossings_r.shape)
    crossing_index = np.broadcast_to(np.arange(n_crossings)[:, None], crossings_r.shape)
    fieldline_lost = lost_step >= 0

    result = PoincareTraceResult(
        r=crossings_r.reshape(-1),
        z=crossings_z.reshape(-1),
        crossing_index=crossing_index.reshape(-1),
        fieldline_index=fieldline_index.reshape(-1),
        fieldline_lost=fieldline_lost,
        lost_step=lost_step,
        lost_crossing=lost_crossing,
    )

    if output_filename is not None:
        np.savez(
            output_filename,
            r=result.r,
            z=result.z,
            crossing_index=result.crossing_index,
            fieldline_index=result.fieldline_index,
            fieldline_lost=result.fieldline_lost,
            lost_step=result.lost_step,
            lost_crossing=result.lost_crossing,
            n_crossings=np.array([n_crossings]),
            steps_per_period=np.array([steps_per_period]),
            integrator=np.array([integrator]),
            wedge_angle=np.array([mesh.wedge_angle]),
            has_delta_b=np.array([delta_b_field is not None]),
        )

    return result


def plot_poincare_map(result, *, ax=None, s=0.2, color="black", **scatter_kwargs):
    """Plot a Poincare trace result as an R-Z scatter map."""

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    ax.scatter(result.r, result.z, s=s, c=color, linewidths=0, **scatter_kwargs)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_poincare_heatmap(
    result,
    *,
    ax=None,
    bins=500,
    range=None,
    log_scale=True,
    cmap=None,
    cbar=True,
    **imshow_kwargs,
):
    """Plot a Poincare point-density heatmap.

    Points are counted on a regular R-Z histogram grid. The default colormap
    goes from black through orange to yellow.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(
            "poincare_black_orange_yellow",
            ["#000000", "#d95f02", "#ffd84d"],
        )

    valid = np.isfinite(result.r) & np.isfinite(result.z)
    counts, r_edges, z_edges = np.histogram2d(
        result.r[valid],
        result.z[valid],
        bins=bins,
        range=range,
    )
    image = counts.T
    if log_scale:
        image = np.log1p(image)

    extent = [r_edges[0], r_edges[-1], z_edges[0], z_edges[-1]]
    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=cmap,
        interpolation="nearest",
        **imshow_kwargs,
    )
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    if cbar:
        label = "log(1 + count)" if log_scale else "count"
        ax.figure.colorbar(im, ax=ax, label=label)
    return im, counts, r_edges, z_edges
