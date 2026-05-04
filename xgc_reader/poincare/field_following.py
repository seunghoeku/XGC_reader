"""Equilibrium-field following helpers for Poincare analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import adios2
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline


_TWO_PI = 2.0 * np.pi


def _as_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(()).item()
    return value


def _read_optional(reader, name, default=None):
    try:
        return reader.read(name)
    except Exception:
        return default


@dataclass
class FieldLinePoint:
    """A point projected along the background field."""

    r: float
    z: float
    phi: float


@dataclass
class FieldLinePoints:
    """Points projected along the background field."""

    r: np.ndarray
    z: np.ndarray
    phi: np.ndarray


def _is_scalar_like(value):
    return np.asarray(value).shape == ()


def _maybe_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr


class EquilibriumMagneticField:
    """Background magnetic field from ``xgc.equil.bp``.

    The implementation mirrors the 2D XGC-Devel magnetic field path:

    - use bicubic interpolation of ``eq_psi_rz`` on the rectangular R-Z grid
    - compute ``dpsi/dR`` and ``dpsi/dZ`` from the same spline coefficients
    - use cubic interpolation of ``eq_I(psi)`` for ``B_phi = I(psi) / R``
    - follow field lines using ``dR/dphi = Br / Bphi * R`` and
      ``dZ/dphi = Bz / Bphi * R``
    """

    def __init__(
        self,
        path_or_xgc="./",
        *,
        filename="xgc.equil.bp",
        plane_index=0,
        bp_sign=None,
        bt_sign=None,
        bounds="clip",
        bphi_floor=1.0e-50,
    ):
        if hasattr(path_or_xgc, "path"):
            path = Path(path_or_xgc.path)
        else:
            path = Path(path_or_xgc)

        self.path = path
        self.filename = filename
        self.bounds = bounds
        self.bphi_floor = bphi_floor

        with adios2.FileReader(str(path / filename)) as f:
            self.eq_mr = int(_as_scalar(f.read("eq_mr")))
            self.eq_mz = int(_as_scalar(f.read("eq_mz")))
            self.eq_min_r = float(_as_scalar(f.read("eq_min_r")))
            self.eq_max_r = float(_as_scalar(f.read("eq_max_r")))
            self.eq_min_z = float(_as_scalar(f.read("eq_min_z")))
            self.eq_max_z = float(_as_scalar(f.read("eq_max_z")))

            self.eq_psi_grid = np.asarray(f.read("eq_psi_grid"), dtype=float)
            self.eq_psi_norm = float(_as_scalar(_read_optional(f, "eq_psi_norm", 1.0)))
            self.eq_x_psi = _read_optional(f, "eq_x_psi")
            if self.eq_x_psi is not None:
                self.eq_x_psi = float(_as_scalar(self.eq_x_psi))
            self.bp_sign = float(_as_scalar(_read_optional(f, "bp_sign", 1.0) if bp_sign is None else bp_sign))
            self.bt_sign = float(_as_scalar(_read_optional(f, "bt_sign", 1.0) if bt_sign is None else bt_sign))

            psi_rz = np.asarray(f.read("eq_psi_rz"), dtype=float)
            if psi_rz.ndim == 3:
                psi_rz = psi_rz[plane_index]
            self.eq_psi_rz = self._as_zr(psi_rz, "eq_psi_rz")

            eq_i = _read_optional(f, "eq_I")
            self.eq_I = None if eq_i is None else np.asarray(eq_i, dtype=float)

            bphi_rz = _read_optional(f, "eq_B_phi_rz")
            if bphi_rz is not None:
                bphi_rz = np.asarray(bphi_rz, dtype=float)
                if bphi_rz.ndim == 3:
                    bphi_rz = bphi_rz[plane_index]
                self.eq_B_phi_rz = self._as_zr(bphi_rz, "eq_B_phi_rz")
            else:
                self.eq_B_phi_rz = None

        self.rgrid = np.linspace(self.eq_min_r, self.eq_max_r, self.eq_mr)
        self.zgrid = np.linspace(self.eq_min_z, self.eq_max_z, self.eq_mz)

        self.psi_spline = RectBivariateSpline(
            self.zgrid,
            self.rgrid,
            self.eq_psi_rz,
            kx=3,
            ky=3,
        )

        if self.eq_I is not None and self.eq_I.size > 0:
            psi_order = np.argsort(self.eq_psi_grid)
            self.eq_psi_grid = self.eq_psi_grid[psi_order]
            self.eq_I = self.eq_I[psi_order]
            self.I_spline = CubicSpline(self.eq_psi_grid, self.eq_I, extrapolate=True)
            self.bphi_spline = None
        elif self.eq_B_phi_rz is not None:
            self.I_spline = None
            self.bphi_spline = RectBivariateSpline(
                self.zgrid,
                self.rgrid,
                self.eq_B_phi_rz,
                kx=3,
                ky=3,
            )
        else:
            raise ValueError("xgc.equil.bp must contain either eq_I or eq_B_phi_rz")

    def _as_zr(self, values, name):
        if values.shape == (self.eq_mz, self.eq_mr):
            return values
        if values.shape == (self.eq_mr, self.eq_mz):
            return values.T
        raise ValueError(
            f"{name} has shape {values.shape}, expected "
            f"({self.eq_mz}, {self.eq_mr}) or ({self.eq_mr}, {self.eq_mz})"
        )

    def _bound_rz(self, r, z):
        r_arr = np.asarray(r, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        if self.bounds == "clip":
            r_arr = np.clip(r_arr, self.eq_min_r, self.eq_max_r)
            z_arr = np.clip(z_arr, self.eq_min_z, self.eq_max_z)
        elif self.bounds == "raise":
            outside = (
                (r_arr < self.eq_min_r)
                | (r_arr > self.eq_max_r)
                | (z_arr < self.eq_min_z)
                | (z_arr > self.eq_max_z)
            )
            if np.any(outside):
                raise ValueError("Requested point lies outside xgc.equil.bp R-Z bounds")
        return r_arr, z_arr

    def psi(self, r, z):
        """Return bicubic-interpolated poloidal flux."""

        rb, zb = self._bound_rz(r, z)
        return self.psi_spline.ev(zb, rb)

    def psi_and_gradient(self, r, z):
        """Return ``psi, dpsi/dR, dpsi/dZ`` from bicubic coefficients."""

        rb, zb = self._bound_rz(r, z)
        psi = self.psi_spline.ev(zb, rb)
        dpsi_dr = self.psi_spline.ev(zb, rb, dx=0, dy=1)
        dpsi_dz = self.psi_spline.ev(zb, rb, dx=1, dy=0)
        return psi, dpsi_dr, dpsi_dz

    def I_value(self, psi):
        """Return XGC's poloidal current function ``I(psi)=R B_phi``."""

        if self.I_spline is None:
            raise ValueError("eq_I is not available for this equilibrium")
        psi_i = np.maximum(0.0, psi)
        if self.eq_x_psi is not None:
            psi_i = np.minimum(psi_i, self.eq_x_psi)
        return self.bt_sign * self.I_spline(psi_i)

    def bvec(self, r, z, phi=0.0):
        """Return background magnetic field components ``(Br, Bz, Bphi)``."""

        del phi
        r_arr = np.asarray(r, dtype=float)
        psi, dpsi_dr, dpsi_dz = self.psi_and_gradient(r_arr, z)
        r_safe = np.where(r_arr == 0.0, np.finfo(float).tiny, r_arr)

        br = -dpsi_dz / r_safe * self.bp_sign
        bz = dpsi_dr / r_safe * self.bp_sign

        if self.I_spline is not None:
            bphi = self.I_value(psi) / r_safe
        else:
            rb, zb = self._bound_rz(r, z)
            bphi = self.bphi_spline.ev(zb, rb)

        bphi = np.where(bphi == 0.0, self.bphi_floor, bphi)
        return br, bz, bphi

    def derivs(self, r, z, phi=0.0):
        """Return ``dR/dphi`` and ``dZ/dphi`` for equilibrium field following."""

        br, bz, bphi = self.bvec(r, z, phi)
        r_arr = np.asarray(r, dtype=float)
        return br / bphi * r_arr, bz / bphi * r_arr

    def follow_field(self, r, z, phi_org, phi_dest, *, ff_step=4, ff_order=4):
        """Follow the equilibrium magnetic field from ``phi_org`` to ``phi_dest``.

        ``r``, ``z``, ``phi_org``, and ``phi_dest`` may be scalars or
        broadcast-compatible arrays. Array input is advanced in one vectorized
        Runge-Kutta loop.
        """

        if ff_step < 1:
            raise ValueError("ff_step must be >= 1")
        if ff_order not in (1, 2, 4):
            raise ValueError("ff_order must be 1, 2, or 4")

        scalar_result = all(_is_scalar_like(v) for v in (r, z, phi_org, phi_dest))
        r_arr, z_arr, phi_arr, phi_dest_arr = np.broadcast_arrays(
            np.asarray(r, dtype=float),
            np.asarray(z, dtype=float),
            np.asarray(phi_org, dtype=float),
            np.asarray(phi_dest, dtype=float),
        )
        r_cur = r_arr.copy()
        z_cur = z_arr.copy()
        phi_cur = phi_arr.copy()
        dphi = (phi_dest_arr - phi_cur) / float(ff_step)

        for _ in range(ff_step):
            if ff_order == 1:
                k1_r, k1_z = self.derivs(r_cur, z_cur, phi_cur)
                r_cur = r_cur + dphi * k1_r
                z_cur = z_cur + dphi * k1_z
            elif ff_order == 2:
                half = 0.5 * dphi
                k1_r, k1_z = self.derivs(r_cur, z_cur, phi_cur)
                k2_r, k2_z = self.derivs(
                    r_cur + half * k1_r,
                    z_cur + half * k1_z,
                    phi_cur + half,
                )
                r_cur = r_cur + dphi * k2_r
                z_cur = z_cur + dphi * k2_z
            else:
                half = 0.5 * dphi
                sixth = dphi / 6.0
                k1_r, k1_z = self.derivs(r_cur, z_cur, phi_cur)
                k2_r, k2_z = self.derivs(
                    r_cur + half * k1_r,
                    z_cur + half * k1_z,
                    phi_cur + half,
                )
                k3_r, k3_z = self.derivs(
                    r_cur + half * k2_r,
                    z_cur + half * k2_z,
                    phi_cur + half,
                )
                k4_r, k4_z = self.derivs(
                    r_cur + dphi * k3_r,
                    z_cur + dphi * k3_z,
                    phi_cur + dphi,
                )
                r_cur = r_cur + sixth * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
                z_cur = z_cur + sixth * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
            phi_cur = phi_cur + dphi

        if scalar_result:
            return FieldLinePoint(float(r_cur), float(z_cur), float(phi_dest_arr))
        return FieldLinePoints(r_cur, z_cur, phi_dest_arr.copy())


def nearest_midplane(phi, nphi, *, phi0=0.0, period=_TWO_PI):
    """Return the nearest toroidal midplane angle to ``phi``.

    ``phi`` may be a scalar or an array.
    """

    if nphi < 1:
        raise ValueError("nphi must be >= 1")
    dphi = period / float(nphi)
    scalar_result = _is_scalar_like(phi)
    phi_arr = np.asarray(phi, dtype=float)
    wrapped = (phi_arr - phi0) % period
    index = np.rint(wrapped / dphi).astype(int) % nphi
    phi_dest = phi0 + index * dphi
    delta = ((phi_dest - phi_arr + 0.5 * period) % period) - 0.5 * period
    phi_nearest = phi_arr + delta
    if scalar_result:
        return float(phi_nearest), int(index)
    return phi_nearest, index


def follow_field_to_nearest_midplane(
    magnetic_field,
    r,
    z,
    phi,
    *,
    nphi,
    phi0=0.0,
    period=_TWO_PI,
    ff_step=4,
    ff_order=4,
):
    """Follow point(s) to the nearest toroidal midplane.

    Parameters
    ----------
    magnetic_field
        ``EquilibriumMagneticField`` instance.
    r, z, phi
        Starting coordinates. These may be scalars or broadcast-compatible
        arrays.
    nphi
        Number of toroidal midplanes in ``period``.
    phi0
        Angle of midplane index 0.
    period
        Toroidal period represented by ``nphi`` planes. Use ``2*pi / wedge_n``
        for wedge simulations.
    ff_step, ff_order
        Runge-Kutta subdivision count and order, matching XGC-Devel's
        ``follow_field`` options.
    """

    phi_dest, midplane_index = nearest_midplane(phi, nphi, phi0=phi0, period=period)
    point = magnetic_field.follow_field(
        r,
        z,
        phi,
        phi_dest,
        ff_step=ff_step,
        ff_order=ff_order,
    )
    return point, midplane_index
