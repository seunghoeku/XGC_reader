"""Canonical-space distribution utilities.

This module provides helpers to average a distribution function on a
regular (energy, mu, P_phi) grid using weighted deposition.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .core import XGCDistribution

try:
    import numba as _numba
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    _numba = None
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]
    _HAS_NUMBA = False


def _distribution_without_perp_jacobian(dist: XGCDistribution) -> np.ndarray:
    """Convert f(v_perp, v_para) to d^3v form by removing v_perp Jacobian."""
    f_in = dist.f if dist.has_maxwellian else dist.f_g
    f_out = np.zeros_like(f_in)
    f_out[:, 1 : dist.vgrid.nvperp, :] = (
        f_in[:, 1 : dist.vgrid.nvperp, :]
        / dist.vgrid.vperp[np.newaxis, 1 : dist.vgrid.nvperp, np.newaxis]
    )
    f_out[:, 0, :] = f_in[:, 0, :] / (dist.vgrid.vperp[1] / 3.0)
    return f_out


def _cell_index_and_frac(x: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cell index [0, n-2] and fractional offset [0, 1] for linear weights."""
    idx = np.searchsorted(bins, x, side="right") - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    dx = bins[idx + 1] - bins[idx]
    frac = np.where(dx > 0.0, (x - bins[idx]) / dx, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    return idx, frac


def _tsc_1d_weights(x: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TSC (quadratic B-spline) weights for neighbors (i0-1, i0, i0+1)."""
    i0 = np.searchsorted(bins, x)
    i0 = np.clip(i0, 1, len(bins) - 2)

    dx = bins[i0 + 1] - bins[i0]
    dx = np.where(dx > 0.0, dx, 1.0)
    u = (x - bins[i0]) / dx

    def phi(u_: np.ndarray) -> np.ndarray:
        a = np.abs(u_)
        return np.where(a < 0.5, 0.75 - u_ * u_, np.where(a < 1.5, 0.5 * (1.5 - a) ** 2, 0.0))

    w_m1 = phi(u + 1.0)
    w_0 = phi(u)
    w_p1 = phi(u - 1.0)
    return i0, w_m1, w_0, w_p1


def _dunavant_7_rule() -> Tuple[np.ndarray, np.ndarray]:
    """Return Dunavant 7-point barycentric coordinates and weights."""
    bary = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
        ],
        dtype=np.float64,
    )
    weights = np.array(
        [
            0.225000000000000,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ],
        dtype=np.float64,
    )
    return bary, weights


def canonical_coordinates(
    xr,
    dist: XGCDistribution,
    potential: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (E, mu, P_phi) arrays shaped like (nn, nvperp, nvpdata) except mu.

    Returns:
        E:    (nn, nvperp, nvpdata) [J]
        mu:   (nn, nvperp)          [J/T]
        Pphi: (nn, nvperp, nvpdata)
    """
    nn = dist.nnodes
    nvperp = dist.vgrid.nvperp
    nvpdata = dist.vgrid.nvpdata

    if potential is None:
        potential = np.zeros(nn, dtype=np.float64)
    else:
        potential = np.asarray(potential, dtype=np.float64)
        if potential.shape[0] != nn:
            raise ValueError(f"potential length {potential.shape[0]} does not match nnodes {nn}")

    bmag = np.sqrt(xr.bfield[0, :] ** 2 + xr.bfield[1, :] ** 2 + xr.bfield[2, :] ** 2)
    e_charge = XGCDistribution.E_CHARGE
    mass = dist.mass

    v_norm = np.sqrt(dist.fg_temp_ev * e_charge / mass)
    vperp_phys = dist.vgrid.vperp[None, :, None] * v_norm[:, None, None]
    vpara_phys = dist.vgrid.vpara[None, None, :] * v_norm[:, None, None]

    e_perp = 0.5 * mass * vperp_phys**2
    e_para = 0.5 * mass * vpara_phys**2
    e_total = e_perp + e_para
    e_total += e_charge * potential[:, None, None]

    mu = (0.5 * mass * (dist.vgrid.vperp[None, :] * v_norm[:, None]) ** 2) / bmag[:, None]

    pphi = xr.mesh.psi[:, None, None] + (
        mass / e_charge
    ) * xr.mesh.r[:, None, None] * vpara_phys * xr.bfield[2, :, None, None] / bmag[:, None, None]

    return e_total, mu, pphi


def _cell_edges_from_centers(centers: np.ndarray, clamp_nonnegative: bool = False) -> np.ndarray:
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    if clamp_nonnegative:
        edges[0] = max(edges[0], 0.0)
    return edges


def _scatter_add_bincount(target_flat: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> None:
    """Sparse accumulation helper using unique+bincount."""
    uniq, inv = np.unique(idx, return_inverse=True)
    target_flat[uniq] += np.bincount(inv, weights=weights)


if _HAS_NUMBA:

    @njit(cache=True)
    def _phi_tsc(u: float) -> float:
        a = abs(u)
        if a < 0.5:
            return 0.75 - u * u
        if a < 1.5:
            t = 1.5 - a
            return 0.5 * t * t
        return 0.0


    @njit(cache=True)
    def _tsc_scalar(x: float, bins: np.ndarray) -> Tuple[int, float, float, float]:
        i0 = np.searchsorted(bins, x)
        nb = bins.size
        if i0 < 1:
            i0 = 1
        elif i0 > nb - 2:
            i0 = nb - 2
        dx = bins[i0 + 1] - bins[i0]
        if dx <= 0.0:
            dx = 1.0
        u = (x - bins[i0]) / dx
        return i0, _phi_tsc(u + 1.0), _phi_tsc(u), _phi_tsc(u - 1.0)


    @njit(parallel=True, cache=True)
    def _analytic_fm_accumulate_openmp(
        triangles: np.ndarray,
        tri_area: np.ndarray,
        bary_q: np.ndarray,
        w_cfg: np.ndarray,
        den: np.ndarray,
        temp_ev: np.ndarray,
        flow: np.ndarray,
        fg_temp_ev: np.ndarray,
        potential: np.ndarray,
        psi: np.ndarray,
        r: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        vp_flat: np.ndarray,
        vpa_flat: np.ndarray,
        meas_base: np.ndarray,
        e_bins: np.ndarray,
        mu_bins: np.ndarray,
        pphi_bins: np.ndarray,
        mass: float,
        e_charge: float,
        ev_to_joule: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ntri = triangles.shape[0]
        nsamp = vp_flat.size
        nen = e_bins.size
        nmu = mu_bins.size
        npphi = pphi_bins.size
        nbins = nen * nmu * npphi
        nthreads = _numba.get_num_threads()
        sw_tls = np.zeros((nthreads, nbins), dtype=np.float64)
        ww_tls = np.zeros((nthreads, nbins), dtype=np.float64)

        for itri in prange(ntri):
            area = tri_area[itri]
            if area <= 0.0:
                continue
            tid = _numba.get_thread_id()
            sw = sw_tls[tid]
            ww = ww_tls[tid]

            n0 = triangles[itri, 0]
            n1 = triangles[itri, 1]
            n2 = triangles[itri, 2]

            for iq in range(7):
                b0w = bary_q[iq, 0]
                b1w = bary_q[iq, 1]
                b2w = bary_q[iq, 2]
                cfg_weight = area * w_cfg[iq]

                den_q = b0w * den[n0] + b1w * den[n1] + b2w * den[n2]
                temp_q = b0w * temp_ev[n0] + b1w * temp_ev[n1] + b2w * temp_ev[n2]
                flow_q = b0w * flow[n0] + b1w * flow[n1] + b2w * flow[n2]
                fg_q = b0w * fg_temp_ev[n0] + b1w * fg_temp_ev[n1] + b2w * fg_temp_ev[n2]
                pot_q = b0w * potential[n0] + b1w * potential[n1] + b2w * potential[n2]
                psi_q = b0w * psi[n0] + b1w * psi[n1] + b2w * psi[n2]
                r_q = b0w * r[n0] + b1w * r[n1] + b2w * r[n2]
                bb0_q = b0w * b0[n0] + b1w * b0[n1] + b2w * b0[n2]
                bb1_q = b0w * b1[n0] + b1w * b1[n1] + b2w * b1[n2]
                bb2_q = b0w * b2[n0] + b1w * b2[n1] + b2w * b2[n2]
                bmag_q = np.sqrt(bb0_q * bb0_q + bb1_q * bb1_q + bb2_q * bb2_q)
                if bmag_q <= 0.0:
                    continue

                if temp_q < 1e-12:
                    temp_q = 1e-12
                if fg_q < 1e-30:
                    fg_q = 1e-30

                v_n_q = np.sqrt(fg_q * e_charge / mass)
                inv_temp_j = 1.0 / (temp_q * ev_to_joule)
                temp_fac = den_q * np.sqrt(fg_q) / (temp_q**1.5)
                pphi_coef = (mass / e_charge) * r_q * bb2_q / bmag_q

                for isamp in range(nsamp):
                    vperp_phys = vp_flat[isamp] * v_n_q
                    vpara_phys = vpa_flat[isamp] * v_n_q
                    en = 0.5 * mass * (
                        (vpara_phys - flow_q) * (vpara_phys - flow_q) + vperp_phys * vperp_phys
                    ) * inv_temp_j
                    f_val = temp_fac * np.exp(-en)

                    e_i = e_charge * pot_q + 0.5 * mass * (vperp_phys * vperp_phys + vpara_phys * vpara_phys)
                    mu_i = 0.5 * mass * vperp_phys * vperp_phys / bmag_q
                    pphi_i = psi_q + pphi_coef * vpara_phys
                    wt = cfg_weight * fg_q * meas_base[isamp]
                    val = f_val * wt

                    ie0, we_m1, we_0, we_p1 = _tsc_scalar(e_i, e_bins)
                    jm0, wm_m1, wm_0, wm_p1 = _tsc_scalar(mu_i, mu_bins)
                    kp0, wp_m1, wp_0, wp_p1 = _tsc_scalar(pphi_i, pphi_bins)

                    ie_m1 = ie0 - 1
                    if ie_m1 < 0:
                        ie_m1 = 0
                    ie_p1 = ie0 + 1
                    if ie_p1 > nen - 1:
                        ie_p1 = nen - 1

                    jm_m1 = jm0 - 1
                    if jm_m1 < 0:
                        jm_m1 = 0
                    jm_p1 = jm0 + 1
                    if jm_p1 > nmu - 1:
                        jm_p1 = nmu - 1

                    kp_m1 = kp0 - 1
                    if kp_m1 < 0:
                        kp_m1 = 0
                    kp_p1 = kp0 + 1
                    if kp_p1 > npphi - 1:
                        kp_p1 = npphi - 1

                    for ie_idx, we in ((ie_m1, we_m1), (ie0, we_0), (ie_p1, we_p1)):
                        for jm_idx, wm in ((jm_m1, wm_m1), (jm0, wm_0), (jm_p1, wm_p1)):
                            w_em = we * wm
                            for kp_idx, wp in ((kp_m1, wp_m1), (kp0, wp_0), (kp_p1, wp_p1)):
                                lin = (ie_idx * nmu + jm_idx) * npphi + kp_idx
                                w_tot = w_em * wp
                                sw[lin] += w_tot * val
                                ww[lin] += w_tot * wt

        sum_weighted_flat = np.zeros(nbins, dtype=np.float64)
        sum_weights_flat = np.zeros(nbins, dtype=np.float64)
        for t in range(nthreads):
            sum_weighted_flat += sw_tls[t]
            sum_weights_flat += ww_tls[t]
        return sum_weighted_flat, sum_weights_flat


def average_analytic_maxwellian_emu_pphi(
    xr,
    dist: XGCDistribution,
    potential: np.ndarray | None = None,
    bins: Tuple[int, int, int] = (180, 180, 300),
    nq_vperp: int = 3,
    nq_vpara: int = 3,
    openmp: bool = True,
    openmp_threads: int | None = None,
) -> Dict[str, np.ndarray]:
    """Average analytic Maxwellian in (E, mu, P_phi) using velocity quadrature.

    Maxwellian moments come from dist.den, dist.temp_ev, dist.flow, dist.fg_temp_ev.
    If numba is available, openmp=True enables a parallel CPU path.
    """
    if nq_vperp < 1 or nq_vpara < 1:
        raise ValueError("nq_vperp and nq_vpara must be >= 1")
    if not hasattr(xr.mesh, "cnct"):
        raise AttributeError("xr.mesh.cnct is required")

    nen, nmu, npphi = bins
    nn = dist.nnodes
    nvperp = dist.vgrid.nvperp
    nvpdata = dist.vgrid.nvpdata

    # Canonical ranges from center-grid values (for robust bin ranges).
    E_ref, mu_ref, Pphi_ref = canonical_coordinates(xr=xr, dist=dist, potential=potential)
    e_bins = np.linspace(np.min(E_ref), np.max(E_ref), nen)
    mu_bins = np.linspace(0.0, np.max(mu_ref), nmu)
    pphi_bins = np.linspace(np.min(Pphi_ref), np.max(Pphi_ref), npphi)

    sum_weighted = np.zeros((nen, nmu, npphi), dtype=np.float64)
    sum_weights = np.zeros((nen, nmu, npphi), dtype=np.float64)
    sum_weighted_flat = sum_weighted.ravel()
    sum_weights_flat = sum_weights.ravel()

    e_charge = XGCDistribution.E_CHARGE
    mass = dist.mass
    vperp_c = dist.vgrid.vperp
    vpara_c = dist.vgrid.vpara
    vperp_edges = _cell_edges_from_centers(vperp_c, clamp_nonnegative=True)
    vpara_edges = _cell_edges_from_centers(vpara_c, clamp_nonnegative=False)
    gperp_x, gperp_w = np.polynomial.legendre.leggauss(nq_vperp)
    gpara_x, gpara_w = np.polynomial.legendre.leggauss(nq_vpara)
    # Precompute all velocity-quadrature samples once to avoid Python nested loops.
    vp_lo = vperp_edges[:-1, None]
    vp_hi = vperp_edges[1:, None]
    vp_pts = 0.5 * ((vp_hi - vp_lo) * gperp_x[None, :] + (vp_hi + vp_lo))
    vp_w = 0.5 * (vp_hi - vp_lo) * gperp_w[None, :]
    vpa_lo = vpara_edges[:-1, None]
    vpa_hi = vpara_edges[1:, None]
    vpa_pts = 0.5 * ((vpa_hi - vpa_lo) * gpara_x[None, :] + (vpa_hi + vpa_lo))
    vpa_w = 0.5 * (vpa_hi - vpa_lo) * gpara_w[None, :]
    quad_shape = (nvperp, nq_vperp, nvpdata, nq_vpara)
    vp_flat = np.broadcast_to(vp_pts[:, :, None, None], quad_shape).reshape(-1)
    vpa_flat = np.broadcast_to(vpa_pts[None, None, :, :], quad_shape).reshape(-1)
    vw_flat = (vp_w[:, :, None, None] * vpa_w[None, None, :, :]).reshape(-1)
    meas_base = np.sqrt(1.0 / (2.0 * np.pi)) * vp_flat * vw_flat

    r = xr.mesh.r
    z = xr.mesh.z
    psi = xr.mesh.psi
    b0 = xr.bfield[0, :]
    b1 = xr.bfield[1, :]
    b2 = xr.bfield[2, :]
    triangles = np.asarray(xr.mesh.cnct, dtype=np.int64)
    bary_q, w_cfg = _dunavant_7_rule()

    if potential is None:
        potential = np.zeros(nn, dtype=np.float64)
    else:
        potential = np.asarray(potential, dtype=np.float64)

    def _deposit(e_i: np.ndarray, mu_i: np.ndarray, pphi_i: np.ndarray, val: np.ndarray, w: np.ndarray) -> None:
        ie0, we_m1, we_0, we_p1 = _tsc_1d_weights(e_i, e_bins)
        jm0, wm_m1, wm_0, wm_p1 = _tsc_1d_weights(mu_i, mu_bins)
        kp0, wp_m1, wp_0, wp_p1 = _tsc_1d_weights(pphi_i, pphi_bins)
        ie_m1 = np.clip(ie0 - 1, 0, nen - 1)
        ie_0 = np.clip(ie0, 0, nen - 1)
        ie_p1 = np.clip(ie0 + 1, 0, nen - 1)
        jm_m1 = np.clip(jm0 - 1, 0, nmu - 1)
        jm_0 = np.clip(jm0, 0, nmu - 1)
        jm_p1 = np.clip(jm0 + 1, 0, nmu - 1)
        kp_m1 = np.clip(kp0 - 1, 0, npphi - 1)
        kp_0 = np.clip(kp0, 0, npphi - 1)
        kp_p1 = np.clip(kp0 + 1, 0, npphi - 1)
        for ie_idx, we in ((ie_m1, we_m1), (ie_0, we_0), (ie_p1, we_p1)):
            for jm_idx, wm in ((jm_m1, wm_m1), (jm_0, wm_0), (jm_p1, wm_p1)):
                w_em = we * wm
                np.add.at(sum_weighted, (ie_idx, jm_idx, kp_m1), w_em * wp_m1 * val)
                np.add.at(sum_weighted, (ie_idx, jm_idx, kp_0), w_em * wp_0 * val)
                np.add.at(sum_weighted, (ie_idx, jm_idx, kp_p1), w_em * wp_p1 * val)
                np.add.at(sum_weights, (ie_idx, jm_idx, kp_m1), w_em * wp_m1 * w)
                np.add.at(sum_weights, (ie_idx, jm_idx, kp_0), w_em * wp_0 * w)
                np.add.at(sum_weights, (ie_idx, jm_idx, kp_p1), w_em * wp_p1 * w)

    def _progress_iter(it, total, desc):
        try:
            from tqdm.auto import tqdm  # type: ignore

            return tqdm(it, total=total, desc=desc)
        except Exception:
            return it

    def _sample_cfg_and_accumulate(
        den_q: float,
        temp_q: float,
        flow_q: float,
        fg_q: float,
        psi_q: float,
        r_q: float,
        b2_q: float,
        bmag_q: float,
        pot_q: float,
        cfg_weight: float,
    ) -> None:
        if bmag_q <= 0.0:
            return
        temp_q = max(temp_q, 1e-12)
        fg_q = max(fg_q, 1e-30)
        v_n_q = np.sqrt(fg_q * e_charge / mass)
        vperp_phys = vp_flat * v_n_q
        vpara_phys = vpa_flat * v_n_q
        en = 0.5 * mass * ((vpara_phys - flow_q) ** 2 + vperp_phys**2) / (temp_q * XGCDistribution.EV_TO_JOULE)
        f_q = den_q * np.exp(-en) / (temp_q**1.5) * np.sqrt(fg_q)

        e_i = e_charge * pot_q + 0.5 * mass * (vperp_phys**2 + vpara_phys**2)
        mu_i = 0.5 * mass * vperp_phys**2 / bmag_q
        pphi_i = psi_q + (mass / e_charge) * r_q * vpara_phys * b2_q / bmag_q

        wt = cfg_weight * fg_q * meas_base
        _deposit(
            e_i=e_i,
            mu_i=mu_i,
            pphi_i=pphi_i,
            val=f_q * wt,
            w=wt,
        )

    r0 = r[triangles[:, 0]]
    r1 = r[triangles[:, 1]]
    r2 = r[triangles[:, 2]]
    z0 = z[triangles[:, 0]]
    z1 = z[triangles[:, 1]]
    z2 = z[triangles[:, 2]]
    tri_area = 0.5 * np.abs((r1 - r0) * (z2 - z0) - (r2 - r0) * (z1 - z0))

    use_openmp = bool(openmp and _HAS_NUMBA)
    if use_openmp:
        prev_threads = None
        if openmp_threads is not None and openmp_threads > 0:
            prev_threads = _numba.get_num_threads()
            _numba.set_num_threads(int(openmp_threads))
        try:
            sum_weighted_flat, sum_weights_flat = _analytic_fm_accumulate_openmp(
                triangles=triangles,
                tri_area=tri_area,
                bary_q=bary_q,
                w_cfg=w_cfg,
                den=np.asarray(dist.den, dtype=np.float64),
                temp_ev=np.asarray(dist.temp_ev, dtype=np.float64),
                flow=np.asarray(dist.flow, dtype=np.float64),
                fg_temp_ev=np.asarray(dist.fg_temp_ev, dtype=np.float64),
                potential=np.asarray(potential, dtype=np.float64),
                psi=np.asarray(psi, dtype=np.float64),
                r=np.asarray(r, dtype=np.float64),
                b0=np.asarray(b0, dtype=np.float64),
                b1=np.asarray(b1, dtype=np.float64),
                b2=np.asarray(b2, dtype=np.float64),
                vp_flat=np.asarray(vp_flat, dtype=np.float64),
                vpa_flat=np.asarray(vpa_flat, dtype=np.float64),
                meas_base=np.asarray(meas_base, dtype=np.float64),
                e_bins=np.asarray(e_bins, dtype=np.float64),
                mu_bins=np.asarray(mu_bins, dtype=np.float64),
                pphi_bins=np.asarray(pphi_bins, dtype=np.float64),
                mass=float(mass),
                e_charge=float(e_charge),
                ev_to_joule=float(XGCDistribution.EV_TO_JOULE),
            )
            sum_weighted[:, :, :] = sum_weighted_flat.reshape(sum_weighted.shape)
            sum_weights[:, :, :] = sum_weights_flat.reshape(sum_weights.shape)
        finally:
            if prev_threads is not None:
                _numba.set_num_threads(prev_threads)
    else:
        for itri, tri_nodes in _progress_iter(enumerate(triangles), triangles.shape[0], "analytic fm avg"):
            area = tri_area[itri]
            if area <= 0.0:
                continue
            for iq in range(7):
                bary = bary_q[iq]
                cfg_weight = area * w_cfg[iq]
                den_q = float(np.dot(bary, dist.den[tri_nodes]))
                temp_q = float(np.dot(bary, dist.temp_ev[tri_nodes]))
                flow_q = float(np.dot(bary, dist.flow[tri_nodes]))
                fg_q = float(np.dot(bary, dist.fg_temp_ev[tri_nodes]))
                pot_q = float(np.dot(bary, potential[tri_nodes]))
                psi_q = float(np.dot(bary, psi[tri_nodes]))
                r_q = float(np.dot(bary, r[tri_nodes]))
                b0_q = float(np.dot(bary, b0[tri_nodes]))
                b1_q = float(np.dot(bary, b1[tri_nodes]))
                b2_q = float(np.dot(bary, b2[tri_nodes]))
                bmag_q = np.sqrt(b0_q * b0_q + b1_q * b1_q + b2_q * b2_q)
                _sample_cfg_and_accumulate(den_q, temp_q, flow_q, fg_q, psi_q, r_q, b2_q, bmag_q, pot_q, cfg_weight)

    fmean = np.where(sum_weights > 0.0, sum_weighted / sum_weights, 0.0)
    return {
        "fmean": fmean,
        "sum_weighted": sum_weighted,
        "sum_weights": sum_weights,
        "E_bins": e_bins,
        "mu_bins": mu_bins,
        "pphi_bins": pphi_bins,
        "E": E_ref,
        "mu": mu_ref,
        "Pphi": Pphi_ref,
    }


def interpolate_fmean_to_velocity_grid(
    fmean: np.ndarray,
    E: np.ndarray,
    mu: np.ndarray,
    Pphi: np.ndarray,
    E_bins: np.ndarray,
    mu_bins: np.ndarray,
    pphi_bins: np.ndarray,
) -> np.ndarray:
    """Interpolate fmean(E,mu,Pphi) back to (nn, nvperp, nvpdata) using TSC."""

    nn, nvperp, nvpdata = E.shape
    nen, nmu, npphi = fmean.shape

    def _expand_to_e_shape(arr: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.shape == E.shape:
            return arr
        if arr.ndim == 2 and arr.shape == (nn, nvperp):
            return np.repeat(arr[:, :, None], nvpdata, axis=2)
        if arr.ndim == 3 and arr.shape == (nn, nvperp, 1):
            return np.repeat(arr, nvpdata, axis=2)
        if arr.ndim == 3 and arr.shape == (nn, 1, nvpdata):
            return np.repeat(arr, nvperp, axis=1)
        raise ValueError(
            f"{name} shape {arr.shape} is incompatible with E shape {E.shape}. "
            f"Expected one of {(nn, nvperp)}, {(nn, nvperp, nvpdata)}, "
            f"{(nn, nvperp, 1)}, {(nn, 1, nvpdata)}"
        )

    E_flat = np.asarray(E).ravel()
    mu_flat = _expand_to_e_shape(mu, "mu").ravel()
    Pphi_flat = _expand_to_e_shape(Pphi, "Pphi").ravel()
    if not (E_flat.size == mu_flat.size == Pphi_flat.size):
        raise ValueError(
            "E, mu, Pphi flattened sizes must match: "
            f"{E_flat.size}, {mu_flat.size}, {Pphi_flat.size}"
        )
    fsrc = np.asarray(fmean)
    ie0, we_m1, we_0, we_p1 = _tsc_1d_weights(E_flat, E_bins)
    jm0, wm_m1, wm_0, wm_p1 = _tsc_1d_weights(mu_flat, mu_bins)
    kp0, wp_m1, wp_0, wp_p1 = _tsc_1d_weights(Pphi_flat, pphi_bins)
    we_sum = np.where((we_m1 + we_0 + we_p1) > 0.0, we_m1 + we_0 + we_p1, 1.0)
    wm_sum = np.where((wm_m1 + wm_0 + wm_p1) > 0.0, wm_m1 + wm_0 + wm_p1, 1.0)
    wp_sum = np.where((wp_m1 + wp_0 + wp_p1) > 0.0, wp_m1 + wp_0 + wp_p1, 1.0)
    we_m1, we_0, we_p1 = we_m1 / we_sum, we_0 / we_sum, we_p1 / we_sum
    wm_m1, wm_0, wm_p1 = wm_m1 / wm_sum, wm_0 / wm_sum, wm_p1 / wm_sum
    wp_m1, wp_0, wp_p1 = wp_m1 / wp_sum, wp_0 / wp_sum, wp_p1 / wp_sum

    ie_m1 = np.clip(ie0 - 1, 0, nen - 1)
    ie_0 = np.clip(ie0, 0, nen - 1)
    ie_p1 = np.clip(ie0 + 1, 0, nen - 1)
    jm_m1 = np.clip(jm0 - 1, 0, nmu - 1)
    jm_0 = np.clip(jm0, 0, nmu - 1)
    jm_p1 = np.clip(jm0 + 1, 0, nmu - 1)
    kp_m1 = np.clip(kp0 - 1, 0, npphi - 1)
    kp_0 = np.clip(kp0, 0, npphi - 1)
    kp_p1 = np.clip(kp0 + 1, 0, npphi - 1)

    out_flat = np.zeros_like(E_flat, dtype=fmean.dtype)
    for ie_idx, we in ((ie_m1, we_m1), (ie_0, we_0), (ie_p1, we_p1)):
        for jm_idx, wm in ((jm_m1, wm_m1), (jm_0, wm_0), (jm_p1, wm_p1)):
            w_em = we * wm
            out_flat += w_em * wp_m1 * fsrc[ie_idx, jm_idx, kp_m1]
            out_flat += w_em * wp_0 * fsrc[ie_idx, jm_idx, kp_0]
            out_flat += w_em * wp_p1 * fsrc[ie_idx, jm_idx, kp_p1]

    return out_flat.reshape(nn, nvperp, nvpdata)


def average_distribution_emu_pphi(
    xr,
    dist: XGCDistribution,
    potential: np.ndarray | None = None,
    bins: Tuple[int, int, int] = (180, 180, 300),
    bins_fm: Tuple[int, int, int] | None = None,
    separate_fm: bool = False,
) -> Dict[str, np.ndarray]:
    """Average distribution on a regular (energy, mu, P_phi) grid.

    Args:
        xr: XGC reader instance with mesh and magnetic field data loaded.
        dist: Distribution object.
        potential: Electrostatic potential array per node (same nnodes).
        bins: Number of points in (E, mu, P_phi) grid.
        Uses fixed settings:
        - dunavant7 in configuration space
        - TSC in canonical space
        - progress enabled
        - always back interpolate
        - always renormalize TSC weights
        - linear back interpolation
        - linear interpolation for f
        bins_fm: Optional bins for Maxwellian branch when `separate_fm=True`.
            If None, uses `bins`.
        separate_fm: Use analytic Maxwellian averaging + original f_g averaging and sum.

    Returns:
        Dictionary with keys:
        - fmean: averaged distribution on (E, mu, P_phi) grid
        - sum_weighted: weighted sum accumulator
        - sum_weights: weights accumulator
        - E_bins, mu_bins, pphi_bins
        - E, mu, Pphi (canonical coordinates at phase-space points)
    """
    method = "tsc"
    config_interp = "dunavant7"
    progress = True
    back_interpolate = True
    renormalize_tsc_weights = True
    back_interp_space = "linear"
    f_interp_space = "linear"

    if not hasattr(xr.mesh, "cnct"):
        raise AttributeError("xr.mesh.cnct is required")

    if bins_fm is None:
        bins_fm = bins

    # Composition mode:
    # 1) Maxwellian part from analytic velocity quadrature.
    # 2) Numerical part from original method on f_g (after remove_maxwellian()).
    # 3) Sum both (and sum after back interpolation when requested).
    if separate_fm:
        # Keep full-f d^3v representation from the original distribution
        # (before any remove_maxwellian call), as requested.
        f_wo_perp_full_before = _distribution_without_perp_jacobian(dist)

        fm_out = average_analytic_maxwellian_emu_pphi(
            xr=xr,
            dist=dist,
            potential=potential,
            bins=bins_fm,
        )

        removed_maxwellian_comp = False
        if dist.has_maxwellian:
            dist.remove_maxwellian()
            removed_maxwellian_comp = True

        try:
            fg_out = average_distribution_emu_pphi(
                xr=xr,
                dist=dist,
                potential=potential,
                bins=bins,
                separate_fm=False,
            )
        finally:
            if removed_maxwellian_comp:
                dist.add_maxwellian()

        out = dict(fg_out)
        out["fmean_numerical"] = fg_out["fmean"]
        out["fmean_fm"] = fm_out["fmean"]
        out["sum_weighted_numerical"] = fg_out["sum_weighted"]
        out["sum_weighted_fm"] = fm_out["sum_weighted"]
        out["f_wo_perp"] = f_wo_perp_full_before

        # If canonical bins differ between numerical and fm branches, direct
        # canonical-grid summation is undefined. In that case, combine only in
        # velocity space after back interpolation.
        if fg_out["fmean"].shape == fm_out["fmean"].shape:
            out["fmean"] = fg_out["fmean"] + fm_out["fmean"]
            out["sum_weighted"] = fg_out["sum_weighted"] + fm_out["sum_weighted"]
        else:
            out["fmean"] = fg_out["fmean"]
            out["sum_weighted"] = fg_out["sum_weighted"]
            out["fmean_combined_on_grid"] = False

        if back_interpolate:
            if "fmean_at_ijk" in fg_out:
                fmean_at_ijk_numerical = fg_out["fmean_at_ijk"]
            else:
                fmean_at_ijk_numerical = interpolate_fmean_to_velocity_grid(
                    fmean=fg_out["fmean"],
                    E=fg_out["E"],
                    mu=fg_out["mu"],
                    Pphi=fg_out["Pphi"],
                    E_bins=fg_out["E_bins"],
                    mu_bins=fg_out["mu_bins"],
                    pphi_bins=fg_out["pphi_bins"],
                )
            fmean_at_ijk_fm = interpolate_fmean_to_velocity_grid(
                fmean=fm_out["fmean"],
                E=fm_out["E"],
                mu=fm_out["mu"],
                Pphi=fm_out["Pphi"],
                E_bins=fm_out["E_bins"],
                mu_bins=fm_out["mu_bins"],
                pphi_bins=fm_out["pphi_bins"],
            )
            out["fmean_at_ijk_numerical"] = fmean_at_ijk_numerical
            out["fmean_at_ijk_fm"] = fmean_at_ijk_fm
            out["fmean_at_ijk"] = fmean_at_ijk_numerical + fmean_at_ijk_fm

        return out

    nen, nmu, npphi = bins
    nn = dist.nnodes
    nvperp = dist.vgrid.nvperp
    nvpdata = dist.vgrid.nvpdata

    E, mu, Pphi = canonical_coordinates(xr=xr, dist=dist, potential=potential)
    f_wo_perp = _distribution_without_perp_jacobian(dist)
    f_wo_perp_for_interp = f_wo_perp

    e_bins = np.linspace(np.min(E), np.max(E), nen)
    mu_bins = np.linspace(0.0, np.max(mu), nmu)
    pphi_bins = np.linspace(np.min(Pphi), np.max(Pphi), npphi)

    sum_weighted = np.zeros((nen, nmu, npphi), dtype=np.float64)
    sum_weights = np.zeros((nen, nmu, npphi), dtype=np.float64)
    sum_weighted_flat = sum_weighted.ravel()
    sum_weights_flat = sum_weights.ravel()

    triangles = np.asarray(xr.mesh.cnct, dtype=np.int64)
    r = xr.mesh.r
    z = xr.mesh.z
    psi = xr.mesh.psi
    b0 = xr.bfield[0, :]
    b1 = xr.bfield[1, :]
    b2 = xr.bfield[2, :]
    e_charge = XGCDistribution.E_CHARGE
    mass = dist.mass
    vperp = dist.vgrid.vperp
    vpara = dist.vgrid.vpara
    vspace_prefac = np.sqrt(1.0 / (2.0 * np.pi)) * dist.vgrid.dvperp**2 * dist.vgrid.dvpara
    v_n_nodes = np.sqrt(dist.fg_temp_ev * e_charge / mass)
    bary_q, w_q = _dunavant_7_rule()

    if potential is None:
        potential = np.zeros(nn, dtype=np.float64)
    else:
        potential = np.asarray(potential, dtype=np.float64)

    def _deposit_sample(e_q: np.ndarray, mu_q: np.ndarray, pphi_q: np.ndarray, f_q: np.ndarray, vol: float) -> None:
        e_i = e_q.ravel()
        mu_i = np.broadcast_to(mu_q[:, None], (nvperp, nvpdata)).ravel()
        pphi_i = pphi_q.ravel()
        f_i = f_q.ravel()
        val = f_i * vol

        ie0, we_m1, we_0, we_p1 = _tsc_1d_weights(e_i, e_bins)
        jm0, wm_m1, wm_0, wm_p1 = _tsc_1d_weights(mu_i, mu_bins)
        kp0, wp_m1, wp_0, wp_p1 = _tsc_1d_weights(pphi_i, pphi_bins)

        ie_m1 = np.clip(ie0 - 1, 0, nen - 1)
        ie_0 = np.clip(ie0, 0, nen - 1)
        ie_p1 = np.clip(ie0 + 1, 0, nen - 1)
        jm_m1 = np.clip(jm0 - 1, 0, nmu - 1)
        jm_0 = np.clip(jm0, 0, nmu - 1)
        jm_p1 = np.clip(jm0 + 1, 0, nmu - 1)
        kp_m1 = np.clip(kp0 - 1, 0, npphi - 1)
        kp_0 = np.clip(kp0, 0, npphi - 1)
        kp_p1 = np.clip(kp0 + 1, 0, npphi - 1)

        for ie_idx, we in ((ie_m1, we_m1), (ie_0, we_0), (ie_p1, we_p1)):
            for jm_idx, wm in ((jm_m1, wm_m1), (jm_0, wm_0), (jm_p1, wm_p1)):
                w_em = we * wm
                for kp_idx, wp in ((kp_m1, wp_m1), (kp_0, wp_0), (kp_p1, wp_p1)):
                    w_tot = w_em * wp
                    lin_idx = (ie_idx * nmu + jm_idx) * npphi + kp_idx
                    _scatter_add_bincount(sum_weighted_flat, lin_idx, w_tot * val)
                    _scatter_add_bincount(sum_weights_flat, lin_idx, w_tot * vol)

    def _progress_iter(iterable, total: int, desc: str):
        try:
            from tqdm.auto import tqdm  # type: ignore

            return tqdm(iterable, total=total, desc=desc)
        except Exception:
            return iterable

    # Triangle geometric areas in R-Z plane.
    r0 = r[triangles[:, 0]]
    r1 = r[triangles[:, 1]]
    r2 = r[triangles[:, 2]]
    z0 = z[triangles[:, 0]]
    z1 = z[triangles[:, 1]]
    z2 = z[triangles[:, 2]]
    tri_area = 0.5 * np.abs((r1 - r0) * (z2 - z0) - (r2 - r0) * (z1 - z0))

    tri_iter = _progress_iter(enumerate(triangles), total=triangles.shape[0], desc="canonical avg")
    for itri, tri_nodes in tri_iter:
        area = tri_area[itri]
        if area <= 0.0:
            continue

        f_tri = f_wo_perp_for_interp[tri_nodes, :, :]
        v_n_tri = v_n_nodes[tri_nodes]
        pot_tri = potential[tri_nodes]
        r_tri = r[tri_nodes]
        psi_tri = psi[tri_nodes]
        b0_tri = b0[tri_nodes]
        b1_tri = b1[tri_nodes]
        b2_tri = b2[tri_nodes]

        for iq in range(7):
            bary = bary_q[iq]
            cfg_vol = area * w_q[iq]
            f_q = np.tensordot(bary, f_tri, axes=(0, 0))
            v_n_q = float(np.dot(bary, v_n_tri))
            pot_q = float(np.dot(bary, pot_tri))
            r_q = float(np.dot(bary, r_tri))
            psi_q = float(np.dot(bary, psi_tri))
            b0_q = float(np.dot(bary, b0_tri))
            b1_q = float(np.dot(bary, b1_tri))
            b2_q = float(np.dot(bary, b2_tri))
            bmag_q = np.sqrt(b0_q * b0_q + b1_q * b1_q + b2_q * b2_q)
            if bmag_q <= 0.0:
                continue
            fg_temp_q = mass / e_charge * v_n_q * v_n_q
            vol = cfg_vol * (fg_temp_q**1.5) * vspace_prefac
            vperp_phys = vperp[:, None] * v_n_q
            vpara_phys = vpara[None, :] * v_n_q
            e_q = e_charge * pot_q + 0.5 * mass * (vperp_phys**2 + vpara_phys**2)
            mu_q = (0.5 * mass * (vperp * v_n_q) ** 2) / bmag_q
            pphi_line = psi_q + (mass / e_charge) * r_q * (vpara * v_n_q) * b2_q / bmag_q
            pphi_q = np.broadcast_to(pphi_line[None, :], (nvperp, nvpdata))
            _deposit_sample(e_q, mu_q, pphi_q, f_q, vol)

    fmean = np.where(sum_weights > 0.0, sum_weighted / sum_weights, 0.0)
    out = {
        "fmean": fmean,
        "sum_weighted": sum_weighted,
        "sum_weights": sum_weights,
        "E_bins": e_bins,
        "mu_bins": mu_bins,
        "pphi_bins": pphi_bins,
        "E": E,
        "mu": mu,
        "Pphi": Pphi,
        "f_wo_perp": f_wo_perp,
    }
    fmean_at_ijk = interpolate_fmean_to_velocity_grid(
        fmean=fmean,
        E=E,
        mu=mu,
        Pphi=Pphi,
        E_bins=e_bins,
        mu_bins=mu_bins,
        pphi_bins=pphi_bins,
    )
    out["fmean_at_ijk"] = fmean_at_ijk
    return out
