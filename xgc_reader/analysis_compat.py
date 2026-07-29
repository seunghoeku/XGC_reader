"""Compatibility layer backed by :mod:`xgc_analysis`.

The classes in this module deliberately do not import ``xgc_analysis`` at
module-import time.  Existing users can therefore continue to import
``xgc_reader`` in environments where the new backend has not been installed,
while ``backend="analysis"`` provides a clear installation error when used.

Large arrays remain owned by XGC-Analysis.  Compatibility objects expose
references or NumPy views wherever the legacy API permits it, and cache the
few stacked or index-converted arrays that must be materialized.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from matplotlib.tri import Triangulation

from .heat_diagnostics import datahl2
from .matrix_ops import ff_mapping, grad_rz


class AnalysisBackendUnavailable(ImportError):
    """Raised when the optional XGC-Analysis backend is not installed."""


def _import_analysis_api():
    """Import the XGC-Analysis entry points needed by the compatibility layer."""
    try:
        from xgc_analysis.catalog import (
            build_static_buffer,
            open_campaign_catalog,
            open_catalog,
            read_static_variables,
        )
        from xgc_analysis.heatdiag import HeatDiag
        from xgc_analysis.oneddiag import OneDDiag
        from xgc_analysis.simulation import Simulation
    except ModuleNotFoundError as exc:
        if exc.name == "xgc_analysis" or (
            exc.name is not None and exc.name.startswith("xgc_analysis.")
        ):
            raise AnalysisBackendUnavailable(
                "backend='analysis' requires XGC-Analysis. For a sibling "
                "checkout, install it with "
                "`python -m pip install -e ~/Documents/git/XGC-Analysis`."
            ) from exc
        raise

    return {
        "Simulation": Simulation,
        "OneDDiag": OneDDiag,
        "HeatDiag": HeatDiag,
        "build_static_buffer": build_static_buffer,
        "open_catalog": open_catalog,
        "open_campaign_catalog": open_campaign_catalog,
        "read_static_variables": read_static_variables,
    }


def _as_data_array(value):
    """Return an object's underlying array without copying when possible."""
    get_data = getattr(value, "get_data", None)
    if get_data is not None:
        return get_data()
    if hasattr(value, "data"):
        return value.data
    return value


def _as_csr_matrix(value):
    """Return an XGC-Analysis sparse matrix's existing SciPy CSR object."""
    if value is None:
        return None
    get_csr_matrix = getattr(value, "get_csr_matrix", None)
    if get_csr_matrix is not None:
        return get_csr_matrix()
    return getattr(value, "csr", value)


class LegacyMeshView:
    """Expose the historical ``x.mesh`` API over an XGC-Analysis plane."""

    def __init__(self, simulation):
        self._simulation = simulation
        self._mesh = simulation.mesh
        self._plane = self._mesh.get_plane(0)
        self._magnetic_field = simulation.magnetic_field

        # XGC-Analysis normalizes surface indices to zero-based indexing.
        # Historical xgc_reader notebooks subtract one themselves, so this is
        # the one geometry array that must be materialized for compatibility.
        self._legacy_surf_idx = np.asarray(self._plane.surf_idx) + 1
        self._triobj = Triangulation(
            self._plane.rz[:, 0],
            self._plane.rz[:, 1],
            self._plane.nd_connect_list,
        )

    @property
    def r(self):
        return self._plane.rz[:, 0]

    @property
    def z(self):
        return self._plane.rz[:, 1]

    @property
    def rz(self):
        return self._plane.rz

    @property
    def cnct(self):
        return self._plane.nd_connect_list

    @property
    def triobj(self):
        return self._triobj

    @property
    def psi(self):
        return _as_data_array(self._magnetic_field.psi_pd)

    @property
    def surf_idx(self):
        return self._legacy_surf_idx

    @property
    def nnodes(self):
        return self._plane.n_n

    @property
    def wedge_angle(self):
        return self._mesh.wedge_angle

    @property
    def delta_phi(self):
        return self._mesh.delta_phi

    def __getattr__(self, name):
        # Most legacy names (node_vol, psi_surf, surf_len, theta, ...)
        # already match Plane attributes and can be returned by reference.
        return getattr(self._plane, name)


class LegacyF0View:
    """Expose historical f0-mesh names over ``Simulation.velocity_grid``."""

    def __init__(self, velocity_grid):
        if velocity_grid is None:
            raise RuntimeError(
                "XGC-Analysis could not initialize VelocityGrid; "
                "xgc.f0.mesh.bp may be missing or incomplete."
            )
        self._grid = velocity_grid

    @property
    def den0(self):
        return self._grid.den

    @property
    def te0(self):
        values = self._grid.T_ev
        return values[0, :] if values.ndim > 1 else values

    @property
    def ti0(self):
        values = self._grid.T_ev
        return values[-1, :] if values.ndim > 1 else values

    @property
    def ne0(self):
        values = self._grid.den
        return values[0, :] if values.ndim > 1 else values

    @property
    def ni0(self):
        values = self._grid.den
        return values[-1, :] if values.ndim > 1 else values

    @property
    def ue0(self):
        values = self._grid.flow
        return values[0, :] if values.ndim > 1 else values

    @property
    def ui0(self):
        values = self._grid.flow
        return values[-1, :] if values.ndim > 1 else values

    def __getattr__(self, name):
        return getattr(self._grid, name)


class LegacyVolumeView:
    """Expose ``x.vol.od`` without duplicating the 1-D volume array."""

    def __init__(self, plane):
        self._plane = plane

    @property
    def od(self):
        return self._plane.vol_1d


class LegacyOneDView:
    """Flatten XGC-Analysis OneDDiag names into the historical namespace."""

    _DERIVED_ALIASES = {
        "Te": "e.T",
        "Ti": "i.T",
        "Ti2": "i2.T",
        "Ti3": "i3.T",
        "Ti4": "i4.T",
        "Ti5": "i5.T",
        "Ti6": "i6.T",
        "Ti7": "i7.T",
        "Ti8": "i8.T",
        "Ti9": "i9.T",
        "Lte": "e.Lt",
        "Lti": "i.Lt",
        "Lti2": "i2.Lt",
        "Lti3": "i3.Lt",
        "Lti4": "i4.Lt",
        "Lti5": "i5.Lt",
        "Lti6": "i6.Lt",
        "Lti7": "i7.Lt",
        "Lti8": "i8.Lt",
        "Lti9": "i9.Lt",
    }
    _NUMBERED_SPECIES_PREFIXES = (
        "i2",
        "i3",
        "i4",
        "i5",
        "i6",
        "i7",
        "i8",
        "i9",
    )

    def __init__(self, oneddiag):
        self._oneddiag = oneddiag
        self._array_cache: dict[str, np.ndarray] = {}

    @property
    def source(self):
        """Return the underlying XGC-Analysis ``OneDDiag`` object."""
        return self._oneddiag

    @property
    def electron_on(self):
        return self._oneddiag.electron_on

    @property
    def ion2_on(self):
        return "i2" in self._oneddiag.active_species_prefixes

    def _standard_name(self, legacy_name: str):
        # Historical XGC output commonly uses ``i2gc_density...`` without an
        # underscore, while XGC-Analysis exposes ``i2.gc_density...``.
        for prefix in self._NUMBERED_SPECIES_PREFIXES:
            if legacy_name.startswith(prefix):
                remainder = legacy_name[len(prefix):].lstrip("_")
                return f"{prefix}.{remainder}"

        for prefix in ("e", "i"):
            marker = prefix + "_"
            if legacy_name.startswith(marker):
                return f"{prefix}.{legacy_name[len(marker):]}"
        return legacy_name

    def _cached_array(self, standard_name: str, *, derived: bool = False):
        cache_key = ("derived:" if derived else "data:") + standard_name
        if cache_key not in self._array_cache:
            if derived:
                value = self._oneddiag.get_derived_array(standard_name)
            else:
                value = self._oneddiag.get_array(standard_name)
            self._array_cache[cache_key] = value
        return self._array_cache[cache_key]

    def _legacy_psi_mks(self):
        """Expose static psi coordinates with the historical time dimension."""
        cache_key = "compat:psi_mks"
        if cache_key not in self._array_cache:
            psi_mks = np.asarray(self._oneddiag.psi_mks)
            if psi_mks.ndim == 1:
                nstep = np.asarray(self._oneddiag.time).size
                psi_mks = np.broadcast_to(
                    psi_mks,
                    (nstep, psi_mks.size),
                )
            self._array_cache[cache_key] = psi_mks
        return self._array_cache[cache_key]

    def d_dpsi(self, var, psi):
        """Accept legacy 2-D or XGC-Analysis 1-D psi coordinates."""
        coordinate = np.asarray(psi)
        if coordinate.ndim > 1:
            coordinate = coordinate[0, :]
        return self._oneddiag.d_dpsi(var, coordinate)

    def __getattr__(self, name):
        if name in self._DERIVED_ALIASES:
            return self._cached_array(self._DERIVED_ALIASES[name], derived=True)

        if name == "psi_mks":
            return self._legacy_psi_mks()

        if name == "tmask":
            try:
                return self._oneddiag.tmask
            except AttributeError:
                return self._oneddiag.get_time_mask()

        if name in {"psi", "psi00", "time", "step", "gstep"}:
            return getattr(self._oneddiag, name)

        standard_name = self._standard_name(name)
        if self._oneddiag.has_var(standard_name):
            return self._cached_array(standard_name)
        if self._oneddiag.has_derived_var(standard_name):
            return self._cached_array(standard_name, derived=True)

        return getattr(self._oneddiag, name)


class LegacyHeatDiagSpeciesView:
    """Legacy heatdiag2 species arrays backed by one ``HeatDiag`` reader."""

    def __init__(self, owner, prefix):
        self._owner = owner
        self.prefix = prefix

    @property
    def number(self):
        return self._owner._species_array(self.prefix, "number")

    @property
    def para_energy(self):
        return self._owner._species_array(self.prefix, "para_energy")

    @property
    def perp_energy(self):
        return self._owner._species_array(self.prefix, "perp_energy")

    @property
    def potential(self):
        return self._owner._species_array(self.prefix, "potential")


class LegacyHeatDiagView(datahl2):
    """Legacy ``datahl2`` API over arrays owned by XGC-Analysis ``HeatDiag``."""

    _SPECIES_PREFIXES = ("e", "i", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9")
    _TOP_LEVEL_SPECIES_ALIASES = {
        "e_number": (0, "number"),
        "e_para_energy": (0, "para_energy"),
        "e_perp_energy": (0, "perp_energy"),
        "i_number": (1, "number"),
        "i_para_energy": (1, "para_energy"),
        "i_perp_energy": (1, "perp_energy"),
    }

    def __init__(self, heatdiag):
        self._heatdiag = heatdiag
        self._array_cache: dict[str, np.ndarray] = {}

        self.time = self._scalar_series("time")
        if self._has_data("step"):
            self.step = self._scalar_series("step")
        elif self._has_data("gstep"):
            self.step = self._scalar_series("gstep")
        else:
            self.step = np.arange(self.time.size)
        if self._has_data("tindex"):
            self.tindex = self._scalar_series("tindex")

        for name in ("ds", "psi", "r", "z", "strike_angle"):
            setattr(self, name, self._wall_array(name))

        self.sp = []
        for prefix in self._SPECIES_PREFIXES:
            if not self._has_data(f"{prefix}_number"):
                break
            self.sp.append(LegacyHeatDiagSpeciesView(self, prefix))
        self.nsp = len(self.sp)

        self.dt = np.zeros_like(self.time, dtype=float)
        if self.time.size > 1:
            self.dt[1:] = self.time[1:] - self.time[:-1]
            self.dt[0] = self.dt[1]
        self.dt = self.dt[:, np.newaxis]

    @property
    def source(self):
        return self._heatdiag

    def _has_data(self, name):
        return name in self._heatdiag.data

    def _time_array(self, name):
        if name not in self._array_cache:
            self._array_cache[name] = self._heatdiag.get_array(name)
        return self._array_cache[name]

    def _scalar_series(self, name):
        array = np.asarray(self._time_array(name))
        if array.size == 0:
            return array.reshape(0)
        return array.reshape(array.shape[0], -1)[:, 0]

    def _wall_array(self, name):
        array = self._heatdiag.wall_data[name]
        if array.ndim == 2 and array.shape[0] == 1:
            return array[0]
        return array

    def _species_array(self, prefix, suffix):
        name = f"{prefix}_{suffix}"
        array = self._time_array(name)
        nseg = np.asarray(self.ds).shape[-1]
        if array.shape[-1] == nseg + 1:
            legacy_name = f"legacy:{name}"
            if legacy_name not in self._array_cache:
                self._array_cache[legacy_name] = array[..., 1:]
            return self._array_cache[legacy_name]
        return array

    def get_parallel_flux(self):
        """Compute legacy species flux arrays with robust toroidal broadcasting."""
        for species in self.sp:
            energy = np.squeeze(species.para_energy + species.perp_energy)
            number = np.squeeze(species.number)
            dt = self.dt
            area = self.area
            if energy.ndim == 3:
                dt = dt[:, :, np.newaxis]
            elif area.ndim == 3 and area.shape[1] == 1:
                area = area[:, 0, :]
            species.q = energy / dt / area
            species.g = number / dt / area

    def __getattr__(self, name):
        alias = self._TOP_LEVEL_SPECIES_ALIASES.get(name)
        if alias is not None:
            species_index, field_name = alias
            if species_index >= self.nsp:
                raise AttributeError(name)
            cache_key = f"compat:{name}"
            if cache_key not in self._array_cache:
                self._array_cache[cache_key] = np.squeeze(
                    getattr(self.sp[species_index], field_name)
                )
            return self._array_cache[cache_key]
        if name in self._heatdiag.wall_data:
            return self._wall_array(name)
        if name in self._heatdiag.data:
            return self._time_array(name)
        return getattr(self._heatdiag, name)


class AnalysisBackend:
    """Lazy owner of the XGC-Analysis catalog and Simulation."""

    def __init__(
        self,
        location,
        *,
        catalog=None,
        simulation=None,
        api: dict[str, Any] | None = None,
        oned_factory: Callable[..., Any] | None = None,
        heatdiag_factory: Callable[..., Any] | None = None,
    ):
        self.location = os.fspath(location)
        self.catalog = catalog if catalog is not None else getattr(simulation, "catalog", None)
        self.simulation = simulation
        self._api = api
        self._oned_factory = oned_factory
        self._heatdiag_factory = heatdiag_factory

    @property
    def is_campaign(self):
        return self.location.endswith(".aca")

    def _get_api(self):
        if self._api is None:
            self._api = _import_analysis_api()
        return self._api

    def ensure_catalog(self):
        if self.catalog is None:
            api = self._get_api()
            if self.is_campaign:
                self.catalog = api["open_campaign_catalog"](self.location)
            else:
                self.catalog = api["open_catalog"](self.location)
        return self.catalog

    def ensure_simulation(self):
        if self.simulation is None:
            api = self._get_api()
            catalog = self.ensure_catalog()
            simulation_type = api["Simulation"]
            static_buffer = self._build_compatible_static_buffer(
                simulation_type,
                catalog,
                api["build_static_buffer"],
            )
            if self.is_campaign:
                self.simulation = simulation_type(
                    catalog=catalog,
                    static_buffer=static_buffer,
                )
            else:
                self.simulation = simulation_type(
                    directories=[self.location],
                    catalog=catalog,
                    static_buffer=static_buffer,
                )
        return self.simulation

    def supports_simulation(self):
        """Return whether the catalog has XGC-Analysis Simulation inputs."""
        if self.simulation is not None:
            return True

        catalog = self.ensure_catalog()
        simulation_type = self._get_api()["Simulation"]
        products = getattr(catalog, "products", {})
        if any(
            name not in products
            for name in simulation_type.REQUIRED_CATALOG_PRODUCTS
        ):
            return False

        has_text = getattr(catalog, "has_text", None)
        return has_text is not None and all(
            has_text(name)
            for name in simulation_type.REQUIRED_CATALOG_TEXTS
        )

    def legacy_static_source(self):
        """Return a path or existing campaign handle for legacy static readers."""
        if not self.is_campaign:
            return self.location.rstrip(os.sep) + os.sep

        catalog = self.ensure_catalog()
        campaign_reader = getattr(catalog, "campaign_reader", None)
        if campaign_reader is None:
            raise RuntimeError(
                "XGC-Analysis campaign catalog has no open reader for "
                "legacy static-product access."
            )
        return campaign_reader

    @staticmethod
    def _build_compatible_static_buffer(
        simulation_type,
        catalog,
        build_static_buffer,
    ):
        """Prefetch only variables advertised by older or newer XGC outputs."""
        products = getattr(catalog, "products", {})
        requests = {}
        for product_key, requested_variables in (
            simulation_type.STATIC_BUFFER_REQUESTS.items()
        ):
            product = products.get(product_key)
            if product is None:
                continue
            available = getattr(product, "variables", {})
            selected = tuple(
                name for name in requested_variables if name in available
            )
            if selected:
                requests[product_key] = selected

        return build_static_buffer(
            catalog,
            requests,
            optional_products=simulation_type.OPTIONAL_STATIC_BUFFER_PRODUCTS,
        )

    def load_units(self):
        """Read the small units product in full through XGC-Analysis."""
        catalog = self.ensure_catalog()
        product_key = "xgc.units.bp"
        product = getattr(catalog, "products", {}).get(product_key)
        if product is None:
            raise KeyError(f"Catalog does not contain {product_key!r}")

        variable_names = sorted(getattr(product, "variables", {}))
        units = dict(self._get_api()["read_static_variables"](
            catalog,
            product_key,
            variable_names,
        ))
        # Historical XGCa units files may omit wedge information.
        units.setdefault("sml_wedge_n", 1)
        return units

    def mesh_view(self):
        return LegacyMeshView(self.ensure_simulation())

    def f0_view(self):
        return LegacyF0View(self.ensure_simulation().velocity_grid)

    def volume_view(self):
        plane = self.ensure_simulation().mesh.get_plane(0)
        return LegacyVolumeView(plane)

    def bfield_array(self):
        value = self.ensure_simulation().magnetic_field.bfield
        data = _as_data_array(value)
        # XGC-Analysis stores (node, component); historical xgc_reader stores
        # (component, node). Transpose is a NumPy view.
        return data.T

    def gradient_view(self):
        """Expose Plane gradient matrices through the existing legacy class."""
        plane = self.ensure_simulation().mesh.get_plane(0)
        view = grad_rz.__new__(grad_rz)
        view.source = plane
        view.mat_basis = plane.basis
        view.mat_psi_r = _as_csr_matrix(plane.gradient_r_psi)
        view.mat_theta_z = _as_csr_matrix(plane.gradient_z_theta)
        return view

    def field_following_views(self):
        """Expose Plane field-following matrices without rereading BP files."""
        plane = self.ensure_simulation().mesh.get_plane(0)
        views = {}
        for name in ("1dp_fwd", "1dp_rev", "hdp_fwd", "hdp_rev"):
            view = ff_mapping.__new__(ff_mapping)
            view.source = plane
            view.name = name
            view.mat = _as_csr_matrix(getattr(plane, f"ff_{name}", None))
            view.dl = getattr(plane, f"dl_par_{name}", None)
            views[name] = view
        return views

    def load_oned(self, *, mass_overrides=None):
        catalog = self.ensure_catalog()
        factory = self._oned_factory
        if factory is None:
            factory = self._get_api()["OneDDiag"]
        kwargs = {
            "path": getattr(self.simulation, "data_directory", self.location),
            "catalog": catalog,
        }
        if self.simulation is not None:
            kwargs["simulation"] = self.simulation
        oneddiag = factory(**kwargs)
        if mass_overrides:
            mass_by_prefix = getattr(oneddiag, "mass_by_prefix", None)
            post_process = getattr(oneddiag, "post_process", None)
            if mass_by_prefix is None or post_process is None:
                raise TypeError(
                    "OneDDiag implementation does not support species-mass "
                    "overrides."
                )
            mass_by_prefix.update(mass_overrides)
            post_process()
        return LegacyOneDView(oneddiag)

    def load_heatdiag2(self):
        catalog = self.ensure_catalog()
        product = getattr(catalog, "products", {}).get("xgc.heatdiag2.bp")
        if product is not None and not getattr(product, "variables", {}):
            raise RuntimeError(
                "Catalog product 'xgc.heatdiag2.bp' contains no readable "
                "variables."
            )
        factory = self._heatdiag_factory
        if factory is None:
            factory = self._get_api()["HeatDiag"]
        variables = self._heatdiag_variables(catalog, factory)
        kwargs = {"catalog": catalog}
        if self.simulation is not None:
            kwargs["simulation"] = self.simulation
        else:
            kwargs["data_dir"] = self.location
        if variables:
            kwargs["variables"] = variables
        heatdiag = factory(**kwargs)
        return LegacyHeatDiagView(heatdiag)

    @staticmethod
    def _heatdiag_variables(catalog, heatdiag_type):
        """Request all available legacy species channels through HeatDiag."""
        product = getattr(catalog, "products", {}).get("xgc.heatdiag2.bp")
        if product is None:
            return None
        available = set(getattr(product, "variables", {}))
        if not available:
            raise RuntimeError(
                "Catalog product 'xgc.heatdiag2.bp' contains no readable "
                "variables."
            )
        requested = set(getattr(heatdiag_type, "DEFAULT_VARS", ()))
        suffixes = ("_number", "_para_energy", "_perp_energy", "_potential")
        requested.update(
            name for name in available if any(name.endswith(suffix) for suffix in suffixes)
        )
        return sorted(requested & available)

    def close(self):
        catalog = self.catalog
        if catalog is not None:
            close = getattr(catalog, "close", None)
            if close is not None:
                close()


def create_analysis_backend(location, **kwargs):
    """Factory kept separate so tests and downstream integrations can inject it."""
    return AnalysisBackend(Path(location), **kwargs)
