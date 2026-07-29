"""Tests for the reference-based XGC-Analysis compatibility facade."""

from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import numpy as np

from xgc_reader import xgc1
from xgc_reader.analysis_compat import (
    AnalysisBackend,
    LegacyF0View,
    LegacyHeatDiagView,
    LegacyMeshView,
    LegacyOneDView,
    LegacyVolumeView,
)


class _ArrayHolder:
    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


class _SparseHolder:
    def __init__(self, matrix):
        self.matrix = matrix

    def get_csr_matrix(self):
        return self.matrix


class _Catalog:
    def __init__(self):
        self.close_count = 0
        self.products = {}

    def close(self):
        self.close_count += 1


class _Mesh:
    def __init__(self, plane):
        self._plane = plane
        self.wedge_n = 4
        self.wedge_angle = np.pi / 2
        self.delta_phi = np.pi / 8

    def get_plane(self, _index=0):
        return self._plane


class _OneDDiag:
    def __init__(self):
        self.psi = np.array([0.2, 0.8, 1.0])
        self.psi00 = np.array([0.1, 0.7, 0.9])
        self.psi_mks = np.array([1.0, 2.0, 3.0])
        self.time = np.array([0.0, 1.0])
        self.step = np.array([10, 20])
        self.gstep = self.step
        self.electron_on = True
        self.active_species_prefixes = ["e", "i", "i2"]
        self.data = {
            "i.gc_density_df_1d": np.arange(6.0).reshape(2, 3),
            "e.gc_density_df_1d": np.arange(6.0, 12.0).reshape(2, 3),
            "i2.gc_density_df_1d": np.arange(12.0, 18.0).reshape(2, 3),
        }
        self.derived = {
            "i.T": np.full((2, 3), 100.0),
            "e.T": np.full((2, 3), 50.0),
            "i.Lt": np.full((2, 3), 4.0),
            "e.Lt": np.full((2, 3), 5.0),
        }
        self.get_array_calls = {}
        self.get_derived_array_calls = {}

    def has_var(self, name):
        return name in self.data

    def has_derived_var(self, name):
        return name in self.derived

    def get_array(self, name):
        self.get_array_calls[name] = self.get_array_calls.get(name, 0) + 1
        return self.data[name]

    def get_derived_array(self, name):
        self.get_derived_array_calls[name] = (
            self.get_derived_array_calls.get(name, 0) + 1
        )
        return self.derived[name]

    def d_dpsi(self, var, psi):
        return var / psi

    def get_time_mask(self):
        self.tmask = np.array([0, 1])
        return self.tmask


class _HeatDiag:
    def __init__(self):
        self.wall_data = {
            "ds": np.array([[0.1, 0.1, 0.1]]),
            "psi": np.array([[0.8, 1.0, 1.2]]),
            "r": np.array([[1.8, 2.0, 2.2]]),
            "z": np.array([[-1.0, -1.1, -1.2]]),
            "strike_angle": np.zeros((1, 3)),
        }
        shape = (2, 1, 3)
        impurity_shape = (2, 1, 4)
        self.arrays = {
            "time": np.array([[[0.0]], [[1.0]]]),
            "gstep": np.array([[[10]], [[20]]]),
            "tindex": np.array([[[1]], [[2]]]),
            "e_number": np.full(shape, 2.0),
            "e_para_energy": np.full(shape, 3.0),
            "e_perp_energy": np.full(shape, 1.0),
            "e_potential": np.zeros(shape),
            "i_number": np.full(shape, 4.0),
            "i_para_energy": np.full(shape, 5.0),
            "i_perp_energy": np.full(shape, 2.0),
            "i_potential": np.zeros(shape),
            # Numbered species are left in their raw form, including the
            # leading garbage-bin segment, by current XGC-Analysis HeatDiag.
            "i2_number": np.full(impurity_shape, 6.0),
            "i2_para_energy": np.full(impurity_shape, 7.0),
            "i2_perp_energy": np.full(impurity_shape, 3.0),
            "i2_potential": np.zeros(impurity_shape),
        }
        self.data = {name: {} for name in self.arrays}
        self.get_array_calls = {}

    def get_array(self, name):
        self.get_array_calls[name] = self.get_array_calls.get(name, 0) + 1
        return self.arrays[name]


def _make_simulation():
    rz = np.array(
        [
            [1.0, -1.0],
            [2.0, -1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    plane = SimpleNamespace(
        rz=rz,
        nd_connect_list=np.array([[0, 1, 2], [1, 3, 2]]),
        surf_idx=np.array([[0, 1, 2], [1, 2, 3]]),
        surf_len=np.array([3, 3]),
        psi_surf=np.array([0.5, 1.0]),
        node_vol=np.arange(4.0),
        n_n=4,
        vol_1d=np.array([10.0, 20.0]),
        basis=np.array([1]),
        gradient_r_psi=_SparseHolder(np.eye(4)),
        gradient_z_theta=_SparseHolder(np.eye(4) * 2),
        ff_1dp_fwd=_SparseHolder(np.eye(4) * 3),
        ff_1dp_rev=_SparseHolder(np.eye(4) * 4),
        ff_hdp_fwd=_SparseHolder(np.eye(4) * 5),
        ff_hdp_rev=_SparseHolder(np.eye(4) * 6),
        dl_par_1dp_fwd=np.arange(4.0),
        dl_par_1dp_rev=np.arange(4.0) + 1,
        dl_par_hdp_fwd=np.arange(4.0) + 2,
        dl_par_hdp_rev=np.arange(4.0) + 3,
    )
    mesh = _Mesh(plane)

    psi = np.linspace(0.0, 1.0, 4)
    bfield = np.arange(12.0).reshape(4, 3)
    magnetic = SimpleNamespace(
        psi_pd=_ArrayHolder(psi),
        bfield=_ArrayHolder(bfield),
        jpar_bg_pd=_ArrayHolder(np.arange(4.0)),
        eq_x_psi=1.0,
        eq_x_r=2.2,
        eq_x_z=-0.4,
        eq_axis_r=1.7,
        eq_axis_z=0.0,
        eq_axis_b=2.1,
    )
    velocity_grid = SimpleNamespace(
        T_ev=np.arange(12.0).reshape(3, 4),
        den=np.arange(20.0, 32.0).reshape(3, 4),
        flow=np.arange(40.0, 52.0).reshape(3, 4),
        dsmu=0.2,
        dvp=0.3,
        smu_max=2.0,
        vp_max=3.0,
    )
    species = [
        SimpleNamespace(charge_eu=-1.0, mass_au=0.0005),
        SimpleNamespace(charge_eu=1.0, mass_au=2.0),
    ]
    catalog = _Catalog()
    simulation = SimpleNamespace(
        mesh=mesh,
        magnetic_field=magnetic,
        velocity_grid=velocity_grid,
        species=species,
        catalog=catalog,
        data_directory="/fake/run",
        input_params={
            "sml_param": {"sml_dt": 1.0e-7, "sml_totalpe": 128},
            "diag_param": {"diag_1d_period": 10},
        },
    )
    return simulation, plane, catalog


def _configure_units_product(reader, catalog):
    values = {
        "eq_x_psi": 1.0,
        "eq_x_r": 2.2,
        "eq_x_z": -0.4,
        "eq_axis_r": 1.7,
        "eq_axis_z": 0.0,
        "eq_axis_b": 2.1,
        "sml_dt": 1.0e-7,
        "diag_1d_period": 10,
        "sml_wedge_n": 4,
    }
    catalog.products["xgc.units.bp"] = SimpleNamespace(
        variables={name: object() for name in values}
    )
    reader._analysis_backend._api = {
        "read_static_variables": (
            lambda _catalog, _product_key, variables: {
                name: values[name] for name in variables
            }
        )
    }


class CompatibilityViewTests(unittest.TestCase):
    def test_mesh_view_shares_source_arrays_except_index_conversion(self):
        simulation, plane, _catalog = _make_simulation()
        view = LegacyMeshView(simulation)

        self.assertTrue(np.shares_memory(view.r, plane.rz))
        self.assertTrue(np.shares_memory(view.z, plane.rz))
        self.assertIs(view.node_vol, plane.node_vol)
        np.testing.assert_array_equal(view.surf_idx, plane.surf_idx + 1)
        self.assertFalse(np.shares_memory(view.surf_idx, plane.surf_idx))
        self.assertIs(view.triobj, view.triobj)

    def test_f0_and_volume_views_share_source_arrays(self):
        simulation, plane, _catalog = _make_simulation()
        f0 = LegacyF0View(simulation.velocity_grid)
        volume = LegacyVolumeView(plane)

        self.assertTrue(np.shares_memory(f0.te0, simulation.velocity_grid.T_ev))
        self.assertTrue(np.shares_memory(f0.ti0, simulation.velocity_grid.T_ev))
        self.assertTrue(np.shares_memory(f0.ne0, simulation.velocity_grid.den))
        self.assertTrue(np.shares_memory(f0.ni0, simulation.velocity_grid.den))
        self.assertIs(volume.od, plane.vol_1d)

    def test_oned_flattened_arrays_are_materialized_once(self):
        source = _OneDDiag()
        view = LegacyOneDView(source)

        first_density = view.i_gc_density_df_1d
        second_density = view.i_gc_density_df_1d
        first_temperature = view.Ti
        second_temperature = view.Ti
        impurity_density = view.i2gc_density_df_1d

        self.assertIs(first_density, second_density)
        self.assertIs(first_temperature, second_temperature)
        self.assertIs(impurity_density, source.data["i2.gc_density_df_1d"])
        self.assertEqual(source.get_array_calls["i.gc_density_df_1d"], 1)
        self.assertEqual(source.get_derived_array_calls["i.T"], 1)
        self.assertIs(view.Lti, source.derived["i.Lt"])
        self.assertIs(view.Lte, source.derived["e.Lt"])
        np.testing.assert_array_equal(view.tmask, [0, 1])
        self.assertEqual(view.psi_mks.shape, (2, 3))
        self.assertTrue(np.shares_memory(view.psi_mks, source.psi_mks))
        self.assertIs(view.psi_mks, view.psi_mks)
        np.testing.assert_array_equal(
            view.d_dpsi(np.full((2, 3), 2.0), view.psi_mks),
            np.full((2, 3), 2.0) / source.psi_mks,
        )
        self.assertTrue(view.electron_on)
        self.assertTrue(view.ion2_on)

    def test_heatdiag2_view_reuses_species_arrays_and_removes_garbage_bin(self):
        source = _HeatDiag()
        view = LegacyHeatDiagView(source)

        self.assertEqual(view.nsp, 3)
        self.assertTrue(np.shares_memory(view.r, source.wall_data["r"]))
        self.assertIs(view.sp[0].number, source.arrays["e_number"])
        self.assertTrue(
            np.shares_memory(
                view.sp[2].number,
                source.arrays["i2_number"],
            )
        )
        self.assertEqual(view.sp[2].number.shape, (2, 1, 3))
        self.assertIs(view.sp[2].number, view.sp[2].number)
        self.assertEqual(source.get_array_calls["i2_number"], 1)
        self.assertEqual(view.e_perp_energy.shape, (2, 3))
        self.assertTrue(
            np.shares_memory(
                view.e_perp_energy,
                source.arrays["e_perp_energy"],
            )
        )
        self.assertIs(view.e_perp_energy, view.e_perp_energy)
        self.assertEqual(view.i_para_energy.shape, (2, 3))
        np.testing.assert_array_equal(
            view.sp[2].number,
            source.arrays["i2_number"][..., 1:],
        )


class AnalysisBackendFacadeTests(unittest.TestCase):
    def test_analysis_is_the_default_backend(self):
        simulation, _plane, _catalog = _make_simulation()
        reader = xgc1(
            ".",
            change_cwd=False,
            simulation=simulation,
        )
        self.addCleanup(reader.close)

        self.assertEqual(reader.backend, "analysis")
        self.assertIs(reader.simulation, simulation)

    def test_units_reader_fetches_entire_product_without_building_simulation(self):
        simulation, _plane, catalog = _make_simulation()
        catalog.products["xgc.units.bp"] = SimpleNamespace(
            variables={
                "eq_axis_b": object(),
                "eq_axis_r": object(),
                "eq_den_v1": object(),
                "eq_tempi_v1": object(),
                "sml_dt": object(),
                "sml_tran": object(),
            }
        )
        captured = {}

        def read_static_variables(
            selected_catalog,
            product_key,
            variables,
        ):
            captured["catalog"] = selected_catalog
            captured["product_key"] = product_key
            captured["variables"] = list(variables)
            return {
                "eq_axis_b": 2.1,
                "eq_axis_r": 1.7,
                "eq_den_v1": 1.5e19,
                "eq_tempi_v1": 800.0,
                "sml_dt": 1.0e-7,
                "sml_tran": 7,
            }

        backend = AnalysisBackend(
            ".",
            simulation=simulation,
            api={"read_static_variables": read_static_variables},
        )

        units = backend.load_units()

        self.assertEqual(
            captured["variables"],
            [
                "eq_axis_b",
                "eq_axis_r",
                "eq_den_v1",
                "eq_tempi_v1",
                "sml_dt",
                "sml_tran",
            ],
        )
        self.assertIs(captured["catalog"], catalog)
        self.assertEqual(captured["product_key"], "xgc.units.bp")
        self.assertEqual(units["eq_den_v1"], 1.5e19)
        self.assertEqual(units["eq_tempi_v1"], 800.0)
        self.assertEqual(units["sml_tran"], 7)
        self.assertEqual(units["eq_axis_r"], 1.7)
        self.assertEqual(units["sml_dt"], 1.0e-7)
        self.assertEqual(units["sml_wedge_n"], 1)

    def test_facade_load_unitsm_falls_back_for_old_units_file(self):
        catalog = _Catalog()
        reader = xgc1(
            "/fake/old-run",
            backend="analysis",
            change_cwd=False,
            catalog=catalog,
        )
        self.addCleanup(reader.close)

        with patch.object(
            reader,
            "load_unitsm_old",
        ) as old_units_reader:
            reader.load_unitsm()

        old_units_reader.assert_called_once_with()

    def test_ascii_units_parser_handles_fortran_values_and_comments(self):
        reader = xgc1.__new__(xgc1)
        contents = """
            sml_dt = 4.3830033179758812D-007;
            sml_wedge_n = 2; ! comment
            % ignored comment
            diag_1d_period = 5;
        """

        with patch("builtins.open", mock_open(read_data=contents)):
            values = reader.load_m("/fake/units.m")

        self.assertEqual(values["sml_wedge_n"], 2.0)
        self.assertEqual(values["diag_1d_period"], 5.0)
        self.assertAlmostEqual(values["sml_dt"], 4.3830033179758812e-7)

    def test_static_prefetch_filters_variables_missing_from_older_outputs(self):
        catalog = SimpleNamespace(
            products={
                "xgc.mesh.bp": SimpleNamespace(
                    variables={"rz": object(), "node_vol": object()}
                )
            }
        )
        simulation_type = SimpleNamespace(
            STATIC_BUFFER_REQUESTS={
                "xgc.mesh.bp": ("rz", "node_vol", "node_vol_ff0"),
                "xgc.f0.mesh.bp": ("f0_den",),
            },
            OPTIONAL_STATIC_BUFFER_PRODUCTS={"xgc.f0.mesh.bp"},
        )
        captured = {}

        def build_static_buffer(_catalog, requests, *, optional_products):
            captured["requests"] = requests
            captured["optional_products"] = optional_products
            return {"xgc.mesh.bp": {"rz": object(), "node_vol": object()}}

        buffer = AnalysisBackend._build_compatible_static_buffer(
            simulation_type,
            catalog,
            build_static_buffer,
        )

        self.assertEqual(
            captured["requests"],
            {"xgc.mesh.bp": ("rz", "node_vol")},
        )
        self.assertEqual(
            captured["optional_products"],
            {"xgc.f0.mesh.bp"},
        )
        self.assertIn("xgc.mesh.bp", buffer)

    def test_facade_exposes_analysis_objects_without_array_duplication(self):
        simulation, plane, catalog = _make_simulation()
        reader = xgc1(
            ".",
            backend="analysis",
            change_cwd=False,
            simulation=simulation,
        )
        self.addCleanup(reader.close)
        _configure_units_product(reader, catalog)

        reader.load_units()
        reader.setup_mesh()
        reader.setup_f0mesh()
        reader.load_volumes()
        reader.load_bfield()

        self.assertIs(reader.simulation, simulation)
        self.assertIs(reader.catalog, catalog)
        self.assertEqual(reader.psix, 1.0)
        self.assertEqual(reader.sml_dt, 1.0e-7)
        self.assertTrue(np.shares_memory(reader.mesh.r, plane.rz))
        self.assertTrue(
            np.shares_memory(
                reader.f0.te0,
                simulation.velocity_grid.T_ev,
            )
        )
        self.assertIs(reader.vol.od, plane.vol_1d)
        self.assertTrue(
            np.shares_memory(
                reader.bfield,
                simulation.magnetic_field.bfield.data,
            )
        )
        np.testing.assert_array_equal(reader.mesh.msep, plane.surf_idx[1])

    def test_facade_loads_oned_through_cached_legacy_view(self):
        simulation, _plane, catalog = _make_simulation()
        source = _OneDDiag()
        reader = xgc1(
            ".",
            backend="analysis",
            change_cwd=False,
            simulation=simulation,
        )
        self.addCleanup(reader.close)
        _configure_units_product(reader, catalog)
        reader._analysis_backend._oned_factory = lambda **_kwargs: source

        reader.load_units()
        reader.load_oned()

        self.assertIs(reader.od.source, source)
        self.assertTrue(reader.electron_on)
        self.assertTrue(reader.ion2_on)
        self.assertIs(reader.od.i_gc_density_df_1d, source.data["i.gc_density_df_1d"])
        self.assertIs(reader.od.Ti, source.derived["i.T"])
        np.testing.assert_allclose(reader.od.psi00n, source.psi00)
        self.assertEqual(reader.od.beta_e.shape, (2, 3))

    def test_facade_loads_oned_without_requiring_full_simulation(self):
        catalog = _Catalog()
        source = _OneDDiag()
        captured = {}

        def oned_factory(**kwargs):
            captured.update(kwargs)
            return source

        reader = xgc1(
            "/fake/oned-only",
            backend="analysis",
            change_cwd=False,
            catalog=catalog,
        )
        self.addCleanup(reader.close)
        reader._analysis_backend._oned_factory = oned_factory

        reader.load_oned()

        self.assertIs(reader.od.source, source)
        self.assertIs(captured["catalog"], catalog)
        self.assertNotIn("simulation", captured)
        self.assertIsNone(reader.simulation)

    def test_facade_uses_legacy_static_readers_for_incomplete_old_run(self):
        catalog = _Catalog()
        simulation_type = SimpleNamespace(
            REQUIRED_CATALOG_PRODUCTS=(
                "xgc.mesh.bp",
                "xgc.equil.bp",
                "xgc.bfield.bp",
            ),
            REQUIRED_CATALOG_TEXTS=("input",),
        )
        reader = xgc1(
            "/fake/old-run",
            backend="analysis",
            change_cwd=False,
            catalog=catalog,
        )
        self.addCleanup(reader.close)
        reader._analysis_backend._api = {"Simulation": simulation_type}
        mesh = object()
        f0 = object()
        volume = object()

        with (
            patch("xgc_reader.base.meshdata", return_value=mesh) as mesh_reader,
            patch("xgc_reader.base.f0meshdata", return_value=f0) as f0_reader,
            patch("xgc_reader.base.voldata", return_value=volume) as volume_reader,
        ):
            reader.setup_mesh()
            reader.setup_f0mesh()
            reader.load_volumes()

        source = "/fake/old-run/"
        mesh_reader.assert_called_once_with(source)
        f0_reader.assert_called_once_with(source)
        volume_reader.assert_called_once_with(source)
        self.assertIs(reader.mesh, mesh)
        self.assertIs(reader.f0, f0)
        self.assertIs(reader.vol, volume)
        self.assertIsNone(reader.simulation)

    def test_facade_reuses_analysis_gradient_and_mapping_matrices(self):
        simulation, plane, _catalog = _make_simulation()
        reader = xgc1(
            ".",
            backend="analysis",
            change_cwd=False,
            simulation=simulation,
        )
        self.addCleanup(reader.close)

        reader.load_grad_rz()
        reader.load_ff_mapping()

        self.assertIs(
            reader.grad.mat_psi_r,
            plane.gradient_r_psi.get_csr_matrix(),
        )
        self.assertIs(
            reader.grad.mat_theta_z,
            plane.gradient_z_theta.get_csr_matrix(),
        )
        self.assertIs(
            reader.ff_1dp_fwd.mat,
            plane.ff_1dp_fwd.get_csr_matrix(),
        )
        self.assertIs(reader.ff_1dp_fwd.dl, plane.dl_par_1dp_fwd)
        np.testing.assert_array_equal(
            reader.grad.apply_gradient(np.ones(4), component="r"),
            np.ones(4),
        )
        np.testing.assert_array_equal(
            reader.ff_hdp_rev.apply_mapping(np.ones(4)),
            np.full(4, 6.0),
        )

    def test_facade_postprocesses_heatdiag2_without_rereading_arrays(self):
        simulation, _plane, catalog = _make_simulation()
        source = _HeatDiag()
        variable_names = set(source.wall_data) | set(source.arrays)
        catalog.products["xgc.heatdiag2.bp"] = SimpleNamespace(
            variables={name: object() for name in variable_names}
        )
        captured = {}

        def heatdiag_factory(**kwargs):
            captured.update(kwargs)
            return source

        heatdiag_factory.DEFAULT_VARS = tuple(source.wall_data) + (
            "time",
            "gstep",
            "tindex",
        )

        reader = xgc1(
            ".",
            backend="analysis",
            change_cwd=False,
            simulation=simulation,
        )
        self.addCleanup(reader.close)
        _configure_units_product(reader, catalog)
        reader._analysis_backend._heatdiag_factory = heatdiag_factory

        reader.load_units()
        reader.load_heatdiag2()

        self.assertIs(reader.hl2.source, source)
        self.assertEqual(reader.hl2.nsp, 3)
        self.assertIn("i2_number", captured["variables"])
        self.assertTrue(hasattr(reader.hl2, "q_total"))
        self.assertTrue(hasattr(reader.hl2, "g_total"))
        self.assertTrue(np.shares_memory(reader.hl2.sp[0].number, source.arrays["e_number"]))
        self.assertEqual(reader.hl2.sp[0].q.shape, (2, 3))

    def test_facade_rejects_empty_heatdiag2_product(self):
        simulation, _plane, catalog = _make_simulation()
        catalog.products["xgc.heatdiag2.bp"] = SimpleNamespace(variables={})
        reader = xgc1(
            ".",
            backend="analysis",
            change_cwd=False,
            simulation=simulation,
        )
        self.addCleanup(reader.close)

        with self.assertRaisesRegex(
            RuntimeError,
            "contains no readable variables",
        ):
            reader.load_heatdiag2()

    def test_facade_legacy_heatdiag_uses_existing_reader(self):
        simulation, _plane, _catalog = _make_simulation()
        reader = xgc1(
            ".",
            backend="analysis",
            change_cwd=False,
            simulation=simulation,
        )
        self.addCleanup(reader.close)

        with patch("xgc_reader.base.load_heatdiag") as legacy_reader:
            reader.load_heatdiag(read_rz_all=True)

        legacy_reader.assert_called_once_with(reader, read_rz_all=True)

    def test_facade_legacy_heatdiag_rejects_campaign_catalog(self):
        catalog = _Catalog()
        reader = xgc1(
            "/fake/archive.aca",
            backend="analysis",
            change_cwd=False,
            catalog=catalog,
        )
        self.addCleanup(reader.close)

        with self.assertRaisesRegex(
            NotImplementedError,
            "only supports directory datasets",
        ):
            reader.load_heatdiag()

    def test_facade_bfieldm_reuses_analysis_campaign_reader(self):
        campaign_reader = object()
        catalog = _Catalog()
        catalog.campaign_reader = campaign_reader
        reader = xgc1(
            "/fake/archive.aca",
            backend="analysis",
            change_cwd=False,
            catalog=catalog,
        )
        self.addCleanup(reader.close)
        reader.unit_dic = {"eq_axis_r": 1.5}
        bfieldm = SimpleNamespace(
            rmid=np.array([1.0, 1.6, 1.8]),
            psin=np.array([0.2, 0.9, 1.1]),
        )

        with patch("xgc_reader.base.databfm", return_value=bfieldm) as reader_type:
            reader.load_bfieldm()

        reader_type.assert_called_once_with(campaign_reader)
        self.assertIs(reader.bfm, bfieldm)
        np.testing.assert_array_equal(reader.bfm.rmido, [1.6, 1.8])
        np.testing.assert_array_equal(reader.bfm.psino, [0.9, 1.1])

    def test_backend_closes_catalog(self):
        simulation, _plane, catalog = _make_simulation()
        backend = AnalysisBackend(".", simulation=simulation)

        backend.close()

        self.assertEqual(catalog.close_count, 1)


if __name__ == "__main__":
    unittest.main()
