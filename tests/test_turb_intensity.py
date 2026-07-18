from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from xgc_reader import analysis


class _FakeFileReader:
    def __init__(self, _filename, variables):
        self._variables = variables

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, name):
        return self._variables[name]


def _xgc_instance():
    mesh = SimpleNamespace(
        z=np.array([-1.0, 1.0, -1.0, 1.0]),
        psi_surf=np.array([0.25, 0.75]),
        surf_len=np.array([2, 2]),
        surf_idx=np.array([[1, 2], [3, 4]]),
        node_vol=np.array([1.0, 3.0, 2.0, 2.0]),
        delta_phi=0.25,
    )
    return SimpleNamespace(mesh=mesh, psix=1.0, eq_axis_z=0.0)


def _reader_patch(var):
    variables = {'e_den': var, 'time': np.array([1.5])}
    return mock.patch.object(
        analysis.adios2,
        'FileReader',
        side_effect=lambda filename: _FakeFileReader(filename, variables),
    )


class TurbIntensityTest(unittest.TestCase):
    def setUp(self):
        self.var = np.array([
            [2.0, 5.0, 4.0, 8.0],
            [4.0, 3.0, 7.0, 5.0],
            [6.0, 4.0, 10.0, 2.0],
        ])

    def test_resolved_toroidal_mean_matches_historical_average(self):
        xgc = _xgc_instance()
        with _reader_patch(self.var):
            psi_avg, time_avg, intensity_avg = analysis.turb_intensity(
                xgc, 10, 11, 1, toroidal='average'
            )
            psi_res, time_res, intensity_res = analysis.turb_intensity(
                xgc, 10, 11, 1, toroidal='resolved'
            )

        self.assertEqual(intensity_avg.shape, (1, 2))
        self.assertEqual(intensity_res.shape, (1, 3, 2))
        np.testing.assert_array_equal(psi_res, psi_avg)
        np.testing.assert_array_equal(time_res, time_avg)
        np.testing.assert_allclose(intensity_res.mean(axis=1), intensity_avg)

    def test_resolved_mode_preserves_toroidal_variation(self):
        with _reader_patch(self.var):
            _, _, intensity = analysis.turb_intensity(
                _xgc_instance(), 10, 11, 1, toroidal='resolved'
            )

        self.assertFalse(
            np.allclose(intensity[:, 0, :], intensity[:, 1, :])
        )

    def test_rejects_unknown_toroidal_mode(self):
        with self.assertRaisesRegex(ValueError, "toroidal must be"):
            analysis.turb_intensity(
                _xgc_instance(), 10, 11, 1, toroidal='not-a-mode'
            )


if __name__ == '__main__':
    unittest.main()
