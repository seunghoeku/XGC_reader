"""Main XGC1 class and core functionality."""

import numpy as np
import os
import adios2

from .constants import cnst
from .utils import (check_adios2_version, adios2_get_shape, 
                   adios2_read_all_time, adios2_read_one_time, read_one_ad2_var)
from .oned_data import data1, radial_flux_all, heat_flux_all
from .mesh_data import meshdata, f0meshdata
from .volume_data import voldata
from .flux_data import fluxavg, turbdata
from .matrix_ops import (xgc_mat, grad_rz, ff_mapping, load_grad_rz, load_ff_mapping, 
                        convert_3d_grad_all, conv_real2ff, GradPlane, GradParX, write_dAs_ff_for_poincare)
from .heat_diagnostics import (
    datahlp,
    datahl2,
    datahl2_sp,
    load_heatdiag,
    load_heatdiag2,
    postprocess_heatdiag2,
)
from .field_data import databfm
# others.py is now empty - functions moved to appropriate modules
from .report import print_plasma_info, report_heatdiag2, report_profiles, report_turb_2d, turb_2d_report
from .plotting import plot1d_if, contourf_one_var, contourf_ad2_var, show_sep, plot2d
from .geometry import (find_sep_idx, find_surf_idx, find_tmask, find_line_segment,
                      fsa_simple, flux_sum_simple, midplane_var, d_dpsi)
from .analysis import (turb_intensity, source_simple, plot_source_simple, 
                      gyro_radius, find_exb_velocity, find_exb_velocity2, reading_3d_data,
                      prepare_plots, power_spectrum_w_k_with_exb, gam_freq_analytic, midplane, midplane_var_all)


class xgc1(object):
    """Main XGC1 data reader class."""
    
    # Import constants for backward compatibility
    cnst = cnst

    def __init__(
        self,
        path='./',
        *,
        backend='analysis',
        change_cwd=True,
        simulation=None,
        catalog=None,
    ):
        """
        Initialize either cd to a directory to process many files later, or
        open an Adios Campaign Archive now.

        Parameters
        ----------
        path : str or os.PathLike, optional
            XGC output directory or ``.aca`` campaign.
        backend : {"legacy", "analysis"}, optional
            Data-access implementation. ``"analysis"`` is the default and
            uses XGC-Analysis through a compatibility facade. ``"legacy"``
            preserves the existing direct ADIOS2 readers.
        change_cwd : bool, optional
            Preserve the historical behavior of changing into a directory
            dataset. Set to False for new code that does not rely on relative
            file reads.
        simulation, catalog : object, optional
            Pre-built XGC-Analysis objects, primarily for integration and
            testing. They are only valid with ``backend="analysis"``.
        """
        # Check ADIOS2 version compatibility
        check_adios2_version()

        path = os.fspath(path)
        if backend not in {"legacy", "analysis"}:
            raise ValueError("backend must be either 'legacy' or 'analysis'")
        if backend == "legacy" and (simulation is not None or catalog is not None):
            raise ValueError(
                "simulation and catalog are only valid with backend='analysis'"
            )

        self.backend = backend
        self.change_cwd = change_cwd
        self._analysis_backend = None
        is_campaign = path.endswith(".aca")
        location = os.path.abspath(path)

        if backend == "analysis":
            from .analysis_compat import create_analysis_backend

            self.campaign = None
            self.campaign_all_vars = {}
            if is_campaign:
                self.path = ''
            else:
                if change_cwd:
                    os.chdir(location)
                self.path = location.rstrip(os.sep) + os.sep
            self._analysis_backend = create_analysis_backend(
                location,
                simulation=simulation,
                catalog=catalog,
            )
            self.catalog = self._analysis_backend.catalog
            self.simulation = self._analysis_backend.simulation
        elif is_campaign:
            self.campaign = adios2.FileReader(location)
            self.path = ''  # for self.path+filename to able to serve as name in campaign
            # get all variable names and info at once and save for reuse
            self.campaign_all_vars = self.campaign.available_variables()
        else:
            self.campaign = None
            if change_cwd:
                os.chdir(location)
            self.path = location.rstrip(os.sep) + os.sep
            self.campaign_all_vars = {}  # not usable when reading individual files locally

    def close(self):
        """Close resources held by the selected backend."""
        if self._analysis_backend is not None:
            self._analysis_backend.close()
        elif self.campaign:
            self.campaign.close()

    @classmethod
    def load_basic(cls, path='./', **kwargs):
        """Load basic XGC data including units, 1D, mesh, and volumes."""
        instance = cls(path, **kwargs)
        instance.load_unitsm()
        instance.load_oned()
        instance.setup_mesh()
        instance.setup_f0mesh()
        instance.load_volumes()
        return instance

    def load_unitsm(self):
        """For compatibility with older version."""
        if self._analysis_backend is not None:
            try:
                self.load_units()
            except KeyError as exc:
                if "xgc.units.bp" not in str(exc):
                    raise
                self.load_unitsm_old()
            return
        try:
            self.load_units()
        except:
            self.load_unitsm_old()

    def load_units(self):
        """Read in xgc.units.bp file."""
        if self._analysis_backend is not None:
            self.unit_dic = self._analysis_backend.load_units()
            self.psix = self.unit_dic['eq_x_psi']
            self.eq_x_r = self.unit_dic['eq_x_r']
            self.eq_x_z = self.unit_dic['eq_x_z']
            self.eq_axis_r = self.unit_dic['eq_axis_r']
            self.eq_axis_z = self.unit_dic['eq_axis_z']
            self.eq_axis_b = self.unit_dic['eq_axis_b']
            self.sml_wedge_n = self.unit_dic['sml_wedge_n']
            for name in ('sml_dt', 'diag_1d_period'):
                if name in self.unit_dic:
                    setattr(self, name, self.unit_dic[name])
            self._sync_analysis_objects()
            return

        if self.campaign:
            f = self.campaign
            prefix = 'xgc.units.bp/'
        else:
            f = adios2.FileReader(self.path + "xgc.units.bp")
            prefix = ''
            
        self.unit_dic = {}
        self.unit_dic['eq_x_psi'] = f.read(prefix + 'eq_x_psi')
        self.unit_dic['eq_x_r'] = f.read(prefix + 'eq_x_r')
        self.unit_dic['eq_x_z'] = f.read(prefix + 'eq_x_z')
        self.unit_dic['eq_axis_r'] = f.read(prefix + 'eq_axis_r')
        self.unit_dic['eq_axis_z'] = f.read(prefix + 'eq_axis_z')
        self.unit_dic['eq_axis_b'] = f.read(prefix + 'eq_axis_b')
        self.unit_dic['sml_dt'] = f.read(prefix + 'sml_dt')
        self.unit_dic['diag_1d_period'] = f.read(prefix + 'diag_1d_period')

        try:
            self.unit_dic['e_ptl_charge_eu'] = f.read(prefix + 'e_ptl_charge_eu')
            self.unit_dic['e_ptl_mass_au'] = f.read(prefix + 'e_ptl_mass_au')
        except:
            print('No electron particle charge/mass found in xgc.units.bp')
        self.unit_dic['eq_den_v1'] = f.read(prefix + 'eq_den_v1')
        self.unit_dic['eq_tempi_v1'] = f.read(prefix + 'eq_tempi_v1')
        self.unit_dic['i_ptl_charge_eu'] = f.read(prefix + 'i_ptl_charge_eu')
        self.unit_dic['i_ptl_mass_au'] = f.read(prefix + 'i_ptl_mass_au')
        self.unit_dic['sml_dt'] = f.read(prefix + 'sml_dt')
        self.unit_dic['sml_totalpe'] = f.read(prefix + 'sml_totalpe')
        self.unit_dic['sml_tran'] = f.read(prefix + 'sml_tran')
        try:
            self.unit_dic['sml_wedge_n'] = f.read(prefix + 'sml_wedge_n')
        except:
            self.unit_dic['sml_wedge_n'] = 1  # XGCa

        self.psix = self.unit_dic['eq_x_psi']
        self.eq_x_r = self.unit_dic['eq_x_r']
        self.eq_x_z = self.unit_dic['eq_x_z']
        self.eq_axis_r = self.unit_dic['eq_axis_r']
        self.eq_axis_z = self.unit_dic['eq_axis_z']
        self.eq_axis_b = self.unit_dic['eq_axis_b']
        self.sml_dt = self.unit_dic['sml_dt']
        self.sml_wedge_n = self.unit_dic['sml_wedge_n']
        self.diag_1d_period = self.unit_dic['diag_1d_period']

        if not self.campaign:
            f.close()

    def load_unitsm_old(self):
        """Read in units.m file -- for backward compatibility."""
        self.unit_file = self.path + 'units.m'
        self.unit_dic = self.load_m(self.unit_file)
        self.psix = self.unit_dic['psi_x']
        self.eq_x_r = self.unit_dic['eq_x_r']
        self.eq_x_z = self.unit_dic['eq_x_z']
        self.eq_axis_r = self.unit_dic['eq_axis_r']
        self.eq_axis_z = self.unit_dic['eq_axis_z']
        self.eq_axis_b = self.unit_dic['eq_axis_b']
        self.sml_dt = self.unit_dic['sml_dt']
        self.sml_wedge_n = self.unit_dic['sml_wedge_n']
        self.diag_1d_period = self.unit_dic['diag_1d_period']

    def load_oned(self, i_mass=2, i2mass=12):
        """Load xgc.oneddiag.bp and some post process."""
        if self._analysis_backend is not None:
            self.od = self._analysis_backend.load_oned()
            self.electron_on = self.od.electron_on
            self.ion2_on = self.od.ion2_on
            if getattr(self.od, 'psi00', None) is not None and hasattr(self, 'psix'):
                self.od.psi00n = self.od.psi00 / self.psix
            if self.electron_on and hasattr(self, 'eq_axis_b'):
                self.od.beta_e = (
                    self.cnst.echarge
                    * self.od.e_gc_density_df_1d
                    * self.od.Te
                    / (self.eq_axis_b**2 * 0.5 / self.cnst.mu0)
                )
            self._sync_analysis_objects()
            return

        if self.campaign:
            self.od = data1(self.campaign, self.campaign_all_vars, "xgc.oneddiag.bp")
        else:
            self.od = data1(self.path + "xgc.oneddiag.bp")
        self.od.psi = self.od.psi[0, :]
        self.od.psi00 = self.od.psi00[0, :]
        try:
            self.od.psi00n = self.od.psi00 / self.psix
        except:
            print("psix is not defined - call load_units() to get psix to get psi00n")

        #gstep
        try:
            self.od.step = self.od.gstep
        except:
            print('gstep is not defined')
        
        # Temperatures
        try: 
            Teperp = self.od.e_perp_temperature_df_1d
        except:
            print('No electron')
            self.electron_on = False
        else:
            self.electron_on = True
            Tepara = self.od.e_parallel_mean_en_df_1d
            self.od.Te = (Teperp + Tepara) / 3 * 2
        
        # Minority or impurity temperature
        try: 
            Ti2perp = self.od.i2perp_temperature_df_1d
        except:
            print('No Impurity')
            self.ion2_on = False
        else:
            self.ion2_on = True
            Ti2para = self.od.i2parallel_mean_en_df_1d - 0.5 * i2mass * self.cnst.protmass * self.od.i2parallel_flow_df_1d**2 / self.cnst.echarge
            self.od.Ti2 = (Ti2perp + Ti2para) / 3 * 2

        Tiperp = self.od.i_perp_temperature_df_1d
        Tipara = self.od.i_parallel_mean_en_df_1d - 0.5 * i_mass * self.cnst.protmass * self.od.i_parallel_flow_df_1d**2 / self.cnst.echarge
        self.od.Ti = (Tiperp + Tipara) / 3 * 2

        # ExB shear calculation
        if self.electron_on:
            shear = self.od.d_dpsi(self.od.e_poloidal_ExB_flow_1d, self.od.psi_mks)
            self.od.grad_psi_sqr = self.od.e_grad_psi_sqr_1d
        else:
            shear = self.od.d_dpsi(self.od.i_poloidal_ExB_flow_1d, self.od.psi_mks)
            self.od.grad_psi_sqr = self.od.i_grad_psi_sqr_1d
        self.od.shear_r = shear * np.sqrt(self.od.grad_psi_sqr)

        if self.electron_on:
            self.od.density = self.od.e_gc_density_df_1d
        else:
            self.od.density = self.od.i_gc_density_df_1d

        # Gradient scale
        self.od.Ln = self.od.density / self.od.d_dpsi(self.od.density, self.od.psi_mks) / np.sqrt(self.od.grad_psi_sqr)
        self.od.Lti = self.od.Ti / self.od.d_dpsi(self.od.Ti, self.od.psi_mks) / np.sqrt(self.od.grad_psi_sqr)
        if self.electron_on:
            self.od.Lte = self.od.Te / self.od.d_dpsi(self.od.Te, self.od.psi_mks) / np.sqrt(self.od.grad_psi_sqr)
            
        # Plasma beta (electron)
        try:
            self.od.beta_e = self.cnst.echarge * self.od.density * self.od.Te / (self.eq_axis_b**2 * 0.5 / self.cnst.mu0)
        except:
            print('electron beta calculation failed. No electron? units.m not loaded?')

        # Find tmask
        d = self.od.step[1] - self.od.step[0]
        st = self.od.step[0] / d
        ed = self.od.step[-1] / d
        st = st.astype(int)
        ed = ed.astype(int)
        idx = np.arange(st, ed, dtype=int)

        self.od.tmask = idx
        for i in idx:
            tmp = np.argwhere(self.od.step == i * d)
            try: 
                self.od.tmask[i - st] = tmp[-1, -1]
            except:
                print('failed to find tmaks', tmp)

    def load_m(self, fname):
        """Load XGC's ASCII ``key = value;`` parameter files."""
        result = {}
        with open(fname, 'r') as parameter_file:
            for line in parameter_file:
                line = line.split('!', 1)[0].split('%', 1)[0].strip()
                if not line or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                value = value.strip().rstrip(';').strip()
                result[key.strip()] = float(
                    value.replace('D', 'e').replace('d', 'e')
                )
        return result

    def setup_mesh(self):
        """Set up mesh data."""
        if self._analysis_backend is not None:
            if self._analysis_backend.supports_simulation():
                self.mesh = self._analysis_backend.mesh_view()
            else:
                self.mesh = meshdata(
                    self._analysis_backend.legacy_static_source()
                )
            self._sync_analysis_objects()
        elif self.campaign:
            self.mesh = meshdata(self.campaign)
        else:
            self.mesh = meshdata(self.path)

        # Setup separatrix
        if hasattr(self.mesh, 'psi_surf') and hasattr(self, 'psix'):
            self.mesh.isep = np.argmin(abs(self.mesh.psi_surf - self.psix))
            isep = self.mesh.isep
            length = self.mesh.surf_len[isep]
            self.mesh.msep = self.mesh.surf_idx[isep, 0:length] - 1  # zero based

    def setup_f0mesh(self):
        """Set up f0 mesh data."""
        if self._analysis_backend is not None:
            if self._analysis_backend.supports_simulation():
                self.f0 = self._analysis_backend.f0_view()
            else:
                self.f0 = f0meshdata(
                    self._analysis_backend.legacy_static_source()
                )
            self._sync_analysis_objects()
        elif self.campaign:
            self.f0 = f0meshdata(self.campaign)
        else:
            self.f0 = f0meshdata(self.path)

    def load_volumes(self):
        """Load volume data."""
        if self._analysis_backend is not None:
            if self._analysis_backend.supports_simulation():
                self.vol = self._analysis_backend.volume_view()
            else:
                self.vol = voldata(
                    self._analysis_backend.legacy_static_source()
                )
            self._sync_analysis_objects()
        elif self.campaign:
            self.vol = voldata(self.campaign)
        else:
            self.vol = voldata(self.path)

    def load_bfieldm(self):
        """Load magnetic field midplane data with the existing small reader."""
        if self._analysis_backend is not None:
            if self._analysis_backend.is_campaign:
                catalog = self._analysis_backend.ensure_catalog()
                campaign_reader = getattr(catalog, 'campaign_reader', None)
                if campaign_reader is None:
                    raise RuntimeError(
                        "XGC-Analysis campaign catalog has no open reader for "
                        "legacy xgc.bfieldm.bp access."
                    )
                self.bfm = databfm(campaign_reader)
            else:
                self.bfm = databfm(self.path)
            self._sync_analysis_objects()
        elif self.campaign:
            self.bfm = databfm(self.campaign)
        else:
            self.bfm = databfm(self.path)
        
        self.bfm.r0 = self.unit_dic['eq_axis_r']
        n0 = np.nonzero(self.bfm.rmid > self.bfm.r0)[0][0]
        self.bfm.rmido = self.bfm.rmid[n0:]
        self.bfm.psino = self.bfm.psin[n0:]

    def load_bfield(self):
        """Load equilibrium bfield data."""
        if (
            self._analysis_backend is not None
            and self._analysis_backend.supports_simulation()
        ):
            self.bfield = self._analysis_backend.bfield_array()
            magnetic = self._analysis_backend.ensure_simulation().magnetic_field
            if hasattr(magnetic, 'jpar_bg_pd'):
                self.jpar_bg = magnetic.jpar_bg_pd.get_data()
            self._sync_analysis_objects()
            return

        if self._analysis_backend is not None and self._analysis_backend.is_campaign:
            self._load_legacy_bfield(
                self._analysis_backend.legacy_static_source(),
                prefix='xgc.bfield.bp/',
            )
            self._sync_analysis_objects()
            return

        if self.campaign:
            self._load_legacy_bfield(
                self.campaign,
                prefix='xgc.bfield.bp/',
            )
            return

        with adios2.FileReader(self.path + "xgc.bfield.bp") as f:
            self._load_legacy_bfield(f)

    def _load_legacy_bfield(self, reader, prefix=''):
        """Populate legacy bfield arrays from a BP file or campaign handle."""
        bfield = None
        for name in (
            prefix + 'bfield',
            prefix + '/bfield',
            prefix + '/node_data[0]/values',
            prefix + 'node_data[0]/values',
        ):
            try:
                bfield = reader.read(name)
                break
            except Exception:
                continue
        if bfield is None:
            raise KeyError("No bfield variable found in xgc.bfield.bp")

        self.bfield = bfield
        if self.bfield.shape[0] != 3:
            self.bfield = np.transpose(self.bfield)
            print('bfield shape is :', self.bfield.shape)

        for name in (prefix + 'jpar_bg', prefix + '/jpar_bg'):
            try:
                self.jpar_bg = reader.read(name)
                break
            except Exception:
                continue
        else:
            print('No jpar_bg in xgc.bfield.bp')


    def load_heatdiag(self, **kwargs):
        """Load legacy heat diagnostic data with the existing direct reader."""
        if (
            self._analysis_backend is not None
            and self._analysis_backend.is_campaign
        ):
            raise NotImplementedError(
                "The legacy xgc.heatdiag.bp reader only supports directory "
                "datasets, not ADIOS Campaign Archives."
            )
        load_heatdiag(self, **kwargs)

    def load_heatdiag2(self):
        """Load heat diagnostic v2 data."""
        if self._analysis_backend is not None:
            self.hl2 = self._analysis_backend.load_heatdiag2()
            postprocess_heatdiag2(self)
            self._sync_analysis_objects()
            return
        load_heatdiag2(self)

    def _sync_analysis_objects(self):
        """Expose lazily-created XGC-Analysis objects on the facade."""
        if self._analysis_backend is None:
            return
        self.catalog = self._analysis_backend.catalog
        self.simulation = self._analysis_backend.simulation

    def fsa_simple(self, var):
        """Simple flux surface average using mesh data."""
        return fsa_simple(self, var)

    def flux_sum_simple(self, var):
        """Simple summation over flux surface."""
        return flux_sum_simple(self, var)

    def midplane_var(self, var, inboard=False, nr=300, delta_r_axis=0., delta_r_edge=0., return_rmid=False):
        """Extract midplane values of a variable."""
        return midplane_var(self, var, inboard, nr, delta_r_axis, delta_r_edge, return_rmid)

    def midplane_var_all(self, istart, iend, skip, varname='dpot', ftype='3d', nr=300, delta_r_axis=0.):
        """Extract all midplane values of a variable."""
        return midplane_var_all(self, istart, iend, skip, varname=varname, ftype=ftype, nr=nr, delta_r_axis=delta_r_axis)

    def radial_flux_all(self):
        """Get radial flux of energy and particle from 1D data."""
        radial_flux_all(self)

    def heat_flux_all(self):
        """Calculate all heat flux components."""
        heat_flux_all(self)

    def gam_freq_analytic(self):
        """Get GAM analytic GAM frequency."""
        return gam_freq_analytic(self)

    def print_plasma_info(self):
        """Print plasma information."""
        print_plasma_info(self)

    def midplane(self):
        """Get midplane analysis."""
        return midplane(self)
    
    def plot1d_if(self, obj, **kwargs):
        """Plot 1D variable of initial and final time steps."""
        return plot1d_if(self, obj, **kwargs)
    
    def contourf_one_var(self, fig, ax, var, title, **kwargs):
        return contourf_one_var(self, var, fig=fig, ax=ax, title=title)

    def contourf_one_var2(self, var, fig=None, ax=None, title=None, vm=None, cmap='jet', levels=150, cbar=True):
        """Create filled contour plot of variable on mesh."""
        if(fig is None or ax is None):
            fig, ax = plt.subplots()
        return contourf_one_var(self, var, fig=fig, ax=ax, title=title, vm=vm, cmap=cmap, levels=levels, cbar=cbar)

    def contourf_ad2_var(self, filename, var, iphi=0, fig=None, ax=None, title=None, vm=None, cmap='jet', levels=150, cbar=True, time_unit='ms'):
        """Read an ADIOS2 file and plot a filled contour of one variable on the mesh."""
        return contourf_ad2_var(self, filename, var, iphi=iphi, fig=fig, ax=ax, title=title, vm=vm, cmap=cmap, levels=levels, cbar=cbar, time_unit=time_unit)

    def show_sep(self, ax, style='-'):
        """Show separatrix on plot."""
        return show_sep(self, ax, style)
    
    def plot2d(self, filestr, varstr, **kwargs):
        """General 2D plot function."""
        return plot2d(self, filestr, varstr, **kwargs)
    
    def find_sep_idx(self):
        """Find separatrix node indices."""
        return find_sep_idx(self)
    
    def find_surf_idx(self, psi_norm=1.0):
        """Find flux surface node indices."""
        return find_surf_idx(self, psi_norm)
    
    def find_tmask(self, step, max_end=False):
        """Find time mask for time steps."""
        return find_tmask(self, step, max_end)
    
    def find_line_segment(self, n, psi_target, dir='middle'):
        """Find line segment along flux surface."""
        return find_line_segment(self, n, psi_target, dir)
    
    def turb_intensity(
        self,
        istart,
        iend,
        skip,
        vartype='f3d_eden',
        mode='all',
        toroidal='average',
    ):
        """Calculate turbulence intensity from 3D data files."""
        return turb_intensity(
            self, istart, iend, skip, vartype, mode, toroidal
        )
    
    def source_simple(self, step, period, sp='i_', moments='energy', source_type='heat_torque'):
        """Simple source analysis from diagnostic files."""
        return source_simple(self, step, period, sp, moments, source_type)
    
    def plot_source_simple(self, step, period, sp='i_', moments='energy', source_type='heat_torque'):
        """Plot simple source analysis."""
        return plot_source_simple(self, step, period, sp, moments, source_type)
    
    def gyro_radius(self, t_ev, b, mass_au, charge_eu):
        """Calculate gyroradius."""
        return gyro_radius(self, t_ev, b, mass_au, charge_eu)
    
    def find_exb_velocity(self, istart, iend, skip, ms):
        """Find average ExB velocity of line segment."""
        return find_exb_velocity(self, istart, iend, skip, ms)
    
    def power_spectrum_w_k_with_exb(self, istart, iend, skip, skip_exb, psi_target, ns_half, varname='dpot', ftype='3d', remove_n0=True, old_vexb=False):
        """Calculate power spectrum w-k with ExB velocity."""
        return power_spectrum_w_k_with_exb(self, istart, iend, skip, skip_exb, psi_target, ns_half, varname=varname, ftype=ftype, remove_n0=remove_n0, old_vexb=old_vexb)
    
    def gam_freq_analytic(self):
        """Get GAM analytic frequency."""
        return gam_freq_analytic(self)
    
    def midplane(self):
        """Get midplane analysis."""
        return midplane(self)
    
    def load_grad_rz(self):
        """Load gradient R-Z matrices."""
        if (
            self._analysis_backend is not None
            and self._analysis_backend.supports_simulation()
        ):
            self.grad = self._analysis_backend.gradient_view()
            self._sync_analysis_objects()
            return
        self.grad = load_grad_rz(self)
    
    def load_ff_mapping(self):
        """Load field-following mapping matrices."""
        if (
            self._analysis_backend is not None
            and self._analysis_backend.supports_simulation()
        ):
            self.ff_mappings = self._analysis_backend.field_following_views()
            self._sync_analysis_objects()
        else:
            self.ff_mappings = load_ff_mapping(self)
        # Set individual mappings as attributes for backward compatibility
        for name, mapping in self.ff_mappings.items():
            setattr(self, 'ff_' + name, mapping)
    
    def convert_3d_grad_all(self, field):
        """Convert field into gradient representation."""
        return convert_3d_grad_all(self, field)
    
    def adios2_get_shape(self, f, varname):
        """Get shape and step information for ADIOS2 variable."""
        return adios2_get_shape(f, varname)
    
    def adios2_read_all_time(self, f, varname):
        """Read all time steps for a variable from ADIOS2 file."""
        return adios2_read_all_time(f, varname)
    
    def adios2_read_one_time(self, f, varname, step=-1):
        """Read one time step for a variable from ADIOS2 file."""
        return adios2_read_one_time(f, varname, step)
    
    def read_one_ad2_var(self, filestr, varstr, with_time=False):
        """Read one variable from ADIOS2 file with optional time."""
        return read_one_ad2_var(filestr, varstr, with_time)
    
    def report_heatdiag2(self, **kwargs):
        """Generate comprehensive heat diagnostic report with plots."""
        return report_heatdiag2(self, **kwargs)
    
    def report_profiles(self, **kwargs):
        """Generate comprehensive profile reports."""
        return report_profiles(self, **kwargs)
    
    def report_turb_2d(self, **kwargs):
        """Generate 2D turbulence report with heat flux contours."""
        return report_turb_2d(self, **kwargs)
    
    def turb_2d_report(self, **kwargs):
        """Alias for report_turb_2d for backward compatibility."""
        return turb_2d_report(self, **kwargs)
    
    # Additional missing methods
    def find_exb_velocity2(self, istart, iend, skip, ms, only_average=True, return_Er=False):
        """Find ExB velocity with detailed analysis (version 2)."""
        return find_exb_velocity2(self, istart, iend, skip, ms, only_average=only_average, return_Er=return_Er)
    
    def reading_3d_data(self, istart, iend, skip, ms, no_fft=False):
        """Read 3D dpot data and perform FFT analysis."""
        return reading_3d_data(self, istart, iend, skip, ms, no_fft)
    
    def prepare_plots(self, dist, ms, time):
        """Prepare plot arrays for k and omega analysis."""
        return prepare_plots(self, dist, ms, time)
    
    def conv_real2ff(self, field):
        """Convert real space field to field-following representation."""
        return conv_real2ff(self, field)
    
    def GradPlane(self, field):
        """Calculate plane gradient of field."""
        return GradPlane(self, field)
    
    def GradParX(self, field):
        """Calculate parallel gradient of field."""
        return GradParX(self, field)
    
    def write_dAs_ff_for_poincare(self, fnum):
        """Write field-following vector potential for Poincare analysis."""
        return write_dAs_ff_for_poincare(self, fnum)
    
    def profile_reports(self, **kwargs):
        """Wrapper for report_profiles for backward compatibility."""
        return self.report_profiles(**kwargs)
    
    def d_dpsi(self, field):
        """Calculate derivative with respect to psi."""
        return d_dpsi(self, field)
    
    def get_midplane_bp_sep_and_eich_scale(self):
        """Get midplane Bp and Eich scale #14
        lambda_q = C * Bp^s
        C = 0.63
        s = -1.19
        """
        from .heat_diagnostics import get_midplane_bp_sep_and_eich_scale
        return get_midplane_bp_sep_and_eich_scale(self)
