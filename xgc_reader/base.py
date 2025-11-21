"""Main XGC1 class and core functionality."""

import numpy as np
import os
import adios2
from scipy.io import matlab

from .constants import cnst
from .utils import (check_adios2_version, adios2_get_shape, 
                   adios2_read_all_time, adios2_read_one_time, read_one_ad2_var)
from .oned_data import data1, radial_flux_all, heat_flux_all
from .mesh_data import meshdata, f0meshdata
from .volume_data import voldata
from .flux_data import fluxavg, turbdata
from .matrix_ops import (xgc_mat, grad_rz, ff_mapping, load_grad_rz, load_ff_mapping, 
                        convert_3d_grad_all, conv_real2ff, GradPlane, GradParX, write_dAs_ff_for_poincare)
from .heat_diagnostics import datahlp, datahl2, datahl2_sp, load_heatdiag, load_heatdiag2
from .field_data import databfm
# others.py is now empty - functions moved to appropriate modules
from .report import print_plasma_info, report_heatdiag2, report_profiles, report_turb_2d, turb_2d_report
from .plotting import plot1d_if, contourf_one_var, show_sep, plot2d
from .geometry import (find_sep_idx, find_surf_idx, find_tmask, find_line_segment,
                      fsa_simple, flux_sum_simple, midplane_var, d_dpsi)
from .analysis import (turb_intensity, source_simple, plot_source_simple, 
                      gyro_radius, find_exb_velocity, find_exb_velocity2, reading_3d_data,
                      prepare_plots, power_spectrum_w_k_with_exb, gam_freq_analytic, midplane, midplane_var_all)


class xgc1(object):
    """Main XGC1 data reader class."""
    
    # Import constants for backward compatibility
    cnst = cnst

    def __init__(self, path='./'):
        """
        Initialize either cd to a directory to process many files later, or
        open an Adios Campaign Archive now.
        """
        # Check ADIOS2 version compatibility
        check_adios2_version()

        if path.endswith(".aca"):
            self.campaign = adios2.FileReader(path)
            self.path = ''  # for self.path+filename to able to serve as name in campaign
            # get all variable names and info at once and save for reuse
            self.campaign_all_vars = self.campaign.available_variables()
        else:
            self.campaign = None
            os.chdir(path)
            self.path = os.getcwd() + '/'
            self.campaign_all_vars = {}  # not usable when reading individual files locally

    def close(self):
        """Close campaign if open."""
        if self.campaign:
            self.campaign.close()

    @classmethod
    def load_basic(cls, path='./'):
        """Load basic XGC data including units, 1D, mesh, and volumes."""
        instance = cls(path)
        instance.load_units()
        instance.load_oned()
        instance.setup_mesh()
        instance.setup_f0mesh()
        instance.load_volumes()
        return instance

    def load_unitsm(self):
        """For compatibility with older version."""
        try:
            self.load_units()
        except:
            self.load_unitsm_old()

    def load_units(self):
        """Read in xgc.units.bp file."""
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
        """Load MATLAB .m file."""
        return matlab.loadmat(fname)

    def setup_mesh(self):
        """Set up mesh data."""
        if self.campaign:
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
        if self.campaign:
            self.f0 = f0meshdata(self.campaign)
        else:
            self.f0 = f0meshdata(self.path)

    def load_volumes(self):
        """Load volume data."""
        if self.campaign:
            self.vol = voldata(self.campaign)
        else:
            self.vol = voldata(self.path)

    def load_grad_rz(self):
        """Load gradient R-Z data."""
        self.grz = grad_rz(self)

    def load_ff_mapping(self):
        """Load field-following mapping."""
        self.ffm = ff_mapping(self)

    def load_bfieldm(self):
        """Load magnetic field midplane data."""
        if self.campaign:
            self.bfm = databfm(self.campaign)
        else:
            self.bfm = databfm(self.path)
        
        self.bfm.r0 = self.unit_dic['eq_axis_r']
        n0 = np.nonzero(self.bfm.rmid > self.bfm.r0)[0][0]
        self.bfm.rmido = self.bfm.rmid[n0:]
        self.bfm.psino = self.bfm.psin[n0:]

    def load_bfield(self):
        """Load equilibrium bfield data."""
        with adios2.FileReader(self.path + "xgc.bfield.bp") as f:
            try:
                self.bfield = f.read('bfield')
            except: # try older version of bfield
                self.bfield = f.read('/node_data[0]/values')

            if(self.bfield.shape[0]!=3): # not 3xN
                self.bfield = np.transpose(self.bfield)
                print('bfield shape is :', self.bfield.shape)            
    
            try:
                self.jpar_bg = f.read('jpar_bg') # background current
            except:
                print('No jpar_bg in xgc.bfield.bp')



    def load_heatdiag(self, **kwargs):
        """Load heat diagnostic data."""
        load_heatdiag(self, **kwargs)

    def load_heatdiag2(self):
        """Load heat diagnostic v2 data."""
        load_heatdiag2(self)

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
    
    def turb_intensity(self, istart, iend, skip, vartype='f3d_eden', mode='all'):
        """Calculate turbulence intensity from 3D data files."""
        return turb_intensity(self, istart, iend, skip, vartype, mode)
    
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
        self.grad = load_grad_rz(self)
    
    def load_ff_mapping(self):
        """Load field-following mapping matrices."""
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