"""Heat load diagnostic data handling classes."""

import numpy as np
import adios2
from scipy.optimize import curve_fit
from scipy.special import erfc
from functools import singledispatchmethod

from .utils import read_all_steps

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class datahlp(object):
    """Heat load diagnostic output class."""
    
    def __init__(self, filename, irg, read_rz_all=False):
        """
        Initialize heat load diagnostic data.
        
        Parameters
        ----------
        filename : str
            Path to heat diagnostic file
        irg : int
            Region number (0=outer, 1=inner divertor)
        read_rz_all : bool, optional
            Whether to read all R,Z data or just last time step
        """
        with adios2.FileReader(filename) as f:
            # irg is region number 0,1 - outer, inner
            # read file and assign it
            self.vars = f.available_variables()
            for v in self.vars:
                stc = self.vars[v].get("AvailableStepsCount")
                ct = self.vars[v].get("Shape")
                sgl = self.vars[v].get("SingleValue")
                stc = int(stc)
                if ct != '':
                    c = [int(i) for i in ct.split(',')]
                    if len(c) == 1:  # time and step 
                        setattr(self, v, f.read(v, start=[0], count=c, step_selection=[0, stc]))
                    elif len(c) == 2:  # c[0] is irg
                        setattr(self, v, np.squeeze(f.read(v, start=[irg, 0], count=[1, c[1]], step_selection=[0, stc])))
                    elif (len(c) == 3 & read_rz_all):  # ct[0] is irg, read only 
                        setattr(self, v, np.squeeze(f.read(v, start=[irg, 0, 0], count=[1, c[1], c[2]], step_selection=[0, stc])))
                    elif (len(c) == 3):  # read_rz_all is false. ct[0] is irg, read only 
                        setattr(self, v, np.squeeze(f.read(v, start=[irg, 0, 0], count=[1, c[1], c[2]], step_selection=[stc - 1, 1])))
                elif v != 'zsamples' and v != 'rsamples':
                    setattr(self, v, f.read(v, start=[], count=[], step_selection=[0, stc]))  # null list for scalar
            
            # keep last time step
            self.r = self.r[-1, :]
            self.z = self.z[-1, :]

    def post_heatdiag(self, ds):
        """Get some parameters for plots of heat diag."""
        self.drmid = self.rmid * 0  # mem allocation
        self.drmid[1:-1] = (self.rmid[2:] - self.rmid[0:-2]) * 0.5
        self.drmid[0] = self.drmid[1]
        self.drmid[-1] = self.drmid[-2]

        dt = np.zeros_like(self.time)
        dt[1:] = self.time[1:] - self.time[0:-1]
        dt[0] = dt[1]
        rst = np.nonzero(dt < 0)  # index when restart happen
        dt[rst] = dt[rst[0] + 1]
        self.dt = dt

        # get separatrix r
        self.rs = np.interp([1], self.psin, self.rmid)
        
        self.rmidsepmm = (self.rmid - self.rs) * 1E3  # dist from sep in mm

        # get heat
        self.qe = np.transpose(self.e_perp_energy_psi + self.e_para_energy_psi) / dt / ds
        self.qi = np.transpose(self.i_perp_energy_psi + self.i_para_energy_psi) / dt / ds
        self.ge = np.transpose(self.e_number_psi) / dt / ds
        self.gi = np.transpose(self.i_number_psi) / dt / ds

        self.qe = np.transpose(self.qe)
        self.qi = np.transpose(self.qi)
        self.ge = np.transpose(self.ge)
        self.gi = np.transpose(self.gi)

        if hasattr(self, 'ion2_on') and self.ion2_on:
            self.qi2 = np.transpose(self.i2perp_energy_psi + self.i2para_energy_psi) / dt / ds
            self.gi2 = np.transpose(self.i2number_psi) / dt / ds
            self.qi2 = np.transpose(self.qi2)
            self.gi2 = np.transpose(self.gi2)

        self.qt = self.qe + self.qi
        if hasattr(self, 'ion2_on') and self.ion2_on:
            self.qt = self.qt + self.qi2

        # imx=self.qt.argmax(axis=1)
        mx = np.amax(self.qt, axis=1)
        self.lq_int = mx * 0  # mem allocation

        for i in range(mx.shape[0]):
            self.lq_int[i] = np.sum(self.qt[i, :] * self.drmid) / mx[i]

    def total_heat(self, wedge_n):
        """Getting total heat (radially integrated) to inner/outer divertor."""
        qe = wedge_n * (np.sum(self.e_perp_energy_psi, axis=1) + np.sum(self.e_para_energy_psi, axis=1))
        qi = wedge_n * (np.sum(self.i_perp_energy_psi, axis=1) + np.sum(self.i_para_energy_psi, axis=1))
        if hasattr(self, 'ion2_on') and self.ion2_on:
            qi2 = wedge_n * (np.sum(self.i2perp_energy_psi, axis=1) + np.sum(self.i2para_energy_psi, axis=1))

        # find restart point and remove -- 
        # find dt in varying sml_dt after restart

        self.qe_tot = qe / self.dt
        self.qi_tot = qi / self.dt
        if hasattr(self, 'ion2_on') and self.ion2_on:
            self.qi2tot = qi2 / self.dt

    def eich(self, xdata, q0, s, lq, dsep):
        """
        Functions for eich fit
        q(x) =0.5*q0* exp( (0.5*s/lq)^2 - (x-dsep)/lq ) * erfc (0.5*s/lq - (x-dsep)/s)
        """
        return 0.5 * q0 * np.exp((0.5 * s / lq)**2 - (xdata - dsep) / lq) * erfc(0.5 * s / lq - (xdata - dsep) / s)

    def eich_fit1(self, ydata, pmask):
        """Eich fitting of one profile data."""
        q0init = np.max(ydata)
        sinit = 2  # 2mm
        lqinit = 1  # 1mm
        dsepinit = 0.1  # 0.1 mm

        p0 = np.array([q0init, sinit, lqinit, dsepinit])
        if pmask is None:
            pmask = slice(0, ydata.shape[0])

        r_data = self.rmidsepmm[pmask]
        y_data = ydata[pmask]
        try:
            popt, pconv = curve_fit(self.eich, r_data, y_data, p0=p0)
        except:
            popt = [0, 1, 1, 0]
            pconv = np.zeros_like(p0)

        return popt, pconv

    def lambda_q3(self, x, q0, qf, lp, ln, lf, dsep):
        """
        Functions for 3 lambda fit: lp (lambda_q of private flux region),
        ln (lambda_q of near SOL), lf (lambda_q of far SOL)
        q(x) =     q0 * exp( (x-dsep)/lp)   when x<dsep
             =(q0-qf) * exp(-(x-dsep)/ln) + qf * exp(-(x-dsep)/lf) when x>dsep
        """
        dsepl = 0  # not using dsep --> dsepl=dsep to use
        rtn = q0 * np.exp((x - dsepl) / lp)  # only x<dsep will be used.
        ms = np.nonzero(x >= dsepl)
        rtn[ms] = (q0 - qf) * np.exp(-(x[ms] - dsepl) / ln) + qf * np.exp(-(x[ms] - dsepl) / lf)
        return rtn

    def lambda_q3_bound(self, x, q0, qf, lp, ln, delta_l, dsep):
        """
        Alternative 3-lambda fit with bounded parameters.
        Parameter space is q0, qf, lp, ln, delta_l (= lf - ln), dsep
        q(x) =     q0 * exp( (x-dsep)/lp)   when x<dsep
             =(q0-qf) * exp(-(x-dsep)/ln) + qf * exp(-(x-dsep)/(ln+delta_l)) when x>dsep
        """
        dsepl = 0  # not using dsep --> dsepl=dsep to use
        rtn = q0 * np.exp((x - dsepl) / lp)  # only x<dsep will be used.
        ms = np.nonzero(x >= dsepl)
        rtn[ms] = (q0 - qf) * np.exp(-(x[ms] - dsepl) / ln) + qf * np.exp(-(x[ms] - dsepl) / (ln + delta_l))
        return rtn

    def lambda_q3_fit1(self, ydata, pmask):
        """3-lambda fitting of one profile data."""
        q0init = np.max(ydata)
        qfinit = 0.01 * q0init  # 1 percent
        lpinit = 1  # 1mm
        lninit = 2  # 2mm
        lfinit = 4  # 4mm
        dsepinit = 0.01  # 0.01 mm

        p0 = np.array([q0init, qfinit, lpinit, lninit, lfinit - lninit, dsepinit])
        if pmask is None:
            pmask = slice(0, ydata.shape[0])

        r_data = self.rmidsepmm[pmask]
        y_data = ydata[pmask]
        try:
            bounds = ([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
            popt, pconv = curve_fit(self.lambda_q3_bound, r_data, y_data, p0=p0, bounds=bounds)
        except:
            popt = [0, 0, 1, 1, 1, 0]
            pconv = np.zeros_like(p0)

        return popt, pconv

    def eich_fit_all(self, **kwargs):
        """Perform fitting for all time steps."""
        pmask = kwargs.get('pmask', None)
        
        self.lq_eich = np.zeros_like(self.lq_int)  # mem allocation
        self.S_eich = np.zeros_like(self.lq_eich)
        self.dsep_eich = np.zeros_like(self.lq_eich)
        
        for i in range(self.time.size):
            popt, pconv = self.eich_fit1(self.qt[i, :], pmask)
            self.lq_eich[i] = popt[2]
            self.S_eich[i] = popt[1]
            self.dsep_eich[i] = popt[3]

    def lambda_q3_fit_all(self, **kwargs):
        """Lambda q3 fitting for all time steps."""
        pmask = kwargs.get('pmask', None)
        
        self.lp_lq3 = np.zeros_like(self.lq_int)  # mem allocation
        self.ln_lq3 = np.zeros_like(self.lp_lq3)
        self.lf_lq3 = np.zeros_like(self.lp_lq3)
        self.dsep_eich = np.zeros_like(self.lp_lq3)
        
        for i in range(self.time.size):
            popt, pconv = self.lambda_q3_fit1(self.qt[i, :], pmask)
            self.lp_lq3[i] = popt[2]
            self.ln_lq3[i] = popt[3]
            self.lf_lq3[i] = popt[4]
            self.dsep_eich[i] = popt[5]

    def qt_reset(self):
        """Reset qt from qi and qe."""
        self.qt = self.qe + self.qi
        if hasattr(self, 'ion2_on') and self.ion2_on:
            self.qt = self.qt + self.qi2

    def qt_smoothing(self, width, order):
        """Smoothing qt before Eich fit."""
        try:
            from scipy.signal import savgol_filter
            for i in range(self.time.size):
                tmp = self.qt[i, :]
                self.qt[i, :] = savgol_filter(tmp, width, order)
        except ImportError:
            print("Warning: scipy not available for qt_smoothing")


class datahl2_sp(object):
    """Data class for heat diagnostic species data."""
    
    def __init__(self, prefix, f):
        """Initialize species-specific heat load data."""
        self.number = read_all_steps(f, prefix + '_number')[:, :, 1:]
        self.para_energy = read_all_steps(f, prefix + '_para_energy')[:, :, 1:]
        self.perp_energy = read_all_steps(f, prefix + '_perp_energy')[:, :, 1:]
        self.potential = read_all_steps(f, prefix + '_potential')[:, :, 1:]


class datahl2(object):
    """Data class for heatdiag2."""
    
    def __init__(self, filename, datahl2_sp_class=None):
        """Initialize heat load diagnostic v2 data."""
        if datahl2_sp_class is None:
            datahl2_sp_class = datahl2_sp
            
        prefix = ['e', 'i', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9']
        with adios2.FileReader(filename) as f:
            vars = f.available_variables()

            self.time = read_all_steps(f, 'time')
            self.step = read_all_steps(f, 'step')
            self.tindex = read_all_steps(f, 'tindex')
            self.ds = read_all_steps(f, 'ds')
            self.psi = read_all_steps(f, 'psi')
            self.r = read_all_steps(f, 'r')
            self.z = read_all_steps(f, 'z')
            self.strike_angle = read_all_steps(f, 'strike_angle')

            # for each species read particle flux and energy flux as an array.
            max_nsp = 10  # maximum number of species. Any larger integer should work.
            self.sp = []
            for isp in range(max_nsp):
                if prefix[isp] + '_number' in vars:
                    self.sp.append(datahl2_sp_class(prefix[isp], f))
                else:
                    # print('No '+prefix[isp]+' species data in heatdiag2.')
                    break
            if isp == 0:
                print('No electron species data in heatdiag2. Nothing loaded.')

            self.nsp = len(self.sp)
            # set dt
            self.dt = np.zeros_like(self.time)
            self.dt[1:] = self.time[1:] - self.time[0:-1]
            self.dt[0] = self.dt[1]  # assume that the first time step is the same as the second one.
            self.dt = self.dt[:, np.newaxis]

    def get_midplane_conversion(self, psino, rmido):
        """Get midplane conversion of each species."""
        rs = np.interp([1], psino, rmido)
        rmidsepmm = (np.interp(self.psi, psino, rmido) - rs) * 1E3
        return rs, rmidsepmm

    def get_parallel_flux(self):
        """Get parallel flux for each species."""
        for isp in range(self.nsp):
            # heat flux q and particle flux gammas(g)
            self.sp[isp].q = np.squeeze(self.sp[isp].para_energy + self.sp[isp].perp_energy) / self.dt / self.area
            self.sp[isp].g = np.squeeze(self.sp[isp].number) / self.dt / self.area

    def update_total_flux(self):
        """Update total heat flux and particle flux."""
        self.g_total = 0
        self.q_total = 0
        for isp in range(self.nsp):
            self.g_total += self.sp[isp].g
            self.q_total += self.sp[isp].q

    def eich_fit_all(self, pmask=None):
        """Perform fitting for all time steps."""
        self.lq_eich = np.zeros_like(self.time)  # mem allocation
        self.S_eich = np.zeros_like(self.lq_eich)
        self.dsep_eich = np.zeros_like(self.lq_eich)
        
        for i in range(self.time.size):
            try:
                popt, pconv = self.eich_fit1(self.q_total[i, :], pmask=pmask)
            except:
                popt = [0, 0, 1, 0]
                
            self.lq_eich[i] = popt[2]
            self.S_eich[i] = popt[1]
            self.dsep_eich[i] = popt[3]

    def get_divertor(self, outer=True, lower=True):
        """
        Get array index for inner and outer divertor.
        Assume the array index is counter-clockwise. --> need to consider the opposite cases
        """
        # find minimum psi location
        sign_z = 1 if lower else -1
        mask = (self.z - self.eq_axis_z) * sign_z < 0
        i0 = np.argmin(np.where(mask, self.psin, np.inf))

        # find maximum psi location
        sign_r = 1 if outer else -1
        mask = (self.r - self.eq_axis_r) * sign_r > 0
        i1 = np.argmax(np.where(mask, self.psin, -np.inf))

        return i0, i1

    def eich(self, xdata, q0, s, lq, dsep):
        """
        Functions for eich fit
        q(x) =0.5*q0* exp( (0.5*s/lq)^2 - (x-dsep)/lq ) * erfc (0.5*s/lq - (x-dsep)/s)
        """
        return 0.5 * q0 * np.exp((0.5 * s / lq)**2 - (xdata - dsep) / lq) * erfc(0.5 * s / lq - (xdata - dsep) / s)

    def eich_fit1(self, ydata, pmask=None):
        """Eich fitting of one profile data."""
        q0init = np.max(ydata)
        sinit = 2  # 2mm
        lqinit = 1  # 1mm
        dsepinit = 0.1  # 0.1 mm

        p0 = np.array([q0init, sinit, lqinit, dsepinit])
        if pmask is None:
            pmask = slice(0, ydata.shape[0])

        r_data = self.rmidsepmm[pmask]
        y_data = ydata[pmask]
        try:
            popt, pconv = curve_fit(self.eich, r_data, y_data, p0=p0)
        except:
            popt = [0, 1, 1, 0]
            pconv = np.zeros_like(p0)

        return popt, pconv

    def lambda_q3(self, x, q0, qf, lp, ln, lf, dsep):
        """
        Functions for 3 lambda fit: lp (lambda_q of private flux region),
        ln (lambda_q of near SOL), lf (lambda_q of far SOL)
        q(x) =     q0 * exp( (x-dsep)/lp)   when x<dsep
             =(q0-qf) * exp(-(x-dsep)/ln) + qf * exp(-(x-dsep)/lf) when x>dsep
        """
        dsepl = 0  # not using dsep --> dsepl=dsep to use
        rtn = q0 * np.exp((x - dsepl) / lp)  # only x<dsep will be used.
        ms = np.nonzero(x >= dsepl)
        rtn[ms] = (q0 - qf) * np.exp(-(x[ms] - dsepl) / ln) + qf * np.exp(-(x[ms] - dsepl) / lf)
        return rtn

    def lambda_q3_bound(self, x, q0, qf, lp, ln, delta_l, dsep):
        """
        Parameter space is q0, qf, lp, ln, delta_l (= lf - ln), dsep
        """
        # dsepl = 0  # not using dsep --> dsepl=dsep to use
        dsepl = dsep
        rtn = q0 * np.exp((x - dsepl) / lp)  # only x<dsep will be used.
        ms = np.nonzero(x >= dsepl)
        rtn[ms] = (q0 - qf) * np.exp(-(x[ms] - dsepl) / ln) + qf * np.exp(-(x[ms] - dsepl) / (ln + delta_l))
        return rtn

    def lambda_q3_fit1(self, ydata, pmask):
        """3 lambda_q fitting of one profile data."""
        q0init = np.max(ydata)
        qfinit = 0.01 * q0init  # 1 percent
        lpinit = 1  # 1mm
        lninit = 1  # 2mm
        lfinit = 3  # 4mm
        dsepinit = 0.01  # 0.01 mm

        p0 = np.array([q0init, qfinit, lpinit, lninit, lfinit - lninit, dsepinit])
        if pmask is None:
            pmask = slice(0, ydata.shape[0])
        r_data = self.rmidsepmm[pmask]
        y_data = ydata[pmask]
        try:
            bounds = ([0, 0, 0, 0, 0.1, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
            popt, pconv = curve_fit(self.lambda_q3_bound, r_data, y_data, p0=p0, bounds=bounds)
        except:
            popt = [0, 0, 1, 1, 1, 0]
            pconv = np.zeros_like(p0)

        return popt, pconv

    def total_heat(self, wedge_n, pmask=None):
        """
        Getting total heat (radially integrated) to inner/outer divertor.
        """
        if pmask is None:
            pmask = np.ones_like(self.rmidsepmm, dtype=bool)

        for isp in range(self.nsp):
            self.sp[isp].q_para_sum = np.sum(self.sp[isp].para_energy[:, :, pmask], axis=(1, 2))[:, np.newaxis] * wedge_n / self.dt
            self.sp[isp].q_perp_sum = np.sum(self.sp[isp].perp_energy[:, :, pmask], axis=(1, 2))[:, np.newaxis] * wedge_n / self.dt
            self.sp[isp].q_sum = self.sp[isp].q_para_sum + self.sp[isp].q_perp_sum
            self.sp[isp].g_sum = np.sum(self.sp[isp].number[:, :, pmask], axis=(1, 2))[:, np.newaxis] * wedge_n / self.dt


def load_heatdiag(xgc_instance, **kwargs):
    """Load and process heat diagnostic data."""
    read_rz_all = kwargs.get('read_rz_all', False)

    xgc_instance.hl = []
    xgc_instance.hl.append(datahlp(xgc_instance.path + "xgc.heatdiag.bp", 0, read_rz_all))
    xgc_instance.hl.append(datahlp(xgc_instance.path + "xgc.heatdiag.bp", 1, read_rz_all))

    # Check species availability
    for i in [0, 1]:
        try: 
            _ = xgc_instance.hl[i].e_perp_energy_psi
            xgc_instance.hl[i].electron_on = True
        except: 
            xgc_instance.hl[i].electron_on = False

        try: 
            _ = xgc_instance.hl[i].i2perp_energy_psi
            xgc_instance.hl[i].ion2_on = True
        except: 
            xgc_instance.hl[i].ion2_on = False

    # Normalize psi
    for i in [0, 1]:
        try:
            xgc_instance.hl[i].psin = xgc_instance.hl[i].psi[-1, :] / xgc_instance.psix
        except:
            print("psix is not defined - call load_units() to get psix to get psin")

    # Process with bfield data if available
    _process_heatdiag_with_bfield(xgc_instance)


def load_heatdiag2(xgc_instance):
    """Load and process heat diagnostic v2 data."""
    xgc_instance.hl2 = datahl2(xgc_instance.path + "xgc.heatdiag2.bp", datahl2_sp)
    
    # Post process
    wedge_n = xgc_instance.unit_dic['sml_wedge_n']
    it = -1  # keep the last one
    xgc_instance.hl2.psin = xgc_instance.hl2.psi[it, :] / xgc_instance.psix
    
    # Area of each segment with angle factor
    xgc_instance.hl2.area = np.pi * xgc_instance.hl2.r[it, :] * xgc_instance.hl2.ds[it, :] / wedge_n * np.cos(xgc_instance.hl2.strike_angle[it, :])
    xgc_instance.hl2.area = xgc_instance.hl2.area[np.newaxis, :]

    # Get midplane conversion
    try:
        # use bfieldm if loaded
        if hasattr(xgc_instance, 'bfm'):
            psino = xgc_instance.bfm.psino
            rmido = xgc_instance.bfm.rmido
        else:  # get it from xgc.mesh.bp
            print("Warning: bfm not loaded, attempting to use mesh data")
            if hasattr(xgc_instance, 'mesh') and hasattr(xgc_instance, 'midplane_var'):
                psino, tmp, rmido = xgc_instance.midplane_var(xgc_instance.mesh.r, return_rmid=True)
                # both tmp and rmido gives midplane r, but rmido is before the mesh interpolation.
            else:
                print("Warning: using simplified midplane approach")
                psino = xgc_instance.hl2.psin
                rmido = xgc_instance.hl2.r[it, :]

        xgc_instance.hl2.rs, xgc_instance.hl2.rmidsepmm = xgc_instance.hl2.get_midplane_conversion(psino, rmido)
        xgc_instance.hl2.get_parallel_flux()
        xgc_instance.hl2.update_total_flux()

    except Exception as e:
        print(f"Warning: Could not process midplane conversion: {e}")

    # Set equilibrium parameters
    xgc_instance.hl2.eq_axis_r = xgc_instance.eq_axis_r
    xgc_instance.hl2.eq_axis_z = xgc_instance.eq_axis_z


def report_heatdiag2(xgc_instance, is_outer=True, is_lower=True, it=-1, xlim=[-5, 15],
                     lq_ylim=[0, 10], ndata=1000000, fit_mask=None,
                     sp_names=['e', 'i', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9']):
    """
    Report basic analysis of heatdiag2.bp
    Need to specify the divertor region
    ndata is maximum number of data point to be considered.
    fit_mask is the mask for fitting. If None, all data will be used.
    """
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for report_heatdiag2")
        return None

    # select divertor
    i0, i1 = xgc_instance.hl2.get_divertor(outer=is_outer, lower=is_lower)
    sign = 1 if (i0 < i1) else -1
    i1 = i0 + sign * ndata if np.abs(i1 - i0) > ndata else i1

    md = np.arange(i0, i1, sign)
    fig, ax = plt.subplots()
    plt.plot(xgc_instance.hl2.r[0, :], xgc_instance.hl2.z[0, :])
    plt.plot(xgc_instance.hl2.r[0, md], xgc_instance.hl2.z[0, md], 'r-', linewidth=4, label='Divertor')
    plt.legend()
    if hasattr(xgc_instance, 'show_sep'):
        xgc_instance.show_sep(ax, style=',')
    plt.axis('equal')

    # plot total heat flux
    xgc_instance.hl2.total_heat(xgc_instance.unit_dic['sml_wedge_n'], pmask=md)
    plt.subplots()
    for isp in range(len(xgc_instance.hl2.sp)):
        plt.plot(xgc_instance.hl2.time * 1E3, xgc_instance.hl2.sp[isp].q_sum / 1E6, '.', label=sp_names[isp])
    plt.xlabel('Time (ms)')
    plt.ylabel('Total Heat Flux (MW)')
    plt.legend()

    # heat flux profile
    plt.subplots()
    for isp in range(len(xgc_instance.hl2.sp)):
        plt.plot(xgc_instance.hl2.rmidsepmm[md], xgc_instance.hl2.sp[isp].q[it, md] / 1E6, label=sp_names[isp])
    plt.plot(xgc_instance.hl2.rmidsepmm[md], xgc_instance.hl2.q_total[it, md] / 1E6, label='Total')

    plt.xlim(xlim[0], xlim[1])
    plt.ylabel('Parallel heat flux [MW/$m^2$] at the divertor')
    plt.xlabel('Midplane distance from separatrix [mm]')
    plt.legend()

    # fitting one time step
    if fit_mask is None:
        fit_mask = md
    popt, pconv = xgc_instance.hl2.eich_fit1(xgc_instance.hl2.q_total[it, :], pmask=fit_mask)

    eich = xgc_instance.hl2.eich(xgc_instance.hl2.rmidsepmm[fit_mask], popt[0], popt[1], popt[2], popt[3])
    plt.subplots()
    plt.plot(xgc_instance.hl2.rmidsepmm[fit_mask], xgc_instance.hl2.q_total[it, fit_mask], label='XGC')
    plt.plot(xgc_instance.hl2.rmidsepmm[fit_mask], eich, label='Eich Fit')
    plt.xlim(xlim[0], xlim[1])
    plt.title('$\\lambda_q$ = %3.3f mm, S=%3.3f mm, t=%3.3f ms' % (popt[2], popt[1], xgc_instance.hl2.time[it] * 1E3))
    plt.ylabel('Parallel heat flux [W/$m^2$] at the divertor')
    plt.xlabel('Midplane distance from separatrix [mm]')
    plt.legend()

    xgc_instance.hl2.eich_fit_all(pmask=fit_mask)
    plt.subplots()
    plt.plot(xgc_instance.hl2.time * 1E3, xgc_instance.hl2.lq_eich, '.', label='$\\lambda_q$')
    plt.plot(xgc_instance.hl2.time * 1E3, xgc_instance.hl2.S_eich, '.', label='S')
    plt.ylim(lq_ylim[0], lq_ylim[1])
    plt.xlabel('Time [ms]')
    plt.ylabel('$\\lambda_q$, S [mm]')
    plt.legend()

    return md


def get_midplane_bp_sep_and_eich_scale(xgc_instance):
    """
    Get midplane Bp and Eich scale #14
    lambda_q = C * Bp^s
    C = 0.63
    s = -1.19
    """
    # set constants
    C = 0.63
    s = -1.19

    if not hasattr(xgc_instance, 'bfield'):
        from .field_data import load_bfield
        load_bfield(xgc_instance)

    bp_mesh = np.sqrt(xgc_instance.bfield[0, :]**2 + xgc_instance.bfield[1, :]**2)

    if hasattr(xgc_instance, 'midplane_var'):
        psi_mid, bp_mid = xgc_instance.midplane_var(bp_mesh)
    else:
        print("Warning: midplane_var not available, using simplified approach")
        # Simple fallback - this may not be accurate
        return None, None

    # Interpolate bp_mid at psi_mid using 1d interpolation
    psi_sep = [1]
    bp_mid_sep = np.interp(psi_sep, psi_mid, bp_mid)
    lq_eich_scale = C * bp_mid_sep**s
    return bp_mid_sep, lq_eich_scale


def find_tmask(step, max_end=False):
    """
    Find time mask for step selection.
    This handles finding unique time steps in reverse order.
    """
    if max_end:  # determine end point
        ed = np.max(step)  # maximum time step
    else:
        ed = step[-1]  # ending time step

    tmask_rev = []
    p = step.size
    for i in range(ed, 0, -1):  # reverse order
        m = np.nonzero(step[0:p] == i)  # find index that has step i
        try:
            p = m[0][-1]  # exclude zero size
        except:
            pass
        else:
            tmask_rev.append(p)  # only append that has step number
    # tmask is reverse order
    tmask = tmask_rev[::-1]
    return tmask


def _process_heatdiag_with_bfield(xgc_instance):
    """Internal function to process heat diagnostics with bfield data."""
    try:
        from .field_data import load_bfieldm
        load_bfieldm(xgc_instance)

        wedge_n = xgc_instance.unit_dic['sml_wedge_n']
        for i in [0, 1]:
            dpsin = xgc_instance.hl[i].psin[1] - xgc_instance.hl[i].psin[0]
            ds = dpsin / xgc_instance.bfm.dpndrs * 2 * 3.141592 * xgc_instance.bfm.r0 / wedge_n
            xgc_instance.hl[i].rmid = np.interp(xgc_instance.hl[i].psin, xgc_instance.bfm.psino, xgc_instance.bfm.rmido)
            xgc_instance.hl[i].post_heatdiag(ds)
            xgc_instance.hl[i].total_heat(wedge_n)

    except Exception as e:
        print(f"Warning: Could not process bfield data: {e}")