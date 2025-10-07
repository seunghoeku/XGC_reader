"""1D diagnostic data handling."""

import numpy as np
import adios2
from functools import singledispatchmethod


class data1(object):
    """Class for reading data files like xgc.oneddiag.bp."""
    
    @singledispatchmethod
    def __init__(self, filename):
        """Initialize from filename."""
        with adios2.FileReader(filename) as f:
            vars = f.available_variables()
            self.load_data(f, vars, 0)

    @__init__.register(adios2.FileReader)
    def _(self, f: adios2.FileReader, vars: dict, filename: str):
        """Initialize from already open file/campaign."""
        vs = {k: v for (k, v) in vars.items() if k.startswith(filename)}
        self.load_data(f, vs, len(filename + "/"))

    def load_data_slow(self, f: adios2.FileReader, vars: dict, prefix_len: int):
        """Load data using slow method (for compatibility)."""
        for v in vars:
            stc = vars[v].get("AvailableStepsCount")
            ct = vars[v].get("Shape")
            sgl = vars[v].get("SingleValue")
            stc = int(stc)
            if ct != '':
                ct = int(ct)
                data = f.read(v, start=[0], count=[ct], step_selection=[0, stc])
                setattr(self, v[prefix_len:], np.reshape(data, [stc, ct]))
            elif v != 'gsamples' and v != 'samples':
                setattr(self, v[prefix_len:], f.read(v, start=[], count=[], step_selection=[0, stc]))

    def load_data(self, f: adios2.FileReader, vars: dict, prefix_len: int):
        """Load data using optimized method."""
        bIO = f.io.impl  # adios2.bindings.adios2_bindings.IO object with C++ like functions
        bEngine = f.engine.impl
        for v in vars:
            bVar = bIO.InquireVariable(v)
            countList = bVar.Count()
            stc = bVar.Steps()
            if countList:
                ct = countList[0]
                # do Deferred Gets for reading many vars at once
                # 'data' will be filled after the PerformGets() call
                data = np.zeros([stc, ct], dtype=np.double)
                setattr(self, v[prefix_len:], data)
                bVar.SetSelection([[0], [ct]]) 
                bVar.SetStepSelection([0, stc]) 
                bEngine.Get(bVar, data, adios2.bindings.Mode.Deferred)
            elif v != 'gsamples' and v != 'samples':
                setattr(self, v[prefix_len:], f.read(v, start=[], count=[], step_selection=[0, stc]))
        bEngine.PerformGets()

    def d_dpsi(self, var, psi):
        """Radial derivative using psi_mks."""
        dvdp = var * 0  # memory allocation
        dvdp[:, 1:-1] = (var[:, 2:] - var[:, 0:-2]) / (psi[:, 2:] - psi[:, 0:-2])
        dvdp[:, 0] = dvdp[:, 1]
        dvdp[:, -1] = dvdp[:, -2]
        return dvdp


def radial_flux_all(xgc_instance):
    
    #load volume data
    if(not hasattr(xgc_instance,"vol")):
        #self.vol=self.voldata(self.path)
        xgc_instance.load_volumes()
    
    #check reading oneddiag?
    
    #get dpsi
    pmks = xgc_instance.od.psi_mks[0, :]
    dpsi = np.zeros_like(pmks)
    dpsi[1:-1] = 0.5 * (pmks[2:] - pmks[0:-2])
    dpsi[0] = dpsi[1]
    dpsi[-1] = dpsi[-2]
    xgc_instance.od.dvdp = xgc_instance.vol.od / dpsi
    xgc_instance.od.dpsi = dpsi

    nt = xgc_instance.od.time.size
    ec = 1.6E-19  # electron charge
    dvdpall = xgc_instance.od.dvdp * xgc_instance.sml_wedge_n

    # ion flux
    xgc_instance.od.efluxi    = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_en_flux_df_1d * dvdpall
    xgc_instance.od.efluxexbi = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_en_flux_ExB_df_1d * dvdpall
    if hasattr(xgc_instance.od, 'i_radial_en_flux_3db_df_1d'):
        xgc_instance.od.eflux3dbi = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_en_flux_3db_df_1d * dvdpall

    xgc_instance.od.cfluxi    = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.Ti * ec * xgc_instance.od.i_radial_flux_df_1d * dvdpall
    xgc_instance.od.cfluxexbi = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.Ti * ec * xgc_instance.od.i_radial_flux_ExB_df_1d * dvdpall
    xgc_instance.od.pfluxi    = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_flux_df_1d * dvdpall
    xgc_instance.od.pfluxexbi = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_flux_ExB_df_1d * dvdpall

    xgc_instance.od.mfluxi    = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_mom_flux_df_1d * dvdpall # toroidal momentum flux
    xgc_instance.od.mfluxexbi = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_mom_flux_ExB_df_1d * dvdpall

    if xgc_instance.electron_on:
        xgc_instance.od.efluxe    = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.e_radial_en_flux_df_1d * dvdpall
        xgc_instance.od.efluxexbe = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.e_radial_en_flux_ExB_df_1d * dvdpall
        if hasattr(xgc_instance.od, 'e_radial_en_flux_3db_df_1d'):
            xgc_instance.od.eflux3dbe = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.e_radial_en_flux_3db_df_1d * dvdpall

        xgc_instance.od.cfluxe    = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.Te * ec * xgc_instance.od.e_radial_flux_df_1d * dvdpall
        xgc_instance.od.cfluxexbe = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.Te * ec * xgc_instance.od.e_radial_flux_ExB_df_1d * dvdpall
        xgc_instance.od.pfluxe    = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.e_radial_flux_df_1d * dvdpall
        xgc_instance.od.pfluxexbe = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.e_radial_flux_ExB_df_1d * dvdpall

    if xgc_instance.ion2_on:
        xgc_instance.od.efluxi2    = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.i2radial_en_flux_df_1d * dvdpall
        xgc_instance.od.efluxexbi2 = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.i2radial_en_flux_ExB_df_1d * dvdpall
        if hasattr(xgc_instance.od, 'i_radial_en_flux_3db_df_1d'):
            xgc_instance.od.eflux3dbi2 = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.i2radial_en_flux_3db_df_1d * dvdpall

        xgc_instance.od.cfluxi2    = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.Ti2 * ec * xgc_instance.od.i2radial_flux_df_1d * dvdpall
        xgc_instance.od.cfluxexbi2 = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.Ti2 * ec * xgc_instance.od.i2radial_flux_ExB_df_1d * dvdpall
        xgc_instance.od.pfluxi2    = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.i2radial_flux_df_1d * dvdpall
        xgc_instance.od.pfluxexbi2 = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.i2radial_flux_ExB_df_1d * dvdpall
        xgc_instance.od.mfluxi2    = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.i2radial_mom_flux_df_1d * dvdpall
        xgc_instance.od.mfluxexbi2 = xgc_instance.od.i2gc_density_df_1d * xgc_instance.od.i2radial_mom_flux_ExB_df_1d * dvdpall    



def heat_flux_all(xgc_instance):
    """Calculate all heat flux components."""
    radial_flux_all(xgc_instance)