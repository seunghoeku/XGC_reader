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
    """Get radial flux of energy and particle from 1D data."""
    # Load volume data if not loaded
    if not hasattr(xgc_instance, "vol"):
        xgc_instance.load_volumes()
    
    if not hasattr(xgc_instance, "od"):
        raise ValueError("1D data not loaded. Call load_oned() first.")

    # Get dpsi
    pmks = xgc_instance.od.psi_mks[0, :]
    dpsi = np.zeros_like(pmks)
    dpsi[1:-1] = 0.5 * (pmks[2:] - pmks[0:-2])
    dpsi[0] = dpsi[1]
    dpsi[-1] = dpsi[-2]
    xgc_instance.od.dvdp = xgc_instance.vol.od / dpsi
    xgc_instance.od.dpsi = dpsi
    
    dvdpall = xgc_instance.od.dvdp * xgc_instance.sml_wedge_n
    
    # Ion flux
    xgc_instance.od.efluxi = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_en_flux_df_1d * dvdpall
    xgc_instance.od.efluxexbi = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_en_flux_ExB_df_1d * dvdpall
    
    # Check for additional flux components
    if hasattr(xgc_instance.od, 'i_radial_en_flux_3db_df_1d'):
        xgc_instance.od.eflux3dbi = xgc_instance.od.i_gc_density_df_1d * xgc_instance.od.i_radial_en_flux_3db_df_1d * dvdpall

    # Electron flux (if available)
    if hasattr(xgc_instance, 'electron_on') and xgc_instance.electron_on:
        xgc_instance.od.efluxe = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.e_radial_en_flux_df_1d * dvdpall
        xgc_instance.od.efluxexbe = xgc_instance.od.e_gc_density_df_1d * xgc_instance.od.e_radial_en_flux_ExB_df_1d * dvdpall


def heat_flux_all(xgc_instance):
    """Calculate all heat flux components."""
    radial_flux_all(xgc_instance)
    
    def load_data_slow(self, f, vars, prefix_len):
        """Load data slowly using standard ADIOS2 reads."""
        for v in vars:
            stc = vars[v].get("AvailableStepsCount")
            ct = vars[v].get("Shape")  
            sgl = vars[v].get("SingleValue")
            stc = int(stc)
            
            if sgl == "true":
                setattr(self, v[prefix_len:], f.read(v))
            else:
                setattr(self, v[prefix_len:], f.read(v, start=[], count=[], step_selection=[0, stc]))

    def load_data(self, f, vars, prefix_len):
        """Load data using optimized ADIOS2 bindings."""
        try:
            bIO = f.io.impl  # adios2.bindings.adios2_bindings.IO object
            bEngine = f.engine.impl
            for v in vars:
                bVar = bIO.InquireVariable(v)
                countList = bVar.Count()
                if len(countList) == 0:  # scalar
                    setattr(self, v[prefix_len:], f.read(v))
                else:
                    setattr(self, v[prefix_len:], f.read(v, start=[], count=countList, step_selection=[0, -1]))
        except:
            # Fallback to slow method
            self.load_data_slow(f, vars, prefix_len)
    
    # Add these methods to data1 class
    data1.load_data_slow = load_data_slow
    data1.load_data = load_data