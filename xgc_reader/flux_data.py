"""Flux averaging and turbulence data handling."""

import numpy as np
import adios2
from functools import singledispatchmethod


class fluxavg(object):
    """Flux surface averaging class."""
    
    @singledispatchmethod 
    def __init__(self, path):
        """Initialize from path to data directory."""
        with adios2.FileReader(path + "xgc.fluxavg.bp") as f:
            self._load_flux_data(f)
    
    @__init__.register(adios2.FileReader)
    def _(self, f: adios2.FileReader):
        """Initialize from already open file/campaign."""
        self._load_flux_data(f, "xgc.fluxavg.bp/")
    
    def _load_flux_data(self, f: adios2.FileReader, prefix=''):
        """Load flux averaging data from ADIOS2 file."""
        try:
            self.eindex = f.read(prefix + 'eindex')
            self.nelement = f.read(prefix + 'nelement') 
            self.npsi = f.read(prefix + 'npsi')
            self.value = f.read(prefix + 'value')
        except Exception as e:
            print(f"Warning: Could not read flux averaging data: {e}")
            self.eindex = None
            self.nelement = None
            self.npsi = None
            self.value = None


class turbdata(object):
    """
    Data for turbulence intensity analysis.
    Note: Marked as obsolete in original, needs replacement.
    """
    
    def __init__(self, istart, iend, istep, midwidth, mesh, f0):
        """
        Initialize turbulence data analysis.
        
        Parameters
        ----------
        istart, iend, istep : int
            Time step range and increment
        midwidth : float
            Midplane width parameter
        mesh : meshdata
            Mesh data object
        f0 : f0meshdata
            Background distribution data
        """
        # Setup flux surface average
        self.midwidth = midwidth
        self.istart = istart
        self.iend = iend
        self.istep = istep
        
        # Initialize storage
        self.dpot_te_sqr = []
        self.dn_n0_sqr = []
        
        # Read whole data
        for i in range(istart, iend, istep):
            # 3D file name
            filename = "xgc.3d.%5.5d.bp" % (i)
            
            try:
                # Read data
                with adios2.FileReader(filename) as f:
                    dpot = f.read("dpot")
                    dden = f.read("eden")
                    
                    nzeta = dpot.shape[0]
                    print(f"Processing time step {i}, nzeta = {nzeta}")
                    
                    dpotn0 = np.mean(dpot, axis=0)
                    dpot = dpot - dpotn0  # numpy broadcasting
                    
                    # Toroidal average of (dpot/Te)^2
                    var_pot = np.mean(dpot**2, axis=0) / f0.te0**2
                    
                    # Remove n=0 mode from density
                    dden = dden - np.mean(dden, axis=0)
                    var_den = dpot / f0.te0 + dden / f0.ne0
                    var_den = np.mean(var_den**2, axis=0)  # toroidal average
                    
                    # Store results (would need flux surface averaging implementation)
                    self.dpot_te_sqr.append(var_pot)
                    self.dn_n0_sqr.append(var_den)
                    
            except Exception as e:
                print(f"Warning: Could not read 3D data from {filename}: {e}")
                continue