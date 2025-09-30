"""Field data handling classes."""

import numpy as np
import adios2
from functools import singledispatchmethod


class databfm(object):
    """B-field data class."""
    
    @singledispatchmethod
    def __init__(self, path):
        """Initialize from path to data directory."""
        with adios2.FileReader(path + "xgc.bfieldm.bp") as f:
            self._load_bfield_data(f)
    
    @__init__.register(adios2.FileReader)
    def _(self, f: adios2.FileReader):
        """Initialize from already open file/campaign."""
        self._load_bfield_data(f, "xgc.bfieldm.bp/")
    
    def _load_bfield_data(self, f: adios2.FileReader, prefix=''):
        """Load B-field data from ADIOS2 file."""
        self.vars = f.available_variables()
        
        # Read major radius data
        if prefix + 'rmajor' in self.vars:
            v = prefix + 'rmajor'
        else:
            v = prefix + '/bfield/rvec'
        self.rmid = f.read(v)
        
        # Read normalized psi data  
        if prefix + 'psi_n' in self.vars:
            v = prefix + 'psi_n'
        else:
            v = prefix + '/bfield/psi_eq_x_psi'
        self.psin = f.read(v)

    def process_midplane_data(self, unit_dic):
        """Process midplane data for heat load analysis."""
        self.r0 = unit_dic['eq_axis_r']
        
        # Get outside midplane only
        n0 = np.nonzero(self.rmid > self.r0)[0][0]
        self.rmido = self.rmid[n0:]
        self.psino = self.psin[n0:]

        # Find separatrix index and r
        msk = np.argwhere(self.psino > 1)
        n0 = msk[1] if len(msk) > 1 else msk[0]
        self.rs = self.rmido[n0]

        # Get dpdr (normalized psi) at separatrix
        self.dpndrs = (self.psino[n0] - self.psino[n0 - 1]) / (self.rmido[n0] - self.rmido[n0 - 1])

        self.rminor = self.rmido - self.r0


def load_bfieldm(xgc_instance):
    """Load magnetic field midplane data."""
    if xgc_instance.campaign:
        xgc_instance.bfm = databfm(xgc_instance.campaign)
    else:
        xgc_instance.bfm = databfm(xgc_instance.path)
    
    xgc_instance.bfm.process_midplane_data(xgc_instance.unit_dic)


def load_bfield(xgc_instance):
    """Load xgc.bfield.bp -- equilibrium bfield."""
    with adios2.FileReader(xgc_instance.path + "xgc.bfield.bp") as f:
        try:
            xgc_instance.bfield = f.read('bfield')
        except:  # try older version of bfield
            try:
                xgc_instance.bfield = f.read('/bfield')
            except Exception as e:
                print(f"Could not read bfield data: {e}")
                xgc_instance.bfield = None