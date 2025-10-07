"""Volume data handling."""

import adios2
from functools import singledispatchmethod


class voldata(object):
    """Volume data class for reading 1D volume-averaged diagnostics."""
    
    @singledispatchmethod
    def __init__(self, path):
        """Initialize from path to data directory."""
        with adios2.FileReader(path + "xgc.volumes.bp") as f:
            self._load_volume_data(f)
    
    @__init__.register(adios2.FileReader)
    def _(self, f: adios2.FileReader):
        """Initialize from already open file/campaign."""
        self._load_volume_data(f, "xgc.volumes.bp/")
    
    def _load_volume_data(self, f: adios2.FileReader, prefix=''):
        """Load volume data from ADIOS2 file."""
        try:
            self.od = f.read(prefix + "diag_1d_vol")
        except Exception as e:
            print(f"Warning: Could not read diag_1d_vol from xgc.volumes.bp: {e}")
            self.od = None