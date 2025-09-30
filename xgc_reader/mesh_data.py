"""Mesh data handling classes."""

import numpy as np
import adios2
from matplotlib.tri import Triangulation
from functools import singledispatchmethod


class meshdata(object):
    """Mesh data class for 2D contour plot."""
    
    @singledispatchmethod
    def __init__(self, path):
        """Initialize from path to data directory."""
        with adios2.FileReader(path + "xgc.mesh.bp") as fm:
            self.load_mesh(fm)

    @__init__.register(adios2.FileReader)
    def _(self, fm: adios2.FileReader):
        """Initialize from already open file/campaign."""
        self.load_mesh(fm, "xgc.mesh.bp/")

    def load_mesh(self, fm: adios2.FileReader, prefix=''):
        """Load mesh data from ADIOS2 file."""
        rz = fm.read(prefix + 'rz')
        self.cnct = fm.read(prefix + 'nd_connect_list')
        self.r = rz[:, 0]
        self.z = rz[:, 1]
        self.triobj = Triangulation(self.r, self.z, self.cnct)
        
        try:
            self.surf_idx = fm.read(prefix + 'surf_idx')
        except:
            print("No surf_idx in xgc.mesh.bp") 
        else:
            self.surf_len = fm.read(prefix + 'surf_len')
            try:
                self.psi_surf = fm.read(prefix + 'psi_surf')
            except:
                print("failed to read psi_surf in xgc.mesh.bp")
                self.psi_surf = np.arange(0, 1, 1 / self.surf_len.size)
            self.theta = fm.read(prefix + 'theta')

        try:
            self.m_max_surf = fm.read(prefix + 'm_max_surf')
        except:
            print("No m_max_surf in xgc.mesh.bp")

        try:
            self.wall_nodes = fm.read(prefix + 'grid_wall_nodes') - 1  # zero based
        except:
            print("No wall_nodes in xgc.mesh.bp")
        
        self.node_vol = fm.read(prefix + 'node_vol')
        self.node_vol_nearest = fm.read(prefix + 'node_vol_nearest')
        self.qsafety = fm.read(prefix + 'qsafety')
        self.psi = fm.read(prefix + 'psi')
        self.epsilon = fm.read(prefix + 'epsilon')
        self.rmin = fm.read(prefix + 'rmin')
        self.rmaj = fm.read(prefix + 'rmaj')
        
        try:
            self.region = fm.read(prefix + 'region')
        except:
            print("No region in xgc.mesh.bp") 
        
        try:
            self.wedge_angle = fm.read(prefix + 'wedge_angle')
        except:
            print("No wedge_angle in xgc.mesh.bp") 
        
        try:
            self.delta_phi = fm.read(prefix + 'delta_phi')
        except:
            print("No delta_phi in xgc.mesh.bp") 

        self.nnodes = np.size(self.r)  # same as n_n 


class f0meshdata(object):
    """F0 mesh data class for background distribution."""
    
    @singledispatchmethod
    def __init__(self, path):
        """Initialize from path to data directory."""
        with adios2.FileReader(path + "xgc.f0.mesh.bp") as f:
            self.load_f0mesh(f)

    @__init__.register(adios2.FileReader)
    def _(self, f: adios2.FileReader):
        """Initialize from already open file/campaign."""
        self.load_f0mesh(f, "xgc.f0.mesh.bp/")

    def load_f0mesh(self, f: adios2.FileReader, prefix=''):
        """Load F0 mesh data from ADIOS2 file."""
        T_ev = f.read(prefix + 'f0_T_ev')
        den0 = f.read(prefix + 'f0_den')
        
        try:
            flow = f.read(prefix + 'f0_flow')
        except:
            print("No flow in xgc.f0.mesh.bp") 
            flow = np.zeros_like(den0)  # zero flow when flow is not written

        if len(den0.shape) > 1:
            self.ni0 = den0[-1, :]
            self.ti0 = T_ev[-1, :]  # last species. need update for multi ion
            self.ui0 = flow[-1, :]
        else:
            # old format
            self.ni0 = den0
            self.ti0 = T_ev
            self.ui0 = flow
            self.te0 = den0
                
        if T_ev.shape[0] >= 2:
            self.te0 = T_ev[0, :]
            try:
                self.ne0 = den0[0, :]
                self.ue0 = flow[0, :]
            except:
                # old format
                self.ne0 = den0
                self.ue0 = flow
        
        if T_ev.shape[0] >= 3:
            print('multi species - ni0, ti0, ui0 are last species')

        self.dsmu = f.read(prefix + 'f0_dsmu')
        self.dvp = f.read(prefix + 'f0_dvp')
        self.smu_max = f.read(prefix + 'f0_smu_max')
        self.vp_max = f.read(prefix + 'f0_vp_max')