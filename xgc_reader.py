"""Module of the XGC1 loader for regerating general plots using ADIOS2
Some parts are taken from Michael's xgc.py which is taken from Loic's load_XGC_local for BES.
It reads the data from the simulation especially 1D results and other small data output.

TODO
3D data are loaded only when it is specified.
"""

import numpy as np
import os
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
import adios2
import matplotlib.pyplot as plt
from scipy.io import matlab
from scipy.optimize import curve_fit
from scipy.special import erfc
import scipy.sparse as sp
from tqdm.auto import trange, tqdm

class xgc1(object):
    
    class cnst:
        echarge = 1.602E-19
        protmass=  1.67E-27
        mu0 = 4 * 3.141592 * 1E-7

        
    def __init__(self):
        """ 
        initialize it from the current directory.
        not doing much thing. 
        """        
        self.path=os.getcwd()+'/'
    

    def load_unitsm(self):
        """
        read in units file
        """
        self.unit_file = self.path+'units.m'
        self.unit_dic = self.load_m(self.unit_file) #actual reading routine
        self.psix=self.unit_dic['psi_x']
        self.eq_x_r = self.unit_dic['eq_x_r']
        self.eq_x_z = self.unit_dic['eq_x_z']
        self.eq_axis_r = self.unit_dic['eq_axis_r']
        self.eq_axis_z = self.unit_dic['eq_axis_z']
        self.eq_axis_b = self.unit_dic['eq_axis_b']
        self.sml_dt = self.unit_dic['sml_dt']
        self.sml_wedge_n = self.unit_dic['sml_wedge_n']
        self.diag_1d_period = self.unit_dic['diag_1d_period']

    def load_oned(self, i_mass=2, i2mass=12):
        """
        load xgc.oneddiag.bp and some post process
        """
        self.od=self.data1("xgc.oneddiag.bp") #actual reading routine
        self.od.psi=self.od.psi[0,:]
        self.od.psi00=self.od.psi00[0,:]
        try:
            self.od.psi00n=self.od.psi00/self.psix #Normalize 0 - 1(Separatrix)
        except:
            print("psix is not defined - call load_unitsm() to get psix to get psi00n")
        # Temperatures
        try: 
            Teperp=self.od.e_perp_temperature_df_1d
        except:
            print('No electron')
            self.electron_on=False
        else:
            self.electron_on=True
            Tepara=self.od.e_parallel_mean_en_df_1d  #parallel flow ignored, correct it later
            self.od.Te=(Teperp+Tepara)/3*2
        
        #minority or impurity tempearture
        try: 
            Ti2perp=self.od.i2perp_temperature_df_1d
        except:
            print('No Impurity')
            self.ion2_on=False
        else:
            self.ion2_on=True
            Ti2para=self.od.i2parallel_mean_en_df_1d  - 0.5* i2mass * self.cnst.protmass * self.od.i2parallel_flow_df_1d**2 / self.cnst.echarge
            self.od.Ti2=(Ti2perp+Ti2para)/3*2

        Tiperp=self.od.i_perp_temperature_df_1d
        Tipara=self.od.i_parallel_mean_en_df_1d - 0.5* i_mass * self.cnst.protmass * self.od.i_parallel_flow_df_1d**2 / self.cnst.echarge  #parallel flow ignored, correct it later
        self.od.Ti=(Tiperp+Tipara)/3*2

        #ExB shear calculation
        if(self.electron_on):
            shear=self.od.d_dpsi(self.od.e_poloidal_ExB_flow_1d,self.od.psi_mks)
            self.od.grad_psi_sqr = self.od.e_grad_psi_sqr_1d
        else:
            shear=self.od.d_dpsi(self.od.i_poloidal_ExB_flow_1d,self.od.psi_mks)
            self.od.grad_psi_sqr = self.od.i_grad_psi_sqr_1d
        self.od.shear_r=shear * np.sqrt(self.od.grad_psi_sqr)  # assuming electron full-f is almost homogeneouse

        if(self.electron_on):
            self.od.density = self.od.e_gc_density_df_1d
        else:
            self.od.density = self.od.i_gc_density_df_1d

        #gradient scale
        self.od.Ln = self.od.density / self.od.d_dpsi(self.od.density, self.od.psi_mks) / np.sqrt(self.od.grad_psi_sqr)
        self.od.Lti =self.od.Ti      / self.od.d_dpsi(self.od.Ti     , self.od.psi_mks) / np.sqrt(self.od.grad_psi_sqr)
        if(self.electron_on):
            self.od.Lte =self.od.Te  / self.od.d_dpsi(self.od.Te     , self.od.psi_mks) / np.sqrt(self.od.grad_psi_sqr)
            
        #plasma beta (electron)
        # (e n T) / (B^2/2mu0)
        try:
            self.od.beta_e= self.cnst.echarge *self.od.density*self.od.Te /(self.eq_axis_b**2*0.5/self.cnst.mu0)
        except:
            print ('electron beta calculation failed. No electron? units.m not loaded?')

        #find tmask
        d=self.od.step[1]-self.od.step[0]
        st=self.od.step[0]/d
        ed=self.od.step[-1]/d
        st=st.astype(int)
        ed=ed.astype(int)
        idx=np.arange(st,ed, dtype=int)

        self.od.tmask=idx  #mem allocation
        for i in idx:
            tmp=np.argwhere(self.od.step==i*d)
            #self.od.tmask[i-st/d]=tmp[-1,-1]   #LFS zero based, RHS last element
            self.od.tmask[i-st]=tmp[-1,-1]   #LFS zero based, RHS last element

    """ 
        class for reading data file like xgc.oneddiag.bp
        Trying to be general, but used only for xgc.onedidag.bp
    """
    class data1(object):
        def __init__(self,filename):
            with adios2.open(filename,"rra") as self.f:
                #read file and assign it
                self.vars=self.f.available_variables()
                for v in self.vars:
                    stc=self.vars[v].get("AvailableStepsCount")
                    ct=self.vars[v].get("Shape")
                    sgl=self.vars[v].get("SingleValue")
                    stc=int(stc)
                    if ct!='':
                        ct=int(ct)
                        setattr(self,v,self.f.read(v,start=[0], count=[ct], step_start=0, step_count=stc))
                    elif v!='gsamples' and v!='samples' :
                        setattr(self,v,self.f.read(v,start=[], count=[], step_start=0, step_count=stc)) #null list for scalar
        def d_dpsi(self,var,psi):
            """
            radial derivative using psi_mks.
            """
            dvdp=var*0; #memory allocation
            dvdp[:,1:-1]=(var[:,2:]-var[:,0:-2])/(psi[:,2:]-psi[:,0:-2])
            dvdp[:,0]=dvdp[:,1]
            dvdp[:,-1]=dvdp[:,-2]
            return dvdp

    """
    class for head load diagnostic output.
    Only psi space data currently?
    """
    class datahlp(object):
        def __init__(self,filename,irg, read_rz_all=False):
            with adios2.open(filename,"r") as self.f:
                #irg is region number 0,1 - outer, inner
                #read file and assign it
                self.vars=self.f.available_variables()
                for v in self.vars:
                    stc=self.vars[v].get("AvailableStepsCount")
                    ct=self.vars[v].get("Shape")
                    sgl=self.vars[v].get("SingleValue")
                    stc=int(stc)
                    if ct!='':
                        c=[int(i) for i in ct.split(',')]  #
                        if len(c)==1 :  # time and step 
                            setattr(self,v,self.f.read(v,start=[0], count=c, step_start=0, step_count=stc))
                        elif len(c)==2 : # c[0] is irg
                            setattr(self,v,np.squeeze(self.f.read(v,start=[irg,0], count=[1,c[1]], step_start=0, step_count=stc)))
                        elif ( len(c)==3 & read_rz_all ) : # ct[0] is irg, read only 
                            setattr(self,v,np.squeeze(self.f.read(v,start=[irg,0,0], count=[1,c[1],c[2]], step_start=0, step_count=stc)))
                        elif ( len(c)==3 ) : # read_rz_all is false. ct[0] is irg, read only 
                            setattr(self,v,np.squeeze(self.f.read(v,start=[irg,0,0], count=[1,c[1],c[2]], step_start=stc-1, step_count=1)))
                    elif v!='zsamples' and v!='rsamples':
                        setattr(self,v,self.f.read(v,start=[], count=[], step_start=0, step_count=stc)) #null list for scalar
                #keep last time step
                self.r=self.r[-1,:]
                self.z=self.z[-1,:]
        
        """ 
        get some parameters for plots of heat diag

        """
        def post_heatdiag(self,ds):
            #
            """
                self.hl[i].rmid=np.interp(self.hl[i].psin,self.bfm.psino,self.bfm.rmido)
                self.hl[i].drmid=self.hl[irg].rmid*0 # mem allocation
                self.hl[i].drmid=[1:-1]=(self.hl[i].rmid[2:]-self.hl[i].rmid[0:-2])*0.5
                self.hl[i].drmid[0]=self.hl[i].drmid[1]
                self.hl[i].drmid[-1]=self.hl[i].drmid[-2]
            """
            self.drmid=self.rmid*0 # mem allocation
            self.drmid[1:-1]=(self.rmid[2:]-self.rmid[0:-2])*0.5
            self.drmid[0]=self.drmid[1]
            self.drmid[-1]=self.drmid[-2]

            dt = np.zeros_like(self.time)
            dt[1:] = self.time[1:] - self.time[0:-1]
            dt[0] = dt[1]
            rst=np.nonzero(dt<0)  #index when restat happen
            dt[rst]=dt[rst[0]+1]
            self.dt = dt

            #get separatrix r
            self.rs=np.interp([1],self.psin,self.rmid)
            
            self.rmidsepmm=(self.rmid-self.rs)*1E3  # dist from sep in mm

            #get heat
            self.qe=np.transpose(self.e_perp_energy_psi + self.e_para_energy_psi)/dt/ds
            self.qi=np.transpose(self.i_perp_energy_psi + self.i_para_energy_psi)/dt/ds
            self.ge=np.transpose(self.e_number_psi)/dt/ds
            self.gi=np.transpose(self.i_number_psi)/dt/ds

            self.qe = np.transpose(self.qe)
            self.qi = np.transpose(self.qi)
            self.ge = np.transpose(self.ge)
            self.gi = np.transpose(self.gi)

            if(self.ion2_on):
                self.qi2=np.transpose(self.i2perp_energy_psi + self.i2para_energy_psi)/dt/ds
                self.gi2=np.transpose(self.i2number_psi)/dt/ds
                self.qi2 = np.transpose(self.qi2)
                self.gi2 = np.transpose(self.gi2)

            self.qt=self.qe+self.qi
            if(self.ion2_on):
                self.qt=self.qt+self.qi2

            #imx=self.qt.argmax(axis=1)
            mx=np.amax(self.qt,axis=1)
            self.lq_int=mx*0 #mem allocation

            for i in range(mx.shape[0]):
                self.lq_int[i]=np.sum(self.qt[i,:]*self.drmid)/mx[i]

        """
        getting total heat (radially integrated) to inner/outer divertor.
        """
        def total_heat(self,wedge_n):
            qe=wedge_n * (np.sum(self.e_perp_energy_psi,axis=1)+np.sum(self.e_para_energy_psi,axis=1))
            qi=wedge_n * (np.sum(self.i_perp_energy_psi,axis=1)+np.sum(self.i_para_energy_psi,axis=1))
            if(self.ion2_on):
                qi2=wedge_n * (np.sum(self.i2perp_energy_psi,axis=1)+np.sum(self.i2para_energy_psi,axis=1))

            #find restart point and remove -- 

            # find dt in varying sml_dt after restart

            self.qe_tot=qe/self.dt
            self.qi_tot=qi/self.dt
            if(self.ion2_on):
                self.qi2tot=qi2/self.dt

            #compare 2D data 
            #qe2=np.sum(self.e_perp_energy+self.e_para_energy,axis=2)
            #qe2=np.sum(qe2,axis=1)
            #self.qe_tot2=qe2*wedge_n/dt
            #qi2=np.sum(self.i_perp_energy+self.i_para_energy,axis=2)
            #qi2=np.sum(qi2,axis=1)
            #self.qi_tot2=qi2*wedge_n/dt

        """
            Functions for eich fit
            q(x) =0.5*q0* exp( (0.5*s/lq)^2 - (x-dsep)/lq ) * erfc (0.5*s/lq - (x-dsep)/s)
        """
        def eich(self,xdata,q0,s,lq,dsep):
            return 0.5*q0*np.exp((0.5*s/lq)**2-(xdata-dsep)/lq)*erfc(0.5*s/lq-(xdata-dsep)/s)

        """
            Eich fitting of one profile data
        """
        def eich_fit1(self,ydata,pmask):
            q0init=np.max(ydata)
            sinit=2 # 2mm
            lqinit=1 # 1mm
            dsepinit=0.1 # 0.1 mm

            p0=np.array([q0init, sinit, lqinit, dsepinit])
            if(pmask==None):
                popt,pconv = curve_fit(self.eich,self.rmidsepmm,ydata,p0=p0)
            else:
                popt,pconv = curve_fit(self.eich,self.rmidsepmm[pmask],ydata[pmask],p0=p0)

            return popt, pconv

        """
            Smoothing qt before Eich fit
        """
        def qt_smoothing(self,width,order):
            from scipy.signal import savgol_filter

            for i in range(self.time.size):
                tmp = self.qt[i,:]
                self.qt[i,:]=  savgol_filter(tmp,width,order)

        """
            Reset qt from qi and qe
        """
        def qt_reset(self):
            self.qt=self.qe+self.qi
            if(self.ion2_on):
                self.qt=self.qt+self.qi2

        """
            perform fitting for all time steps.
        """
        def eich_fit_all(self,**kwargs):
            # need pmask for generalization?
            pmask = kwargs.get('pmask', None)

            self.lq_eich=np.zeros_like(self.lq_int) #mem allocation
            self.S_eich=np.zeros_like(self.lq_eich)
            self.dsep_eich=np.zeros_like(self.lq_eich)

            for i in range(self.time.size):
                try :
                    popt,pconv = self.eich_fit1(self.qt[i,:],pmask)
                except:
                    popt=[0, 0, 0, 0]
                
                self.lq_eich[i]= popt[2]
                self.S_eich[i] = popt[1]
                self.dsep_eich[i]= popt[3]
    """
        data for bfieldm
    """
    class databfm(object):
        def __init__(self):
            with adios2.open("xgc.bfieldm.bp","r") as self.f:
                self.vars=self.f.available_variables()
                v='/bfield/rvec'
                ct=self.vars[v].get("Shape")
                c=int(ct)
                self.rmid=self.f.read(v,start=[0],count=[c],step_start=0, step_count=1)
                v='/bfield/psi_eq_x_psi'
                ct=self.vars[v].get("Shape")
                c=int(ct)
                self.psin=self.f.read(v,start=[0],count=[c],step_start=0, step_count=1)


    def load_heatdiag(self, **kwargs):
        """
        load xgc.heatdiag.bp and some post process
        """
        read_rz_all = kwargs.get('read_rz_all',False) #read heat load in RZ

        self.hl=[]
        self.hl.append( self.datahlp("xgc.heatdiag.bp",0,read_rz_all) ) #actual reading routine
        self.hl.append( self.datahlp("xgc.heatdiag.bp",1,read_rz_all) )#actual reading routine

        for i in [0,1] :
            try: 
                self.hl[i].e_perp_energy_psi
                self.hl[i].electron_on=True
            except: 
                self.hl[i].electron_on=False

            try: 
                self.hl[i].i2perp_energy_psi
                self.hl[i].ion2_on=True
            except: 
                self.hl[i].ion2_on=False


        for i in [0,1] :
            try:
                self.hl[i].psin=self.hl[i].psi[-1,:]/self.psix #Normalize 0 - 1(Separatrix)
            except:
                print("psix is not defined - call load_unitsm() to get psix to get psin")

        #read bfieldm data if available
        self.load_bfieldm()

        #dt=self.unit_dic['sml_dt']*self.unit_dic['diag_1d_period']
        wedge_n=self.unit_dic['sml_wedge_n']
        for i in [0,1]:
            dpsin=self.hl[i].psin[1]-self.hl[i].psin[0]  #equal dist
            #ds = dR* 2 * pi * R / wedge_n
            ds=dpsin/self.bfm.dpndrs* 2 * 3.141592 * self.bfm.r0 /wedge_n  #R0 at axis is used. should I use Rs?
            self.hl[i].rmid=np.interp(self.hl[i].psin,self.bfm.psino,self.bfm.rmido)
            self.hl[i].post_heatdiag(ds)
            self.hl[i].total_heat(wedge_n)
    """
        Load xgc.bfieldm.bp -- midplane bfield info
    """
    def load_bfieldm(self):
        self.bfm = self.databfm()
        self.bfm.r0=self.unit_dic['eq_axis_r']

        #get outside midplane only
        msk=np.argwhere(self.bfm.rmid>self.bfm.r0)
        n0=msk[0,1]
        self.bfm.rmido=self.bfm.rmid[0,n0:]
        self.bfm.psino=self.bfm.psin[0,n0:]

        #find separtrix index and r
        msk=np.argwhere(self.bfm.psino>1)
        n0=msk[1]
        self.bfm.rs = self.bfm.rmido[n0]

        #get dpdr (normalized psi) at separatrix
        self.bfm.dpndrs = (self.bfm.psino[n0]-self.bfm.psino[n0-1])/(self.bfm.rmido[n0]-self.bfm.rmido[n0-1])

        self.bfm.rminor= self.bfm.rmido - self.bfm.r0

    """
        Load xgc.bfield.bp -- equilibrium bfield 
    """
    #def load_bfield  -- not yet.
    

    """
        load the whole  .m file and return a dictionary contains all the entries.
    """     
    def load_m(self,fname):
        f = open(fname,'r')
        result = {}
        for line in f:
            words = line.split('=')
            key = words[0].strip()
            value = words[1].strip(' ;\n')
            result[key]= float(value)
        f.close()
        return result 

    def plot1d_if(self,obj,**kwargs):
        """
        plot 1D (psi) var of initial and final
        with ylabel of varstr
        Maybe it can be moved to data1 class -- but it might be possible to be used other data type??
        """
        var=kwargs.get('var',None)
        varstr = kwargs.get('varstr', None)
        box = kwargs.get('box', None)
        psi = kwargs.get('psi', None)
        xlim = kwargs.get('xlim', None)
        initial = kwargs.get('initial',True)
        time_legend = kwargs.get('time_legend',True)

        
        if(type(psi).__module__ != np.__name__):  #None or not numpy data
            psi=obj.psi #default psi is obj.psi
            
        if(type(var).__module__ != np.__name__):
            if(varstr==None):   
                print("Either var or varstr should be defined.")
            else:
                var=getattr(obj,varstr) #default var is from varstr
               
        stc=var.shape[0]
        fig, ax=plt.subplots()
        it0=0 #0th time index
        it1=stc-1 # last time index
        tnorm=1E3
        if(time_legend):
            lbl=["t=%3.3f"%(obj.time[it0]*tnorm), "t=%3.3f"%(obj.time[it1]*tnorm)]
        else:
            lbl=["Initial","Final"]

        if(xlim==None):
            if(initial):
                ax.plot(psi,var[it0,],label=lbl[0])
            ax.plot(psi,var[it1,],label=lbl[1])
        else:
            msk=(psi >= xlim[0]) & (psi <= xlim[1])
            if(initial):
                ax.plot(psi[msk],var[it0,msk],label=lbl[0])
            ax.plot(psi[msk],var[it1,msk],label=lbl[1])
                
        ax.legend()
        ax.set(xlabel='Normalized Pol. Flux')
        if(varstr!=None):
            ax.set(ylabel=varstr)
            
        #add time stamp of final?
        return fig, ax   

    """
    setup self.mesh
    """
    def setup_mesh(self):
        self.mesh=self.meshdata()

        #setup separatrix
        self.mesh.isep = np.argmin(abs(self.mesh.psi_surf-self.psix))
        isep=self.mesh.isep
        length=self.mesh.surf_len[isep]
        self.mesh.msep = self.mesh.surf_idx[isep,0:length]-1 # zero based

    """
    setup f0mesh
    """ 
    def setup_f0mesh(self):
       self.f0=self.f0meshdata()

    class meshdata(object):    
        """
        mesh data class for 2D contour plot
        """
        def __init__(self):
            with adios2.open("xgc.mesh.bp","rra") as fm:
                rz=fm.read('rz')
                self.cnct=fm.read('nd_connect_list')
                self.r=rz[:,0]
                self.z=rz[:,1]
                self.triobj = Triangulation(self.r,self.z,self.cnct)
                try:
                    self.surf_idx=fm.read('surf_idx')
                except:
                    print("No surf_idx in xgc.mesh.bp") 
                else:
                    self.surf_len=fm.read('surf_len')
                    self.psi_surf=fm.read('psi_surf')
                    self.theta=fm.read('theta')
                    self.m_max_surf=fm.read('m_max_surf')

                self.node_vol=fm.read('node_vol')
                self.node_vol_nearest=fm.read('node_vol_nearest')
                self.qsafety=fm.read('qsafety')
                self.psi=fm.read('psi')
                self.epsilon=fm.read('epsilon')
                self.rmin=fm.read('rmin')
                self.rmaj=fm.read('rmaj')


    class f0meshdata(object):    
        """
        mesh data class for 2D contour plot
        """
        def __init__(self):
            with adios2.open("xgc.f0.mesh.bp","rra") as f:
                T_ev=f.read('f0_T_ev')
                den0=f.read('f0_den')
                flow=f.read('f0_flow')
                if(flow.size==0):
                    flow=np.zeros_like(den0) #zero flow when flow is not written
                self.ni0=den0[-1,:]
                self.ti0=T_ev[-1,:]  # last species. need update for multi ion
                self.ui0=flow[-1,:]
                if(T_ev.shape[0]>=2):
                    self.te0=T_ev[0,:]
                    self.ne0=den0[0,:]
                    self.ue0=flow[0,:]
                self.dsmu=f.read('f0_dsmu')
                self.dvp =f.read('f0_dvp')
                self.smu_max=f.read('f0_smu_max')
                self.vp_max=f.read('f0_vp_max')

    """
    flux surface average data structure
    Not completed. Use fsa_simple
    """
    class fluxavg(object):
        def __init__(self):
            with adios2.open("xgc.fluxavg.bp","rra") as f:
                eindex=f.read('eindex')
                nelement=f.read('nelement')
                self.npsi=f.read('npsi')
                value=f.read('value')

                


                #setup matrix
                mat = IncrementalCOOMatrix(shape, np.float64)

                for i in range(shape[0]):
                    for j in range(shape[1]):
                        mat.append(i, j, dense[i, j])

                
    class voldata(object):
        """
        read volume data
        """
        def __init__(self):
            with adios2.open("xgc.volumes.bp","rra") as f:
                self.od=f.read("diag_1d_vol")
                #try:
                self.adj_eden=f.read("psn_adj_eden_vol")

    class turbdata(object):
        """
        data for turb intensity
        assuming convert_grid2 for flux average
        Obsolete. need to be replaced by new one.
        """
        def __init__(self,istart,iend,istep,midwidth,mesh,f0):
            # setup flux surface average

            self.midwidth=midwidth
            self.istart=istart
            self.iend=iend
            self.istep=istep

            #setup flux surface average matrix


            # read whole data
            for i in range(istart,iend,istep):
                # 3d file name
                filename= "xgc.3d.%5.5d.bp" % (i)

                #read data
                with adios2.open(filename,"rra") as f:
                    dpot=f.read("dpot")
                    dden=f.read("eden")

                    nzeta=dpot.shape[0]  
                    print(nzeta)  #check correct number
                    dpotn0=np.mean(dpot,axis=0)
                    dpot=dpot-dpotn0 #numpy broadcasting
                    #toroidal average of (dpot/Te)^2
                    var=np.mean(dpot**2,axis=0)/f0.Te0**2
                    #flux surface average of dpot/Te  (midplane only)
                    
                    #self.dpot_te_sqr=

                    dden=dden - np.mean(dden,axis=0)  # remove n=0 mode
                    var=dpot/f0.Te0 + dden/f0.ne0
                    var=np.mean(var**2,axis=0) # toroidal average
                    #flux surface average of dn/n0

                    #self.dn_n0_sqr=
                



    def load_volumes(self):
        """
        setup self.vol
        """
        self.vol=self.voldata()

    def heat_flux_all(self):
        
        #load volume data
        if(not hasattr(self,"vol")):
            self.vol=self.voldata()
        
        #check reading oneddiag?
        
        #get dpsi
        pmks=self.od.psi_mks[0,:]
        dpsi=np.zeros_like(pmks)
        dpsi[1:-1]=0.5*(pmks[2:]-pmks[0:-2])
        dpsi[0]=dpsi[1]
        dpsi[-1]=dpsi[-2]
        self.od.dvdp=self.vol.od/dpsi
        self.od.dpsi=dpsi
        
        nt=self.od.time.size
        ec=1.6E-19  #electron charge
        dvdpall=self.od.dvdp * self.sml_wedge_n
        
        #ion flux
        self.od.efluxi    = self.od.i_gc_density_df_1d * self.od.i_radial_en_flux_df_1d * dvdpall
        self.od.efluxexbi = self.od.i_gc_density_df_1d * self.od.i_radial_en_flux_ExB_df_1d * dvdpall
        self.od.cfluxi    = self.od.i_gc_density_df_1d * self.od.Ti * ec * self.od.i_radial_flux_df_1d * dvdpall
        self.od.cfluxexbi = self.od.i_gc_density_df_1d * self.od.Ti * ec * self.od.i_radial_flux_ExB_df_1d * dvdpall
        self.od.pfluxi    = self.od.i_gc_density_df_1d * self.od.i_radial_flux_df_1d * dvdpall
        self.od.pfluxexbi = self.od.i_gc_density_df_1d * self.od.i_radial_flux_ExB_df_1d * dvdpall
        
        if(self.electron_on):
            self.od.efluxe    = self.od.e_gc_density_df_1d * self.od.e_radial_en_flux_df_1d * dvdpall
            self.od.efluxexbe = self.od.e_gc_density_df_1d * self.od.e_radial_en_flux_ExB_df_1d * dvdpall
            self.od.cfluxe    = self.od.e_gc_density_df_1d * self.od.Te * ec * self.od.e_radial_flux_df_1d * dvdpall
            self.od.cfluxexbe = self.od.e_gc_density_df_1d * self.od.Te * ec * self.od.e_radial_flux_ExB_df_1d * dvdpall
            self.od.pfluxe    = self.od.e_gc_density_df_1d * self.od.e_radial_flux_df_1d * dvdpall
            self.od.pfluxexbe = self.od.e_gc_density_df_1d * self.od.e_radial_flux_ExB_df_1d * dvdpall

        if(self.ion2_on):
            self.od.efluxi2    = self.od.i2gc_density_df_1d * self.od.i2radial_en_flux_df_1d * dvdpall
            self.od.efluxexbi2 = self.od.i2gc_density_df_1d * self.od.i2radial_en_flux_ExB_df_1d * dvdpall
            self.od.cfluxi2    = self.od.i2gc_density_df_1d * self.od.Ti2 * ec * self.od.i2radial_flux_df_1d * dvdpall
            self.od.cfluxexbi2 = self.od.i2gc_density_df_1d * self.od.Ti2 * ec * self.od.i2radial_flux_ExB_df_1d * dvdpall
            self.od.pfluxi2    = self.od.i2gc_density_df_1d * self.od.i2radial_flux_df_1d * dvdpall
            self.od.pfluxexbi2 = self.od.i2gc_density_df_1d * self.od.i2radial_flux_ExB_df_1d * dvdpall
                


    def plot2d(self,filestr,varstr,**kwargs):
        """
        general 2d plot
        filestr: file name
        varstr: variable name
        plane: 0 based plane index - ignored for axisymmetric data
        Improve it to handle box 
        additional var to add: box, levels, cmap, etc

        Obsolete --> try contourf_one_var
        """
        box= kwargs.get('box', None) # rmin, rmax, zmin, zmax
        plane=kwargs.get('plane',0)
        levels = kwargs.get('levels', None)
        cmap = kwargs.get('cmap', 'jet')
        
        f=adios2.open(filestr,'rra')
        var=f.read(varstr)
        fig, ax=plt.subplots()

        if(box!=None):
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
        if(True):
            try:
                cf=ax.tricontourf(self.mesh.triobj,var[plane,], cmap=cmap,extend='both')
            except:
                cf=ax.tricontourf(self.mesh.triobj,var, cmap=cmap, extend='both')
            
            cbar = fig.colorbar(cf)

        if(box!=None):
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])

        #else:
        if(False):
        #if(box!=None):
            Rmin=box[0]
            Rmax=box[1]
            Zmin=box[2]
            Zmax=box[3]

            #ax.set_xlim(Rmin, Rmax)
            #ax.set_ylim(Zmin, Zmax)
            """ 
            #color bar change
            new_clim = (0, 100)
            # find location of the new upper limit on the color bar
            loc_on_cbar = cbar.norm(new_clim[1])
            # redefine limits of the colorbar
            cf.colorbar.set_clim(*new_clim)
            cf.colorbar.set_ticks(np.linspace(*new_clim, 50))
            # redefine the limits of the levels of the contour
            cf.set_clim(*new_clim)
            # updating the contourplot
            cf.changed()
            """
 
            #find subset triobj
            #limit to the user-input ranges
            idxsub = ( (self.mesh.r>=Rmin) & (self.mesh.r<=Rmax) & (self.mesh.z>=Zmin) & (self.mesh.z<=Zmax) )
            rsub=self.mesh.r[idxsub]
            zsub=self.mesh.z[idxsub]


            #find which triangles are in the defined spatial region
            tmp=idxsub[self.mesh.cnct] #idxsub T/F array, same size as R
            goodtri=np.all(tmp,axis=1) #only use triangles who have all vertices in idxsub
            trisub=self.mesh.cnct[goodtri,:]
            #remap indices in triangulation
            indices=np.where(idxsub)[0]
            for i in range(len(indices)):
                trisub[trisub==indices[i]]=i

            trisubobj = Triangulation(rsub,zsub,trisub)

            try:
                cf=ax.tricontourf(trisubobj,var[plane,idxsub], cmap=cmap,extend='both')
            except:
                cf=ax.tricontourf(trisubobj,var[idxsub], cmap=cmap, extend='both')
            
            cbar = fig.colorbar(cf)


        ax.set_title(varstr + " from " + filestr)
        return ax, cf
        

    
    def fsa_simple(self,var,**kwargs):
        """
        simple flux surface average using mesh data
        self.meshdata should be called before

        var: variable to average 
        plane: 0 based plane index - ignored for axisymmetric data
        Improve it to handle box 
        additional var to add: box, levels, cmap, etc
        """
        favg=np.zeros(self.mesh.psi_surf.size)
        for i in range(0,self.mesh.psi_surf.size):
            s1=0
            s2=0
            for j in range(0,self.mesh.surf_len[i]):
                idx=self.mesh.surf_idx[i,j] - 1
                s1=s1+var[idx]*self.mesh.node_vol[i]
                s2=s2+self.mesh.node_vol[i]
            favg[i]=s1/s2
        return favg

    def print_plasma_info(self):
        # print some plasma information (mostly from unit_dic)
        print("magnetic axis (R,Z) = (%5.5f, %5.5f) m" % (self.eq_axis_r, self.eq_axis_z))
        print("magnetic field at axis = %5.5f T" % self.eq_axis_b)
        print("X-point (R,Z) = (%5.5f, %5.5f)" % (self.eq_x_r, self.eq_x_z))
        print("simulation delta t = %e s" % self.sml_dt)
        print("wedge number = %d" % self.sml_wedge_n)
        print("Ion mass = %d" % self.unit_dic['ptl_ion_mass_au'])
        #print("particle number = %e" % (self.unit_dic['sml_totalpe']* self.unit_dic['ptl_num']))
    
    def midplane(self):
        #convert 1d psi coord to r
        self.od.r = np.interp(self.od.psi, self.bfm.psin, self.bfm.rminor)

    '''
    find time index mask to get continuous time stepping (discontinuity from restart)
    '''
    def find_tmask(self, step, max_end=False):
        if(max_end): #determin end point
            ed = np.max(step) # maximum time step
        else:
            ed = step[-1]   #ending time step

        tmask_rev=[]
        p=step.size
        for i in range(ed,0,-1):  #reverse order
            m=np.nonzero(step[0:p]==i) #find index that has step i
            try :
                p = m[0][-1] # exclude zero size 
            except:
                pass
            else:
                tmask_rev.append(p) # only append that has step number
        #tmaks is reverse order
        tmask=tmask_rev[::-1]
        return tmask

    '''
    contour plot of one plane quantity
    '''
    def contourf_one_var(self, fig, ax, var, title='None', vm='None', cmap='jet'):
        if(vm=='None'):
            cf=ax.tricontourf(self.mesh.triobj,var, cmap=cmap,extend='both',levels=150) #,vmin=-vm, vmax=vm)
        elif(vm=='Sigma2'):
            sigma = np.sqrt(np.mean(var*var) - np.mean(var)**2)
            vm = 2 * sigma
            var2=np.minimum(vm,np.maximum(-vm,var))
            cf=ax.tricontourf(self.mesh.triobj,var2, cmap=cmap,extend='both',levels=150,vmin=-vm, vmax=vm)
        else:
            var2=np.minimum(vm,np.maximum(-vm,var))
            cf=ax.tricontourf(self.mesh.triobj,var2, cmap=cmap,extend='both',levels=150,vmin=-vm, vmax=vm)
        cbar = fig.colorbar(cf, ax=ax)
        if(title != 'None'):
            ax.set_title(title)

    #Function for adios reading
    def adios2_get_shape(self, f, varname):
        nstep = int(f.available_variables()[varname]['AvailableStepsCount'])
        shape = f.available_variables()[varname]['Shape']
        lshape = None
        if shape == '':
            ## Accessing Adios1 file
            ## Read data and figure out
            v = f.read(varname)
            lshape = v.shape
        else:
            lshape = tuple([ int(xx.strip(',')) for xx in shape.strip().split() ])
        return (nstep, lshape)

    def adios2_read_all_time(self, f, varname):
        nstep, nsize = self.adios2_get_shape(f,varname)
        
        # how can generalize start??
        if(len(nsize)==1):
            return np.squeeze(f.read(varname, start=(0), count=nsize, step_start=0, step_count=nstep))
        elif(len(nsize)==2):
            return np.squeeze(f.read(varname, start=(0,0), count=nsize, step_start=0, step_count=nstep))
        elif(len(nsize)==3):
            return np.squeeze(f.read(varname, start=(0,0,0), count=nsize, step_start=0, step_count=nstep))


    def adios2_read_one_time(self, f, varname, step=-1):
        
        if(step==-1):
            step=nstep-1 # use last step

        idx=0 #initialize    
        for f1 in f:
            if(idx==step):
                break
            idx=idx+1

        nstep, nsize = self.adios2_get_shape(f1,varname)

        # how can generalize start??
        if(len(nsize)==1):
            return np.squeeze(f1.read(varname, start=(0), count=nsize, step_start=step, step_count=1))
        elif(len(nsize)==2):
            return np.squeeze(f1.read(varname, start=(0,0), count=nsize, step_start=step, step_count=1))
        elif(len(nsize)==3):
            return np.squeeze(f1.read(varname, start=(0,0,0), count=nsize, step_start=step, step_count=1))
        elif(len(nsize)==4):
            return np.squeeze(f1.read(varname, start=(0,0,0,0), count=nsize, step_start=step, step_count=1))

    ''' 
    functions for k-w power spectrum
    '''
    def power_spectrum_w_k_with_exb(self, istart, iend, skip, skip_exb, psi_target, ns_half):
        #find line segment
        ms, psi0, length = self.find_line_segment(ns_half, psi_target)
        #get total distance of the line segment
        dist=np.sum(np.sqrt( (self.mesh.r[ms[0:-1]]-self.mesh.r[ms[1:]])**2 + (self.mesh.z[ms[0:-1]]-self.mesh.z[ms[1:]])**2 ))

        print('psi0=',psi0,'length=',length)
        print('getting ExB velocity...')
        #get exb
        v_exb = self.find_exb_velocity(istart, iend, skip_exb, ms)
        print('v_exb=',v_exb,' m/s')
        #reading data
        print('reading 3d data...')
        dpot4,po,time = self.reading_3d_data(istart, iend, skip, ms)

        #prepare parameters for plot
        k, omega = self.prepare_plots(dist,ms,time)
        print('done.')

        return ms, psi0, v_exb, dpot4, po, k, omega, time

    
    # Find line segment of midplane with psi=psi_target or nearest flux surface
    # Works inside separatrix, but not separatrix or SOL
    def find_line_segment(self, n, psi_target, dir='middle'):
        isurf=np.argmin( np.abs(self.mesh.psi_surf/self.psix-psi_target) )

        #plt.plot(psi_surf)
        msk=self.mesh.surf_idx[isurf,0:self.mesh.surf_len[isurf]] -1 #node index of the surface, -1 for zero base
        #plt.plot(x.mesh.r[msk],x.mesh.z[msk])
        if(dir=='middle'):
            tmp1=msk[-n:]
            tmp2=msk[0:n]
            ms=np.append(tmp1,tmp2)
        elif(dir=='up'):
            ms=msk[0:2*n]
        else:
            ms=msk[-2*n:]
        ax=plt.subplot()
        ax.plot(self.mesh.r[ms],self.mesh.z[ms],'.')
        ax.axis('equal')
        psi0=self.mesh.psi_surf[isurf]/self.psix
    
        dr=self.mesh.r[ms[1:]]-self.mesh.r[ms[0:-1]]
        dz=self.mesh.z[ms[1:]]-self.mesh.z[ms[0:-1]]
        ds=np.sqrt( (dr)**2 + (dz)**2 )
        length=np.sum(ds)

        begin_end_ratio = ds[0]/ds[-1]
        print('ratio=',begin_end_ratio)
        if((begin_end_ratio>1.5) or (begin_end_ratio < 0.7)):
            ms, psi0, length = self.find_line_segment(n, psi_target, dir='up')

        return ms, psi0, length


    '''
    find average ExB velocity of line segment defined with node index ms
    It reads xgc.f3d.*.bp from index (istart, iend, skip)
    and do time average.
    '''
    def find_exb_velocity(self, istart, iend, skip, ms):
        pol_vi = 0
        pol_ve  = 0
        ct= 0
        pbar = tqdm(range(istart,iend,skip))
        for i in pbar:
            f=adios2.open('xgc.f3d.%5.5d.bp' % (i),'rra')

            i_pol_n0_f0=f.read('i_poloidal_flow_n0_f0')
            e_pol_n0_f0=f.read('e_poloidal_flow_n0_f0')
            f.close()

            pol_vi = pol_vi + np.mean(i_pol_n0_f0[ms])
            pol_ve = pol_ve + np.mean(e_pol_n0_f0[ms])

            ct = ct + 1
            #print(i)
        pol_vi = pol_vi/ct
        pol_ve = pol_ve/ct
        v_exb = (pol_vi + pol_ve)/2
        print('pol_vi=', pol_vi, 'pol_ve=', pol_ve)
        return v_exb

    '''
    Reading 3D dpot data of time index (istart, iend, skip) and
    node index ms
    FFT to get power spectrum.
    returns dpot in time-theta index, power spectrum in k-w, and time value
    '''
    def reading_3d_data(self,istart, iend, skip, ms):
        ns=np.size(ms)
        nt=int( (iend-istart)/skip ) +1

        #get nphi
        i=istart
        f=adios2.open('xgc.3d.%5.5d.bp' % (i),'rra')

        dpot=f.read('dpot')
        f.close()
        nphi=np.shape(dpot)[0]

        dpot4=np.zeros((nphi,nt,ns))
        time=np.zeros(nt)
        pbar = tqdm(range(istart,iend+skip,skip))
        for i in pbar:
            f=adios2.open('xgc.3d.%5.5d.bp' % (i),'rra')
            it=int( (i-istart)/skip )
            dpot=f.read('dpot')
            time1=f.read('time')
            f.close()
            dpot2=dpot-np.mean(dpot,axis=0)
            dpot3=dpot2[:,ms]
            #print(nt,it)
            dpot4[:,it,:] = dpot3
            time[it]=time1
            #print(it)

        #fft and average
        for iphi in range(0,nphi-1):
            fc=np.fft.fft2(dpot4[iphi,:,:])
            fc=np.fft.fftshift(fc)
            if(iphi==0):
                po=np.abs(fc)
            else:
                po=po+np.abs(fc)

        return dpot4[0,:,:], po, time

    '''
    get k and omega 
    dist is total distance of line sement. Assume the spacings are even in k_theta evaluation
    time is time in sec, and assumed even spacing.
    '''
    def prepare_plots(self,dist,ms,time):
        ns=np.size(ms)
        nt=np.size(time)

        kmax=2*np.pi/dist*ns
        omax=2*np.pi/(time[-1]-time[0])*nt

        k = np.fft.fftshift(np.fft.fftfreq(ns,1/kmax))
        omega=np.fft.fftshift(np.fft.fftfreq(nt,1/omax))

        return k, omega
    
    '''
    Show separatrix in plot
    ''' 
    def find_sep_idx(self):
        isep = np.argmin(abs(self.mesh.psi_surf-self.psix))
        length=self.mesh.surf_len[isep]
        msep = self.mesh.surf_idx[isep,0:length]-1
        return msep

    def show_sep(self,ax):
        msep=self.find_sep_idx()
        ax.plot(self.mesh.r[msep],self.mesh.z[msep],label='Separatrix')

    '''
    Basic analysis
    '''
    def profile_reports(self,i_name='Main ion',i2_name='Impurity', init_idx=0, end_idx=-1, edge_lim=[0.85,1.05]):

        #show initial profiles
        #temperature
        tunit=1E3
        fig, ax=plt.subplots()
        if(self.electron_on):
            plt.plot(self.od.psi, self.od.Te[0,:]/tunit,label='Elec.')
        plt.plot(self.od.psi, self.od.Ti[0,:]/tunit,label=i_name)
        if(self.ion2_on):
            plt.plot(self.od.psi, self.od.Ti2[0,:]/tunit,'--',label=i2_name)
        
        plt.legend()
        #plt.xlim(0., 1.08)
        #plt.ylim(0, 5.5)
        plt.xlabel('Normalized Pol. Flux')
        plt.ylabel('Temperature (keV)')
        plt.title('Initial Temperature')

        #density
        dunit=1E19
        fig, ax=plt.subplots()
        if(self.electron_on):
            plt.plot(self.od.psi,self.od.e_gc_density_df_1d[0,:]/dunit,label='Elec.')
        plt.plot(self.od.psi,self.od.i_gc_density_df_1d[0,:]/dunit,label=i_name)
        if(self.ion2_on):
            plt.plot(self.od.psi,self.od.i2gc_density_df_1d[0,:]/dunit,'--',label=i2_name)
        plt.legend()
        #plt.xlim(0., 1.08)
        #plt.ylim(0, 5.5)
        plt.xlabel('Normalized Pol. Flux')
        plt.ylabel('Density ($10^{19} m^{-3}$)')
        plt.title('Initial Density')

        #plasma beta (electron) - initial
        fig, ax=plt.subplots()
        bunit=1E-2
        try:
            plt.plot(self.od.psi,self.od.beta_e[0,:]/bunit)
            plt.title('Electron beta (%)')
            plt.xlabel('Normalized Pol. Flux')
            plt.ylabel('$beta_e$ (%)')
        except:
            print('beta_e plot ignored')


        # Edge range
        ie=end_idx
        # to use init_idx, plot1d_if need to be adjusted. 
        # need to pass init_idx ( & restructure the code to send whole array)
        if(self.electron_on):
            self.plot1d_if(self.od,var=self.od.e_gc_density_df_1d[:ie,:],varstr='Density (m^-3)')
            self.plot1d_if(self.od,var=self.od.e_gc_density_df_1d[:ie,:],varstr='Density (m^-3)',xlim=edge_lim)

        self.plot1d_if(self.od,var=self.od.i_gc_density_df_1d[:ie,:],varstr=i_name+' g.c. Density (m^-3)')
        self.plot1d_if(self.od,var=self.od.i_gc_density_df_1d[:ie,:],varstr=i_name+' g.c. Density (m^-3)',xlim=edge_lim)

        if(self.ion2_on):
            self.plot1d_if(self.od,var=self.od.i2gc_density_df_1d[:ie,:],varstr=i2_name+' g.c. Density (m^-3)')
            self.plot1d_if(self.od,var=self.od.i2gc_density_df_1d[:ie,:],varstr=i2_name+' g.c. Density (m^-3)',xlim=edge_lim)

        if(self.electron_on):
            self.plot1d_if(self.od,var=self.od.Te[:ie,:],varstr='Elec. Temperature (eV)')
            self.plot1d_if(self.od,var=self.od.Te[:ie,:],varstr='Elec. Temperature (eV)',xlim=edge_lim)

        self.plot1d_if(self.od,var=self.od.Ti[:ie,:],varstr=i_name+' Temperature (eV)')
        self.plot1d_if(self.od,var=self.od.Ti[:ie,:],varstr=i_name+' Temperature (eV)',xlim=edge_lim)

        if(self.ion2_on):
            self.plot1d_if(self.od,var=self.od.Ti2[:ie,:],varstr=i2_name+' Temperature (eV)')
            self.plot1d_if(self.od,var=self.od.Ti2[:ie,:],varstr=i2_name+' Temperature (eV)',xlim=edge_lim)


        #self.plot1d_if(self.od,var=self.od.e_parallel_mean_en_1d[:,:],varstr='elec full-f',xlim=[0.9, 1.07])
        self.plot1d_if(self.od,var=self.od.i_parallel_flow_df_1d[:ie,:],varstr=i_name+' parallel flow FSA (m/s)')
        if(self.ion2_on):
            self.plot1d_if(self.od,var=self.od.i2parallel_flow_df_1d[:ie,:],varstr=i2_name+' parallel flow FSA (m/s)')




    def turb_2d_report(self,i_name='Main ion',i2_name='Impurity', pm=slice(0,-1),tm=slice(0,-1), wnorm=1E6, cmap='jet'):


        # elec heat flux
        if(self.electron_on):
            fig, ax=plt.subplots()
            cf=ax.contourf(self.od.psi[pm],self.od.time[tm]*1E3,self.od.efluxexbe[tm,pm]/wnorm,levels=50,cmap=cmap)
            fig.colorbar(cf)
            plt.title('Electron Heat Flux by ExB (MW)')
            plt.xlabel('Poloidal Flux')
            plt.ylabel('Time (ms)')

            fig, ax=plt.subplots()
            cf=ax.contourf(self.od.psi[pm],self.od.time[tm]*1E3,self.od.efluxe[tm,pm]/wnorm,levels=50,cmap=cmap)
            fig.colorbar(cf)
            plt.title('Electron Heat Flux (MW)')
            plt.xlabel('Poloidal Flux')
            plt.ylabel('Time (ms)')

        # ion heat flux
        fig, ax=plt.subplots()
        cf=ax.contourf(self.od.psi[pm],self.od.time[tm]*1E3,self.od.efluxexbi[tm,pm]/wnorm,levels=50,cmap=cmap)
        fig.colorbar(cf)
        plt.title('%s Heat Flux by ExB (MW)'%i_name)
        plt.xlabel('Poloidal Flux')
        plt.ylabel('Time (ms)')

        fig, ax=plt.subplots()
        cf=ax.contourf(self.od.psi[pm],self.od.time[tm]*1E3,self.od.efluxi[tm,pm]/wnorm,levels=50,cmap=cmap)
        fig.colorbar(cf)
        plt.title('%s Heat Flux (MW)'%i_name)
        plt.xlabel('Poloidal Flux')
        plt.ylabel('Time (ms)')

        # i2 heat flux
        if(self.ion2_on):
            fig, ax=plt.subplots()
            cf=ax.contourf(self.od.psi[pm],self.od.time[tm]*1E3,self.od.efluxexbi2[tm,pm]/wnorm,levels=50,cmap=cmap)
            fig.colorbar(cf)
            plt.title('%s Heat Flux by ExB (MW)'%i2_name)
            plt.xlabel('Poloidal Flux')
            plt.ylabel('Time (ms)')

            fig, ax=plt.subplots()
            cf=ax.contourf(self.od.psi[pm],self.od.time[tm]*1E3,self.od.efluxi2[tm,pm]/wnorm,levels=50,cmap=cmap)
            fig.colorbar(cf)
            plt.title('%s Heat Flux (MW)'%i2_name)
            plt.xlabel('Poloidal Flux')
            plt.ylabel('Time (ms)')

        #electron PARTICLE flux
        if(self.electron_on):
            fig, ax=plt.subplots()
            cf=ax.contourf(self.od.psi[pm],self.od.time[tm]*1E3,self.od.pfluxexbe[tm,pm],levels=50,cmap='jet')
            fig.colorbar(cf)
            plt.title('Elec Particle Flux by ExB (#/s)')
            plt.xlabel('Poloidal Flux')
            plt.ylabel('Time (ms)')

        #ion PARTICLE flux
        fig, ax=plt.subplots()
        cf=ax.contourf(self.od.psi[pm],self.od.time[tm]*1E3,self.od.pfluxexbi[tm,pm],levels=50,cmap='jet')
        fig.colorbar(cf)
        plt.title('Ion G.C. Particle Flux by ExB (#/s)')
        plt.xlabel('Poloidal Flux')
        plt.ylabel('Time (ms)')



    # midplane value interpolation
    # need array operation if var has toroidal angle
    def midplane_var(self, var):
        maxr = self.mesh.r.max()
        nr = 300

        r_mid = np.linspace(self.eq_axis_r, maxr, nr)
        z_mid = np.linspace(self.eq_axis_z, self.eq_axis_z, nr)
        psi_tri = LinearTriInterpolator(self.mesh.triobj,self.mesh.psi/self.psix)
        psi_mid = psi_tri(r_mid, z_mid )

        var_tri = LinearTriInterpolator(self.mesh.triobj,var)
        var_mid = var_tri(r_mid, z_mid)

        return psi_mid, var_mid

    # read one variable from filestr -- for 3d and f3d files. 
    # it might work with other files, too.
    def read_one_ad2_var(self,filestr,varstr):
        f=adios2.open(filestr,'rra')
        #f.__next__()
        var=f.read(varstr)
        f.close()
        return var


'''
def load_prf(filename):
    import pandas as pd
    #space separated file 
    df = pd.read_csv(filename, sep = r'\s{2,}',engine='python')
    psi = df.index.values
    psi = psi[0:-1]
    var = df.values
    var = var[0:-1]
    return(psi,var)

# read background profile
psi_t, var_t=load_prf('../XGC-1_inputs/temp_cbc_w_0.15.prf')
var_t = var_t[:,0]
x.od.temp0 = np.interp(x.od.psi, psi_t, var_t)
'''
