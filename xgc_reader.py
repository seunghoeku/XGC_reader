"""Module of the XGC1 loader for regerating general plots using ADIOS2
Some parts are taken from Michael's xgc.py which is taken from Loic's load_XGC_local for BES.
It reads the data from the simulation especially 1D results and other small data output.

TODO
3D data are loaded only when it is specified.
"""

import numpy as np
import os
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
import matplotlib.pyplot as plt
from scipy.io import matlab
from scipy.optimize import curve_fit
from scipy.special import erfc
import scipy.sparse as sp
from tqdm.auto import trange, tqdm
from functools import singledispatchmethod

import adios2
adios2_version_minor = int(adios2.__version__[2:adios2.__version__.find('.',2)])
if adios2_version_minor < 10:
   raise RuntimeError(f"Must use adios 2.10 or newer with the xgc_reader module, loaded 2.{adios2_version_minor}\n For 2.9.x version try adios_2_9_x branch")

def read_all_steps(f, var):
    vars=f.available_variables()
    stc=vars[var].get("AvailableStepsCount")
    ct=vars[var].get("Shape")
    stc=int(stc)
    #print(var+':', ct, stc)

    if ct!='':
        c=[int(i) for i in ct.split(',')]  #
        if len(c)==1 :
            return np.reshape(f.read(var,start=[0],    count=c, step_selection=[0,stc]), [stc, c[0]])
        elif len(c)==2 :
            return np.reshape(f.read(var,start=[0,0],  count=c, step_selection=[0,stc]), [stc, c[0], c[1]])
        elif ( len(c)==3 ):
            return np.reshape(f.read(var,start=[0,0,0],count=c, step_selection=[0,stc]), [stc, c[0], c[1], c[2]])
    else:
        return f.read(var, step_selection=[0,stc])
class xgc1(object):
    
    class cnst:
        echarge = 1.602E-19
        protmass=  1.67E-27
        mu0 = 4 * 3.141592 * 1E-7

    def __init__(self, path='./'):
        """ 
        initialize either cd to a directory to process many files later, or
        open an Adios Campaign Archive now.
        """ 

        if path.endswith(".aca"):
            self.campaign = adios2.FileReader(path)
            self.path=''  # for self.path+filename to able to serve as name in campaign
            # get all variable names and info at once and save for reuse
            self.campaign_all_vars = self.campaign.available_variables()
        else:
            self.campaign=None
            os.chdir(path)
            self.path=os.getcwd()+'/'
            self.campaign_all_vars = {} # not usable when reading individual files locally

    def close(self):
        if self.campaign:
            self.campaign.close()

    @classmethod
    def load_basic(cls, path='./'):
        os.chdir(path)
        cls.path=os.getcwd()+'/'
        cls.load_units(cls)
        cls.load_oned(cls)
        cls.setup_mesh(cls)
        cls.setup_f0mesh(cls)
        cls.load_volumes(cls)

    #for compatibility with older version
    def load_unitsm(self):
        try:
            self.load_units()
        except:
            self.load_unitsm_old()

    def load_units(self):
        """
        read in xgc.units.bp file
        """
        if self.campaign:
            f = self.campaign
            prefix = 'xgc.units.bp/'
        else:
            f = adios2.FileReader(self.path+"xgc.units.bp")
            prefix = ''
            
        self.unit_dic = {}
        self.unit_dic['eq_x_psi'] = f.read(prefix+'eq_x_psi')
        self.unit_dic['eq_x_r'] = f.read(prefix+'eq_x_r')
        self.unit_dic['eq_x_z'] = f.read(prefix+'eq_x_z')
        self.unit_dic['eq_axis_r'] = f.read(prefix+'eq_axis_r')
        self.unit_dic['eq_axis_z'] = f.read(prefix+'eq_axis_z')
        self.unit_dic['eq_axis_b'] = f.read(prefix+'eq_axis_b')
        self.unit_dic['sml_dt'] = f.read(prefix+'sml_dt')
        self.unit_dic['diag_1d_period'] = f.read(prefix+'diag_1d_period')

        try:
            self.unit_dic['e_ptl_charge_eu'] = f.read(prefix+'e_ptl_charge_eu')
            self.unit_dic['e_ptl_mass_au'] = f.read(prefix+'e_ptl_mass_au')
        except:
            print('No electron particle charge/mass found in xgc.units.bp')
        self.unit_dic['eq_den_v1'] = f.read(prefix+'eq_den_v1')
        self.unit_dic['eq_tempi_v1'] = f.read(prefix+'eq_tempi_v1')
        self.unit_dic['i_ptl_charge_eu'] = f.read(prefix+'i_ptl_charge_eu')
        self.unit_dic['i_ptl_mass_au'] = f.read(prefix+'i_ptl_mass_au')
        self.unit_dic['sml_dt'] = f.read(prefix+'sml_dt')
        self.unit_dic['sml_totalpe'] = f.read(prefix+'sml_totalpe')
        self.unit_dic['sml_tran'] = f.read(prefix+'sml_tran')
        try:
            self.unit_dic['sml_wedge_n'] = f.read(prefix+'sml_wedge_n')
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
        """
        read in units.m file -- for backward compatibility. not needed for new (>2024?) XGC 
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
        if self.campaign:
            self.od=self.data1(self.campaign, self.campaign_all_vars, "xgc.oneddiag.bp") #actual reading routine
        else:
            self.od=self.data1(self.path+"xgc.oneddiag.bp") #actual reading routine
        self.od.psi=self.od.psi[0,:]
        self.od.psi00=self.od.psi00[0,:]
        try:
            self.od.psi00n=self.od.psi00/self.psix #Normalize 0 - 1(Separatrix)
        except:
            print("psix is not defined - call load_units() to get psix to get psi00n")
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
            try: 
                self.od.tmask[i-st]=tmp[-1,-1]   #LFS zero based, RHS last element
            except:
                print ('failed to find tmaks', tmp)


    """ 
        class for reading data file like xgc.oneddiag.bp
        Trying to be general, but used only for xgc.onedidag.bp
    """
    class data1(object):
        @singledispatchmethod
        def __init__(self, filename):
            with adios2.FileReader(filename) as f:
                vars=f.available_variables()
                self.load_data(f, vars, 0)

        # e.g. data1(adios2.FileReader, vars, 'xgc.oneddiag.bp') 
        # to read selected vars from already open file/campaign
        @__init__.register(adios2.FileReader)
        def _(self, f: adios2.FileReader, vars: dict, filename: str):
            vs = {k:v for (k,v) in vars.items() if k.startswith(filename)}
            self.load_data(f, vs, len(filename+"/"))

        def load_data_slow(self, f: adios2.FileReader, vars: dict, prefix_len: int):
            for v in vars:
                stc=vars[v].get("AvailableStepsCount")
                ct=vars[v].get("Shape")
                sgl=vars[v].get("SingleValue")
                stc=int(stc)
                if ct!='':
                    ct=int(ct)
                    data = f.read(v,start=[0], count=[ct], step_selection=[0, stc])
                    setattr(self,v[prefix_len:],np.reshape(data, [stc, ct]))
                elif v!='gsamples' and v!='samples' :
                    setattr(self,v[prefix_len:],f.read(v,start=[], count=[], step_selection=[0, stc])) #null list for scalar

        def load_data(self, f: adios2.FileReader, vars: dict, prefix_len: int):
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
                    setattr(self,v[prefix_len:],data)
                    bVar.SetSelection([ [0], [ct] ]) 
                    bVar.SetStepSelection([0, stc]) 
                    bEngine.Get(bVar, data, adios2.bindings.Mode.Deferred)
                elif v!='gsamples' and v!='samples' :
                    setattr(self,v[prefix_len:],f.read(v,start=[], count=[], step_selection=[0, stc])) #null list for scalar
            bEngine.PerformGets()

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
            with adios2.FileReader(filename) as f:
                #irg is region number 0,1 - outer, inner
                #read file and assign it
                self.vars=f.available_variables()
                for v in self.vars:
                    stc=self.vars[v].get("AvailableStepsCount")
                    ct=self.vars[v].get("Shape")
                    sgl=self.vars[v].get("SingleValue")
                    stc=int(stc)
                    if ct!='':
                        c=[int(i) for i in ct.split(',')]  #
                        if len(c)==1 :  # time and step 
                            setattr(self,v,f.read(v,start=[0], count=c, step_selection=[0, stc]))
                        elif len(c)==2 : # c[0] is irg
                            setattr(self,v,np.squeeze(f.read(v,start=[irg,0], count=[1,c[1]], step_selection=[0, stc])))
                        elif ( len(c)==3 & read_rz_all ) : # ct[0] is irg, read only 
                            setattr(self,v,np.squeeze(f.read(v,start=[irg,0,0], count=[1,c[1],c[2]], step_selection=[0, stc])))
                        elif ( len(c)==3 ) : # read_rz_all is false. ct[0] is irg, read only 
                            setattr(self,v,np.squeeze(f.read(v,start=[irg,0,0], count=[1,c[1],c[2]], step_selection=[stc-1, 1])))
                    elif v!='zsamples' and v!='rsamples':
                        setattr(self,v,f.read(v,start=[], count=[], step_selection=[0, stc])) #null list for scalar
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
            Functions for 3 lambda fit: lp (lambda_q of private flux region), ln (lambda_q of near SOL), lf (lambda_q of far SOL)
            q(x) =     q0 * exp( (x-dsep)/lp)   when x<dsep
                 =(q0-qf) * exp(-(x-dsep)/ln) + qf * exp(-(x-dsep)/lf) when x>dsep
        """
        def lambda_q3(self,x,q0,qf,lp,ln,lf,dsep):
            
            dsepl =0 # not using dsep --> dsepl=dsep to use
            rtn = q0  * np.exp( (x-dsepl)/lp) # only x<dsep will be used.
            ms=np.nonzero(x>=dsepl)
            rtn[ms] = (q0-qf) * np.exp(-(x[ms]-dsepl)/ln) + qf * np.exp(-(x[ms]-dsepl)/lf)
            return rtn

        """
            3 lambda_q fitting of one profile data
        """
        def lambda_q3_fit1(self,ydata,pmask):
            q0init=np.max(ydata)
            qfinit=0.01*q0init # 1 percent
            lpinit=1 # 1mm
            lninit=2 # 2mm
            lfinit=4 # 4mm
            dsepinit=0.01 # 0.01 mm

            p0=np.array([q0init, qfinit, lpinit, lninit, lfinit, dsepinit])
            if(pmask==None):
                popt,pconv = curve_fit(self.lambda_q3,self.rmidsepmm,ydata,p0=p0)
            else:
                popt,pconv = curve_fit(self.lambda_q3,self.rmidsepmm[pmask],ydata[pmask],p0=p0)

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
        
        def lambda_q3_fit_all(self,**kwargs):
            pmask = kwargs.get('pmask', None)

            self.lp_lq3=np.zeros_like(self.lq_int) #mem allocation
            self.ln_lq3=np.zeros_like(self.lp_lq3)
            self.lf_lq3=np.zeros_like(self.lp_lq3)
            self.dsep_eich=np.zeros_like(self.lp_lq3)

            for i in range(self.time.size):
                try :
                    popt,pconv = self.lambda_q3_fit1(self.qt[i,:],pmask)
                except:
                    popt=[0, 0, 0, 0, 0, 0]
                
                self.lp_lq3[i]= popt[2] 
                self.ln_lq3[i] = popt[3] 
                self.lf_lq3[i] = popt[4]
                self.dsep_eich[i]= popt[5]
    """
        data for bfieldm
    """
    class databfm(object):
        def __init__(self,path):
            with adios2.FileReader(path+"xgc.bfieldm.bp") as f:
                self.vars=f.available_variables()
                if('rmajor' in self.vars):
                    v='rmajor'
                else:
                    v='/bfield/rvec' 
                #ct=self.vars[v].get("Shape")
                #c=int(ct)
                self.rmid=f.read(v) #,start=[0],count=[c],step_selection=[0,1])
                if('psi_n' in self.vars):
                    v='psi_n'
                else:
                    v='/bfield/psi_eq_x_psi'
                #ct=self.vars[v].get("Shape")
                #c=int(ct)
                self.psin=f.read(v) #,start=[0],count=[c],step_selection=[0,1])


    def load_heatdiag(self, **kwargs):
        """
        load xgc.heatdiag.bp and some post process
        """
        read_rz_all = kwargs.get('read_rz_all',False) #read heat load in RZ

        self.hl=[]
        self.hl.append( self.datahlp(self.path+"xgc.heatdiag.bp",0,read_rz_all) ) #actual reading routine
        self.hl.append( self.datahlp(self.path+"xgc.heatdiag.bp",1,read_rz_all) )#actual reading routine

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

    #data class for each species data of heatdiag2
    class datahl2_sp(object):
        def __init__(self, prefix, f):
            self.number = read_all_steps(f, prefix + '_number')[:,:,1:]
            self.para_energy = read_all_steps(f, prefix + '_para_energy')[:,:,1:]
            self.perp_energy = read_all_steps(f, prefix + '_perp_energy')[:,:,1:]
            self.potential = read_all_steps(f, prefix + '_potential')[:,:,1:]

    # data class for heatdiag2
    class datahl2(object):
        def __init__(self,filename, datahl2_sp):

            prefix = ['e', 'i', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9']
            with adios2.FileReader(filename) as f:
                vars=f.available_variables()

                self.time = read_all_steps(f, 'time')
                self.step = read_all_steps(f, 'step')
                self.tindex = read_all_steps(f, 'tindex')
                self.ds = read_all_steps(f, 'ds')
                self.psi = read_all_steps(f, 'psi')
                self.r = read_all_steps(f, 'r')
                self.z = read_all_steps(f, 'z')
                self.strike_angle = read_all_steps(f, 'strike_angle')

                # for each species read particle flux and energy flux as an array.
                max_nsp = 10 # maximum number of species. Any larger integer should work.
                self.sp=[]
                for isp in range(max_nsp):
                    if(prefix[isp]+'_number' in vars):
                        self.sp.append( datahl2_sp(prefix[isp],f) )
                    else:
                        #print('No '+prefix[isp]+' species data in heatdiag2.')
                        break
                if(isp==0):
                    print('No electron species data in heatdiag2. Nothing loaded.')

                self.nsp = len(self.sp)
                #set dt
                self.dt = np.zeros_like(self.time)
                self.dt[1:] = self.time[1:] - self.time[0:-1]
                self.dt[0] = self.dt[1] # assume that the first time step is the same as the second one.
                self.dt=self.dt[:,np.newaxis]

        def get_midplane_conversion(self,psino,rmido, psix, wedge_n):
            """
            get midplane conversion of each species
            """
            self.rs = np.interp([1],psino,rmido)
            self.rmidsepmm = (np.interp(self.psin,psino,rmido) - self.rs)  * 1E3

        def get_parallel_flux(self):
            for isp in range(self.nsp):
                # heat flux q and particle flux gammas(g)
                self.sp[isp].q = np.squeeze(self.sp[isp].para_energy + self.sp[isp].perp_energy)/self.dt/self.area
                self.sp[isp].g = np.squeeze(self.sp[isp].number)/self.dt/self.area

        def update_total_flux(self):
            """
            update total heat flux and particle flux
            """
            self.g_total = 0
            self.q_total = 0
            for isp in range(self.nsp):
                self.g_total += self.sp[isp].g
                self.q_total += self.sp[isp].q

        def get_divertor(self, outer=True, lower=True):
            """
            get array index for inner and outer divertor
            Assume the array index is conter-clockwise. --> need to consider the opposite cases
            """
            # find minimum psi location
            sign_z = 1 if lower else -1
            mask = (self.z-self.eq_axis_z) * sign_z < 0
            i0 = np.argmin(np.where(mask, self.psin, np.inf))

            # find maximum psi location
            sign_r = 1 if outer else -1
            mask = (self.r-self.eq_axis_r) * sign_r > 0
            i1 = np.argmax(np.where(mask, self.psin, -np.inf))

            return i0,i1

        """
            Functions for eich fit
            q(x) =0.5*q0* exp( (0.5*s/lq)^2 - (x-dsep)/lq ) * erfc (0.5*s/lq - (x-dsep)/s)
        """
        def eich(self,xdata,q0,s,lq,dsep):
            return 0.5*q0*np.exp((0.5*s/lq)**2-(xdata-dsep)/lq)*erfc(0.5*s/lq-(xdata-dsep)/s)

        """
            Eich fitting of one profile data
        """
        def eich_fit1(self,ydata,pmask=None):
            q0init=np.max(ydata)
            sinit=2 # 2mm
            lqinit=1 # 1mm
            dsepinit=0.1 # 0.1 mm

            p0=np.array([q0init, sinit, lqinit, dsepinit])
            if(pmask is None):
                popt,pconv = curve_fit(self.eich,self.rmidsepmm,ydata,p0=p0)
            else:
                popt,pconv = curve_fit(self.eich,self.rmidsepmm[pmask],ydata[pmask],p0=p0)

            return popt, pconv

        """
            perform fitting for all time steps.
        """
        def eich_fit_all(self,pmask=None):

            self.lq_eich=np.zeros_like(self.time) #mem allocation
            self.S_eich=np.zeros_like(self.lq_eich)
            self.dsep_eich=np.zeros_like(self.lq_eich)

            for i in range(self.time.size):
                try :
                    popt,pconv = self.eich_fit1(self.q_total[i,:],pmask=pmask)
                except:
                    popt=[0, 0, 0, 0]

                self.lq_eich[i]= popt[2]
                self.S_eich[i] = popt[1]
                self.dsep_eich[i]= popt[3]

        """
        getting total heat (radially integrated) to inner/outer divertor.
        """
        def total_heat(self,wedge_n, pmask=None):
            if(pmask is None):
                pmask=np.ones_like(self.rmidsepmm,dtype=bool)

            for isp in range(self.nsp):
                self.sp[isp].q_para_sum=np.sum(self.sp[isp].para_energy[:,:,pmask],axis=(1,2))[:,np.newaxis]*wedge_n/self.dt
                self.sp[isp].q_perp_sum=np.sum(self.sp[isp].perp_energy[:,:,pmask],axis=(1,2))[:,np.newaxis]*wedge_n/self.dt
                self.sp[isp].q_sum=self.sp[isp].q_para_sum+self.sp[isp].q_perp_sum
                self.sp[isp].g_sum=np.sum(self.sp[isp].number[:,:,pmask],axis=(1,2))[:,np.newaxis]*wedge_n/self.dt

    # load xgc.heatdiag2.bp and some postprocess
    def load_heatdiag2(self):
        self.hl2 = self.datahl2(self.path+"xgc.heatdiag2.bp", self.datahl2_sp)
        #print('loading heatdiag2 done')

        # post process
        # calculate normalized psi and area at the target
        wedge_n = self.unit_dic['sml_wedge_n']
        it=-1 # keep the last one
        self.hl2.psin=self.hl2.psi[it,:]/self.psix
         #area of each segment with angle factor. 2pi * (r1+r2)/2 * ds / wedge_n * cos(angle)
        self.hl2.area=np.pi*self.hl2.r[it,:]*self.hl2.ds[it,:]/wedge_n * np.cos(self.hl2.strike_angle[it,:])
        self.hl2.area = self.hl2.area[np.newaxis,:]

        # use bfieldm if loaded
        if(hasattr(self, 'bfm')):
            psino=self.bfm.psino
            rmido=self.bfm.rmido
        else: #get it from xgc.mesh.bp
            psino, rmido = self.midplane_var(self.mesh.r)

        # get midplane conversion
        #plt.plot(psino,rmido)
        self.hl2.get_midplane_conversion(psino, rmido, self.psix, wedge_n)
        self.hl2.get_parallel_flux()
        self.hl2.update_total_flux()

        # get divertor index
        self.hl2.eq_axis_r = self.eq_axis_r
        self.hl2.eq_axis_z = self.eq_axis_z

    # report basic analysis of heatdiag2.bp
    # Need to specify the divertor region
    # ndata is maximum number of data point to be considered.
    # fit_mask is the mask for fitting. If None, all data will be used.
    def report_heatdiag2(self, is_outer=True, is_lower=True, it=-1, xlim=[-5, 15], lq_ylim=[0, 10], ndata=1000000, fit_mask=None, sp_names=['e', 'i', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9']):
        #select divertor
        i0, i1 = self.hl2.get_divertor(outer=is_outer, lower=is_lower)
        sign= 1 if (i0<i1) else -1 
        i1 = i0 + sign*ndata if np.abs(i1-i0)>ndata else i1

        md = np.arange(i0,i1,sign)
        fig, ax = plt.subplots()
        plt.plot(self.hl2.r[0,:],self.hl2.z[0,:])
        plt.plot(self.hl2.r[0,md],self.hl2.z[0,md],'r-',linewidth=4,label='Divertor')
        plt.legend()
        self.show_sep(ax, style=',')
        plt.axis('equal')

        #plot total heat flux
        self.hl2.total_heat(self.sml_wedge_n, pmask=md)
        plt.subplots()
        for isp in range(len(self.hl2.sp)):
            plt.plot(self.hl2.time*1E3, self.hl2.sp[isp].q_sum/1E6, '.',label=sp_names[isp])
        plt.xlabel('Time (ms)')
        plt.ylabel('Total Heat Flux (MW)')
        plt.legend()

        #heat flux profile
        plt.subplots()
        for isp in range(len(self.hl2.sp)):
            plt.plot(self.hl2.rmidsepmm[md], self.hl2.sp[isp].q[it,md]/1E6,label=sp_names[isp])
        plt.plot(self.hl2.rmidsepmm[md], self.hl2.q_total[it,md]/1E6,label='Total')

        plt.xlim(xlim[0], xlim[1])
        plt.ylabel('Parallel heat flux [MW/$m^2$] at the divertor')
        plt.xlabel('Midplane distance from separatrix [mm]')
        plt.legend()

        #fitting one time step
        if fit_mask is None:
            fit_mask = md
        popt,pconv = self.hl2.eich_fit1(self.hl2.q_total[it,:], pmask=fit_mask)
        eich = self.hl2.eich(self.hl2.rmidsepmm[md], popt[0], popt[1], popt[2], popt[3])
        plt.subplots()
        plt.plot(self.hl2.rmidsepmm[md], self.hl2.q_total[it,md],label='XGC')
        plt.plot(self.hl2.rmidsepmm[md], eich,label='Eich Fit')
        plt.xlim(xlim[0], xlim[1])
        plt.title('$\\lambda_q$ = %3.3f mm, S=%3.3f mm, t=%3.3f ms'%(popt[2],popt[1],self.hl2.time[it]*1E3))
        plt.ylabel('Parallel heat flux [W/$m^2$] at the divertor')
        plt.xlabel('Midplane distance from separatrix [mm]')
        plt.legend()

        self.hl2.eich_fit_all(pmask=fit_mask)
        plt.subplots()
        plt.plot(self.hl2.time*1E3, self.hl2.lq_eich, '.', label='$\\lambda_q$')
        plt.plot(self.hl2.time*1E3, self.hl2.S_eich, '.', label='S')
        plt.ylim(lq_ylim[0], lq_ylim[1])
        plt.xlabel('Time [ms]')
        plt.ylabel('$\\lambda_q$, S [mm]')
        plt.legend()

        return md
    """
        Load xgc.bfieldm.bp -- midplane bfield info
    """
    def load_bfieldm(self):
        self.bfm = self.databfm(self.path)
        self.bfm.r0=self.unit_dic['eq_axis_r']
        plt.plot(self.bfm.rmid)
        #get outside midplane only
        #msk=np.argwhere(self.bfm.rmid>self.bfm.r0)
        #print(msk)
        #n0=msk[0]
        n0 = np.nonzero(self.bfm.rmid > self.bfm.r0)[0][0]
        self.bfm.rmido=self.bfm.rmid[n0:]
        self.bfm.psino=self.bfm.psin[n0:]

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
    def load_bfield(self):
        with adios2.FileReader(self.path+"xgc.bfield.bp") as f:
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
        if self.campaign:
            self.mesh = self.meshdata(self.campaign)
        else:
            self.mesh = self.meshdata(self.path)

        #setup separatrix
        self.mesh.isep = np.argmin(abs(self.mesh.psi_surf-self.psix))
        isep=self.mesh.isep
        length=self.mesh.surf_len[isep]
        self.mesh.msep = self.mesh.surf_idx[isep,0:length]-1 # zero based

    """
    setup f0mesh
    """ 
    def setup_f0mesh(self):
        if self.campaign:
            self.f0 = self.f0meshdata(self.campaign)
        else:
            self.f0 = self.f0meshdata(self.path)

    class meshdata(object):    
        """
        mesh data class for 2D contour plot
        """
        @singledispatchmethod
        def __init__(self,path):
            with adios2.FileReader(path+"xgc.mesh.bp") as fm:
                self.load_mesh(fm)

        @__init__.register(adios2.FileReader)
        def _(self, fm: adios2.FileReader):
            self.load_mesh(fm, "xgc.mesh.bp/")

        def load_mesh(self, fm: adios2.FileReader, prefix=''):
            rz=fm.read(prefix+'rz')
            self.cnct=fm.read(prefix+'nd_connect_list')
            self.r=rz[:,0]
            self.z=rz[:,1]
            self.triobj = Triangulation(self.r,self.z,self.cnct)
            try:
                self.surf_idx=fm.read(prefix+'surf_idx')
            except:
                print("No surf_idx in xgc.mesh.bp") 
            else:
                self.surf_len=fm.read(prefix+'surf_len')
                try:
                    self.psi_surf=fm.read(prefix+'psi_surf')
                except:
                    print("failed to read psi_surf in xgc.mesh.bp")
                    self.psi_surf=np.arange(0,1,1/self.surf_len.size)
                self.theta=fm.read(prefix+'theta')

            try:
                self.m_max_surf=fm.read(prefix+'m_max_surf')
            except:
                print("No m_max_surf in xgc.mesh.bp")

            try:
                self.wall_nodes = fm.read(prefix+'grid_wall_nodes') -1 #zero based
            except:
                print("No wall_nodes in xgc.mesh.bp")
            self.node_vol=fm.read(prefix+'node_vol')
            self.node_vol_nearest=fm.read(prefix+'node_vol_nearest')
            self.qsafety=fm.read(prefix+'qsafety')
            self.psi=fm.read(prefix+'psi')
            self.epsilon=fm.read(prefix+'epsilon')
            self.rmin=fm.read(prefix+'rmin')
            self.rmaj=fm.read(prefix+'rmaj')
            try:
                self.region=fm.read(prefix+'region')
            except:
                print("No region in xgc.mesh.bp") 
            try:
                self.wedge_angle=fm.read(prefix+'wedge_angle')
            except:
                print("No wedge_angle in xgc.mesh.bp") 
            try:
                self.delta_phi=fm.read(prefix+'delta_phi')
            except:
                print("No delta_phi in xgc.mesh.bp") 

            self.nnodes = np.size(self.r) # same as n_n 

    class f0meshdata(object):    
        """
        mesh data class for 2D contour plot
        """
        @singledispatchmethod
        def __init__(self,path):
            with adios2.FileReader(path+"xgc.f0.mesh.bp") as f:
                self.load_f0mesh(f)

        @__init__.register(adios2.FileReader)
        def _(self, f: adios2.FileReader):
            self.load_f0mesh(f, "xgc.f0.mesh.bp/")

        def load_f0mesh(self, f: adios2.FileReader, prefix=''):
            T_ev=f.read(prefix+'f0_T_ev')
            den0=f.read(prefix+'f0_den')
            try:
                flow=f.read(prefix+'f0_flow')
            except:
                print("No flow in xgc.f0.mesh.bp") 
                flow=np.zeros_like(den0) #zero flow when flow is not written

            if(len(den0.shape)>1):
                self.ni0=den0[-1,:]
                self.ti0=T_ev[-1,:]  # last species. need update for multi ion
                self.ui0=flow[-1,:]
            else:
                # old format
                self.ni0=den0
                self.ti0=T_ev
                self.ui0=flow
                self.te0=den0
                
            if(T_ev.shape[0]>=2):
                self.te0=T_ev[0,:]
                try:
                    self.ne0=den0[0,:]
                    self.ue0=flow[0,:]
                except:
                    # old format
                    self.ne0=den0
                    self.ue0=flow
            if(T_ev.shape[0]>=3):
                print('multi species - ni0, ti0, ui0 are last species')


            self.dsmu=f.read(prefix+'f0_dsmu')
            self.dvp =f.read(prefix+'f0_dvp')
            self.smu_max=f.read(prefix+'f0_smu_max')
            self.vp_max=f.read(prefix+'f0_vp_max')

    """
    flux surface average data structure
    Not completed. Use fsa_simple
    """
    class fluxavg(object):
        def __init__(self,path):
            with adios2.FileReader(path + "xgc.fluxavg.bp") as f:
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
        def __init__(self,path):
            with adios2.FileReader(path+"xgc.volumes.bp") as f:
                self.od=f.read("diag_1d_vol")

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
                with adios2.FileReader(filename) as f:
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
        self.vol=self.voldata(self.path)

    def heat_flux_all(self):
        self.radial_flux_all()

    # get radial flux of energy and particle from 1D data
    def radial_flux_all(self):
        
        #load volume data
        if(not hasattr(self,"vol")):
            #self.vol=self.voldata(self.path)
            self.load_volumes()
        
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
        if hasattr(self.od,'i_radial_en_flux_3db_df_1d'):
            self.od.eflux3dbi = self.od.i_gc_density_df_1d * self.od.i_radial_en_flux_3db_df_1d * dvdpall

        self.od.cfluxi    = self.od.i_gc_density_df_1d * self.od.Ti * ec * self.od.i_radial_flux_df_1d * dvdpall
        self.od.cfluxexbi = self.od.i_gc_density_df_1d * self.od.Ti * ec * self.od.i_radial_flux_ExB_df_1d * dvdpall
        self.od.pfluxi    = self.od.i_gc_density_df_1d * self.od.i_radial_flux_df_1d * dvdpall
        self.od.pfluxexbi = self.od.i_gc_density_df_1d * self.od.i_radial_flux_ExB_df_1d * dvdpall
        
        self.od.mfluxi    = self.od.i_gc_density_df_1d * self.od.i_radial_mom_flux_df_1d * dvdpall # toroidal momentum flux
        self.od.mfluxexbi = self.od.i_gc_density_df_1d * self.od.i_radial_mom_flux_ExB_df_1d * dvdpall
        

        if(self.electron_on):
            self.od.efluxe    = self.od.e_gc_density_df_1d * self.od.e_radial_en_flux_df_1d * dvdpall
            self.od.efluxexbe = self.od.e_gc_density_df_1d * self.od.e_radial_en_flux_ExB_df_1d * dvdpall
            if hasattr(self.od,'e_radial_en_flux_3db_df_1d'):
                self.od.eflux3dbe = self.od.e_gc_density_df_1d * self.od.e_radial_en_flux_3db_df_1d * dvdpall

            self.od.cfluxe    = self.od.e_gc_density_df_1d * self.od.Te * ec * self.od.e_radial_flux_df_1d * dvdpall
            self.od.cfluxexbe = self.od.e_gc_density_df_1d * self.od.Te * ec * self.od.e_radial_flux_ExB_df_1d * dvdpall
            self.od.pfluxe    = self.od.e_gc_density_df_1d * self.od.e_radial_flux_df_1d * dvdpall
            self.od.pfluxexbe = self.od.e_gc_density_df_1d * self.od.e_radial_flux_ExB_df_1d * dvdpall

        if(self.ion2_on):
            self.od.efluxi2    = self.od.i2gc_density_df_1d * self.od.i2radial_en_flux_df_1d * dvdpall
            self.od.efluxexbi2 = self.od.i2gc_density_df_1d * self.od.i2radial_en_flux_ExB_df_1d * dvdpall
            if hasattr(self.od,'i_radial_en_flux_3db_df_1d'):
                self.od.eflux3dbi2 = self.od.i2gc_density_df_1d * self.od.i2radial_en_flux_3db_df_1d * dvdpall

            self.od.cfluxi2    = self.od.i2gc_density_df_1d * self.od.Ti2 * ec * self.od.i2radial_flux_df_1d * dvdpall
            self.od.cfluxexbi2 = self.od.i2gc_density_df_1d * self.od.Ti2 * ec * self.od.i2radial_flux_ExB_df_1d * dvdpall
            self.od.pfluxi2    = self.od.i2gc_density_df_1d * self.od.i2radial_flux_df_1d * dvdpall
            self.od.pfluxexbi2 = self.od.i2gc_density_df_1d * self.od.i2radial_flux_ExB_df_1d * dvdpall
            self.od.mfluxi2    = self.od.i2gc_density_df_1d * self.od.i2radial_mom_flux_df_1d * dvdpall
            self.od.mfluxexbi2 = self.od.i2gc_density_df_1d * self.od.i2radial_mom_flux_ExB_df_1d * dvdpall                


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
        
        f=adios2.FileReader(filestr)
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
        

    
    def fsa_simple(self,var):
        """
        simple flux surface average using mesh data
        self.meshdata should be called before

        var: variable to average 
        """
        favg=np.zeros(self.mesh.psi_surf.size)
        for i in range(0,self.mesh.psi_surf.size):
            s1=0
            s2=0
            for j in range(0,self.mesh.surf_len[i]):
                idx=self.mesh.surf_idx[i,j] - 1
                s1=s1+var[idx]*self.mesh.node_vol[idx]
                s2=s2+self.mesh.node_vol[idx]
            favg[i]=s1/s2
            if(np.isnan(favg[i])):
                print("NaN found at psi=%f" % self.mesh.psi_surf[i], 's1,s2=',s1,s2)
        return favg
    
    def flux_sum_simple(self,var):
        """
        simple summation over surface - not good when non-aligned points are nearby
        self.meshdata should be called before
        """
        favg=np.zeros(self.mesh.psi_surf.size)
        for i in range(0,self.mesh.psi_surf.size):
            s1=0
            for j in range(0,self.mesh.surf_len[i]):
                idx=self.mesh.surf_idx[i,j] - 1
                s1=s1+var[idx]
            favg[i]=s1
        return favg

    def print_plasma_info(self):
        # print some plasma information (mostly from unit_dic)
        print("magnetic axis (R,Z) = (%5.5f, %5.5f) m" % (self.eq_axis_r, self.eq_axis_z))
        print("magnetic field at axis = %5.5f T" % self.eq_axis_b)
        print("X-point (R,Z) = (%5.5f, %5.5f)" % (self.eq_x_r, self.eq_x_z))
        print("simulation delta t = %e s" % self.sml_dt)
        print("wedge number = %d" % self.sml_wedge_n)
        print("Ion mass = %d" % self.unit_dic['i_ptl_mass_au'])
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
    def contourf_one_var(self, fig, ax, var, title='None', vm='None', cmap='jet', levels=150, cbar=True):
        if(vm=='None'):
            cf=ax.tricontourf(self.mesh.triobj,var, cmap=cmap,extend='both',levels=levels) #,vmin=-vm, vmax=vm)
        elif(vm=='Sigma2'):
            sigma = np.sqrt(np.mean(var*var) - np.mean(var)**2)
            vm = 2 * sigma
            var2=np.minimum(vm,np.maximum(-vm,var))
            cf=ax.tricontourf(self.mesh.triobj,var2, cmap=cmap,extend='both',levels=levels,vmin=-vm, vmax=vm)
        else:
            var2=np.minimum(vm,np.maximum(-vm,var))
            cf=ax.tricontourf(self.mesh.triobj,var2, cmap=cmap,extend='both',levels=levels,vmin=-vm, vmax=vm)
        if(cbar):
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
    def power_spectrum_w_k_with_exb(self, istart, iend, skip, skip_exb, psi_target, ns_half, old_vexb=False):
        #find line segment
        ms, psi0, length = self.find_line_segment(ns_half, psi_target)

        print('psi0=',psi0,'length=',length) 
        print('getting ExB velocity...')
        #get exb
        if(old_vexb):
            v_exb = self.find_exb_velocity(istart, iend, skip_exb, ms)
        else:
            v_exb = self.find_exb_velocity2(istart, iend, skip_exb, ms)

        print('v_exb=',v_exb,' m/s')
        #reading data
        print('reading 3d data...')
        dpot4,po,time = self.reading_3d_data(istart, iend, skip, ms)

        #prepare parameters for plot
        k, omega = self.prepare_plots(length,ms,time)
        print('done.')

        return ms, psi0, v_exb, dpot4, po, k, omega, time, length

    
    # Find line segment of midplane with psi=psi_target or nearest flux surface
    # Works inside separatrix, but not separatrix or SOL
    def find_line_segment(self, n, psi_target, dir='middle'):
        isurf=np.argmin( np.abs(self.mesh.psi_surf/self.psix-psi_target) )

        #plt.plot(psi_surf)
        msk=self.mesh.surf_idx[isurf,0:self.mesh.surf_len[isurf]] -1 #node index of the surface, -1 for zero base
        #plt.plot(x.mesh.r[msk],x.mesh.z[msk])
        #core mesh or SOL
        if(self.mesh.psi_surf[isurf]<0.99999*self.psix):
            #core below
            if(dir=='middle'):
                tmp1=msk[-n:]
                tmp2=msk[0:n]
                ms=np.append(tmp1,tmp2)
            elif(dir=='up'):
                ms=msk[0:2*n]
            else:
                ms=msk[-2*n:]
        else:
            #SOL below
            #1. find segments low field side and abvoe X-point
            msk2=np.nonzero( np.logical_and(self.mesh.r[msk]>self.eq_x_r,self.mesh.z[msk]>self.eq_x_z) )
            msk3=msk[msk2]
            imid=np.argmin(np.abs(self.mesh.z[msk3]-self.eq_axis_z))
            ms=msk3[imid-n:imid+n]

        ax=plt.subplot()
        ax.plot(self.mesh.r[ms],self.mesh.z[ms],'.')
        ax.axis('equal')
        psi0=self.mesh.psi_surf[isurf]/self.psix
    
        dr=self.mesh.r[ms[1:]]-self.mesh.r[ms[0:-1]]
        dz=self.mesh.z[ms[1:]]-self.mesh.z[ms[0:-1]]
        ds=np.sqrt( (dr)**2 + (dz)**2 )
        length=np.sum(ds)

        begin_end_ratio = ds[0]/ds[-1]
        if((begin_end_ratio>1.5) or (begin_end_ratio < 0.7)):
            if(dir=='middle'):
                print('ratio=',begin_end_ratio, 'trying upper side')
                ms, psi0, length = self.find_line_segment(n, psi_target, dir='up')
            elif(dir=='up'):
                print('ratio=',begin_end_ratio, 'trying lower side')
                ms, psi0, length = self.find_line_segment(n, psi_target, dir='down')
            else:
                print('ratio=',begin_end_ratio, 'Failed to find line segment')
                return np.array([]), 0, 0

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
            f=adios2.FileReader('xgc.f3d.%5.5d.bp' % (i))

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
    find avearage ExB velocity of line segment defined with node index ms
    It reads epsi of xgc.3d.*.bp from index (istart, iend, skip)
    and calculate ExB velocity in time.
    '''
    def find_exb_velocity2(self, istart, iend, skip, ms, only_average=True):

        bt = self.bfield[2,ms]
        b2 = np.sqrt(self.bfield[0,ms]**2 + self.bfield[1,ms]**2 + self.bfield[2,ms]**2)

        pbar = tqdm(range(istart,iend,skip))
        for count, istep in enumerate(pbar):
            with adios2.FileReader('xgc.3d.%5.5d.bp' % (istep)) as f:
                epsi=f.read('epsi') # E_r
                try:
                    time1=f.read('time')
                except:
                    time1=istep*self.sml_dt
                v_exb1=epsi[:,ms]*bt/b2 # ExB velocity
                v_exb1=v_exb1[np.newaxis,:,:]

            if(count==0):
                v_exb = v_exb1
                time = time1
            else:
                v_exb = np.concatenate((v_exb,v_exb1),axis=0)
                time = np.vstack((time,time1))
        if(only_average):
            v_exb = np.mean(v_exb,axis=(0,1,2))
            return v_exb # only return averaged ExB velocity
        else:
            return v_exb, time # return all ExB velocity in (time, toroidal angle, poloidal index) and time array


    '''
    Reading 3D dpot data of time index (istart, iend, skip) and
    node index ms
    FFT to get power spectrum.
    returns dpot in time-theta index, power spectrum in k-w, and time value
    '''
    def reading_3d_data(self,istart, iend, skip, ms, no_fft=False):
        ns=np.size(ms)
        nt=int( (iend-istart)/skip ) +1

        #get nphi
        i=istart
        f=adios2.FileReader('xgc.3d.%5.5d.bp' % (i))

        dpot=f.read('dpot')
        f.close()
        nphi=np.shape(dpot)[0]

        dpot4=np.zeros((nphi,nt,ns))
        time=np.zeros(nt)
        pbar = tqdm(range(istart,iend+skip,skip))
        for i in pbar:
            f=adios2.FileReader('xgc.3d.%5.5d.bp' % (i))
            it=int( (i-istart)/skip )
            dpot=f.read('dpot')
            try:
                time1=f.read('time')
            except:
                time1=i*self.sml_dt
            f.close()
            dpot2=dpot-np.mean(dpot,axis=0)
            dpot3=dpot2[:,ms]
            #print(nt,it)
            dpot4[:,it,:] = dpot3
            time[it]=time1
            #print(it)


        if(no_fft):
            return dpot4, time # return whole dpot4 data to process later
        
        
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

    def show_sep(self,ax, style='-'):
        msep=self.find_sep_idx()
        ax.plot(self.mesh.r[msep],self.mesh.z[msep],style,label='Separatrix')


    def find_surf_idx(self, psi_norm=1.0):
        isep0 = np.argmin(abs(self.mesh.psi_surf-self.psix))
        if(psi_norm<1.0):
            psi_surf=self.mesh.psi_surf[:isep0]
        else:
            psi_surf=self.mesh.psi_surf
        isep = np.argmin(abs(psi_surf-self.psix*psi_norm))

        length=self.mesh.surf_len[isep]
        msep = self.mesh.surf_idx[isep,0:length]-1
        return msep
    
    '''
    Turbulence intensity
    '''
    def turb_intensity(self,istart, iend, skip, vartype='f3d_eden', mode='all'):
        
        # file type
        if(vartype=='f3d_eden'): 
            # use dn/n0
            fname='f3d'
            vname='e_den'
        elif(vartype=='3d_dpot'): 
            # use dphi/Te
            fname='3d'
            vname='dpot'
        elif(vartype=='f3d_iTperp'):
            # use dTperp/Ti0
            fname='f3d'
            vname='i_T_perp'
        else:
            print('unknown vartype type')
            return

        print('using '+vname+' of '+fname)
        err_msg_count =0 
        msk = np.zeros_like(self.mesh.z) +1
        if(mode=='all'):
            # use all planes
            pass    
        elif(mode=='upper'):
            # use upper half planes
            msk[self.mesh.z < self.eq_axis_z] = 0
        elif(mode=='lower'):
            # use lower half planes
            msk[self.mesh.z > self.eq_axis_z] = 0


        pbar = tqdm(range(istart,iend,skip))
        for count, i in enumerate(pbar):
            with adios2.FileReader('xgc.'+fname+'.%5.5d.bp' % (i)) as f:
                it=int( (i-istart)/skip )
                var=f.read(vname)
                time1=f.read('time')
                if(var.shape[0]>200): # 200 is maximum plane number
                    var=np.transpose(var) 

            if(fname=='f3d'): # variables has f_0. Normalization will be toroidal average.
                var2=var-np.mean(var,axis=0)  # delta-n or delta-T
                var0=np.mean(var,axis=0)      # n(n=0) or T0(n=0)
            else: 
                # '3d' -- var is dpot
                var2=var-np.mean(var,axis=0)
                try:
                    var0=self.f0.te0
                except:
                    if(err_msg_count==0):
                        print('f0.te0 is not available. Use 1 for normalization')
                        err_msg_count=1
                    var0=1

            dns = var2*var2
            dns = np.mean(dns,axis=0) # toroidal average
            dns = dns/(var0*var0)     # normalization with n0
            dns_surf = self.fsa_simple(dns*msk) # flux surface average with masking
            
            #print(it)
            if(count==0):
                turb_int=dns_surf
                time=time1
            else:
                turb_int = np.vstack((turb_int, dns_surf))
                time = np.append(time, time1)
                
        #fft and average
        return turb_int, time


    '''
    Basic analysis
    '''
    def profile_reports(self,i_name='Main ion',i2_name='Impurity', init_idx=0, end_idx=-1, edge_lim=[0.85,1.05]):       #wrapper for backward compatibility
        self.report_profiles(i_name=i_name, i2_name=i2_name, init_idx=init_idx, end_idx=end_idx, edge_lim=edge_lim)

    def report_profiles(self,i_name='Main ion',i2_name='Impurity', init_idx=0, end_idx=-1, edge_lim=[0.85,1.05]):

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
        self.report_turb_2d(i_name=i_name,i2_name=i2_name, pm=pm,tm=tm, wnorm=wnorm, cmap=cmap)

    def report_turb_2d(self,i_name='Main ion',i2_name='Impurity', pm=slice(0,-1),tm=slice(0,-1), wnorm=1E6, cmap='jet'):


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
    def midplane_var(self, var, inboard=False, nr=300, delta_r_axis = 0., delta_r_edge = 0., return_rmid=False):
        maxr = self.mesh.r.max() - delta_r_edge
        minr = self.mesh.r.min() + delta_r_edge

        if(inboard):
            r_mid = np.linspace(minr, self.eq_axis_r-delta_r_axis, nr)
        else:
            r_mid = np.linspace(self.eq_axis_r+delta_r_axis, maxr, nr)
        z_mid = np.linspace(self.eq_axis_z, self.eq_axis_z, nr)
        psi_tri = LinearTriInterpolator(self.mesh.triobj,self.mesh.psi/self.psix)
        psi_mid = psi_tri(r_mid, z_mid )

        var_tri = LinearTriInterpolator(self.mesh.triobj,var)
        var_mid = var_tri(r_mid, z_mid)

        if(return_rmid):
            return psi_mid, var_mid, r_mid
        else:   
            return psi_mid, var_mid

    #get GAM anlytic GAM frequency based on 1D diag and psi_surf
    def gam_freq_analytic(self):
        #finding region 1
        psi_surf=self.mesh.psi_surf/self.psix
        msk=np.nonzero(np.logical_and(psi_surf[:-1]<=1 , psi_surf[1:]>1))
        m=slice(0,msk[0][0])

        q = np.interp(self.od.psi, psi_surf[m], self.mesh.qsafety[m])
        f = (1+1/(2*q*q)) * np.sqrt( self.cnst.echarge*(self.od.Te+self.od.Ti)/self.cnst.protmass/self.unit_dic['ptl_ion_mass_au'] )/ (2*np.pi* self.eq_axis_r)
        return f


    # read one variable from filestr -- for 3d and f3d files. 
    # it might work with other files, too.
    def read_one_ad2_var(self,filestr,varstr, with_time=False):
        f=adios2.FileReader(filestr)
        #f.__next__()
        var=f.read(varstr)
        if(with_time):
            try:
                time=f.read('time')
            except:
                time=0
        f.close()
        if(with_time):
            return var, time
        else:
            return var


    class xgc_mat(object):
        def create_sparse_xgc(self,nelement,eindex,value,m=None,n=None):
            """Create Python sparse matrix from XGC data"""
            from scipy.sparse import csr_matrix
            
            if m is None: m = nelement.size
            if n is None: n = nelement.size
            
            #format for Python sparse matrix
            indptr = np.insert(np.cumsum(nelement),0,0)
            indices = np.empty((indptr[-1],))
            data = np.empty((indptr[-1],))
            for i in range(nelement.size):
                    indices[indptr[i]:indptr[i+1]] = eindex[i,0:nelement[i]]
                    data[indptr[i]:indptr[i+1]] = value[i,0:nelement[i]]
            #create sparse matrices
            spmat = csr_matrix((data,indices,indptr),shape=(m,n))
            return spmat


    class grad_rz(xgc_mat):
        """
        gradient operation
        """
        def __init__(self,path):
            with adios2.FileReader(path+"xgc.grad_rz.bp") as f:
                try:
                    # Flag indicating whether gradient is (R,Z) or (psi,theta)
                    self.mat_basis = f.read('basis')

                    # Set up matrix for psi/R derivative
                    nelement       = f.read('nelement_r')
                    eindex         = f.read('eindex_r')-1
                    value          = f.read('value_r')
                    nrows          = f.read('m_r')
                    ncols          = f.read('n_r')
                    self.mat_psi_r=self.create_sparse_xgc(nelement,eindex,value,m=nrows,n=ncols)

                    # Set up matrix for theta/Z derivative
                    nelement       = f.read('nelement_z')
                    eindex         = f.read('eindex_z')-1
                    value          = f.read('value_z')
                    nrows          = f.read('m_z')
                    ncols          = f.read('n_z')
                    self.mat_theta_z=self.create_sparse_xgc(nelement,eindex,value,m=nrows,n=ncols)

                except:
                    self.mat_psi_r   = 0
                    self.mat_theta_z = 0
                    self.mat_basis   = 0

    def load_grad_rz(self):
        self.grad = self.grad_rz(self.path) # need to fix

    "ff_mappings"
    class ff_mapping(xgc_mat):
        def __init__(self,ff_name,path):
                fn       ='xgc.ff_'+ff_name+'.bp'
                with adios2.FileReader(fn) as f:
                    nelement = f.read('nelement')
                    eindex   = f.read('eindex')-1
                    value    = f.read('value')
                    nrows    = f.read('nrows')
                    ncols    = f.read('ncols')
                    dl_par   = f.read('dl_par')
                    self.mat=self.create_sparse_xgc(nelement, eindex, value, m=nrows, n=ncols)
                    #
                    self.dl  = dl_par

    def load_ff_mapping(self):
        map_names = ["1dp_fwd","1dp_rev","hdp_fwd","hdp_rev"]
        for ff_name in map_names:
            #tmp=ff_mapping(ff_name)
            self.__setattr__('ff_'+ff_name,self.ff_mapping(ff_name,path))

    # Converts field into field-following representation (projection to midplane of
    # of a toroidal section)
    # input is expected to have shape (nnode,nphi,dim_field) or (nnode,nphi) (assuming dim_field=1)
    # output will be (nphi,nnode,dim_field,2) for dim_field=3 or
    # (nphi,nnode,2) for dim_field=1
    def conv_real2ff(self,field):
        if (field.ndim==3):
            field_work = field
        elif (field.ndim==2):
            field_work = np.zeros((field.shape[0],field.shape[1],1),dtype=field.dtype)
            field_work[:,:,0] = field[:,:]
        else:
            print("conv_real2ff: input field has wrong shape.")
            return -1
        fdim = field_work.shape[2]
        nphi = field_work.shape[1]
        field_ff = np.zeros((field_work.shape[0],nphi,fdim,2),dtype=field_work.dtype)
        for iphi in range(nphi):
            iphi_l  = iphi-1 if iphi>0 else nphi-1
            iphi_r  = iphi
            for j in range(fdim):
                field_ff[:,iphi,j,0] = self.ff_hdp_rev.mat.dot(field_work[:,iphi_l,j])
                field_ff[:,iphi,j,1] = self.ff_hdp_fwd.mat.dot(field_work[:,iphi_r,j])
        field_ff = np.transpose(field_ff,(1,0,2,3))
        if fdim==1:
            field_ff = (np.transpose(field_ff,(0,1,3,2)))[:,:,:]
        #
        return field_ff



    # Calculates the (psi,theta)/(R,Z) derivative of field. from RH xgc.py
    def GradPlane(self,field):
        if field.ndim>2:
            print("GradPlane: Wrong array shape of field, must be (nnode,nphi) or (nnode)")
            return -1
        nnode = field.shape[0]
        if field.ndim==2:
            field_loc = field
            nphi = field.shape[1]
        else:
            nphi           = 1
            field_loc      = np.zeros((nnode,nphi),dtype=field.dtype)
            field_loc[:,0] = field
        grad_field = np.zeros((nnode,nphi,2),dtype=field.dtype)
        for iphi in range(nphi):
            grad_field[:,iphi,0] = self.grad.mat_psi_r.dot(field_loc[:,iphi])
            grad_field[:,iphi,1] = self.grad.mat_theta_z.dot(field_loc[:,iphi])
        return grad_field


    # Calculates the 2nd order accurate finite difference derivative
    # of field along the magnetic field, i.e., b.grad(field)
    def GradParX(self,field):
        if field.ndim!=2:
            print("GradParX: Wrong array shape of field, must be (nnode,nphi)",field.shape)
            return -1
        nphi  = field.shape[1]
        nnode = field.shape[0]
        if nnode!=self.ff_1dp_fwd.mat.shape[0]:
            return -1
        sgn   = np.sign(self.bfield[2,0]) # toroidal field at the magnetic axis
        l_l   = self.ff_1dp_rev.dl
        l_r   = self.ff_1dp_fwd.dl
        l_tot = l_r+l_l
        bdotgrad_field = np.zeros_like(field)
        for iphi in range(nphi):
            iphi_l  = iphi-1 if iphi>0 else nphi-1
            iphi_r  = np.fmod((iphi+1),nphi)
            field_l = self.ff_1dp_rev.mat.dot(field[:,iphi_l])
            field_r = self.ff_1dp_fwd.mat.dot(field[:,iphi_r])
            #
            bdotgrad_field[:,iphi] = sgn * (-(    l_r)/(l_l*l_tot)*field_l        \
                                            +(l_r-l_l)/(l_l*l_r  )*field[:,iphi]  \
                                            +(    l_l)/(l_r*l_tot)*field_r        )
        return bdotgrad_field


    def convert_3d_grad_all(self,field):
        if field.ndim!=2:
            print("convert_3d_grad_all: Wrong array shape of field, must be (nnode,nphi)")
            return -1
        nphi  = field.shape[1]
        nnode = field.shape[0]
        grad_field = np.zeros((nnode,nphi,3))
        grad_field[:,:,0:2] = self.GradPlane(field)
        grad_field[:,:,2]   = self.GradParX(field)
        return grad_field


    def write_dAs_ff_for_poincare(self,fnum):
        #load As
        fn='xgc.3d.%5.5d.bp'%fnum
        with adios2.FileReader(fn) as f:
            As = f.read('apars').transpose()
            print('Read As[%d,%d] from '%(As.shape[0],As.shape[1]) +fn)
            print('As',As.shape)

            # Calculate grad(As) and transform As and grad(As) to
            # field-following representation
            dAs        = self.convert_3d_grad_all(As)
            print('dAs',dAs.shape)

            As_phi_ff  = self.conv_real2ff(As)
            print('As_phi_ff',As_phi_ff.shape)

            dAs_phi_ff = -self.conv_real2ff(dAs)
            print('dAs_phi_ff',dAs_phi_ff.shape)

            # Write Adios file with perturbed vector potential in
            # field-following representation
            import adios2 as ad
            fn2='xgc.dAs.%5.5d.bp'%fnum
            fbp   = ad.open(fn2,"w")
            nphi  = dAs_phi_ff.shape[0]
            nnode = dAs_phi_ff.shape[1]
            fbp.write("nphi",np.array([nphi]))
            fbp.write("nnode",np.array([nnode]))
            # For some reason the numpy data layout for these variables is not
            # C-style --> make contiguous.
            dum = np.ascontiguousarray(As_phi_ff)
            fbp.write("As_phi_ff",dum, dum.shape, [0]*len(dum.shape), dum.shape)
            dum = np.ascontiguousarray(dAs_phi_ff)
            fbp.write("dAs_phi_ff",dum, dum.shape, [0]*len(dum.shape), dum.shape)
            fbp.close()

    #f-source 1D 
    # sp = 'e_', 'i_', 'i2', 'i3', ...
    # moments = 'density', 'energy', 'torque'
    # source_type = 'collision', 'heat_torque', 'neutral', 'pellet', 'radiation', 'total', total2'
    def source_simple(self, step, period, sp='i_', moments='energy', source_type='heat_torque'):

        with adios2.FileReader("xgc.fsourcediag.%5.5d.bp"%step) as f:
            var=f.read(sp+moments+'_change_'+source_type)
            den=f.read(sp+'density_' +source_type)
            vol=f.read(sp+'volume_'  +source_type) 

        dt=period * self.sml_dt
        change_per_time=var*den*vol*self.sml_wedge_n/dt
        var_1d = self.flux_sum_simple(change_per_time)
        sum_1d = np.cumsum(var_1d)
        print('Total change=',np.sum(change_per_time))
        return var_1d, sum_1d
    
    def plot_source_simple(self, step, period, sp='i_', moments='energy', source_type='heat_torque'):

        var_1d, sum_1d = self.source_simple(step, period, sp=sp, moments=moments, source_type= source_type)
        plt.plot(self.od.psi,sum_1d)
        plt.xlabel('Normalized Pol. Flux')
        plt.title(sp+moments+'_'+source_type)
        #ax.set(xlabel='Normalized Pol. Flux')

    # gyroradius calculation
    # t_ev: temperature in eV (can be array)
    # b: magnetic field in Tesla (can be array)
    # mass_au: mass in atomic unit (scalar)
    # charge_eu: charge in electron unit (scalar)
    def gyro_radius(self, t_ev, b, mass_au, charge_eu):
        mass = mass_au * self.cnst.protmass
        return 1/(charge_eu * b) * np.sqrt(mass*t_ev/self.cnst.echarge)
