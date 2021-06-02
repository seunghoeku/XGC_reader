"""Module of the XGC1 loader for regerating general plots using ADIOS2
Some parts are taken from Michael's xgc.py which is taken from Loic's load_XGC_local for BES.
It reads the data from the simulation especially 1D results and other small data output.

TODO
3D data are loaded only when it is specified.
"""

import numpy as np
import os
from matplotlib.tri import Triangulation
import adios2
import matplotlib.pyplot as plt
from scipy.io import matlab
from scipy.optimize import curve_fit
from scipy.special import erfc
import scipy.sparse as sp


class xgc1(object):
    
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

    def load_oned(self):
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
        Tiperp=self.od.i_perp_temperature_df_1d
        Tipara=self.od.i_parallel_mean_en_df_1d  #parallel flow ignored, correct it later
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
            with adios2.open(filename,"r") as self.f:
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
        def __init__(self,filename,irg):
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
                        elif len(c)==3 : # ct[0] is irg, read only 
                            setattr(self,v,np.squeeze(self.f.read(v,start=[irg,0,0], count=[1,c[1],c[2]], step_start=0, step_count=stc)))
                    elif v!='zsamples' and v!='rsamples':
                        setattr(self,v,self.f.read(v,start=[], count=[], step_start=0, step_count=stc)) #null list for scalar
                #keep last time step
                self.r=self.r[-1,:]
                self.z=self.z[-1,:]
        
        """ 
        get some parameters for plots of heat diag

        """
        def post_heatdiag(self,dt,ds):
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

            #get separatrix r
            self.rs=np.interp([1],self.psin,self.rmid)
            
            self.rmidsepmm=(self.rmid-self.rs)*1E3  # dist from sep in mm

            #get heat
            self.qe=(self.e_perp_energy_psi + self.e_para_energy_psi)/dt/ds
            self.qi=(self.i_perp_energy_psi + self.i_para_energy_psi)/dt/ds
            self.ge=self.e_number_psi/dt/ds
            self.gi=self.i_number_psi/dt/ds
            self.qt=self.qe+self.qi
            #imx=self.qt.argmax(axis=1)
            mx=np.amax(self.qt,axis=1)
            self.lq_int=mx*0 #mem allocation

            for i in range(mx.shape[0]):
                self.lq_int[i]=np.sum(self.qt[i,:]*self.drmid)/mx[i]

        """
        getting total heat (radially integrated) to inner/outer divertor.
        """
        def total_heat(self,dt,wedge_n):
            qe=wedge_n * (np.sum(self.e_perp_energy_psi,axis=1)+np.sum(self.e_para_energy_psi,axis=1))
            qi=wedge_n * (np.sum(self.i_perp_energy_psi,axis=1)+np.sum(self.i_para_energy_psi,axis=1))

            #find restart point and remove -- 

            # find dt in varying sml_dt after restart

            self.qe_tot=qe/dt
            self.qi_tot=qi/dt
            
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
            perform fitting for all time steps.
        """
        def eich_fit_all(self,**kwargs):
            # need pmask for generalization?
            pmask = kwargs.get('pmask', None)

            self.lq_eich=self.lq_int*0 #mem allocation

            for i in range(self.time.size):
                try :
                    popt,pconv = self.eich_fit1(self.qt[i,:],pmask)
                except:
                    popt=[0, 0, 0, 0]
                
                self.lq_eich[i]= popt[2]
    
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


    def load_heatdiag(self):
        """
        load xgc.heatdiag.bp and some post process
        """
        self.hl=[]
        self.hl.append( self.datahlp("xgc.heatdiag.bp",0) ) #actual reading routine
        self.hl.append( self.datahlp("xgc.heatdiag.bp",1) )#actual reading routine

        for i in [0,1] :
            try:
                self.hl[i].psin=self.hl[i].psi[0,:]/self.psix #Normalize 0 - 1(Separatrix)
            except:
                print("psix is not defined - call load_unitsm() to get psix to get psin")

        #read bfieldm data if available
        self.load_bfieldm()

        dt=self.unit_dic['sml_dt']*self.unit_dic['diag_1d_period']
        wedge_n=self.unit_dic['sml_wedge_n']
        for i in [0,1]:
            dpsin=self.hl[i].psin[1]-self.hl[i].psin[0]  #equal dist
            #ds = dR* 2 * pi * R / wedge_n
            ds=dpsin/self.bfm.dpndrs* 2 * 3.141592 * self.bfm.r0 /wedge_n  #R0 at axis is used. should I use Rs?
            self.hl[i].rmid=np.interp(self.hl[i].psin,self.bfm.psino,self.bfm.rmido)
            self.hl[i].post_heatdiag(dt,ds)
            self.hl[i].total_heat(dt,wedge_n)

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


                
    def load_m(self,fname):
        """load the whole  .m file and return a dictionary contains all the entries.
        """
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
        
        if(type(psi).__module__ != np.__name__):  #None or not numpy data
            psi=obj.psi #default psi is obj.psi
            
        if(type(var).__module__ != np.__name__):
            if(varstr==None):   
                print("Either var or varstr should be defined.")
            else:
                var=getattr(obj,varstr) #default var is from varstr
               
        stc=var.shape[0]
        fig, ax=plt.subplots()
        lbl=["Initial","Final"]
        if(xlim==None):
            if(initial):
                ax.plot(psi,var[0,],label='Initial')
            ax.plot(psi,var[stc-1,],label='Final')
        else:
            msk=(psi >= xlim[0]) & (psi <= xlim[1])
            if(initial):
                ax.plot(psi[msk],var[0,msk],label='Initial')
            ax.plot(psi[msk],var[stc-1,msk],label='Final')
                
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
            with adios2.open("xgc.mesh.bp","r") as fm:
                rz=fm.read('rz')
                self.cnct=fm.read('/cell_set[0]/node_connect_list')
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
                self.node_vol=fm.read('node_vol')
                self.qsafety=fm.read('qsafety')
                self.psi=fm.read('psi')


    class f0meshdata(object):    
        """
        mesh data class for 2D contour plot
        """
        def __init__(self):
            with adios2.open("xgc.f0.mesh.bp","r") as f:
                T_ev=f.read('f0_T_ev')
                den0=f.read('f0_den')
                flow=f.read('f0_flow')
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
    """
    class fluxavg(object):
        def __init__(self):
            with adios2.open("xgc.fluxavg.bp","r") as f:
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
            with adios2.open("xgc.volumes.bp","r") as f:
                self.od=f.read("diag_1d_vol")
                #try:
                self.adj_eden=f.read("psn_adj_eden_vol")

    class turbdata(object):
        """
        data for turb intensity
        assuming convert_grid2 for flux average
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
                with adios2.open(filename,"r") as f:
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
        
        
        self.od.efluxe    = self.od.e_gc_density_df_1d * self.od.e_radial_en_flux_df_1d * dvdpall
        self.od.efluxexbe = self.od.e_gc_density_df_1d * self.od.e_radial_en_flux_ExB_df_1d * dvdpall
        self.od.cfluxe    = self.od.e_gc_density_df_1d * self.od.Te * ec * self.od.e_radial_flux_df_1d * dvdpall
        self.od.cfluxexbe = self.od.e_gc_density_df_1d * self.od.Te * ec * self.od.e_radial_flux_ExB_df_1d * dvdpall
        self.od.pfluxe    = self.od.e_gc_density_df_1d * self.od.e_radial_flux_df_1d * dvdpall
        self.od.pfluxexbe = self.od.e_gc_density_df_1d * self.od.e_radial_flux_ExB_df_1d * dvdpall
        


    def plot2d(self,filestr,varstr,**kwargs):
        """
        general 2d plot
        filestr: file name
        varstr: variable name
        plane: 0 based plane index - ignored for axisymmetric data
        Improve it to handle box 
        additional var to add: box, levels, cmap, etc
        """
        box= kwargs.get('box', None) # rmin, rmax, zmin, zmax
        plane=kwargs.get('plane',0)
        levels = kwargs.get('levels', None)
        cmap = kwargs.get('cmap', 'jet')
        
        f=adios2.open(filestr,'r')
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
        print("particle number = %e" % (self.unit_dic['sml_totalpe']* self.unit_dic['ptl_num']))
    
