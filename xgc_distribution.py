
from typing import *
import xgc_reader
import numpy as np
import adios2
import matplotlib.pyplot as plt
import warnings
class VelocityGrid:
    """
    A class to represent a velocity grid for a distribution function.

    Attributes
    ----------
    nvperp : int
        Number of perpendicular velocity points.
    nvpara : int
        Number of parallel velocity points in one direction.
    nvpdata : int
        Number of parallel velocity data points (2 * nvpara + 1).
    vpara : numpy.ndarray
        Array of parallel velocities ranging from -vpara_max to vpara_max.
    vperp : numpy.ndarray
        Array of perpendicular velocities ranging from 0 to vperp_max.
    dvperp : float
        Increment in perpendicular velocity.
    dvpara : float
        Increment in parallel velocity.
    vpara_max : float
        Maximum parallel velocity.
    vperp_max : float
        Maximum perpendicular velocity.

    Methods
    -------
    __init__(nvperp: int, nvpara: int, vperp_max: float, vpara_max: float)
        Initializes the velocity grid with given parameters.
    """

    def __init__(self, nvperp: int, nvpara: int,
                 vperp_max: float, vpara_max: float):
        self.nvperp = nvperp
        self.nvpara = nvpara
        self.nvpdata = nvpara*2 +1

        self.vpara=np.linspace(-vpara_max, vpara_max, self.nvpdata)
        self.vperp=np.linspace(0, vperp_max, nvperp)

        self.dvperp = self.vperp[1] - self.vperp[0]
        self.dvpara = self.vpara[1] - self.vpara[0]

        self.vpara_max = vpara_max
        self.vperp_max = vperp_max
        
class XGCDistribution:
    """
    A class to represent the XGC distribution.
    Attributes
    ----------
    PROTON_MASS : float
        Mass of a proton in kilograms.
    E_CHARGE : float
        Elementary charge in coulombs.
    EV_TO_JOULE : float
        Conversion factor from electron volts to joules.
    MU0_FACTOR : float
        Factor applied at zero perpendicular velocity.
    Methods
    -------
    __init__(vgrid, nnodes, f, den, temp_ev, flow, fg_temp_ev, mass, has_maxwellian=True)
        Initializes the XGCDistribution object with given parameters.
    from_xgc_output(cls, filename, var_string='i_f', dir='./', time_step=0, mass_au=2.0)
        Class method to initialize the object from an XGC output file.
    from_xgc_input(cls, filename)
        Class method to initialize the object from an XGC input file.
    update_maxwellian_moments(self, x)
        Updates the moments of the Maxwellian distribution function.
    remove_maxwellian(self, get=False, add=False)
        Removes or adds the Maxwellian component from/to the distribution function.
    add_maxwellian(self)
        Adds the Maxwellian component to the distribution function.
    get_maxwellian(self)
        Returns the Maxwellian component of the distribution function.
    fix_axis_value(self, var, sz=10)
        Fixes the axis value by averaging the first few elements.
    save(self, filename="xgc.f_init.bp")
        Saves the distribution function and related data to a file.
    zero_out_fg(self)
        Zeros out the fg component of the distribution function.
    """
    # Class constants
    PROTON_MASS: ClassVar[float] = 1.6726219e-27
    E_CHARGE: ClassVar[float] = 1.602176634e-19
    EV_TO_JOULE: ClassVar[float] = 1.602176634e-19
    MU0_FACTOR: ClassVar[float] = 1/3
    
    #initialize from data passing
    def __init__(self, vgrid: 'VelocityGrid', nnodes: int, f: np.ndarray, den: np.ndarray, temp_ev: np.ndarray, flow: np.ndarray, fg_temp_ev: np.ndarray, mass: float, charge: float, has_maxwellian=True, has_boltzmann=False, dpot=None) -> None:
        self.vgrid = vgrid
        self.nnodes = nnodes
        self.den = den
        self.temp_ev = temp_ev
        self.flow = flow
        self.fg_temp_ev = fg_temp_ev
        self.mass = mass
        self.charge = charge
        self.has_maxwellian = has_maxwellian
        self.has_boltzmann = has_boltzmann

        if(has_maxwellian):
            self.f = f
        else:
            self.f_g = f

        if(has_boltzmann):
            self.dpot = dpot
            self.density_with_boltzmann = self.den * self.exp_ad(self.dpot/self.temp_ev)

        self.check_nan()


    #initialize from adios2 file like xgc.f0.00000.bp
    @classmethod
    def from_xgc_output(cls, filename, var_string='i_f', dir = './', time_step = 0, mass_au=2.0, charge_num = 1.0, has_electron=True, use_initial_moments=False):
        # read the specific species.
        if(var_string=='e_f'):
            idx = 0
        elif(var_string=='i_f'):
            idx = 1
        else:
            if var_string.startswith('i'):
                idx = int(var_string[1]) if var_string[1].isdigit() else 0
            else:
                raise ValueError(f"Unsupported var_string: {var_string}") 
        if(not has_electron):
            idx = idx-1

        print('species index:', idx)

        # electron distribution has boltzmann factor - need to read dpot, too
        if(var_string=='e_f'):
            has_boltzmann = True
        else:
            has_boltzmann = False
        with adios2.FileReader(dir+'/'+filename) as file:
            it=time_step
            ftmp=file.read(var_string) #,start=[],count=[],step_start=it, step_count=1)
            print('Reading data with shape:',np.shape(ftmp))
            step=file.read('step') #,start=[],count=[],step_start=it, step_count=1)
            ftmp = np.squeeze(ftmp) # remove time step dimension
            if(ftmp.ndim == 4):
                ftmp = np.mean(ftmp,axis=0) # toroidal average
            ftmp = np.swapaxes(ftmp, 0, 1)
            print('Loading '+var_string + ' from '+ filename+ '. The toroidal averaged distribution has dim of', np.shape(ftmp))
            nvperp = ftmp.shape[1]
            nvpara = int((ftmp.shape[2]-1)/2)
            nnodes=ftmp.shape[0]

            if(has_boltzmann):
                dpot = file.read('dpot')
                if(dpot.ndim == 2): # ndim=1 for XGCa.
                    dpot = np.mean(dpot,axis=0) # toroidal average -- assume that they are axisymmetric.
            else:
                dpot = None
        with adios2.FileReader(dir + 'xgc.f0.mesh.bp') as file:
            vp_max = file.read('f0_vp_max')
            smu_max = file.read('f0_smu_max')
            vgrid = VelocityGrid(nvperp, nvpara, smu_max, vp_max)
            
            t_tmp = file.read('f0_fg_T_ev')
            if(t_tmp.ndim != 2):
                t_tmp = file.read('f0_T_ev') # for old version

            fg_temp_ev = t_tmp[idx,:]  # need species index handling
            if(use_initial_moments):
                den= file.read('f0_den')
                temp_ev= file.read('f0_T_ev')
                flow = file.read('f0_flow')
                den = den[idx,:]
                temp_ev = temp_ev[idx,:]
                flow = flow[idx,:]
            else:
                den = np.zeros(nnodes)
                temp_ev = np.zeros(nnodes)
                flow = np.zeros(nnodes)

        mass = mass_au * cls.PROTON_MASS
        charge = charge_num * cls.E_CHARGE

        return cls(vgrid, nnodes, ftmp, den, temp_ev, flow, fg_temp_ev, mass, charge, has_maxwellian=True, has_boltzmann=has_boltzmann, dpot=dpot)

    #initialize from distribution input file -- same format as that of save()
    @classmethod
    def from_xgc_input(cls, filename):
        with adios2.FileReader(filename) as fr:
            f_g = fr.read("f_g")
            den = fr.read("density")
            temp_ev = fr.read("temperature")
            flow = fr.read("flow")
            fg_temp_ev = fr.read("fg_temp_ev")
            mass = fr.read("mass")
            charge = fr.read("charge")
            vgrid = VelocityGrid(fr.read("nvperp"), fr.read("nvpara"), fr.read("vperp_max"), fr.read("vpara_max"))
            nnodes = fr.read("nnodes")
            has_boltzmann=fr.read("has_boltzmann")
            dpot = fr.read("dpot")

        return cls(vgrid, nnodes, f_g, den, temp_ev, flow, fg_temp_ev, mass, charge, has_maxwellian=False, has_boltzmann=has_boltzmann, dpot=dpot)

    # check nan values in the data
    def check_nan(self):
        if(self.has_maxwellian):
            if(np.isnan(self.f).any()):
                print('f has nan')
        else:
            if(np.isnan(self.f_g).any()):
                print('f_g has nan')

        if(np.isnan(self.den).any()):
            print('den has nan')
        if(np.isnan(self.temp_ev).any()):
            print('temp_ev has nan')
        if(np.isnan(self.flow).any()):
            print('flow has nan')
        if(np.isnan(self.fg_temp_ev).any()):
            print('fg_temp_ev has nan')

    # keep maxwellian component and remove the rest
    def clear_nonmaxwellian(self):
        if(self.has_maxwellian):
            self.remove_maxwellian()
        self.f_g[:,:,:] = 0.0


    # update maxwellian distribution function
    def update_maxwellian_moments(self, xr, do_flux_average=True, no_update=False): # x required for flux surface average

        # update_maxwellian_moments only for full-f distribution including maxwellian component.
        if(not self.has_maxwellian):
            self.add_maxwellian()

        # apply mu0_factor at zero vperp
        self.f[:,0,:] = self.f[:,0,:] * self.MU0_FACTOR

        #get vspace volume
        vspace_vol = self.fg_temp_ev * np.sqrt(1/(np.pi*2)) * self.vgrid.dvperp * self.vgrid.dvpara

        # get moments

        # particle density
        ptls = np.sum(self.f, axis=(1, 2))
        if(np.isnan(ptls).any()):
            print('ptls has nan')
        den = ptls * vspace_vol

        # flow
        flow = np.sum(self.f * self.vgrid.vpara[np.newaxis, np.newaxis, :], axis=(1, 2))
        if(np.isnan(flow).any()):
            print( 'flow has nan:')

        flow = flow/ ptls * np.sqrt(self.fg_temp_ev * self.EV_TO_JOULE/ self.mass) 

        # temperature
        en1 = np.add.outer(self.vgrid.vperp**2/2, self.vgrid.vpara**2/2) # check factoer 1/2        
        temp = np.sum(self.f * en1[np.newaxis,:, :], axis=(1, 2)) /ptls * self.fg_temp_ev  # mean enrgy
        temp = temp - 0.5 * flow**2 * self.mass/self.EV_TO_JOULE # moving frame
        temp = temp * 2 / 3 # mean energy to temperature

        # flux surface average
        if(do_flux_average):
            den = flux_surface_average(den, xr)
            if(np.isnan(den).any()):
                print('den has nan after flux average')

            temp = flux_surface_average(temp, xr) 

            flow = flux_surface_average(flow/xr.mesh.r, xr)*xr.mesh.r # u/R (parallel rotation freq) flux average
            if(np.isnan(flow).any()):
                print('flow has nan after flux average')


        # undo multiplication by MU0_FACTOR
        self.f[:,0,:] = self.f[:,0,:] / self.MU0_FACTOR

        # update maxwellian moments
        if(no_update):
            return den, temp, flow
        else:
            self.den = den
            self.flow = flow
            self.temp_ev = temp


    # add or remove maxwellian to f_g or from f
    def remove_maxwellian(self, get=False, add=False):        
        if(get):
            fm = np.zeros((self.nnodes, self.vgrid.nvperp, self.vgrid.nvpdata))
        else: # check invalid cases and set parameters
            #check invalid cases
            if(not add and not self.has_maxwellian):
                print('remove_maxwellian: No maxwellian component to remove')
                return
            if(add and self.has_maxwellian):
                print('remove_maxwellian: Maxwellian component already exists')
                return
            
            #set parameters
            if not add: # remove maxwellian component
                sign = -1
                f_array = self.f
            else: # add maxwellian component
                sign = 1
                f_array = self.f_g


        #actual calculation
        for i in range(self.vgrid.nvperp):
            for j in range(self.vgrid.nvpdata):
                v_n = np.sqrt(self.fg_temp_ev*self.E_CHARGE/self.mass)
                en =0.5* self.mass * ((self.vgrid.vpara[j]*v_n-self.flow)**2 + (self.vgrid.vperp[i]*v_n)**2) / (self.temp_ev*self.EV_TO_JOULE) # normalized energy by T
                if np.any(en<0):
                    print('Negative energy:', en[en <0])
                    print('Negative energy index:', np.where(en<0))
                    en[en<0]=0
                tmp_exp = np.exp(-en)
                tmp = self.den * tmp_exp / (self.temp_ev)**1.5 * self.vgrid.vperp[i]  * np.sqrt(self.fg_temp_ev)

                # mimic `vgrid.inv_mu0_factor = 1/3` correction for ivr==0
                if i==0:
                    tmp = self.den * tmp_exp / (self.temp_ev)**1.5 * self.vgrid.dvperp/3. * np.sqrt(self.fg_temp_ev)

                if(self.has_boltzmann):
                    tmp = tmp * self.exp_ad(self.dpot/self.temp_ev)

                if(get):
                    fm[:,i,j] = tmp
                else:
                    f_array[:,i,j]=f_array[:,i,j] + sign * tmp

        # when get=True, return the maxwellian component        
        if(get):
            return fm
        
        
        if(sign==-1): # remove maxwellian component
            self.has_maxwellian = False
            self.f_g = self.f
            delattr(self,'f')
        if(sign==1): # add maxwellian component
            self.has_maxwellian = True
            self.f = self.f_g
            delattr(self,'f_g')

    # adding maxwellian to f_g and rename it to f. simple interface to remove maxwellian
    def add_maxwellian(self):
        self.remove_maxwellian(add=True)

    # get maxwellian component. It does not change the distribution data
    def get_maxwellian(self):
        return self.remove_maxwellian(get=True)


    # save data structure to adios2 file
    def save(self, filename="xgc.f_init.bp"):
        #when writing to file, remove maxwellian component
        if(self.has_maxwellian):
            self.remove_maxwellian()

        with adios2.Stream(filename, mode='w') as fw:
            adios2_write_array(fw, "f_g", self.f_g) # maxwellian component is removed.
            adios2_write_array(fw, "density", self.den)
            adios2_write_array(fw, "temperature", self.temp_ev)
            adios2_write_array(fw, "flow", self.flow)
            adios2_write_array(fw, "fg_temp_ev", self.fg_temp_ev)
            fw.write("mass", np.array([self.mass]))
            fw.write("charge", np.array([self.charge]))

            fw.write("nvpara", np.array([self.vgrid.nvpara], dtype=np.int32))
            fw.write("nvperp", np.array([self.vgrid.nvperp], dtype=np.int32))
            fw.write("vpara_max", np.array([self.vgrid.vpara_max], dtype=np.float64))
            fw.write("vperp_max", np.array([self.vgrid.vperp_max], dtype=np.float64))
            fw.write("nnodes", np.array([self.nnodes], dtype=np.int32))
            if(self.has_boltzmann):
                adios2_write_array(fw, "density_with_boltzmann", self.density_with_boltzmann) # This needs only for electron
            if hasattr(self, "dpot"):
                adios2_write_array(fw, "dpot", self.dpot) # This needs only for electron

    # zero out f_g 
    def zero_out_fg(self):
        if(self.has_maxwellian):
            delattr(self,'f')
        self.f_g = np.zeros((self.nnodes, self.vgrid.nvperp, self.vgrid.nvpdata))
        self.has_maxwellian = False

    # setup canonical maxwellian
    # correction is to correct the psi_c value for being close to psi
    def canonical_maxwellian(self, xr, psi_den, den_c, psi_temp, temp_ev_c, correction):
        fcm = np.zeros((self.nnodes, self.vgrid.nvperp, self.vgrid.nvpdata))

        bmag = np.sqrt(xr.bfield[0,:]**2 + xr.bfield[1,:]**2 + xr.bfield[2,:]**2)
        q_m = self.E_CHARGE/self.mass
        m_q = 1/q_m
        #actual calculation
        for i in range(self.vgrid.nvperp):
            for j in range(self.vgrid.nvpdata):
                v_n = np.sqrt(self.fg_temp_ev*q_m)
                vpara = self.vgrid.vpara[j]*v_n # no flow
                vperp = self.vgrid.vperp[i]*v_n
                en =0.5* self.mass * (vpara**2 + vperp**2) 
                psi_c = xr.mesh.psi + m_q * xr.mesh.r * xr.bfield[2,:]/bmag * vpara
                mu = 0.5 * self.mass * vperp**2 / bmag
                if(correction):
                    h = en - mu * xr.eq_axis_b
                    h = np.maximum(h,0.0) # hevyside function multiplied.
                    psi_c = psi_c - np.sign(vpara)* m_q * xr.eq_axis_r * np.sqrt(2*h/self.mass)
                
                den = np.interp(psi_c, psi_den, den_c)
                temp_ev = np.interp(psi_c, psi_temp, temp_ev_c)
                en = en/ (temp_ev*self.EV_TO_JOULE) # normalized energy by T
                fcm[:,i,j] = den * np.exp(-en) / (temp_ev)**1.5 * self.vgrid.vperp[i]  * np.sqrt(self.fg_temp_ev)
        return fcm

    # resize the distribution function with new # of nodes.
    def resize(self) -> None:
        self.vgrid = VelocityGrid(self.vgrid.nvperp, self.vgrid.nvpara, self.vgrid.vperp_max, self.vgrid.vpara_max)

        self.den = np.zeros(self.nnodes)
        self.temp_ev = np.zeros(self.nnodes)
        self.flow = np.zeros(self.nnodes)
        self.fg_temp_ev = np.zeros(self.nnodes)
        if(self.has_maxwellian):
            self.f = np.zeros((self.nnodes, self.vgrid.nvperp, self.vgrid.nvpdata))
        else:
            self.f_g = np.zeros((self.nnodes, self.vgrid.nvperp, self.vgrid.nvpdata))

    # Get the canonical maxwellian distribution function iteratively to match the moments
    # psi_den and psi_temp are unnormlized psi values
    def set_canonical_maxwellian_iterative(self, xr, psi_den, den_target_in, psi_temp, temp_ev_target_in, correction=True, tol=1e-2, max_iter=30, show_fig=False):

        # interpolate the target moments to oned psi 
        psi = xr.od.psi*xr.psix
        den_target=np.interp(psi, psi_den, den_target_in)
        temp_ev_target=np.interp(psi, psi_temp, temp_ev_target_in)

        den_c = den_target.copy()
        temp_ev_c = temp_ev_target.copy()
        den_norm = np.sqrt(np.mean(den_target**2))
        temp_ev_norm = np.sqrt(np.mean(temp_ev_target**2))
        
        if(show_fig):
            fig, ax = plt.subplots(2,2)
            ax[0,0].plot(psi/xr.psix, den_target, label='target')
            ax[0,1].plot(psi/xr.psix, temp_ev_target, label='target')

        for iter in range(max_iter):
            self.set_canonical_maxwellian(xr, psi, den_c, psi , temp_ev_c, correction)
            
            # get the difference between the target and the calculated moments
            den_diff = den_target - np.interp(psi, xr.mesh.psi_surf, xr.fsa_simple(self.den))
            temp_ev_diff = temp_ev_target - np.interp(psi, xr.mesh.psi_surf, xr.fsa_simple(self.temp_ev))
            error_den = np.sqrt(np.mean(den_diff**2))/den_norm 
            error_temp = np.sqrt(np.mean(temp_ev_diff**2))/temp_ev_norm
            error = np.sqrt(error_den**2 + error_temp**2)            
            print('Iter:', iter, 'Error:', error, 'Error_den:', error_den, 'Error_temp:', error_temp)
            if(error < tol):
                break

            # update the canonical maxwellian moments
            if(error_den > error_temp):
                den_c = den_c + den_diff
            else:
                temp_ev_c = temp_ev_c + temp_ev_diff
            
            if(show_fig):
                ax[1,0].plot(psi/xr.psix, den_c, label='iter'+str(iter))                
                ax[1,1].plot(psi/xr.psix, temp_ev_c, label='iter'+str(iter))
                ax[0,0].plot(xr.mesh.psi/xr.psix, self.den, label='iter'+str(iter))
                ax[0,1].plot(xr.mesh.psi/xr.psix, self.temp_ev, label='iter'+str(iter))

        if(show_fig):
            ax[0,0].set_title('Density local')
            ax[0,1].set_title('Temperature local')
            ax[1,0].set_title('Density canonical')
            ax[1,1].set_title('Temperature canonical')
            ax[0,0].legend()
            ax[0,1].legend()
            ax[1,0].legend()
            ax[1,1].legend()
            plt.show()



    # psi_den and psi_temp are unnormlized psi values
    def set_canonical_maxwellian(self, xr, psi_den, den_c, psi_temp, temp_ev_c, correction=True):
        if(not self.has_maxwellian):
            delattr(self,'f_g')
        self.f = self.canonical_maxwellian(xr, psi_den, den_c, psi_temp, temp_ev_c, correction)
        self.has_maxwellian = True
        self.update_maxwellian_moments(xr)

    # contour plot at the given node
    def contour_plot(self, nnode):
        if(self.has_maxwellian):
            f=self.f
        else:
            f=self.f_g

        contour = plt.contourf(self.vgrid.vpara, self.vgrid.vperp, f[nnode,:,:])
        plt.colorbar(contour)


    def exp_ad(self, x):
        a: Final=0.7
        tmp = np.exp(-np.abs(x) + a) * (1-a)
        tmp2 = np.where(x < -a, tmp, 1.0+x)
        return (np.where(x>a, 2-tmp, tmp2))

    # remove boltzmann factor in maxwellian component
    # Still maxwellian w/o botzmann factor is kept.
    def remove_boltzmann(self):
        if(not self.has_boltzmann):
            print('No Boltzmann factor to remove')
            return
    
        # It asssumes that the original distribution already has boltzmann factor
        # If maxwellian is removed, that must be wrong and it gives worng f_g.
        # So, add back maxwellian first to get the original f.
        self.add_maxwellian()
        bltz_factor = self.exp_ad(self.dpot/self.temp_ev)

        #actual calculation - this for loop is based on remove_maxwellian()
        for i in range(self.vgrid.nvperp):
            for j in range(self.vgrid.nvpdata):
                v_n = np.sqrt(self.fg_temp_ev*self.E_CHARGE/self.mass)
                en =0.5* self.mass * ((self.vgrid.vpara[j]*v_n-self.flow)**2 + (self.vgrid.vperp[i]*v_n)**2) / (self.temp_ev*self.EV_TO_JOULE) # normalized energy by T
                if np.any(en<0):
                    print('Negative energy:', en[en <0])
                    print('Negative energy index:', np.where(en<0))
                    en[en<0]=0
                tmp_exp = np.exp(-en)
                tmp = self.den * tmp_exp / (self.temp_ev)**1.5 * self.vgrid.vperp[i]  * np.sqrt(self.fg_temp_ev)

                # mimic `vgrid.inv_mu0_factor = 1/3` correction for ivr==0
                if i==0:
                    tmp = self.den * tmp_exp / (self.temp_ev)**1.5 * self.vgrid.dvperp/3. * np.sqrt(self.fg_temp_ev)

                # remove maxwellian with botzmann factor and add maxwellian without boltzmann factor.
                self.f[:,i,j] = self.f[:,i,j] + tmp* (1-bltz_factor)

        self.has_boltzmann = False
        
# interpolate flux surface moments to mesh
def interp_flux_surface_moments(psi_surf, moments_surf, xr):
    # find last point of psi_surf that monotonically increases
    for i in range(len(psi_surf)-1,0,-1):
        if psi_surf[i] > psi_surf[i-1]:
            break
    var_mesh = np.interp(xr.mesh.psi, psi_surf[:i+1], moments_surf[:i+1])  # region 3 need to ba handled separately
    return var_mesh

#flux surface average using xgc_reader data and scatter back to the mesh
# private region need to be implemented.
def flux_surface_average(var, xr):

    # check if NaN exists
    if(np.isnan(var).any()):
        print('NaN exists in var')
        idx=np.isnan(var)[0]
        print('NaN at:', idx, var[idx])
        
    fsa_surf = xr.fsa_simple(var)
    var_favg = interp_flux_surface_moments(xr.mesh.psi_surf, fsa_surf, xr)
    msk=xr.mesh.region>2
    var_favg[msk] = var[msk] # use original value for private region
    return (var_favg)

        
# write an array to an adios2 file - upto 4D array.        
def adios2_write_array(file, name, data):
    if (data.ndim == 1):
        start=[0]
        count=[data.shape[0]]
    elif (data.ndim == 2):
        start=[0,0]
        count=[data.shape[0], data.shape[1]]
    elif (data.ndim == 3):
        start=[0,0,0]
        count=[data.shape[0], data.shape[1], data.shape[2]]
    elif (data.ndim == 4):
        start=[0,0,0,0]
        count= [data.shape[0], data.shape[1], data.shape[2], data.shape[3]]
    else:
        print("Data dimension not supported")
        
    file.write(name, np.ascontiguousarray(data), data.shape, start=start, count=count)


# Convert distribution to another mesh
# The basic algorithm is that interpolate the normalized distribution function to the new mesh.
# This approach could cause some issues when the normalization factor (fg_temp_ev) is varying a lot.
# Algorithm:
# for each velocity node point
#  0. Assume that nvperp, nvpara, vperp_max, vpara_max are same for the new mesh.
#  1. interpolate the normalization factor to the new mesh.
#  2. interpolate the distribution function to the new mesh.
def convert_distribution(dist, xr_old, xr_new, update_moments=True, remove_maxwellian=True):
    
    import copy
    import matplotlib.tri as tri
    from scipy.spatial import cKDTree
    
    # currently it interpolates the distribution.
    # alternative approach is to interpolate the moments and then f_g
    # 0 add maxwellian
    if(not dist.has_maxwellian):
        dist.add_maxwellian()

    # 0.1 copy the VelocityGrid object
    dist_new=copy.deepcopy(dist)

    # 0.2 create new dist object with new mesh -- nnodes, mass, has_maxwellian are same.
    dist_new.nnodes = xr_new.mesh.nnodes
    dist_new.resize()

    # 0.3 find the points that are outside the mesh - to avoid NaN from interpolation
    tf = xr_old.mesh.triobj.get_trifinder()
    triangle_indices=tf(xr_new.mesh.r,xr_new.mesh.z)
    outside = triangle_indices==-1

    # 0.4 replace the outside points with the nearest points
    points = np.column_stack((xr_old.mesh.r, xr_old.mesh.z))
    query_points = np.column_stack((xr_new.mesh.r[outside], xr_new.mesh.z[outside]))
    tree = cKDTree(points)    # Create a cKDTree from the reference points
    distances, indices = tree.query(query_points) # Query the tree to find the nearest points
    nearest_points = points[indices] # Get the nearest points

    r = np.copy(xr_new.mesh.r)
    z = np.copy(xr_new.mesh.z)
    r[outside] = nearest_points[:,0]
    z[outside] = nearest_points[:,1]

    # 1. interpolate fg_temp_ev (+moments) to the new mesh
    interpolator = tri.LinearTriInterpolator(xr_old.mesh.triobj, dist.fg_temp_ev)
    dist_new.fg_temp_ev = interpolator(r, z)
    interpolator = tri.LinearTriInterpolator(xr_old.mesh.triobj, dist.den)
    dist_new.den = interpolator(r, z)
    interpolator = tri.LinearTriInterpolator(xr_old.mesh.triobj, dist.temp_ev)
    dist_new.temp_ev = interpolator(r, z)
    interpolator = tri.LinearTriInterpolator(xr_old.mesh.triobj, dist.flow)
    dist_new.flow = interpolator(r, z)
    
    # 2. interpolate the distribution function to the new mesh

    # 2.1 interpolate dist.f to dist_new.f
    for i in range(dist.vgrid.nvperp):
        for j in range(dist.vgrid.nvpdata):
            interpolator = tri.LinearTriInterpolator(xr_old.mesh.triobj, dist.f[:,i,j])
            dist_new.f[:,i,j] = interpolator(r, z)

    # 2.2 update moments
    if(update_moments):
        dist_new.update_maxwellian_moments(xr_new)

    # 2.3 remove maxwellian from dist_new.f
    if(update_moments and remove_maxwellian):
        dist_new.remove_maxwellian()

    return dist_new

# adjust private flux density
#1. set flat private flux density or decaying density like input
#2. scale f as den_new / dist.den
#3. update maxwellian part
def adjust_private_flux_density(dist, xr, decay_factor=1, decay_width=1.5E-2):
    
    #1. set flat private flux density or decaying density like input    
    msk = xr.mesh.region>2
    den_new = np.zeros_like(dist.den[msk])

    #get density at the X-point (separatrix)
    ix = np.argmin((xr.mesh.r-xr.eq_x_r)**2 + (xr.mesh.z-xr.eq_x_z)**2)
    den_new = dist.den[ix] # set to the density at the X-point
    
    # adding decay factor
    
    factor=(1-decay_factor)*np.exp(-abs(xr.mesh.psi[msk]-xr.psix)/decay_width)+decay_factor;
    den_new= factor * den_new

    #2. scale f as den_new / dist.den
    
    #2.1 add maxwellian
    dist.add_maxwellian()

    #2.2 scale f
    for i in range(dist.vgrid.nvperp):
        for j in range(dist.vgrid.nvpdata):
            dist.f[msk,i,j] = dist.f[msk,i,j] * den_new / dist.den[msk]

    #2.3 update moments
    dist.update_maxwellian_moments(xr)

#get local moments
def get_local_moments(dist, xr):
    den, temp_ev, flow = dist.update_maxwellian_moments(xr, do_flux_average=False, no_update=True)
    return den, temp_ev, flow


# regularize density near X-point region
# default radius is 10cm
# decay factor is (psi-psix)/
def regularize_near_x(dist, xr, radius=0.1, del_psi_n = 0.01):
    #1. define region
    rr = np.sqrt( (xr.mesh.r - xr.eq_x_r)**2 + (xr.mesh.z - xr.eq_x_z)**2 )
    msk =  rr < radius
    tmp = np.abs(xr.mesh.psi[msk]/xr.psix -1) / del_psi_n
    factor = np.exp(-tmp**2) * (1- rr[msk]/radius)**2

    den_l, temp_l, flow_l = get_local_moments(dist, xr)
    den_f, temp_f, flow_f = dist.update_maxwellian_moments(xr, do_flux_average=True, no_update=True)

    factor2 = den_f[msk]/den_l[msk] * factor + (1-factor)

    #2. scale f
    for i in range(dist.vgrid.nvperp):
        for j in range(dist.vgrid.nvpdata):
            dist.f[msk,i,j] = dist.f[msk,i,j] * factor2
