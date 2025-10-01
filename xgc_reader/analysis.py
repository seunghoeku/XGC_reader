"""Advanced analysis functions for turbulence, sources, and plasma physics calculations."""

import numpy as np
import adios2

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: matplotlib not available for analysis plotting.")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x


def turb_intensity(xgc_instance, istart, iend, skip, vartype='f3d_eden', mode='all'):
    """
    Calculate turbulence intensity from 3D data files.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh and data
    istart : int
        Starting time step
    iend : int
        Ending time step
    skip : int
        Time step increment
    vartype : str, optional
        Variable type ('f3d_eden', '3d_dpot', 'f3d_iTperp')
    mode : str, optional
        Analysis mode ('all', 'upper', 'lower')
        
    Returns
    -------
    psi_n : array_like
        Normalized psi coordinates
    turb_int : array_like
        Turbulence intensity profile
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    
    # File and variable type selection
    if vartype == 'f3d_eden': 
        fname = 'f3d'
        vname = 'e_den'
    elif vartype == '3d_dpot': 
        fname = '3d'
        vname = 'dpot'
    elif vartype == 'f3d_iTperp':
        fname = 'f3d'
        vname = 'i_T_perp'
    else:
        raise ValueError('Unknown vartype: ' + vartype)

    print('Using ' + vname + ' of ' + fname)
    err_msg_count = 0 
    msk = np.zeros_like(xgc_instance.mesh.z) + 1
    
    if mode == 'all':
        pass    
    elif mode == 'upper':
        msk[xgc_instance.mesh.z < xgc_instance.eq_axis_z] = 0
    elif mode == 'lower':
        msk[xgc_instance.mesh.z > xgc_instance.eq_axis_z] = 0
    else:
        raise ValueError('Unknown mode: ' + mode)

    turb_intensity_list = []
    pbar = tqdm(range(istart, iend, skip))
    
    for count, i in enumerate(pbar):
        try:
            with adios2.FileReader('xgc.' + fname + '.%5.5d.bp' % (i)) as f:
                it = int((i - istart) / skip)
                var = f.read(vname)
                time1 = f.read('time')
                if var.shape[0] > 200:  # 200 is maximum plane number
                    var = np.transpose(var) 
        except:
            print(f"Warning: Could not read file for step {i}")
            continue

        if fname == 'f3d':  # variables has f_0. Normalization will be toroidal average.
            var2 = var - np.mean(var, axis=0)  # delta-n or delta-T
            var0 = np.mean(var, axis=0)        # n(n=0) or T0(n=0)
        else: 
            # '3d' -- var is dpot
            var2 = var - np.mean(var, axis=0)
            try:
                var0 = xgc_instance.f0.te0
            except:
                if err_msg_count == 0:
                    print('f0.te0 is not available. Use 1 for normalization')
                    err_msg_count = 1
                var0 = 1

        dns = var2 * var2
        dns = np.mean(dns, axis=0)  # toroidal average
        dns = dns / (var0 * var0)   # normalization with n0
        
        # Import fsa_simple from geometry module
        from .geometry import fsa_simple
        dns_surf = fsa_simple(xgc_instance, dns * msk)  # flux surface average with masking
        turb_intensity_list.append(dns_surf)

    # Average over time
    turb_int = np.mean(turb_intensity_list, axis=0)
    psi_n = xgc_instance.mesh.psi_surf / xgc_instance.psix
    
    return psi_n, turb_int


def source_simple(xgc_instance, step, period, sp='i_', moments='energy', source_type='heat_torque'):
    """
    Simple source analysis from diagnostic files.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance
    step : int
        Time step
    period : int
        Period
    sp : str, optional
        Species prefix
    moments : str, optional
        Moment type
    source_type : str, optional
        Source type
        
    Returns
    -------
    var_1d : array_like
        1D source profile
    sum_1d : array_like
        Cumulative sum
    """
    try:
        with adios2.FileReader("xgc.fsourcediag.%5.5d.bp" % step) as f:
            var = f.read(sp + moments + '_change_' + source_type)
            den = f.read(sp + 'density_' + source_type)
            vol = f.read(sp + 'volume_' + source_type) 
    except:
        raise ValueError(f"Could not read source diagnostic file for step {step}")

    dt = period * xgc_instance.sml_dt
    change_per_time = var * den * vol * xgc_instance.sml_wedge_n / dt
    
    # Import flux_sum_simple from geometry module
    from .geometry import flux_sum_simple
    var_1d = flux_sum_simple(xgc_instance, change_per_time)
    sum_1d = np.cumsum(var_1d)
    print('Total change=', np.sum(change_per_time))
    return var_1d, sum_1d


def plot_source_simple(xgc_instance, step, period, sp='i_', moments='energy', source_type='heat_torque'):
    """
    Plot simple source analysis.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance
    step : int
        Time step
    period : int
        Period
    sp : str, optional
        Species prefix
    moments : str, optional
        Moment type
    source_type : str, optional
        Source type
        
    Returns
    -------
    fig, ax : matplotlib objects
        Figure and axes objects
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions")
    
    var_1d, sum_1d = source_simple(xgc_instance, step, period, sp=sp, moments=moments, source_type=source_type)
    
    fig, ax = plt.subplots()
    ax.plot(xgc_instance.od.psi, sum_1d)
    ax.set_xlabel('Normalized Pol. Flux')
    ax.set_title(sp + moments + '_' + source_type)
    
    return fig, ax


def gyro_radius(xgc_instance, t_ev, b, mass_au, charge_eu):
    """
    Calculate gyroradius.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with constants
    t_ev : array_like
        Temperature in eV
    b : array_like
        Magnetic field in Tesla
    mass_au : float
        Mass in atomic units
    charge_eu : float
        Charge in electron units
        
    Returns
    -------
    rho : array_like
        Gyroradius
    """
    mass = mass_au * xgc_instance.cnst.protmass
    return 1 / (charge_eu * b) * np.sqrt(mass * t_ev / xgc_instance.cnst.echarge)


def find_exb_velocity(xgc_instance, istart, iend, skip, ms):
    """
    Find average ExB velocity of line segment defined with node index ms.
    It reads xgc.f3d.*.bp from index (istart, iend, skip) and do time average.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance
    istart : int
        Starting step
    iend : int
        Ending step
    skip : int
        Skip interval
    ms : array_like
        Node indices for line segment
        
    Returns
    -------
    v_exb : float
        Average ExB velocity
    """
    pol_vi = 0
    pol_ve = 0
    ct = 0
    pbar = tqdm(range(istart, iend, skip))
    
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
def find_exb_velocity2(xgc_instance, istart, iend, skip, ms, only_average=True):

    bt = xgc_instance.bfield[2,ms]
    b2 = np.sqrt(xgc_instance.bfield[0,ms]**2 + xgc_instance.bfield[1,ms]**2 + xgc_instance.bfield[2,ms]**2)

    pbar = tqdm(range(istart,iend,skip))
    for count, istep in enumerate(pbar):
        with adios2.FileReader('xgc.3d.%5.5d.bp' % (istep)) as f:
            epsi=f.read('epsi') # E_r
            try:
                time1=f.read('time')
            except:
                time1=istep*xgc_instance.sml_dt
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




def power_spectrum_w_k_with_exb(xgc_instance, istart, iend, skip, skip_exb, psi_target, ns_half, old_vexb=False):
    """
    Calculate power spectrum w-k with ExB velocity.
    """
    # Find line segment
    from .geometry import find_line_segment
    ms, psi0, length = find_line_segment(xgc_instance, ns_half, psi_target)
    
    print('psi0=', psi0, 'length=', length) 
    print('getting ExB velocity...')
    
    #get exb
    if(old_vexb):
        v_exb = find_exb_velocity(xgc_instance, istart, iend, skip_exb, ms)
    else:
        v_exb = find_exb_velocity2(xgc_instance, istart, iend, skip_exb, ms)

    print('v_exb=',v_exb,' m/s')
    #reading data
    print('reading 3d data...')
    dpot4,po,time = reading_3d_data(xgc_instance, istart, iend, skip, ms)

    #prepare parameters for plot
    k, omega = prepare_plots(xgc_instance, length,ms,time)
    print('done.')
        
    return ms, psi0, v_exb, dpot4, po, k, omega, time, length


def gam_freq_analytic(xgc_instance):
    """
    Get GAM analytic GAM frequency based on 1D diag and psi_surf.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh, 1D data, and constants
        
    Returns
    -------
    f : array_like
        GAM frequency
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    if not hasattr(xgc_instance, 'od'):
        raise ValueError("1D data not loaded. Call load_oned() first.")
        
    # Finding region 1
    psi_surf = xgc_instance.mesh.psi_surf / xgc_instance.psix
    msk = np.nonzero(np.logical_and(psi_surf[:-1] <= 1, psi_surf[1:] > 1))
    m = slice(0, msk[0][0])

    q = np.interp(xgc_instance.od.psi, psi_surf[m], xgc_instance.mesh.qsafety[m])
    f = (1 + 1 / (2 * q * q)) * np.sqrt(xgc_instance.cnst.echarge * (xgc_instance.od.Te + xgc_instance.od.Ti) / xgc_instance.cnst.protmass / xgc_instance.unit_dic['i_ptl_mass_au']) / (2 * np.pi * xgc_instance.eq_axis_r)
    return f


def midplane(xgc_instance):
    """
    Get midplane analysis (placeholder for more complex implementation).
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh data
        
    Returns
    -------
    psi_mid : array_like
        Normalized psi at midplane
    r_mid : array_like
        R coordinates at midplane
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    
    # Simple implementation - extract midplane psi and R
    from .geometry import midplane_var
    psi_mid, r_mid = midplane_var(xgc_instance, xgc_instance.mesh.psi, return_rmid=False)
    return psi_mid, r_mid


def find_exb_velocity2(xgc_instance, istart, iend, skip, ms, only_average=True):
    """
    Find ExB velocity with detailed analysis (version 2).
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with magnetic field data
    istart : int
        Starting step
    iend : int
        Ending step
    skip : int
        Skip interval
    ms : array_like
        Node indices
    only_average : bool, optional
        Return only average velocity
        
    Returns
    -------
    v_exb : float or array_like
        ExB velocity (averaged or full array)
    time : array_like, optional
        Time array (if only_average=False)
    """
    if not hasattr(xgc_instance, 'bfield'):
        raise ValueError("Magnetic field data not loaded. Call load_bfield() first.")
    
    bt = xgc_instance.bfield[2, ms]
    b2 = np.sqrt(xgc_instance.bfield[0, ms]**2 + 
                 xgc_instance.bfield[1, ms]**2 + 
                 xgc_instance.bfield[2, ms]**2)
    
    pbar = tqdm(range(istart, iend, skip))
    for count, istep in enumerate(pbar):
        try:
            with adios2.FileReader('xgc.3d.%5.5d.bp' % istep) as f:
                epsi = f.read('epsi')  # E_r
                try:
                    time1 = f.read('time')
                except:
                    time1 = istep * xgc_instance.sml_dt
                    
                v_exb1 = epsi[:, ms] * bt / b2  # ExB velocity
                v_exb1 = v_exb1[np.newaxis, :, :]
                
            if count == 0:
                v_exb = v_exb1
                time = time1
            else:
                v_exb = np.concatenate((v_exb, v_exb1), axis=0)
                time = np.vstack((time, time1))
        except:
            print(f"Warning: Could not read 3d file for step {istep}")
            continue
    
    if only_average:
        v_exb = np.mean(v_exb, axis=(0, 1, 2))
        return v_exb
    else:
        return v_exb, time


def reading_3d_data(xgc_instance, istart, iend, skip, ms, no_fft=False):
    """
    Read 3D dpot data and perform FFT analysis.
    """
    
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

def prepare_plots(xgc_instance, dist, ms, time):
    """
    Prepare plot arrays for k and omega analysis.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance
    dist : float
        Distance
    ms : array_like
        Node indices
    time : array_like
        Time array
        
    Returns
    -------
    k : array_like
        k array
    omega : array_like
        omega array
    """
    ns = np.size(ms)
    nt = np.size(time)
    kmax = 2 * np.pi / dist * ns
    omax = 2 * np.pi / (time[-1] - time[0]) * nt
    k = np.fft.fftshift(np.fft.fftfreq(ns, 1 / kmax))
    omega = np.fft.fftshift(np.fft.fftfreq(nt, 1 / omax))
    return k, omega