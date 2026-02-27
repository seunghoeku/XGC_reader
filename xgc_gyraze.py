'''
Export XGC distribution data to gyraze format

It utilize xgc_reader and xgc_distribution module
'''

import xgc_reader as xr
import xgc_distribution as xd

import numpy as np
import os
import h5py

def generate_F_mps_content(Eperp, Epara, f):
    """
    Generate input files for F.
    
    Parameters:
    -----------
    vperp : np.ndarray
        Perpendicular velocity grid
    vpara : np.ndarray
        Parallel velocity grid
    f : np.ndarray
        Distribution function f(vperp, vpara)
        
    Returns:
    --------
    args_text : str
        Content for the args file (grid info)
    content_text : str
        Content for the data file (distribution data)
    """
    
    neperp = len(Eperp)
    nepara = len(Epara)
    
    # Generate args content
    args_text = ' '.join(map(str, Eperp)) + '\n' + ' '.join(map(str, Epara))
    
    
    # Write as a block of numbers
    import io
    s = io.StringIO()
    
    content_text = ''
    # f is expected to be (nvperp, nvpara)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            s.write(f'{f[i,j]:.16e} ')
        s.write('\n')
        
    content_text += s.getvalue()
    
    return args_text, content_text

def single_distribution_writer(x, idist, edist, node, step, dir, filename='gyraze_input.h5', return_distribution=False, use_hdf5=True, tags=None):
    # get distribution function and change the jacobian
    idist.add_maxwellian()
    edist.add_maxwellian()
    fi = np.squeeze(idist.f[node,:,:])
    fe = np.squeeze(edist.f[node,:,:])


    #get vspace volume
    i_vspace_vol = idist.fg_temp_ev[node] * np.sqrt(1/(np.pi*2)) * idist.vgrid.dvperp * idist.vgrid.dvpara
    e_vspace_vol = edist.fg_temp_ev[node] * np.sqrt(1/(np.pi*2)) * edist.vgrid.dvperp * edist.vgrid.dvpara

    vspace_vol = [i_vspace_vol, e_vspace_vol]

    # set the density is the same
    for i, f in enumerate([fi, fe]):
        # apply mu0_factor at boundary of vperp
        f[0,:] = f[0,:] * idist.MU_VOL_FAC
        f[-1,:] = f[-1,:] * idist.MU_VOL_FAC
        
        #apply vp_vol_fac at boundary of vpara
        f[:,0] = f[:,0] * idist.VP_VOL_FAC
        f[:,-1] = f[:,-1] * idist.VP_VOL_FAC


        # particle density
        ptls = np.sum(f)
        if(np.isnan(ptls).any()):
            print('ptls has nan')
        den = ptls * vspace_vol[i]

        if(i==0):
            ni = den
        else:
            ne = den

        # undo boundary factor
        f[0,:] = f[0,:]/idist.MU_VOL_FAC
        f[-1,:] = f[-1,:]/idist.MU_VOL_FAC
        
        f[:,0] = f[:,0]/idist.VP_VOL_FAC
        f[:,-1] = f[:,-1]/idist.VP_VOL_FAC

    #remove density factor in distribution function -- both the same density
    fi = fi / ni
    fe = fe / ne

    #change vperp jacobian
    fi[1:,:]=fi[1:,:]/idist.vgrid.vperp[1:,np.newaxis]
    fe[1:,:]=fe[1:,:]/edist.vgrid.vperp[1:,np.newaxis]
    fi[0,:]=fi[0,:]/idist.vgrid.vperp[1]*3
    fe[0,:]=fe[0,:]/edist.vgrid.vperp[1]*3

    # convert v to energy (eV)
    i_en_perp = idist.vgrid.vperp**2/2*idist.fg_temp_ev[node]
    i_en_para = idist.vgrid.vpara**2/2*idist.fg_temp_ev[node]

    e_en_perp = edist.vgrid.vperp**2/2*edist.fg_temp_ev[node]
    e_en_para = edist.vgrid.vpara**2/2*edist.fg_temp_ev[node]

    #determine the direction to wall - use positive density side
    ivp0 = np.argmin(np.abs(idist.vgrid.vpara))
    sum_minus = np.sum(fi[:,0:ivp0+1])
    sum_plus = np.sum(fi[:,ivp0:])

    if(sum_minus>sum_plus):
        direction = -1
    else:
        direction = 1

    # pick the side to the wall
    if(direction == -1):
        fi = fi[:,0:ivp0+1]
        fe = fe[:,0:ivp0+1]
        i_en_para=i_en_para[0:ivp0+1]
        e_en_para=e_en_para[0:ivp0+1]
    else:
        fi = fi[:,ivp0:]
        fe = fe[:,ivp0:]
        i_en_para=i_en_para[ivp0:]
        e_en_para=e_en_para[ivp0:]

    # smoothing operation
    # not implemented yet


    # Ensure directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    # Generate content
    fi_args, fi_content = generate_F_mps_content(i_en_perp, i_en_para,fi)
    fe_args, fe_content = generate_F_mps_content(e_en_perp, e_en_para,fe)
    
    if(use_hdf5):
        # Write to HDF5
        filename2 = os.path.join(dir, filename)
        
        with h5py.File(filename2, 'a') as hf:
            # Create group for this node
            group_name = f'step_{step}_node_{node}'
            if group_name in hf:
                del hf[group_name]
            grp = hf.create_group(group_name)
            
            dt = h5py.string_dtype(encoding='utf-8')
        
            # Write Datasets
            grp.create_dataset('Fi_mpe_args.txt', data=fi_args, dtype=h5py.string_dtype(encoding='utf-8'))
            grp.create_dataset('Fi_mpe.txt', data=fi_content, dtype=h5py.string_dtype(encoding='utf-8'))
            grp.create_dataset('Fe_mpe_args.txt', data=fe_args, dtype=h5py.string_dtype(encoding='utf-8'))
            grp.create_dataset('Fe_mpe.txt', data=fe_content, dtype=h5py.string_dtype(encoding='utf-8'))
            
            if(tags is not None):
                for key, value in tags.items():
                    grp.attrs[key] = value
            else:
                print('tags is not provided')


    else:
        # Write to text files
        fi_args_file = os.path.join(dir, 'Fi_mpe_args.txt')
        fi_content_file = os.path.join(dir, 'Fi_mpe.txt')
        fe_args_file = os.path.join(dir, 'Fe_mpe_args.txt')
        fe_content_file = os.path.join(dir, 'Fe_mpe.txt')
        
        with open(fi_args_file, 'w') as f:
            f.write(fi_args)
        
        with open(fi_content_file, 'w') as f:
            f.write(fi_content)
        
        with open(fe_args_file, 'w') as f:
            f.write(fe_args)
        
        with open(fe_content_file, 'w') as f:
            f.write(fe_content) 
        
        if(tags is not None):
            with open(os.path.join(dir, 'tags.txt'), 'w') as f:
                for key, value in tags.items():
                    f.write(f'{key}: {value}\n')
    

    # return the distribution instead of writing to file
    if(return_distribution):
        return fi, fe, i_en_perp, i_en_para, e_en_perp, e_en_para
    
    
def gyraze_input_single_time_step(x, step, node_list, mass_au, target_dir='./', filename='gyraze_input.h5', use_hdf5=True):
    """
    Generate gyraze input files for a single distribution.
    """
    
    f0_file = "/xgc.f0.%05d.bp"%step
    edist = xd.XGCDistribution.from_xgc_output(f0_file, time_step=0, mass_au=mass_au[0], var_string='e_f', dir=x.path, has_electron=True, use_initial_moments=True)
    idist = xd.XGCDistribution.from_xgc_output(f0_file, time_step=0, mass_au=mass_au[1], var_string='i_f', dir=x.path, has_electron=True, use_initial_moments=True)
    
    # read sheath potential
    sheath_pot = read_sheath_potential(x, step)
    
    
    for node in node_list:
        
        # get angle 
        angle, nearest_wall_node = get_graze_angle(x,node)
        rho_e_lambda_d = get_rho_e_lambda_d(x,edist,node)
        potential = get_sheath_pot(x,node,sheath_pot,nearest_wall_node) # nearest_wall_node is the index of the nearest wall node
    
        tags = {
            "angle": angle,
            "rho_e_lambda_d": rho_e_lambda_d,
            "potential": potential,
            "step": step,
            "node": node,
            "mass_au": mass_au,
            "source_dir": x.path
        }
        
        single_distribution_writer(x, idist, edist, node, step, dir=target_dir, filename=filename, use_hdf5=use_hdf5, tags=tags)
        
def get_graze_angle(x,node):
    """
    Get the graze angle for a given node.
    """
    
    wall_node_region_no=100
    
    nd_wall = np.where(x.mesh.region == wall_node_region_no)[0]
    dists = np.sqrt((x.mesh.r[nd_wall] - x.mesh.r[node])**2 + (x.mesh.z[nd_wall] - x.mesh.z[node])**2)
    wall_idx = np.argmin(dists)
    nearest_wall_node = nd_wall[wall_idx]

    second_wall_idx = np.argsort(dists)[1]
    second_nearest_wall_node = nd_wall[second_wall_idx]
    
    
    dr = x.mesh.r[nearest_wall_node] - x.mesh.r[second_nearest_wall_node]
    dz = x.mesh.z[nearest_wall_node] - x.mesh.z[second_nearest_wall_node]

    vec_wall = np.array([dz, -dr, 0])
    vec_B = x.bfield[:,node]

    graze_angle =   np.arcsin(np.abs(np.dot(vec_wall, vec_B)) / (np.linalg.norm(vec_wall) * np.linalg.norm(vec_B)))

    return graze_angle, nearest_wall_node

def get_rho_e_lambda_d(x,edist,node):
    """
    Get the rho_e_lambda_d for a given node.
    """

    sqrt_te_ev= 1 # 1eV for arbitrary value, because is cancled out in the ratio
    ne = edist.den[node]
    bmag = np.sqrt(x.bfield[0,node]**2 + x.bfield[1,node]**2 + x.bfield[2,node]**2)
    rho_e = 2.38E-2 * sqrt_te_ev/(bmag*1E4)
    lambda_d = 7.43 * sqrt_te_ev/ np.sqrt(ne/1E6)
    rho_e_lambda_d = rho_e/lambda_d
    
    return rho_e_lambda_d

def get_sheath_pot(x,node,sheath_pot,nearest_wall_node):
    """
    Get the potential for a given node.
    """
    
    wall_idx = np.argwhere(x.mesh.grid_wall_nodes==nearest_wall_node)[0][0]
    return sheath_pot[wall_idx]

def read_sheath_potential(x, step):
    """
    Read the sheath potential from the source directory.
    """
    import adios2
    
    # read sheath potential of all time steps
    with adios2.FileReader(x.path+'/xgc.sheathdiag.bp') as f:
        vars = f.available_variables()
        var = vars['sheath_pot']
        ct = var.get("Shape")
        stc = int(var.get("AvailableStepsCount"))
        sheath_pot = f.read('sheath_pot', start=[], count=[], step_selection=[0, stc])

    with adios2.FileReader(x.path+'/xgc.mesh.bp') as f:
        x.mesh.grid_wall_nodes = f.read('grid_wall_nodes') -1 # subtract 1 to convert to 0-based index


    # find the corresponding time slice for the sheath potential
    #it = np.argmin(np.abs(x.od.steps-step)) 
    # the above give the first occurrence. 
    # following code to find the last occurrence
    a = np.abs(x.od.step-step)
    m = a.min()
    it = np.flatnonzero(a==m)[-1]
    
    return sheath_pot[it,:]
    


