"""Geometry and grid utility functions."""

import numpy as np
from matplotlib.tri import LinearTriInterpolator

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: matplotlib not available for geometry visualization.")


def find_sep_idx(xgc_instance):
    """
    Find separatrix node indices.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh data and psix
        
    Returns
    -------
    msep : array_like
        Node indices for separatrix
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    if not hasattr(xgc_instance, 'psix'):
        raise ValueError("Units not loaded. Call load_units() first.")
        
    isep = np.argmin(abs(xgc_instance.mesh.psi_surf - xgc_instance.psix))
    length = xgc_instance.mesh.surf_len[isep]
    msep = xgc_instance.mesh.surf_idx[isep, 0:length] - 1  # zero-based indexing
    return msep


def find_surf_idx(xgc_instance, psi_norm=1.0):
    """
    Find flux surface node indices for given normalized psi.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh data and psix
    psi_norm : float, optional
        Normalized psi value (default 1.0 for separatrix)
        
    Returns
    -------
    msep : array_like
        Node indices for flux surface
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    if not hasattr(xgc_instance, 'psix'):
        raise ValueError("Units not loaded. Call load_units() first.")
        
    isep0 = np.argmin(abs(xgc_instance.mesh.psi_surf - xgc_instance.psix))
    
    if psi_norm < 1.0:
        psi_surf = xgc_instance.mesh.psi_surf[:isep0]
    else:
        psi_surf = xgc_instance.mesh.psi_surf
        
    isep = np.argmin(abs(psi_surf - xgc_instance.psix * psi_norm))
    
    length = xgc_instance.mesh.surf_len[isep]
    msep = xgc_instance.mesh.surf_idx[isep, 0:length] - 1  # zero-based indexing
    return msep


def find_tmask(xgc_instance, step, max_end=False):
    """
    Find time mask for time steps.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance (for compatibility)
    step : array_like
        Step array
    max_end : bool, optional
        Use maximum time step as end point
        
    Returns
    -------
    tmask : list
        Time mask indices
    """
    if max_end:
        ed = np.max(step)
    else:
        ed = step[-1]

    tmask_rev = []
    p = step.size
    for i in range(ed, 0, -1):  # reverse order
        m = np.nonzero(step[0:p] == i)[0]  # find index that has step i
        try:
            p = m[-1]  # exclude zero size 
        except IndexError:
            pass
        else:
            tmask_rev.append(p)  # only append that has step number
    
    # tmask is reverse order
    tmask = tmask_rev[::-1]
    return tmask


def find_line_segment(xgc_instance, n, psi_target, dir='middle'):
    """
    Find line segment along flux surface for analysis.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh data
    n : int
        Half number of points in segment
    psi_target : float
        Target normalized psi value
    dir : str, optional
        Direction ('middle', 'up', 'down')
        
    Returns
    -------
    ms : array_like
        Node indices of line segment
    psi0 : float
        Actual normalized psi of surface
    length : float
        Physical length of segment
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    if not hasattr(xgc_instance, 'psix'):
        raise ValueError("Units not loaded. Call load_units() first.")
        
    isurf = np.argmin(np.abs(xgc_instance.mesh.psi_surf / xgc_instance.psix - psi_target))
    
    # Node index of the surface, -1 for zero base
    msk = xgc_instance.mesh.surf_idx[isurf, 0:xgc_instance.mesh.surf_len[isurf]] - 1
    
    # Core mesh or SOL
    if xgc_instance.mesh.psi_surf[isurf] < 0.99999 * xgc_instance.psix:
        # Core region
        if dir == 'middle':
            tmp1 = msk[-n:]
            tmp2 = msk[0:n]
            ms = np.append(tmp1, tmp2)
        elif dir == 'up':
            ms = msk[0:2*n]
        else:  # down
            ms = msk[-2*n:]
    else:
        # SOL region - find segments low field side and above X-point
        if hasattr(xgc_instance, 'eq_x_r') and hasattr(xgc_instance, 'eq_x_z'):
            msk2 = np.nonzero(np.logical_and(
                xgc_instance.mesh.r[msk] > xgc_instance.eq_x_r,
                xgc_instance.mesh.z[msk] > xgc_instance.eq_x_z
            ))[0]
            msk3 = msk[msk2]
            imid = np.argmin(np.abs(xgc_instance.mesh.z[msk3] - xgc_instance.eq_axis_z))
            ms = msk3[imid-n:imid+n]
        else:
            # Fallback for missing X-point data
            ms = msk[len(msk)//2-n:len(msk)//2+n]
    
    # Optional visualization
    if plt is not None:
        try:
            ax = plt.subplot()
            ax.plot(xgc_instance.mesh.r[ms], xgc_instance.mesh.z[ms], '.')
            ax.axis('equal')
        except:
            pass  # Skip plotting if mesh data incomplete
    
    psi0 = xgc_instance.mesh.psi_surf[isurf] / xgc_instance.psix
    
    # Calculate segment length
    dr = xgc_instance.mesh.r[ms[1:]] - xgc_instance.mesh.r[ms[0:-1]]
    dz = xgc_instance.mesh.z[ms[1:]] - xgc_instance.mesh.z[ms[0:-1]]
    ds = np.sqrt(dr**2 + dz**2)
    length = np.sum(ds)
    
    # Check segment quality
    if len(ds) > 0:
        begin_end_ratio = ds[0] / ds[-1]
        if (begin_end_ratio > 1.5) or (begin_end_ratio < 0.7):
            if dir == 'middle':
                print('ratio=', begin_end_ratio, 'trying upper side')
                return find_line_segment(xgc_instance, n, psi_target, dir='up')
            elif dir == 'up':
                print('ratio=', begin_end_ratio, 'trying lower side')
                return find_line_segment(xgc_instance, n, psi_target, dir='down')
            else:
                print('ratio=', begin_end_ratio, 'Failed to find line segment')
                return np.array([]), 0, 0
    
    return ms, psi0, length


def fsa_simple(xgc_instance, var):
    """
    Simple flux surface average using mesh data (optimized version).
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with loaded mesh data
    var : array_like
        Variable to average
        
    Returns
    -------
    favg : array_like
        Flux surface averaged values
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    
    mesh = xgc_instance.mesh
    n_surf = mesh.psi_surf.size
    favg = np.zeros(n_surf)
    
    # Vectorized computation where possible
    for i in range(n_surf):
        surf_len = mesh.surf_len[i]
        if surf_len > 0:
            # Get indices for this surface (convert to 0-based)
            indices = mesh.surf_idx[i, :surf_len] - 1
            
            # Vectorized operations
            var_vals = var[indices]
            vol_vals = mesh.node_vol[indices]
            
            # Compute weighted average
            s1 = np.sum(var_vals * vol_vals)
            s2 = np.sum(vol_vals)
            
            if s2 > 0:
                favg[i] = s1 / s2
            else:
                favg[i] = 0.0
                
            # Check for NaN with more informative error
            if np.isnan(favg[i]):
                print(f"NaN found at surface {i}, psi={mesh.psi_surf[i]:.6f}, s1={s1:.6e}, s2={s2:.6e}")
    
    return favg


def flux_sum_simple(xgc_instance, var):
    """
    Simple summation over surface (optimized version).
    
    Note: Not good when non-aligned points are nearby.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with loaded mesh data
    var : array_like
        Variable to sum
        
    Returns
    -------
    favg : array_like
        Flux surface summed values
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    
    mesh = xgc_instance.mesh
    n_surf = mesh.psi_surf.size
    favg = np.zeros(n_surf)
    
    # Vectorized computation
    for i in range(n_surf):
        surf_len = mesh.surf_len[i]
        if surf_len > 0:
            # Get indices for this surface (convert to 0-based)
            indices = mesh.surf_idx[i, :surf_len] - 1
            # Vectorized sum
            favg[i] = np.sum(var[indices])
    
    return favg


def midplane_var(xgc_instance, var, inboard=False, nr=300, delta_r_axis=0., delta_r_edge=0., return_rmid=False):
    """
    Extract midplane values of a variable.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with loaded mesh data
    var : array_like
        Variable to extract at midplane
    inboard : bool, optional
        If True, extract inboard midplane
    nr : int, optional
        Number of radial points
    delta_r_axis : float, optional
        Delta R from magnetic axis
    delta_r_edge : float, optional
        Delta R from edge
    return_rmid : bool, optional
        If True, return R midplane coordinates
        
    Returns
    -------
    psi_mid : array_like
        Normalized psi at midplane
    var_mid : array_like
        Variable values at midplane
    r_mid : array_like, optional
        R coordinates at midplane (if return_rmid=True)
    """
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
        
    maxr = xgc_instance.mesh.r.max() - delta_r_edge
    minr = xgc_instance.mesh.r.min() + delta_r_edge

    if inboard:
        r_mid = np.linspace(minr, xgc_instance.eq_axis_r - delta_r_axis, nr)
    else:
        r_mid = np.linspace(xgc_instance.eq_axis_r + delta_r_axis, maxr, nr)
    z_mid = np.linspace(xgc_instance.eq_axis_z, xgc_instance.eq_axis_z, nr)
    
    psi_tri = LinearTriInterpolator(xgc_instance.mesh.triobj, xgc_instance.mesh.psi / xgc_instance.psix)
    psi_mid = psi_tri(r_mid, z_mid)

    var_tri = LinearTriInterpolator(xgc_instance.mesh.triobj, var)
    var_mid = var_tri(r_mid, z_mid)

    # Remove nan
    mask = np.ma.getmaskarray(psi_mid)
    r_mid = r_mid[~mask]
    psi_mid = np.asarray(psi_mid.compressed())
    var_mid = np.asarray(var_mid.compressed())

    if return_rmid:
        return psi_mid, var_mid, r_mid
    else:   
        return psi_mid, var_mid


def d_dpsi(xgc_instance, field):
    """Calculate derivative with respect to psi."""
    if not hasattr(xgc_instance, 'grad') or xgc_instance.grad is None:
        print("Warning: d_dpsi requires gradient matrices to be loaded")
        return field
    
    # Apply psi derivative using gradient matrix
    if hasattr(xgc_instance.grad, 'mat_psi_r') and xgc_instance.grad.mat_psi_r is not None:
        return xgc_instance.grad.mat_psi_r @ field
    else:
        return field