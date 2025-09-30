"""Plotting and visualization functions."""

import numpy as np
import adios2

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: matplotlib not available. Plotting functions will not work.")


def plot1d_if(xgc_instance, obj, **kwargs):
    """
    Plot 1D (psi) variable of initial and final time steps.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance (not used, kept for compatibility)
    obj : data1
        Data object with time series data
    var : array_like, optional
        Variable to plot
    varstr : str, optional
        Variable name to extract from obj
    psi : array_like, optional
        Psi coordinates (defaults to obj.psi)
    xlim : tuple, optional
        X-axis limits
    initial : bool, optional
        Include initial time step (default True)
    time_legend : bool, optional
        Use time values in legend (default True)
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions")
    
    var = kwargs.get('var', None)
    varstr = kwargs.get('varstr', None)
    box = kwargs.get('box', None)
    psi = kwargs.get('psi', None)
    xlim = kwargs.get('xlim', None)
    initial = kwargs.get('initial', True)
    time_legend = kwargs.get('time_legend', True)
    
    if psi is None or not isinstance(psi, np.ndarray):
        psi = obj.psi
        
    if var is None or not isinstance(var, np.ndarray):
        if varstr is None:   
            raise ValueError("Either var or varstr should be defined.")
        else:
            var = getattr(obj, varstr)
           
    stc = var.shape[0]
    fig, ax = plt.subplots()
    it0 = 0  # 0th time index
    it1 = stc - 1  # last time index
    tnorm = 1E3
    
    if time_legend and hasattr(obj, 'time'):
        lbl = ["t=%3.3f" % (obj.time[it0] * tnorm), "t=%3.3f" % (obj.time[it1] * tnorm)]
    else:
        lbl = ["Initial", "Final"]

    if xlim is None:
        if initial:
            ax.plot(psi, var[it0, :], label=lbl[0])
        ax.plot(psi, var[it1, :], label=lbl[1])
    else:
        msk = (psi >= xlim[0]) & (psi <= xlim[1])
        if initial:
            ax.plot(psi[msk], var[it0, msk], label=lbl[0])
        ax.plot(psi[msk], var[it1, msk], label=lbl[1])
            
    ax.legend()
    ax.set(xlabel='Normalized Pol. Flux')
    if varstr is not None:
        ax.set(ylabel=varstr)
    
    return fig, ax


def contourf_one_var(xgc_instance, fig, ax, var, title='None', vm='None', cmap='jet', levels=150, cbar=True):
    """
    Create a filled contour plot of a variable on the mesh.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh data
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    var : array_like
        Variable to plot
    title : str, optional
        Plot title
    vm : str or float, optional
        Value limits ('None', 'Sigma2', or numeric value)
    cmap : str, optional
        Colormap name
    levels : int, optional
        Number of contour levels
    cbar : bool, optional
        Add colorbar
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions")
    
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    
    if vm == 'None':
        cf = ax.tricontourf(xgc_instance.mesh.triobj, var, cmap=cmap, extend='both', levels=levels)
    elif vm == 'Sigma2':
        sigma = np.sqrt(np.mean(var * var) - np.mean(var)**2)
        vm_val = 2 * sigma
        var2 = np.minimum(vm_val, np.maximum(-vm_val, var))
        cf = ax.tricontourf(xgc_instance.mesh.triobj, var2, cmap=cmap, extend='both', 
                           levels=levels, vmin=-vm_val, vmax=vm_val)
    else:
        var2 = np.minimum(vm, np.maximum(-vm, var))
        cf = ax.tricontourf(xgc_instance.mesh.triobj, var2, cmap=cmap, extend='both', 
                           levels=levels, vmin=-vm, vmax=vm)
    
    if cbar:
        cbar_obj = fig.colorbar(cf, ax=ax)
    
    if title != 'None':
        ax.set_title(title)
    
    return cf


def show_sep(xgc_instance, ax, style='-'):
    """
    Show separatrix on the plot.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh data
    ax : matplotlib.axes.Axes
        Axes object
    style : str, optional
        Line style for separatrix
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions")
    
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    
    if not hasattr(xgc_instance.mesh, 'msep'):
        raise ValueError("Separatrix not found. Ensure mesh is properly loaded with psix.")
    
    msep = xgc_instance.mesh.msep
    ax.plot(xgc_instance.mesh.r[msep], xgc_instance.mesh.z[msep], style, label='Separatrix')


def plot2d(xgc_instance, filestr, varstr, **kwargs):
    """
    General 2D plot function (basic implementation).
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with mesh data
    filestr : str
        File name
    varstr : str
        Variable name
    box : tuple, optional
        Spatial limits (rmin, rmax, zmin, zmax)
    plane : int, optional
        Plane index (default 0)
    levels : int, optional
        Number of levels
    cmap : str, optional
        Colormap
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting functions")
    
    if not hasattr(xgc_instance, 'mesh'):
        raise ValueError("Mesh data not loaded. Call setup_mesh() first.")
    
    box = kwargs.get('box', None)
    plane = kwargs.get('plane', 0)
    levels = kwargs.get('levels', None)
    cmap = kwargs.get('cmap', 'jet')
    
    with adios2.FileReader(filestr) as f:
        var = f.read(varstr)
    
    fig, ax = plt.subplots()

    if box is not None:
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
    
    try:
        cf = ax.tricontourf(xgc_instance.mesh.triobj, var[plane, :], cmap=cmap, extend='both')
    except (IndexError, TypeError):
        cf = ax.tricontourf(xgc_instance.mesh.triobj, var, cmap=cmap, extend='both')
        
    cbar = fig.colorbar(cf)

    if box is not None:
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
    
    return fig, ax