"""Report and information display functions."""

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: matplotlib not available for reporting plots.")


def print_plasma_info(xgc_instance):
    """Print some plasma information (mostly from unit_dic)."""
    print("magnetic axis (R,Z) = (%5.5f, %5.5f) m" % (xgc_instance.eq_axis_r, xgc_instance.eq_axis_z))
    print("magnetic field at axis = %5.5f T" % xgc_instance.eq_axis_b)
    print("X-point (R,Z) = (%5.5f, %5.5f)" % (xgc_instance.eq_x_r, xgc_instance.eq_x_z))
    print("simulation delta t = %e s" % xgc_instance.sml_dt)


def report_heatdiag2(xgc_instance, is_outer=True, is_lower=True, it=-1, xlim=[-5, 15], 
                    lq_ylim=[0, 10], ndata=1000000, fit_mask=None, 
                    sp_names=['e', 'i', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9']):
    """
    Generate comprehensive heat diagnostic report with plots.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with loaded heat diagnostic data
    is_outer : bool, optional
        Select outer divertor
    is_lower : bool, optional  
        Select lower divertor
    it : int, optional
        Time index (-1 for last)
    xlim : list, optional
        X-axis limits for plots
    lq_ylim : list, optional
        Y-axis limits for lambda_q plots
    ndata : int, optional
        Maximum number of data points
    fit_mask : array_like, optional
        Mask for fitting
    sp_names : list, optional
        Species names for labeling
    """
    if plt is None:
        raise ImportError("matplotlib is required for heat diagnostic reporting")
    
    if not hasattr(xgc_instance, 'hl2'):
        raise ValueError("Heat diagnostic data not loaded. Call load_heatdiag2() first.")
    
    # Select divertor
    i0, i1 = xgc_instance.hl2.get_divertor(outer=is_outer, lower=is_lower)
    sign = 1 if (i0 < i1) else -1 
    i1 = i0 + sign * ndata if np.abs(i1 - i0) > ndata else i1

    md = np.arange(i0, i1, sign)
    
    # Plot divertor location
    fig, ax = plt.subplots()
    plt.plot(xgc_instance.hl2.r[0, :], xgc_instance.hl2.z[0, :])
    plt.plot(xgc_instance.hl2.r[0, md], xgc_instance.hl2.z[0, md], 'r-', linewidth=4, label='Divertor')
    plt.legend()
    
    # Show separatrix if available
    try:
        from .plotting import show_sep
        show_sep(xgc_instance, ax, style=',')
    except:
        pass
    
    plt.axis('equal')

    # Plot total heat flux
    xgc_instance.hl2.total_heat(xgc_instance.sml_wedge_n, pmask=md)
    plt.figure()
    
    for isp in range(len(xgc_instance.hl2.sp)):
        if isp < len(sp_names):
            plt.plot(xgc_instance.hl2.time * 1E3, xgc_instance.hl2.sp[isp].q_sum / 1E6, 
                    '.', label=sp_names[isp])
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Total Heat Flux (MW)')
    plt.legend()
    plt.title('Total Heat Flux vs Time')


def report_profiles(xgc_instance, i_name='Main ion', i2_name='Impurity', 
                   init_idx=0, end_idx=-1, edge_lim=[0.85, 1.05]):
    """
    Generate comprehensive profile reports.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with loaded 1D data
    i_name : str, optional
        Main ion species name
    i2_name : str, optional
        Impurity species name  
    init_idx : int, optional
        Initial time index
    end_idx : int, optional
        End time index (-1 for last)
    edge_lim : list, optional
        Edge region limits for zoomed plots
    """
    if plt is None:
        raise ImportError("matplotlib is required for profile reporting")
    
    if not hasattr(xgc_instance, 'od'):
        raise ValueError("1D data not loaded. Call load_oned() first.")
    
    # Show initial profiles - temperature
    tunit = 1E3
    fig, ax = plt.subplots()
    
    if hasattr(xgc_instance, 'electron_on') and xgc_instance.electron_on:
        plt.plot(xgc_instance.od.psi, xgc_instance.od.Te[0, :] / tunit, label='Elec.')
    
    plt.plot(xgc_instance.od.psi, xgc_instance.od.Ti[0, :] / tunit, label=i_name)
    
    if hasattr(xgc_instance, 'ion2_on') and xgc_instance.ion2_on:
        plt.plot(xgc_instance.od.psi, xgc_instance.od.Ti2[0, :] / tunit, '--', label=i2_name)
    
    plt.legend()
    plt.xlabel('Normalized Pol. Flux')
    plt.ylabel('Temperature (keV)')
    plt.title('Initial Temperature')

    # Density
    dunit = 1E19
    fig, ax = plt.subplots()
    
    if hasattr(xgc_instance, 'electron_on') and xgc_instance.electron_on:
        plt.plot(xgc_instance.od.psi, xgc_instance.od.e_gc_density_df_1d[0, :] / dunit, label='Elec.')
    
    plt.plot(xgc_instance.od.psi, xgc_instance.od.i_gc_density_df_1d[0, :] / dunit, label=i_name)
    
    if hasattr(xgc_instance, 'ion2_on') and xgc_instance.ion2_on:
        plt.plot(xgc_instance.od.psi, xgc_instance.od.i2gc_density_df_1d[0, :] / dunit, '--', label=i2_name)
    
    plt.legend()
    plt.xlabel('Normalized Pol. Flux')
    plt.ylabel('Density ($10^{19} m^{-3}$)')
    plt.title('Initial Density')

    # Plasma beta (electron) - initial
    fig, ax = plt.subplots()
    bunit = 1E-2
    
    try:
        if hasattr(xgc_instance.od, 'beta_e'):
            plt.plot(xgc_instance.od.psi, xgc_instance.od.beta_e[0, :] / bunit)
            plt.title('Electron beta (%)')
            plt.xlabel('Normalized Pol. Flux')
            plt.ylabel('$\\beta_e$ (%)')
    except:
        print('beta_e plot ignored')

    # Evolution plots using plot1d_if
    ie = end_idx
    
    # Import plot1d_if function
    from .plotting import plot1d_if
    
    if hasattr(xgc_instance, 'electron_on') and xgc_instance.electron_on:
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.e_gc_density_df_1d[:ie, :], varstr='Density (m^-3)')
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.e_gc_density_df_1d[:ie, :], varstr='Density (m^-3)', xlim=edge_lim)

    plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.i_gc_density_df_1d[:ie, :], varstr=i_name + ' g.c. Density (m^-3)')
    plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.i_gc_density_df_1d[:ie, :], varstr=i_name + ' g.c. Density (m^-3)', xlim=edge_lim)

    if hasattr(xgc_instance, 'ion2_on') and xgc_instance.ion2_on:
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.i2gc_density_df_1d[:ie, :], varstr=i2_name + ' g.c. Density (m^-3)')
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.i2gc_density_df_1d[:ie, :], varstr=i2_name + ' g.c. Density (m^-3)', xlim=edge_lim)

    # Temperature evolution
    if hasattr(xgc_instance, 'electron_on') and xgc_instance.electron_on:
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.Te[:ie, :], varstr='Elec. Temperature (eV)')
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.Te[:ie, :], varstr='Elec. Temperature (eV)', xlim=edge_lim)

    plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.Ti[:ie, :], varstr=i_name + ' Temperature (eV)')
    plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.Ti[:ie, :], varstr=i_name + ' Temperature (eV)', xlim=edge_lim)

    if hasattr(xgc_instance, 'ion2_on') and xgc_instance.ion2_on:
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.Ti2[:ie, :], varstr=i2_name + ' Temperature (eV)')
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.Ti2[:ie, :], varstr=i2_name + ' Temperature (eV)', xlim=edge_lim)

    # Flow plots
    plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.i_parallel_flow_df_1d[:ie, :], varstr=i_name + ' parallel flow FSA (m/s)')
    
    if hasattr(xgc_instance, 'ion2_on') and xgc_instance.ion2_on:
        plot1d_if(xgc_instance, xgc_instance.od, var=xgc_instance.od.i2parallel_flow_df_1d[:ie, :], varstr=i2_name + ' parallel flow FSA (m/s)')


def report_turb_2d(xgc_instance, i_name='Main ion', i2_name='Impurity', 
                  pm=slice(0, -1), tm=slice(0, -1), wnorm=1E6, cmap='jet'):
    """
    Generate 2D turbulence report with heat flux contours.
    
    Parameters
    ----------
    xgc_instance : xgc1
        XGC instance with loaded 1D data and flux calculations
    i_name : str, optional
        Main ion species name
    i2_name : str, optional
        Impurity species name
    pm : slice, optional
        Psi mask for plotting
    tm : slice, optional
        Time mask for plotting  
    wnorm : float, optional
        Normalization factor for flux (default 1E6 for MW)
    cmap : str, optional
        Colormap for contour plots
    """
    if plt is None:
        raise ImportError("matplotlib is required for turbulence reporting")
    
    if not hasattr(xgc_instance, 'od'):
        raise ValueError("1D data not loaded. Call load_oned() first.")
    
    # Electron heat flux plots
    if hasattr(xgc_instance, 'electron_on') and xgc_instance.electron_on:
        if hasattr(xgc_instance.od, 'efluxexbe'):
            fig, ax = plt.subplots()
            cf = ax.contourf(xgc_instance.od.psi[pm], xgc_instance.od.time[tm] * 1E3, 
                           xgc_instance.od.efluxexbe[tm, pm] / wnorm, levels=50, cmap=cmap)
            fig.colorbar(cf)
            plt.title('Electron Heat Flux by ExB (MW)')
            plt.xlabel('Poloidal Flux')
            plt.ylabel('Time (ms)')

        if hasattr(xgc_instance.od, 'efluxe'):
            fig, ax = plt.subplots()
            cf = ax.contourf(xgc_instance.od.psi[pm], xgc_instance.od.time[tm] * 1E3,
                           xgc_instance.od.efluxe[tm, pm] / wnorm, levels=50, cmap=cmap)
            fig.colorbar(cf)
            plt.title('Electron Heat Flux (MW)')
            plt.xlabel('Poloidal Flux')
            plt.ylabel('Time (ms)')

    # Ion heat flux plots
    if hasattr(xgc_instance.od, 'efluxi'):
        fig, ax = plt.subplots()
        cf = ax.contourf(xgc_instance.od.psi[pm], xgc_instance.od.time[tm] * 1E3,
                       xgc_instance.od.efluxi[tm, pm] / wnorm, levels=50, cmap=cmap)
        fig.colorbar(cf)
        plt.title(f'{i_name} Heat Flux (MW)')
        plt.xlabel('Poloidal Flux')
        plt.ylabel('Time (ms)')

    if hasattr(xgc_instance.od, 'efluxexbi'):
        fig, ax = plt.subplots()
        cf = ax.contourf(xgc_instance.od.psi[pm], xgc_instance.od.time[tm] * 1E3,
                       xgc_instance.od.efluxexbi[tm, pm] / wnorm, levels=50, cmap=cmap)
        fig.colorbar(cf)
        plt.title(f'{i_name} Heat Flux by ExB (MW)')
        plt.xlabel('Poloidal Flux')
        plt.ylabel('Time (ms)')


def turb_2d_report(xgc_instance, i_name='Main ion', i2_name='Impurity', 
                  pm=slice(0, -1), tm=slice(0, -1), wnorm=1E6, cmap='jet'):
    """Alias for report_turb_2d for backward compatibility."""
    return report_turb_2d(xgc_instance, i_name, i2_name, pm, tm, wnorm, cmap)