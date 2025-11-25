""" Some Python utilities for general use
"""

import os, sys
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import erfc

import adios2

def Try(func, verbose=True):
    try:
        func()
        if verbose: print(f'{func.__name__:<15} -> Succeed')
    except Exception as exc:
        print(f'{func.__name__:<15} -> Failed')
        print(f"Error: {exc}")
        print("Traceback details:")
        traceback.print_exc()

#--- basic XGC IO helpers
def gen_arr(start, interval, number):
    return np.arange(start, start + interval*number, interval)

def get_existing_steps(fdir='./', header='3d'):
    existing_dirs = []
    for (dirpath, dirnames, filenames) in os.walk(fdir):
        existing_dirs.extend(dirnames)
        break

    existing = sorted([x for x in dirnames if f'xgc.{header}' in x], key=lambda x: int(''.join(filter(str.isdigit, x))))

    existing_steps = []
    regex = re.compile(r'\d+')
    for _indx, _dir in enumerate(existing):
        existing_steps.append(int(regex.findall(_dir)[1]))
    
    return np.array(existing_steps)

def closest_existing_step(step_in, fdir='./', header='3d'):
    existing_steps = get_existing_steps(fdir, header)
    step_out = existing_steps[np.argmin(abs(existing_steps-step_in))]
    return step_out

# select N steps around 'step_base' in 'existing_steps'
def select_step_window(existing_steps, step_base, N):
    istep_base = np.argmin(abs(existing_steps-step_base))
    
    half = N//2 # divide and round down
    start = max(0, istep_base-half)
    end = start+N-1

    # if 'end' goes past the total, shift window backward
    if end > len(existing_steps)-1:
        end = len(existing_steps)-1
        start = max(0, end-N+1)

    return existing_steps[start:end+1]

def convert_time_to_step(xr, tm, target_time): # target_time in [ms]
    step_tm = xr.od.step[tm]
    time_tm = xr.od.time[tm]

    _target_time = time_tm[np.argmin(abs(time_tm-(np.array(target_time)/1e3)))]*1e3 # [ms]
    _itm = np.argmin(abs(time_tm-_target_time/1e3))
    _istep = tm[_itm]    
    return _istep

def convert_step_to_time(xr, tm, target_step): # []
    step_tm = xr.od.step[tm]
    time_tm = xr.od.time[tm]

    _target_step = step_tm[np.argmin(abs(step_tm-np.array(target_step)))]
    _itm = np.argmin(abs(step_tm-_target_step))
    _itime = tm[_itm]
    return _itime

def get_adios2_var(filestr, varstr):
    f=adios2.FileReader(filestr)
    #f.__next__()
    var=f.read(varstr)
    f.close()
    return var

def get_adios2_var_step(filestr, varstr, step):
    f=adios2.FileReader(filestr)
    vars=f.available_variables()
    ct=vars[varstr].get("Shape")
    ct=[int(i) for i in ct.split(',')]
    data = f.read(varstr, start=[0]*len(ct), count=ct, step_selection=[step, 1])
    f.close()
    return data

def get_adios2_var_allstep(filestr, varstr):
    f=adios2.FileReader(filestr)
    vars=f.available_variables()
    stc=vars[varstr].get("AvailableStepsCount")
    ct=vars[varstr].get("Shape")
    stc=int(stc)

    if ct!='':
        c=[int(i) for i in ct.split(',')]  #
        if len(c)==1 :
            return np.reshape(f.read(varstr, start=[0],    count=c, step_selection=[0,stc]), [stc, c[0]])
        elif len(c)==2 :
            return np.reshape(f.read(varstr, start=[0,0],  count=c, step_selection=[0,stc]), [stc, c[0], c[1]])
        elif ( len(c)==3 ):
            return np.reshape(f.read(varstr, start=[0,0,0],count=c, step_selection=[0,stc]), [stc, c[0], c[1], c[2]])
    else:
        return f.read(varstr, step_selection=[0,stc]) 



def get_3d_array(dir_run, step, varstr, header='3d', op_name="0th plane", verbose=True):
    operations = {"0th plane": lambda v: v[0,:],
                  "mean"     : lambda v: np.mean(v, axis=0),
                  "w/o n=0"  : lambda v: v[0,:]-np.mean(v, axis=0) }

    var = get_adios2_var(f"{dir_run}/xgc.{header}.{step:05}.bp", varstr)

    op_name_used = "no op."
    if var.ndim==2:
        # some arrays are [inode, iphi]. Force it to be [iphi, inode]
        if var.shape[1] < var.shape[0]:
            var = var.T

        op_name_used = op_name
        # op_name_used = "mean"
        # op_name_used = "w/o n=0"
    
        var = operations[op_name_used](var) # One of 1) "0th plane", 2) "mean", and 3) "w/o n=0"
        
        if verbose: print(f"reading f3d file: {varstr}, operation [{op_name_used}]")
    else:
        if verbose: print(f"reading f3d file: {varstr}, operation [{op_name_used}] (`no op.` is forced as `var.ndim != 2`)")
    
    return var, op_name_used

def get_f3d_components(dir_run, step, varstr, op_name="0th plane", verbose=False):
    #--- Read xgc.f3d file for suffixes
    sufxs = ["_n0_f0", "_n0_df", "_turb_f0", "_turb_df"] # suffixes
    descs = [r"$\bar{f}_{A}$", r"$\bar{f}_{NA}$", r"$\tilde{f}_{A}$", r"$\tilde{f}_{NA}$"] # descriptions for each suffixes
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    # Container
    var = {}
    total = None  # placeholder for accumulating the total
    
    # Load and annotate
    for _sufx, _desc, _c in zip(sufxs, descs, colors):
        try:
            _data, _op_name = get_3d_array(dir_run, step, varstr+_sufx, header="f3d", op_name=op_name, verbose=verbose)
            
            # simple sanity checks
            if _data.ndim != 1:       raise ValueError("The target values from `xgc.f3d` are expected to be toroidal averaged quantity")                
            if np.isnan(_data).any(): raise ValueError("NaN detected")
            
            var[_sufx[1:]] = { # strip leading underscore for cleaner keys
                "data": _data,
                "desc": _desc,
                "c": _c
            }

            # Accumulate total
            total = _data if total is None else total + _data
            
        except FileNotFoundError:
            print(f"Suffix '{_sufx}' not found for {varstr}")

    # Add total if any component was found
    if total is not None:
        var["total"] = {
            "data": total,
            "desc": "Total",
            "c": "k"
        }
    
    return var

#--- Velocity contour plots
def contour_vgrid_f0(ax, iphi, inode, f0_f, vpara=np.linspace(-4,4,29), vperp=np.linspace(0,4,33), title="", use_log=False, draw_contour=False, **kwargs):
    if f0_f.ndim ==4:
        _f = f0_f[iphi,:,inode,:] # [iphi, iperp, inode, ipara]
    else:
        _f = f0_f[:,inode,:]
        print(f"'iphi' {iphi} is given, but it seems that toroidally averaged f0_f is given. So ignoring 'iphi'")

    if use_log:
        title = title+" (log)"
        cntr = ax.contourf(vpara, vperp, np.log(_f[:,:]), **kwargs)
    else:
        cntr = ax.contourf(vpara, vperp, _f[:,:], **kwargs)
        
    plt.colorbar(cntr)
    cntr.colorbar.formatter.set_powerlimits((0,0))
    ax.set_xlabel('vpara')
    ax.set_ylabel('vperp')
    ax.set_title(title)
    
    if draw_contour:
        levels=[1e-2,1e-1,1e0,1e1,1e2]
        cs = ax.contour(vpara, vperp, _f[:,:], levels=levels, colors='white', linewidths=1.5)
        ax.clabel(cs, fmt="%.2f", colors='white', fontsize=10)

#--- Drawing line
def find_mesh_indice(r,z,r0,z0):
    dist2 = (r - r0)**2 + (z - z0)**2  # squared distance
    return np.argmin(dist2)

def gen_line_pt(pt1, pt2, x):
    slope = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    C = pt1[1] - (pt2[1]-pt1[1])/(pt2[0]-pt1[0])*pt1[0]
    y = slope * x + C
    return x, y

def gen_line_theta(pt1, theta, x):
    slope = np.tan(theta)
    C = pt1[1] - slope*pt1[0]
    y = slope * x + C
    return x, y

def gen_line_theta_dist(pt1, theta, dist=1.0, dist_backward=0.0):
    # x = np.linspace(pt1[0],pt1[0]+np.cos(theta)*dist,1000)
    # slope = np.tan(theta)
    # C = pt1[1] - slope*pt1[0]
    # y = slope * x + C
    # return x, y

    epsilon = 1e-8  # tolerance for floating-point comparison
    if np.isclose(np.cos(theta), 0.0, atol=epsilon):
        # Vertical line
        x = np.full(1000, pt1[0])  # constant x
        if np.sin(theta) > 0:
            y = np.linspace(pt1[1], pt1[1] + dist, 1000)
        else:
            y = np.linspace(pt1[1], pt1[1] - dist, 1000)
    else:
        x = np.linspace(pt1[0] - np.cos(theta)*dist_backward, pt1[0] + np.cos(theta)*dist, 1000)
        slope = np.tan(theta)
        C = pt1[1] - slope * pt1[0]
        y = slope * x + C
    return x, y

def pt_to_theta(pt1, pt2): # rad
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    theta = np.atan2(dy, dx)
    return theta

#--- Eich fit
def eich(xdata,q0,s,lq,dsep):
  return 0.5*q0*np.exp((0.5*s/lq)**2-(xdata-dsep)/lq)*erfc(0.5*s/lq-(xdata-dsep)/s)

def eich_fit1(ydata,rmidsepmm,pmask=None):
  q0init=np.max(ydata)  
  sinit=0.1 # 1mm
  lqinit=3 # 3mm
  dsepinit=0.1 # 0.1 mm

  p0=np.array([q0init, sinit, lqinit, dsepinit])
  if(pmask==None):
      popt,pconv = curve_fit(eich,rmidsepmm,ydata,p0=p0)
  else:
      popt,pconv = curve_fit(eich,rmidsepmm[pmask],ydata[pmask],p0=p0)

  return popt, pconv
def deg2rad(deg):
    return deg * np.pi / 180

def get_OMP_index(xr, isurf, hfs=False):
    inodes = xr.mesh.surf_idx[isurf, 0:xr.mesh.surf_len[isurf]]-1
    if not hfs:
        msk = (xr.mesh.r[inodes] > xr.eq_axis_r) # LFS
    else:
        msk = (xr.mesh.r[inodes] < xr.eq_axis_r) # HFS
    
    idx = np.argmin(abs(xr.mesh.z[inodes[msk]]-xr.eq_axis_z))
    idx_omp = inodes[msk][idx]
    inode_idx_omp = np.argmin(abs(inodes-idx_omp))
    
    return inode_idx_omp
