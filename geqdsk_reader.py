"""

Reading GEQDSK file based on freeqdsk module
In addition to the data from GEQDSK, it adds additonal data.
rgrid: R grid for psi data
zgrid: Z grid for psi data
rmin: minimum R in R grid
zmin: minimum Z in Z grid
rmid_sep: Miplane R at separatrix
bp_midsep : Poloidal B-field at (rmid_sep, zmagx)
B0 : B at magnetic axis (can be different from bcenter). It is evaluated using Fpol/R0
BT_midsep: Toroidal B-field at (rmid_sep, zmagx)
a_out: minor radius of outer midplane (rmid_sep-r_axis)
a_all: (max(r) - min(r))/2 of separatrix
"""

"""
Following is the comment from freeqdsk


The Fortran format for the header can be expressed as ``(a48,3i4)``. This is followed
by 4 lines of floats describing a tokamak plasma equilibrium. Each line contains 5
floats, following the Fortran format ``(5e16.9)``. These floats are:

====== ====== ====== ====== ======
rdim   zdim   rcentr rleft  zmid
rmagx  zmagx  simagx sibdry bcentr
cpasma simagx        rmagx
zmagx         sibdry
====== ====== ====== ====== ======

The blank spaces are ignored, and are usually set to zero. Note that ``rmagx``,
``zmagx``, ``simagx``, and ``sibdry`` are duplicated. The meaning of these
floats are:

======= ========================================================================
rdim    Width of computational domain in R direction, float [meter]
zdim    Height of computational domain in Z direction, float [meter]
rcentr  Reference value of R, float [meter]
rleft   R at left (inner) boundary, float [meter]
zmid    Z at middle of domain, float [meter]
rmagx   R at magnetic axis (0-point), float [meter]
zmagx   Z at magnetic axis (0-point), float [meter]
simagx  Poloidal flux :math:`\psi` at magnetic axis, float [weber / radian]
sibdry  Poloidal flux :math:`\psi` at plasma boundary, float [weber / radian]
bcentr  Vacuum toroidal magnetic field at rcentr, float [tesla]
cpasma  Plasma current, float [ampere]
======= ========================================================================

This is then followed by a series of grids:

======= ========================================================================
fpol    Poloidal current function :math:`F(\psi)=RB_t`, 1D array [meter * tesla]
pres    Plasma pressure :math:`p(\psi)`, 1D array [pascal]
ffprime :math:`FF'(\psi)`, 1D array [meter**2 * tesla**2 * radian / weber]
pprime  :math:`p'(\psi)`, 1D array [pascal * radian / weber]
psi     Poloidal flux :math:`\psi`, 2D array [weber / radian]
qpsi    Safety factor :math:`q(\psi)`, 1D array [dimensionless]
======= ========================================================================

The 1D arrays are expressed on a linearly spaced :math:`\psi` grid which may be
generated using ``numpy.linspace(simagx, sibdry, nx)``. The 2D :math:`\psi` grid is
instead expressed on a linearly spaced  grid extending the range
``[rleft, rleft + rdim]`` in the R direction and ``[zmid - zdim/2, zmid + zdim/2]``
in the Z direction. Each grid is printed over multiple lines using the Fortran
format ``(5e16.9)``, with the final line containing some blank spaces if the total
grid size is not a multiple of 5. Note that the ``psi`` grid is expressed in a
flattened state using Fortran ordering, meaning it increments in the columns
direction first, then in rows.

The G-EQDSK file then gives information on the plasma boundary and the surrounding
limiter contour. The next line gives the dimensions of these grids in the format
``(2i5)``:

======= ========================================================================
nbdry   Number of points in the boundary grid, int
nlim    Number of points in the limiter grid, int
======= ========================================================================

Finally, the boundary and limiter grids are specified as lists of ``(R, Z)``
coordinates, again using the format ``(5e16.9)``:

======= ========================================================================
rbdry   R of boundary points, 1D array [meter]
zbdry   Z of boundary points, 1D array [meter]
rlim    R of limiter points, 1D array [meter]
zlim    Z of limiter points, 1D array [meter]
======= ======================================================================== 

"""
import numpy as np
import matplotlib.pyplot as plt

def geqdsk_reader(filename, overide_wall=None):
    from freeqdsk import geqdsk
    
    with open(filename, "r") as f:
        g = geqdsk.read(f)
    #make (r,z) grid
    mw=g['nx']
    mh=g['ny']
    dr=g['rdim']/(mw-1)
    dz=g['zdim']/(mh-1)
    g['zmin']=g['zmid']-g['zdim']/2
    g['rmin']=g['rleft']
    
    g['rgrid'] = g['rmin'] + np.arange(mw) * dr
    g['zgrid'] = g['zmin'] + np.arange(mh) * dz

    if overide_wall is not None:
        read_wall(g,overide_wall)
    
    add_values(g)
    return g

def read_wall(g, filename):
    #read wall data
    with open(filename, "r") as f:
        [n] = map(int, f.readline().strip().split())
        #allocate array
        r=np.zeros(n)
        z=np.zeros(n)

        for l in range(n):
            [r[l], z[l]]=map(float, f.readline().strip().split() )

        g['rlim']=r
        g['zlim']=z


def add_values(g):
    #define spline function
    from scipy.interpolate import interp1d, RectBivariateSpline
    psi_rbs = RectBivariateSpline(g['rgrid'], g['zgrid'], g['psi'])
    
    #midplane
    nrmid = 1000
    try:
        rmid_max = np.max(g['rlim'])  # maximum limiter r is maximum of rmid
        has_wall = True        
    except:
        print('Warning: No limiter data in GEQDSK file. Use overide_wall="[FILE_NAME]" to provide wall data')
        has_wall = False

    if has_wall:
        drmid = (rmid_max - g['rmagx']) / (nrmid - 1)
        rmid = g['rmagx'] + drmid * np.arange(nrmid)
        zmid = g['zmagx']
        psimid = psi_rbs(rmid, zmid).flatten()
        g['rmid']=rmid
        g['psimid']=psimid
        #plt.plot(rmid,psimid)

        #find midplane r
        interp_func = interp1d(psimid, rmid)
        rmid_sep = interp_func(g['sibdry'])
        g['rmid_sep'] = rmid_sep

        #bp at (rmid_sep,zmid)
        dpdr=psi_rbs.partial_derivative(1,0)
        dpdz=psi_rbs.partial_derivative(0,1)
        dpdr_ms=dpdr(rmid_sep,zmid)
        dpdz_ms=dpdz(rmid_sep,zmid)
        bp_midsep = np.sqrt(dpdr_ms**2 + dpdz_ms**2)/rmid_sep
        g['bp_midsep']=bp_midsep

        # find Bt at midplane - separatrix
        g['BT_midsep'] = g['fpol'][-1] / rmid_sep
        g['a_out'] = rmid_sep - g['rmagx']


    # Other quantities
    g['B0'] = g['fpol'][0] / g['rmagx']    
    g['a_all'] = (np.max(g['rbdry']) - np.min(g['rbdry'])) / 2
    g['n_gb'] = g['cpasma'] / 1E6 / (np.pi * g['a_all']**2) #g['cpasma'] is current
    return 

def show_geqdsk(g):
    plt.contour(g['rgrid'], g['zgrid'], g['psi'].transpose(), 100,cmap='jet')
    plt.axis('equal')
    plt.plot(g['rbdry'], g['zbdry'], 'k', linewidth=2)
    plt.plot(g['rlim'], g['zlim'], 'k', linewidth=2)
    plt.plot(g['rmagx'], g['zmagx'], 'kx')

    print(f'Axis R, rmagx = {g['rmagx']} m')
    print(f'Axis B, B0 = {g['B0']} T')
    print(f'radius at midplane-separatrix rmid_sep= {g['rmid_sep']} m')


def find_x_point(rgrid, zgrid, psi, initial_guess, bd_dim):
    from scipy.optimize import minimize
    from scipy.interpolate import interp1d, RectBivariateSpline

    """
    Attempt to find a saddle point of the psi_rbs function.

    Parameters:
    - rgrid: 1D array of R grid points
    - zgrid: 1D array of Z grid points
    - psi: 2D array of psi values on the R-Z grid
    - initial_guess: Tuple (r, z) as the initial guess for the optimization
    - bd_dim: Tuple (dr,dz) as the bounds relative to initial_guess
    Returns:
    - A dictionary with the saddle point (r, z) and success status.
    """
    psi_rbs = RectBivariateSpline(rgrid, zgrid, psi)

    def gradient_magnitude(point):
        r, z = point
        dpdr = psi_rbs.ev(r, z, dx=1, dy=0)
        dpdz = psi_rbs.ev(r, z, dx=0, dy=1)
        return np.sqrt(dpdr**2 + dpdz**2)

    ri,zi = initial_guess
    dr,dz = bd_dim
    options={'disp': False, 'xtol': 1E-7}
    result = minimize(gradient_magnitude, initial_guess, method='Powell', bounds=[(ri-dr, ri+dr), (zi-dz, zi+dz)],options=options)

    if result.success:
        
        print('grad_x',gradient_magnitude(result.x))
        r_saddle, z_saddle = result.x
        # Check the Hessian's determinant at the saddle point for confirmation
        d2pdr2 = psi_rbs.ev(r_saddle, z_saddle, dx=2, dy=0)
        d2pdz2 = psi_rbs.ev(r_saddle, z_saddle, dx=0, dy=2)
        d2pdzdr = psi_rbs.ev(r_saddle, z_saddle, dx=1, dy=1)
        Hessian = np.array([[d2pdr2, d2pdzdr], [d2pdzdr, d2pdz2]])
        det_Hessian = np.linalg.det(Hessian)

        if det_Hessian < 0:
            return {'r': r_saddle, 'z': z_saddle, 'success': True, 'message': 'Saddle point found.'}
        else:
            return {'success': False, 'message': 'Optimization was successful, but the point is not a saddle point.'}
    else:
        return {'success': False, 'message': 'Optimization failed to converge.'}


