"""Module of eqd file reader (XGC input file for background magnetic field geometry)

"""

import numpy as np
import os
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
import matplotlib.pyplot as plt
import math

class eqd_class(object):
    class cnst:
        echarge = 1.602E-19
        protmass=  1.67E-27
        mu0 = 4 * 3.141592 * 1E-7

        
    def __init__(self, filename=None, verbose=False):
        """ 
        initialize it from the current directory.
        not doing much thing. 
        """        
        self.path=os.getcwd()+'/'
        if(filename!=None):
            self.read_eqd(filename, verbose=verbose)
    
    #read eqd file
    def read_eqd(self,filename, verbose):            
        with open(filename, 'r') as file:
            self.filename=filename
            self.header=file.readline()
            #read dimensions
            [self.mr, self.mz, self.mpsi] = map(int, file.readline().strip().split())
            if(verbose):
                print('mr, mz, mpsi=', self.mr, self.mz, self.mpsi)
            [self.min_r, self.max_r, self.min_z, self.max_z]= map(float, file.readline().strip().split())
            if(verbose):
                print('min_r, max_r, min_z, max_z=', self.min_r, self.max_r, self.min_z, self.max_z)
            # read const
            [self.axis_r, self.axis_z, self.axis_b]=map(float, file.readline().strip().split())
            if(verbose):
                print('axis_r, axis_z, axis_b=', self.axis_r, self.axis_z, self.axis_b)
            [self.x_psi, self.x_r, self.x_z]=map(float, file.readline().strip().split())
            if(verbose):
                print('x_psi, x_r, x_z=', self.x_psi, self.x_r, self.x_z)
            # read psi_grid & I (R*BT)
            vars_per_line=4
            nl=math.ceil(self.mpsi/vars_per_line) # number of lines to read
            
            self.psi_grid=[]
            for l in range(nl):
                self.psi_grid.extend(list(map(float, file.readline().split())))
            if(verbose):
                print('psi_grid=', self.psi_grid[0],'...',self.psi_grid[-1])

            self.I=[]
            for l in range(nl):
                self.I.extend(list(map(float, file.readline().split())))
            if(verbose):
                print('I=', self.I[0],'...',self.I[-1])

            #read psi_rz
            nl=math.ceil(self.mr*self.mz/vars_per_line)
            psi_rz=[]
            for l in range(nl):
                psi_rz.extend(list(map(float, file.readline().split())))
            self.psi_rz = np.array(psi_rz).reshape((self.mr, self.mz)) # check mr,mz order
            if(verbose):
                print('psi_rz=', self.psi_rz[0,0],'...',self.psi_rz[-1,-1])

            #read end flag
            [end_flag]=map(int, file.readline().strip().split())
            if(end_flag!=-1):
                print('Error: end flag is not -1. end_flag= %d'%end_flag)

            #read limiter data
            [nlim]=map(int, file.readline().strip().split())
            if(nlim>0):
                rlim=[]
                zlim=[]
                for i in range(nlim):
                    [r,z]=map(float, file.readline().strip().split())
                    rlim.append(r)
                    zlim.append(z)
                self.rlim=np.array(rlim)
                self.zlim=np.array(zlim)
                [end_flag]=map(int, file.readline().strip().split())
                if(end_flag!=-1):
                    print('Error: end flag is not -1. end_flag= %d'%end_flag)
            else:
                print('Warning: limiter data not found.')

            if(verbose):
                print('rlim=', self.rlim[0],'...',self.rlim[-1])


            #read boundary data
            [nbdry]=map(int, file.readline().strip().split())
            if(nbdry>0):
                rbdry=[]
                zbdry=[]
                for i in range(nbdry):
                    [r,z]=map(float, file.readline().strip().split())
                    rbdry.append(r)
                    zbdry.append(z)
                self.rbdry=np.array(rbdry)
                self.zbdry=np.array(zbdry)
                [end_flag]=map(int, file.readline().strip().split())
                if(end_flag!=-1):
                    print('Error: end flag is not -1. end_flag= %d'%end_flag)
            else:
                print('Warning: boundary data not found.')

            if(verbose):
                print('rbdry=', self.rbdry[0],'...',self.rbdry[-1])

            #post process
            # setup rgrid and zgrid
            self.rgrid=np.linspace(self.min_r, self.max_r, num=self.mr)
            self.zgrid=np.linspace(self.min_z, self.max_z, num=self.mz)

    #writing eqd file. It requires freeqdsk 
    def write_eqd(self,filename):
        from freeqdsk._fileutils import write_array
        with open(filename, 'w') as file:
            self.header
            file.write(f"{self.header}\n")
            f_fmt='4(e19.12,1x)'
            write_array((self.mr,self.mz,self.mpsi),file,'3I8')
            write_array((self.min_r,self.max_r,self.min_z,self.max_z),file,f_fmt)
            write_array((self.axis_r,self.axis_z,self.axis_b),file,f_fmt)
            write_array((self.x_psi,self.x_r,self.x_z),file,f_fmt)
            write_array(self.psi_grid,file,f_fmt)
            write_array(self.I,file,f_fmt)
            write_array(self.psi_rz,file,f_fmt)
            file.write('-1\n')
            #limiter and separatrix information is ignored.
            
            if hasattr(self,'rlim'):
                eqd_nlim = len(self.rlim)
                file.write(f'{eqd_nlim}\n')

                #5003 format(e19.13,' ',e19.13)
                f_fmt='2(e19.12,1x)'
                eqd_rlim = self.rlim
                eqd_zlim = self.zlim # Assumes zlim exists if rlim exists and is the same length
                for i in range(eqd_nlim):
                    write_array((eqd_rlim[i],eqd_zlim[i]),file,f_fmt)
                file.write('-1\n')
            else:
                print("Warning: eqd.rlim not found. Skipping limiter data.")

            # bdry data
            if hasattr(self, 'rbdry'):
                eqd_nbdry = len(self.rbdry)
                file.write(f'{eqd_nbdry}\n')

                eqd_rbdry = self.rbdry
                eqd_zbdry = self.zbdry # Assumes zbdry exists if rbdry exists and is the same length
                for i in range(eqd_nbdry):
                    file.write(f'{eqd_rbdry[i]:+17.12e} {eqd_zbdry[i]:+17.12e}\n')
                file.write('-1\n')
            else:
                print("Warning: eqd.rbdry not found. Skipping boundary data.")


def get_eqd_from_eqdsk(g):
    """
    Create an eqd_class object from
    the dictionary of eqdsk data.
    """

    eqd = eqd_class()
    eqd.header = 'Equilibrium data generated by efit2eqd.py from efit data.'
    eqd.mr=g['nx']
    eqd.mz=g['ny']
    eqd.mpsi=g['nx']
    eqd.min_r=g['rgrid'][0]
    eqd.max_r=g['rgrid'][-1]
    eqd.min_z=g['zgrid'][0]
    eqd.max_z=g['zgrid'][-1]

    #refine the magnetic axis
    #  - find the local extrema or saddle near the magnetic axis of eqdsk.
    raf, zaf, psiaf  = refine_axis_or_x_spline_optimize(g['rgrid'], g['zgrid'], g['psi'], g['rmagx'], g['zmagx'], search_radius=0.1)
    print(f"Refined magnetic axis: ({raf}, {zaf}, {psiaf})")
    print(f"Initial magnetic axis: ({g['rmagx']}, {g['zmagx']}, {g['simagx']})")
    eqd.axis_r=raf
    eqd.axis_z=zaf
    eqd.axis_b=np.abs(g['B0']) # maybe update B0 as well with Fpol/R_axis ? 

    # refine the X-point
    #  - find the local extrema or saddle near the X-point of eqdsk.
    # assume lower divertor is the primary X-point
    ix = np.argmin(g['zbdry'])
    rx = g['rbdry'][ix]
    zx = g['zbdry'][ix]

    rxf, zxf, psixf  = refine_axis_or_x_spline_optimize(g['rgrid'], g['zgrid'], g['psi'], rx, zx, search_radius=0.1)
    print(f"Refined X-point: ({rxf}, {zxf}, {psixf})")
    print(f"Initial X-point: ({rx}, {zx}, {g['sibdry']})")

    # set axis psi is zero 
    eqd.psi_rz=g['psi']-psiaf
    psixf = psixf - psiaf
    # set X-point psi is positive
    if(psixf<0):
        eqd.psi_rz=-eqd.psi_rz
        psixf=-psixf
    
    eqd.x_r=rxf
    eqd.x_z=zxf  
    eqd.x_psi=psixf
    eqd.psi_grid=np.linspace(0, psixf, eqd.mpsi)
    eqd.I = np.abs(g['fpol'])  # set fpol to positive

    eqd.rlim = g['rlim']
    eqd.zlim = g['zlim']
    eqd.rbdry = g['rbdry'] 
    eqd.zbdry = g['zbdry']

    return eqd






# the following is for searching magnetic axis or X-point
# This has overlapping feature with find_x_point of geqdsk_reader.py
# need to merge them in the future.

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

def refine_axis_or_x_spline_optimize(rgrid, zgrid, psi, rmaxis, zmaxis, search_radius=0.1):
    """
    Finds the local extrema of psi near (rmaxis, zmaxis) using 2D cubic spline interpolation
    and scipy.optimize.minimize to refine the magnetic axis location.

    Args:
        rgrid (np.array): 1D array of R grid points.
        zgrid (np.array): 1D array of Z grid points.
        psi (np.array): 2D array of psi values on the (rgrid, zgrid) grid.
        rmaxis (float): Initial guess for R of the magnetic axis.
        zmaxis (float): Initial guess for Z of the magnetic axis.
        search_radius (float): Radius around (rmaxis, zmaxis) to constrain the search.

    Returns:
        tuple: Refined (rmaxis, zmaxis, psi_value) location, where psi_value is the value of psi at the refined axis.
    """

    # Create the 2D cubic spline interpolator
    interp_func = RectBivariateSpline(rgrid, zgrid, psi)

    # Define the objective function 
    def objective_function(x):
        r = x[0]
        z = x[1]

        # get gradient
        grad = interp_func(r, z, dx=1, dy=0)**2 + interp_func(r, z, dx=0, dy=1)**2
        grad = grad[0][0]

        return grad  # RectBivariateSpline returns a 2D array

    # Initial guess
    initial_guess = [rmaxis, zmaxis]

    # Define the bounds for the search
    bounds = [(rmaxis - search_radius, rmaxis + search_radius),
              (zmaxis - search_radius, zmaxis + search_radius)]

    # Perform the optimization
    result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)

    if result.success:
        refined_rmaxis = result.x[0]
        refined_zmaxis = result.x[1]
        return refined_rmaxis, refined_zmaxis, interp_func(refined_rmaxis, refined_zmaxis)[0][0]
    else:
        print("Warning: Optimization failed. Returning initial guess.")
        return rmaxis, zmaxis, interp_func(rmaxis, zmaxis)[0][0]