"""Module of eqd file reader (XGC input file for background magnetic field geometry)

"""

import numpy as np
import os
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
import matplotlib.pyplot as plt
import math

class eqd(object):
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


