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

        
    def __init__(self, filename=None):
        """ 
        initialize it from the current directory.
        not doing much thing. 
        """        
        self.path=os.getcwd()+'/'
        if(filename!=None):
            self.read_eqd(filename)
    
    #read eqd file
    def read_eqd(self,filename):            
        with open(filename, 'r') as file:
            self.filename=filename
            self.header=file.readline()
            #read dimensions
            [self.mr, self.mz, self.mpsi] = map(int, file.readline().strip().split())
            [self.min_r, self.max_r, self.min_z, self.max_z]= map(float, file.readline().strip().split())
            # read const
            [self.axis_r, self.axis_z, self.axis_b]=map(float, file.readline().strip().split())
            [self.x_psi, self.x_r, self.x_z]=map(float, file.readline().strip().split())
            # read psi_grid & I (R*BT)
            vars_per_line=4
            nl=math.ceil(self.mpsi/vars_per_line) # number of lines to read
            
            self.psi_grid=[]
            for l in range(nl):
                self.psi_grid.extend(list(map(float, file.readline().split())))

            self.I=[]
            for l in range(nl):
                self.I.extend(list(map(float, file.readline().split())))

            #read psi_rz
            nl=math.ceil(self.mr*self.mz/vars_per_line)
            psi_rz=[]
            for l in range(nl):
                psi_rz.extend(list(map(float, file.readline().split())))
            self.psi_rz = np.array(psi_rz).reshape((self.mr, self.mz)) # check mr,mz order

            #read end flag
            [end_flag]=map(int, file.readline().strip().split())
            if(end_flag!=-1):
                print('Error: end flag is not -1. end_flag= %d'%end_flag)

            #post process
            # setup rgrid and zgrid
            self.rgrid=np.linspace(self.min_r, self.max_r, num=self.mr)
            self.zgrid=np.linspace(self.min_z, self.max_z, num=self.mr)

    #writing eqd file. It requires freeqdsk 
    def write_eqd(self,filename):
        from freeqdsk._fileutils import write_array
        with open(filename, 'w') as file:
            self.header
            file.write(f"{self.header}")
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



