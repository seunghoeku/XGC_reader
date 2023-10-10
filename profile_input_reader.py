"""Module of profile file reader 
(XGC input file for temperature/density/flow profile)

The format is
# of data points
p1 v1
p2 v2
p3 v3
.
.
.
-1

p are normalized poloidal flux
v are value in SI (Teperature in eV)
At the end of the file "-1" is added to check the vailidity
"""

import numpy as np
import os
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
import matplotlib.pyplot as plt
import math

class prf(object):
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
            self.read_prf(filename)
    
    #read prf file
    def read_prf(self,filename):            
        with open(filename, 'r') as file:
            self.filename=filename
            #read dimensions
            [self.n] = map(int, file.readline().strip().split())
            #allocate array
            self.psi=np.zeros(self.n)
            self.var=np.zeros(self.n)

            for l in range(self.n):
                [self.psi[l], self.var[l]]=map(float, file.readline().strip().split() )
            
            #read end flag
            [end_flag]=map(int, file.readline().strip().split())
            if(end_flag!=-1):
                print('Error: end flag is not -1. end_flag= %d'%end_flag)

    #write prf file
    def write_prf(self,filename):
        with open(filename,'w') as file:
            file.write("%8d\n"%(self.n))
            for l in range(self.n):
                file.write("%19.13e  %19.13e\n"%(self.psi[l],self.var[l]))
            file.write("-1\n")
