
# coding: utf-8

# In[1]:

import numpy as np
import random
from __future__ import print_function
import matplotlib.pyplot as plt
from scipy import optimize, stats
import scipy.interpolate
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import sys
from multiprocessing import Pool
import pylab
import os
from matplotlib.colors import LogNorm
import itertools
get_ipython().magic(u'matplotlib inline')


# In[2]:

####coding for my numerical paper to produce nice pots of CII emissivity

def extrapolate_emissivity_data(filename, x, y, z, index):
    data = np.genfromtxt(filename)
    interp_val = scipy.interpolate.interp1d(data[:,0],data[:,index])
    radius = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    #usage example    print(extrapolate_emissivity_data('data_emissivity_test.dat', x, y, z))
    return interp_val(radius)

def polar_angle():
    return(np.arccos((2.0*random.random()-1.0)))   #2.0*np.pi*random.random()

def azimuthial_angle():
    return (2.0*np.pi*random.random())


# In[3]:

file = 'data_emissivity_test.dat'
data = np.genfromtxt(file)
radius_edge = max(data[:,0])
for m in np.arange(len(data[0,:])-1)+1:
    plt.plot(data[:,0],data[:,m])  
plt.savefig('cii_emissivity_profiles.pdf', bbox_inches='tight')




# In[4]:

n_photons = 400000
t_max = [0.0]
angle_viewed_from = 0.0  #np.pi/2.0
counter_to_plot = []
theory = []

for l in np.arange(len(data[0,:])-1)+1:
    for m in t_max:
        counter = -1.0
        xtwod_projected = [0.0]
        ytwod_projected = [0.0]
        weights_projected = [(1.0/(4.0*np.pi))*extrapolate_emissivity_data(file, 0.0, 0.0, 0.0, l)]
        for k in range(n_photons):
            x = 0.0
            y = 0.0
            z = 0.0
            r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
            while r >= 0.0 and r < radius_edge :
                distance = -np.log(random.random())*radius_edge
                theta = polar_angle()
                phi = azimuthial_angle()
                x = x + distance*np.sin(theta)*np.cos(phi)
                y = y + distance*np.sin(theta)*np.sin(phi)
                z = z + distance*np.cos(theta)
                r = np.sqrt(x**2.0 + y**2.0 + z**2.0) 
                counter = counter +1.0
                if r < radius_edge:
                    xtwod_projected.append(x)
                    ytwod_projected.append(y)
                    weight = (1.0/(4.0*np.pi))*extrapolate_emissivity_data(file, x, y, z, l)
                    weights_projected.append(weight)           
        print(counter/n_photons)
        print(m)
        counter_to_plot.append(counter/n_photons)
        H, xedges, yedges = np.histogram2d(xtwod_projected,ytwod_projected,bins=500.0, weights=weights_projected)
        H = np.rot90(H)
        H = np.flipud(H)
        Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero   
        plt.pcolormesh(xedges,yedges,Hmasked, norm=LogNorm(vmin=Hmasked.min().min(), vmax=Hmasked.max().max()))
        plt.xlabel('x (r/Rmax)')
        plt.ylabel('y (r/Rmax)')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('[CII] Emissivity')
        #plt.show()
        plt.savefig('test_index' + str(l) +'.pdf', bbox_inches='tight')
        plt.close()



# In[ ]:



