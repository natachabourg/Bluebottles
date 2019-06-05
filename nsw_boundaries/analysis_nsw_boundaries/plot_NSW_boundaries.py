# -*- coding: utf-8 -*-
"""

@author : Natacha
"""

"""
Read Shapefile, plot NSW boundaries, create a grid 
"""
import numpy as np
#import pandas as pd
#import geopandas as gpd
import scipy as sc
import pylab as py
import matplotlib.pyplot as plt
import scipy.io as sio #to load matlab file


def GetLonLat(lonmin, lonmax, latmin, latmax, stepsizelon): #generate 2 longitude et latitude vectors
    steps=(lonmax-lonmin)/stepsizelon
    stepsizelat=(latmax-latmin)/steps
    lon=np.arange(lonmin,lonmax,stepsizelon) 
    lat=np.arange(latmin,latmax,stepsizelat)
    ax=[]
    ay=[]
    for i in range(0,len(lon)):
        for j in range(0,len(lat)):
            ax.append(lon[i])
            ay.append(lat[j])
    return (ax,ay)

def LimitBoundaries(lon_i, lat_i, lon_min): #return lon and lat for every lon>lon_min
    newlat=[]
    newlon=[]
    for i in range (0,len(lat_i)):
        if (lon_i[i]>lon_min):
            newlat.append(lat_i[i])
            newlon.append(lon_i[i])
    return newlon, newlat

#Get NSW borders
lonmin, latmin, lonmax, latmax = 148.,-38.,174.,-25.
stepsizelon=0.5
lon, lat=GetLonLat(lonmin, lonmax, latmin, latmax, stepsizelon)
plt.plot(lon, lat, marker='.', color='k', linestyle='none')

"""Load & read Matlab file"""
mat_boundaries = sio.loadmat('../raw_nsw_boundaries/NSW_boundary.mat')
lat_nsw=mat_boundaries['lat_nsw'][0]
lon_nsw=mat_boundaries['lon_nsw'][0]
lon_nsw, lat_nsw = LimitBoundaries(lon_nsw, lat_nsw, 149.5)

plt.plot(lon_nsw, lat_nsw)
plt.show()