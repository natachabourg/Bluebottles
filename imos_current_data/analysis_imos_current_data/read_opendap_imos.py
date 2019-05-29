#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Laurent Besnard
# Institute: IMOS / eMII
# email address: laurent.besnard@utas.edu.au
# Website: http://imos.org.au/  https://github.com/aodn/imos_user_code_library
# May 2013; Last revision: 20-May-2013
#
# Copyright 2013 IMOS
# The script is distributed under the terms of the GNUv3 General Public License
import numpy as np
from netCDF4 import Dataset
from numpy import meshgrid
from matplotlib.pyplot import (figure, pcolor, colorbar, xlabel, ylabel, 
                               title, show)
#from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile

######## GHRSST – L3P mosaic
IMOS_cur_URL = 'http://thredds.aodn.org.au/thredds/dodsC/IMOS/OceanCurrent/GSLA/DM00/yearfiles/IMOS_OceanCurrent_HV_2019_C-20190520T232835Z.nc.gz'
 
IMOS_cur_DATA = Dataset(IMOS_cur_URL) 

step = 1 # we take one point out of 'step'. Only to make it faster to plot
vcur = IMOS_cur_DATA.variables['VCUR'][0,::step,::step]
ucur = IMOS_cur_DATA.variables['UCUR'][0,::step,::step]
lat =IMOS_cur_DATA.variables['LATITUDE'][::step]
lon = IMOS_cur_DATA.variables['LONGITUDE'][::step]
[lon_mesh,lat_mesh] = meshgrid(lon,lat)  #we create a matrix of similar size to be used afterwards with pcolor

figure1 =  figure(num=1, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
pcolor(lon_mesh,lat_mesh,vcur)

#title(srs_L3P_DATA.title + '-' +  srs_L3P_DATA.start_date)
xlabel(IMOS_cur_DATA.variables['LONGITUDE'].long_name +  ' in ' + IMOS_cur_DATA.variables['LONGITUDE'].units)
ylabel(IMOS_cur_DATA.variables['LATITUDE'].long_name +  ' in ' + IMOS_cur_DATA.variables['LATITUDE'].units)

cbar = colorbar()
cbar.ax.set_ylabel(IMOS_cur_DATA.variables['VCUR'].long_name + '\n in ' + IMOS_cur_DATA.variables['VCUR'].units)

figure2 =  figure(num=2, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
pcolor(lon_mesh,lat_mesh,ucur)

#title(srs_L3P_DATA.title + '-' +  srs_L3P_DATA.start_date)
xlabel(IMOS_cur_DATA.variables['LONGITUDE'].long_name +  ' in ' + IMOS_cur_DATA.variables['LONGITUDE'].units)
ylabel(IMOS_cur_DATA.variables['LATITUDE'].long_name +  ' in ' + IMOS_cur_DATA.variables['LATITUDE'].units)

cbar = colorbar()
cbar.ax.set_ylabel(IMOS_cur_DATA.variables['UCUR'].long_name + '\n in ' + IMOS_cur_DATA.variables['UCUR'].units)
show() 

f=Dataset('IMOS.nc','w', format='NETCDF4')
latt = f.createDimension('lat',len(lat))
lonn = f.createDimension('lon',len(lon))
time = f.createDimension('time',None)
U=f.createVariable('U', np.float64,('lat','lon','time'))
V=f.createVariable('V', np.float64,('lat','lon','time'))
U.units = 'm/s'
V.units = 'm/s'

U = ucur[:]
V = vcur[:]


