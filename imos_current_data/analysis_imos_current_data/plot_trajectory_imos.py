# -*- coding: utf-8 -*-
"""

@author : Natacha 
"""

from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, ErrorCode
import numpy as np
import scipy as sc
import math
from datetime import timedelta, datetime
from operator import attrgetter
import copy_plottrajectoriesfile as cpt
import matplotlib.animation as animation
import imageio
import matplotlib.pyplot as plt
import scipy.io as sio #to load matlab file

"""
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

points = np.random.rand(1000, 2)
values = func(points[:,0], points[:,1])

grid_z0 = sc.interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
plt.subplot(222)
plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
plt.title('Nearest')
plt.show()
"""

def PlotImosFile(filenames):
    """
    Plot trajectory of particles in a current field (input in .nc) 
    """
                
    variables = {'U' : 'UCUR',
                'V' : 'VCUR'}
    dimensions = {'lat' : 'LATITUDE',
                  'lon' : 'LONGITUDE',
                  'time' : 'TIME'}

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    lati=fieldset.U.lat
    longi=fieldset.U.lon
    points=[lati,longi]
    ucur=fieldset.U.data
    vcur=fieldset.V.data
    new_x=np.linspace(24,36,100)
    new_y=np.linspace(150,160,100)
    neww=np.meshgrid(new_x,new_y)
    u_new = sc.interpolate.griddata(points, ucur, neww, method='nearest')
    
         
    def DeleteParticle(particle, fieldset, time):
        particle.delete()
        
    images = []

    
    #pset = ParticleSet.from_line(fieldset=fieldset,   
                            #   pclass=JITParticle,
                            #    size=10,
                             #   start=(151.259, -33.919),
                            #    finish=(151.257,-33.923))
    
    
    pset=ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=151.259, lat=-33.92, time=datetime(2019, 3, 9, 2))

    for i in range(1):
        pset.execute(AdvectionRK4,                 #
                    runtime=timedelta(hours=i),    # the total length of the run
                    dt=-timedelta(minutes=5),      # the timestep of the kernel
                    recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                    output_file=pset.ParticleFile(name="IMOSTRY.nc", outputdt=timedelta(hours=1)))
        
        plt.plot(lon_nsw, lat_nsw)
        pset.show(domain={'N':-24, 'S':-36, 'E':160, 'W':150}, field='vector',vmin=0.,vmax=1.3, land=False)
        
        
        pset.show(domain={'N':-15, 'S':-42, 'E':170, 'W':140}, field='vector',vmin=0.,vmax=1.3, savefile='../outputs_imos_current_data/make_gif/particl_revers'+str(i)+'.png')
        #images.append(imageio.imread('../outputs_imos_current_data/make_gif/particl_revers'+str(i)+'.png'))
    #imageio.mimsave('../outputs_imos_current_data/particle_advection_reverse.gif', images)
    
    cpt.plotTrajectoriesFileModified('IMOSTRY.nc',mode='movie2d',tracerlon='LONGITUDE',tracerlat='LATITUDE',tracerfield='UCUR');
   #tracerfile=filenames,
                   #                  tracerlon='LONGITUDE',
                #                     tracerlat='LATITUDE',
                #                     tracerfield='UCUR');


filenames = "../raw_imos_current_data/IMOS_OceanCurrent_HV_2019_C-20190520T232835Z.nc"
PlotImosFile(filenames)

"""
nsw boundaries

    mat_boundaries = sio.loadmat('../../nsw_boundaries/raw_nsw_boundaries/NSW_boundary.mat')
    lat_nsw=mat_boundaries['lat_nsw'][0]
    lon_nsw=mat_boundaries['lon_nsw'][0]
    lon_nsw, lat_nsw = LimitBoundaries(lon_nsw, lat_nsw, 149.5)


"""