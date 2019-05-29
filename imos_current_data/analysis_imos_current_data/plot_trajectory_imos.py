# -*- coding: utf-8 -*-
"""

@author : Natacha 
"""

from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, ErrorCode
import numpy as np
import math
from datetime import timedelta, datetime
from operator import attrgetter
import copy_plottrajectoriesfile as cpt
import matplotlib.animation as animation
import imageio

def PlotImosFile(filenames):
    
    """
    Plot trajectory of particles in a current field (input in .nc) 
    """
                
    variables = {'U' : 'UCUR',
                'V' : 'VCUR'}
    dimensions = {'lat' : 'LATITUDE',
                  'lon' : 'LONGITUDE',
                  'time' : 'TIME'}

    def WestVel(particle, fieldset, time):
        if time > 86400:
            uvel = -0.02
            particle.lon += uvel * particle.dt
      
    def DeleteParticle(particle, fieldset, time):
        particle.delete()
    images = []

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    pset = ParticleSet.from_line(fieldset=fieldset,   # the fields on which the particles are advected
                                pclass=JITParticle,# the type of particles (JITParticle or ScipyParticle)
                                size=10,
                                start=(152., -30),
                                finish=(159.,-34))# a vector of release latitudes

    k_WestVel = pset.Kernel(WestVel)
    for i in range(70):
        pset.execute(AdvectionRK4,                 #
                    runtime=timedelta(hours=i),    # the total length of the run
                    dt=timedelta(minutes=5),      # the timestep of the kernel
                    recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                    output_file=pset.ParticleFile(name="IMOSTRY.nc", outputdt=timedelta(hours=1)))
        pset.show(domain={'N':-15, 'S':-42, 'E':170, 'W':140}, field='vector',vmin=0.,vmax=1.3, savefile='../outputs_imos_current_data/make_gif/particl'+str(i)+'.png')
        images.append(imageio.imread('../outputs_imos_current_data/make_gif/particl'+str(i)+'.png'))
    imageio.mimsave('../outputs_imos_current_data/particle_advection.gif', images)
    
   #cpt.plotTrajectoriesFileModified('IMOSTRY.nc',mode='movie2d', tracerlon='LONGITUDE',tracerlat='LATITUDE',tracerfield='UCUR');
    """tracerfile=filenames,
                                     tracerlon='LONGITUDE',
                                     tracerlat='LATITUDE',
                                     tracerfield='UCUR');"""

filenames = "../raw_imos_current_data/IMOS_OceanCurrent_HV_2019_C-20190520T232835Z.nc"
PlotImosFile(filenames)
