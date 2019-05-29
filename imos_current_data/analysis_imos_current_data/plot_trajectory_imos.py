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

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
    pset = ParticleSet.from_line(fieldset=fieldset,   # the fields on which the particles are advected
                                pclass=JITParticle,# the type of particles (JITParticle or ScipyParticle)
                                size=10,
                                start=(152., -30),
                                finish=(159.,-34))# a vector of release latitudes


    k_WestVel = pset.Kernel(WestVel)

    pset.execute(AdvectionRK4,                 # the kernel (which defines how particles move)
                runtime=timedelta(days=6),    # the total length of the run
                dt=timedelta(minutes=5),      # the timestep of the kernel
                recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
                output_file=pset.ParticleFile(name="IMOSTRY.nc", outputdt=timedelta(hours=1)))
    #pset.show(field=fieldset.U, show_time=datetime(2019, 3, 10, 2))
    #pset.show(field=fieldset.U, show_time=datetime(2019, 3, 20, 2))


    cpt.plotTrajectoriesFileModified('IMOSTRY.nc');
    """tracerfile=filenames,
                                     tracerlon='LONGITUDE',
                                     tracerlat='LATITUDE',
                                     tracerfield='UCUR');"""

filenames = "IMOS_OceanCurrent_HV_2019_C-20190520T232835Z.nc"
PlotImosFile(filenames)
