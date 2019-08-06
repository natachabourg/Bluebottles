#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:00:41 2019

@author: amandine
"""
#%reset -f
#%matplotlib qt 
#%matplotlib inline 
from matplotlib import pyplot as plt
import numpy as np
from datetime import date
import datetime
import matplotlib.dates as mdates
#import glob
import pandas as pd
import matplotlib as mpl

file = 'month.csv'
df = pd.read_csv(file)
simul = (df['simul'][:].values.tolist())
total_obs = (df['total obs'][:]).values.astype('float')
diff = (df['diff/month'][:]).values.astype('float')

# Def categories symbols
matching_0deg = [s for s in simul if "0deg" in s]
matching_0deg_ind = [i for i in range(len(simul)) if "0deg" in simul[i]]
matching_randdeg = [s for s in simul if "randeg" in s]
matching_randdeg_ind = [i for i in range(len(simul)) if "randeg" in simul[i]]
matching_varydeg = [s for s in simul if "varydeg" in s]
matching_varydeg_ind = [i for i in range(len(simul)) if "varydeg" in simul[i]]
matching_justwinds = [s for s in simul if "justwinds" in s]
matching_justwinds_ind = [i for i in range(len(simul)) if "justwinds" in simul[i]]

 # Def categories colours                         
matching_CD004_ind = [i for i in range(len(simul)) if "CD004_" in simul[i]]
matching_CD01_ind = [i for i in range(len(simul)) if "CD01_" in simul[i]]
matching_CD015_ind = [i for i in range(len(simul)) if "CD015_" in simul[i]]
# Assign colours                                   
matching_colour = np.NaN*np.zeros(len(simul))                       
matching_colour[matching_CD004_ind] = 1
matching_colour[matching_CD01_ind] = 2
matching_colour[matching_CD015_ind] = 3

# Plot!
min_value=1
max_value=4 # nb of colours +1 -> to change when adding colour options
cmap = plt.cm.jet
bounds = [i for i in np.linspace(min_value,max_value,max_value)]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)                                                
fig = plt.figure()
plt.scatter(diff[matching_0deg_ind],total_obs[matching_0deg_ind],marker='x',s=100,c=matching_colour[matching_0deg_ind],label='0deg',norm=norm)
plt.scatter(diff[matching_randdeg_ind],total_obs[matching_randdeg_ind],marker='o',s=100,c=matching_colour[matching_randdeg_ind],label='randdeg',norm=norm)
plt.scatter(diff[matching_varydeg_ind],total_obs[matching_varydeg_ind],marker='*',s=100,c=matching_colour[matching_varydeg_ind],label='varydeg',norm=norm)
plt.scatter(diff[matching_justwinds_ind],total_obs[matching_justwinds_ind],marker='+',s=1000,c=matching_colour[matching_justwinds_ind],label='justwinds',norm=norm)

# Plot colorbar 
cbar = plt.colorbar(boundaries=bounds)
cbar.set_ticks(np.array(bounds)[0:-1]+0.5)
cbar_labels = ['CD004','CD1','CD015']   
cbar.ax.set_yticklabels(cbar_labels, fontsize=10)
# Plot legend 
plt.xlabel('Difference with obs/month')
plt.ylabel("% of BB beached")
plt.legend(markerscale=0.7, scatterpoints=1, fontsize=10)
plt.show()
fig.savefig(file[0:-4] + '.pdf', bbox_inches='tight') 


