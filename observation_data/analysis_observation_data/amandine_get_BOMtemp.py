# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:54:25 2016

@author: amandine
"""
#%reset -f


import pandas as pd
from matplotlib import pyplot as plt
import glob
from datetime import date
import numpy as np
import matplotlib.dates as mdates

YEARS = np.arange(1991,2019)    # TO CHANGE!!!
MHWPeriod = [1991,2019]


############################ AIR temp
list_mhws = []
list_clim = []
list_sst = []
list_sst_time = []
IDs = []

list_FILES = glob.glob('/home/nfs/z3340777/hdrive/My_documents/AUSTRALIE/MHW/CODE_tide_gauges/file_airTEMP*.csv')
N_FILES = len(list_FILES)

for f in range(N_FILES):
    
    FILE = list_FILES[f]
    print(FILE)
    ID = FILE[82:90]
        
    df = pd.read_csv(FILE)    
    df1 = df.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! grrrr
    print(df1.head())
    
    #ADCP_date = df1['DATE'][:]
    BOMairTEMP_time = df1['TIME'][:].values.astype('int')  - 366    ### CAREFUL! To account for the difference between Matlab and Python date numbers...
    # date.fromordinal(ADCP_time[0])
    BOMairTEMP = df1['AirTemperature_daily'][:].values
        
    
    t = BOMairTEMP_time
    sst  = BOMairTEMP  
        

    # Trend on raw data
    from scipy import stats
    nogaps = (~np.isnan(t) & ~np.isnan(sst))
    slope, intercept, r_value, p_value, std_err = stats.linregress(t[nogaps],sst[nogaps]) 
    print(slope*365*10*10)
    yhat = intercept + slope*t
    
    plt.figure()
    ax=plt.subplot()
    plt.plot_date(t,sst, fmt='b-', tz=None, xdate=True,ydate=False)
    plt.xlabel('Time')
    plt.ylabel('Daily air temperature')
    plt.title(ID, size=20)
    plt.plot_date(t,yhat, fmt='r-', tz=None, xdate=True,ydate=False)
    plt.text(0.1, 0.05,'slope: ' + "{:10.2f}".format(slope*365*100) + '[$^\circ$C/ century]', ha='left', va='center', transform=ax.transAxes)
    plt.text(0.7, 0.05,'p value: ' + "{:10.4f}".format(p_value), ha='left', va='center', transform=ax.transAxes)
    plt.savefig('PLOTS_BOM/Plot_temp_trend_' + str(ID) + '_ahws.png')
    plt.show()    


    ## Run MHW
#    import marineHeatWaves_AS as mhw    
    ClimatologyPeriod=[1992,2016] #[1992,2016]
    import marineHeatWaves_AS_v2 as mhw_v2
    mhws, clim = mhw_v2.detect(t, sst, climatologyPeriod=ClimatologyPeriod,MHWPeriod=MHWPeriod,smoothPercentileWidth=31,minDuration=3)
    # Make sure there is no interpolation or replacement with climatology...
    sst[clim['missing']]=np.nan
    
    mhws['n_events']
    mhwname = 'MHWS'
    
    # Make sure there is no interpolation or replacement with climatology...
    sst[clim['missing']]=np.nan

    ## write in file
    list_mhws.append(mhws)
    list_clim.append(clim)    
    list_sst.append(sst)
    list_sst_time.append(t)
    IDs.append(ID)

    ############################
    ###### Save the data in file
    import shelve
    d = shelve.open("SSAVE_BOMair_" + ID)  # open -- file may get suffix added by low-level
    d['BOMairTEMP_time'] = BOMairTEMP_time              # store data at key (overwrites old data if    
    d['BOMairTEMP'] = sst              # store data at key (overwrites old data if    
    d['BOMair_mhws'] = mhws              # store data at key (overwrites old data if    
    d['BOMair_clim'] = clim              # store data at key (overwrites old data if    
    d['BOMair_t'] = t              # store data at key (overwrites old data if    
    
    #data = d['list']              # retrieve a COPY of data at key (raise KeyError
    d.close()                  # close it


## Save the data in file
import shelve
d = shelve.open('SSAVE_BOM_mhws_AIR')  # open -- file may get suffix added by low-level                          # library
d['list_mhws'] = list_mhws              # store data at key (overwrites old data if    
d['list_clim'] = list_clim              # store data at key (overwrites old data if    
d['list_sst'] = list_sst              # store data at key (overwrites old data if    
d['list_sst_time'] = list_sst_time              # store data at key (overwrites old data if    
d['t'] = t              # store data at key (overwrites old data if    
d['IDs'] = ID              # store data at key (overwrites old data if    

#data = d['list']              # retrieve a COPY of data at key (raise KeyError
d.close()                  # close it


####################################
###### Plot events for all depths
ts = date(1992,1,1).toordinal()
te = date(2019,1,1).toordinal()
L = len(list_mhws)
plt.figure(figsize=(25,5))
#N_events = np.zeros(L)
for d in range(L):
    ax=plt.subplot(L,1,d+1)
#    plt.bar(list_mhws[d]['date_peak'], 1+np.zeros(len(list_mhws[d]['date_peak'])), width=10, color=(0.7,0.7,0.7))
    plt.bar(list_mhws[d]['date_start'], 1+np.zeros(len(list_mhws[d]['date_start'])), width=list_mhws[d]['duration'], facecolor='steelblue', edgecolor='steelblue')
    plt.xlim(ts, te)
    plt.ylabel('BOM ' + str(IDs[d]))
    ax.set_yticklabels([])
    years = mdates.YearLocator()   # every year
    if d+1 < L:
        ax.set_xticklabels([])
    if d+1 == L:
        plt.xlabel('Dates')        
        ax.xaxis.set_major_locator(years)
    if d == 0:
        plt.title('BOM, AHWs events', size=15)
plt.savefig('PLOTS_BOM/Plot_events_ALLdepths_ahw.png')
plt.show()
    
    
####################################
####################################
    
####################################
sst_nan = []

###### Compute Nb events per years / + Nb missing data per year
#L = len(list_mhws)
t_MHW_time = grid = [[0] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: nan except where days MHS, then value day
t_MHW_time_year = grid = [[0] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: nan except where days MHS, then value day
t_MHW_time_bool = grid = [[0] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: True and False for each day for MHW
t_year = np.zeros(len(t))   # array of years
sst_nan =  [[np.nan] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: nan except where days MHS, then value day
sst_nan_year =  [[0] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: nan except where days MHS, then value day
YEARS_MHWS_nb = np.zeros(shape=(L,len(YEARS)))  # array of nb events per year
YEARS_NaN_nb = np.zeros(shape=(L,len(YEARS)))   # array of nb missing data per year

for i in range(len(t)):    # loop time
    t_year[i] =  date.fromordinal(t[i]).year     
for d in range(L):
    sst_nan_year[d] = sst_nan[d]*t_year

for d in range(L):
    t0 = t*0*np.NaN
    t00 = t*0*np.NaN
    for i in range(list_mhws[d]['n_events']):    # loop events
        aaa = ((t >= list_mhws[d]['time_start'][i]) & (t <= list_mhws[d]['time_end'][i]+1))
        t0[aaa] = t[aaa]
        t00[aaa] = t_year[aaa]        
    t_MHW_time[d] = t0
    t_MHW_time_bool[d] = ~np.isnan(t_MHW_time[d])
    t_MHW_time_year[d] = t00
    sst_nan_year[d] = sst_nan[d]*t_year                         
    for y in range(len(YEARS)):            
        YEARS_MHWS_nb[d,y] = (np.array(t_MHW_time_year[d]) == YEARS[y]).sum()
        YEARS_NaN_nb[d,y] = (np.array(sst_nan_year[d]) == YEARS[y]).sum()


# Percentage MHW days over the non gap days
YEARS_MHWS_nb_percNoNan = YEARS_MHWS_nb /(367 - YEARS_NaN_nb) *100
plt.pcolor(YEARS,range(3),YEARS_MHWS_nb_percNoNan)
plt.colorbar()
plt.xticks(YEARS,rotation='vertical')
plt.show()
#
trend = np.array(range(L),dtype = 'float')
trend_p_value = np.array(range(L),dtype = 'float')
trend_r_value = np.array(range(L),dtype = 'float')
for d in range(L):
#    depth = DEPTHS[d]
    y =YEARS_MHWS_nb_percNoNan[d,:]
    from scipy import stats
    nogaps = (~np.isnan(YEARS) & ~np.isnan(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(YEARS[nogaps],y[nogaps]) 
    yhat = intercept + slope*YEARS    
    trend[d] = slope
    trend_p_value[d] = p_value
    trend_r_value[d] = r_value
    
    plt.figure(figsize=(15,5))
    ax = plt.subplot(1,1,1)
    plt.plot(YEARS,y,'bo-')
    plt.plot(YEARS,yhat,'r-',linewidth=2)
    plt.title('BOM: % AHW days / year, ' + str(IDs[d]), size=20)
    plt.ylabel('[%]')
    plt.text(0.1, 0.9,'slope: ' + "{:10.2f}".format(slope) + '[%/ year]', ha='left', va='center', transform=ax.transAxes)
    plt.text(0.5, 0.9,'p value: ' + "{:10.5f}".format(p_value), ha='left', va='center', transform=ax.transAxes)
    plt.savefig('PLOTS_BOM/Plot_AHWs_trends' + str(IDs[d]) + '.png')
    plt.show()
###### Plot
plt.figure(figsize=(5,8))
ax = plt.subplot(1,1,1)
plt.plot(trend,range(3),'o-',color = 'steelblue')    # Trend by century
plt.xlabel('Trend [% / year]')
plt.ylabel('Depth [m]')
plt.gca().invert_yaxis()
plt.title('BOM: trends AHW days / year, ' + str(IDs[d]), size=20)
for d in range(L):
    plt.text(-0.1, d,'p value: ' + "{:5.2f}".format(trend_p_value[d]))
plt.savefig('PLOTS_BOM/Plot_AHWs_trends_all.png')
plt.show()


###### Plot events for all depths YEARS
plt.figure(figsize=(8,13))
#N_events = np.zeros(L)
for d in range(L):
    ax=plt.subplot(L,1,d+1)
    plt.plot(YEARS,YEARS_MHWS_nb[d,:],'x-', lw=2, color='steelblue')
    ax.fill_between(YEARS,YEARS_MHWS_nb[d,:], YEARS_MHWS_nb[d,:]+YEARS_NaN_nb[d,:], facecolor='steelblue', alpha=0.1)
    plt.xlim(1991, 2019)
    plt.ylim(0,150)    
    plt.ylabel('z=' + str((IDs[d])) + 'm')
    plt.yticks([50,100,150], size=8)
    ax.yaxis.tick_right()
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.1)
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.1)
    plt.xticks(np.arange(1991,2019,1),rotation='vertical')
    if d+1 < L:
        ax.set_xticklabels([])
    if d+1 == L:
        plt.xlabel('Years') 
    if d == 0:
        plt.title('BOM, AHWs days / years' + str(IDs[d]), size=15)
plt.savefig('PLOTS_BOM/Plot_days_ALLstations_years_AHWs.png')
plt.show()
 

   
    
################################################################################################################################################    
############################ WATER temp    
################################################################################################################################################    
list_mhws = []
list_clim = []
list_sst = []
list_sst_time = []
IDs = []

list_FILES = glob.glob('/home/nfs/z3340777/hdrive/My_documents/AUSTRALIE/MHW/CODE_tide_gauges/file_waterTEMP*.csv')
N_FILES = len(list_FILES)

for f in range(N_FILES):
    
    FILE = list_FILES[f]
    print(FILE)
    ID = FILE[84:92]
        
    df = pd.read_csv(FILE)    
    df1 = df.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! grrrr
    print(df1.head())
    
    #ADCP_date = df1['DATE'][:]
    BOMwaterTEMP_time = df1['TIME'][:].values.astype('int')  - 366    ### CAREFUL! To account for the difference between Matlab and Python date numbers...
    # date.fromordinal(ADCP_time[0])
    BOMwaterTEMP = df1['WaterTemperature_daily'][:].values
    
    t = BOMwaterTEMP_time
    sst  = BOMwaterTEMP  

    # Trend on raw data
    from scipy import stats
    nogaps = (~np.isnan(t) & ~np.isnan(sst))
    slope, intercept, r_value, p_value, std_err = stats.linregress(t[nogaps],sst[nogaps]) 
    print(slope*365*10*10)
    yhat = intercept + slope*t
    
    plt.figure()
    ax=plt.subplot()
    plt.plot_date(t,sst, fmt='b-', tz=None, xdate=True,ydate=False)
    plt.xlabel('Time')
    plt.ylabel('Daily water temperature')
    plt.title(ID, size=20)
    plt.plot_date(t,yhat, fmt='r-', tz=None, xdate=True,ydate=False)
    plt.text(0.1, 0.05,'slope: ' + "{:10.2f}".format(slope*365*100) + '[$^\circ$C/ century]', ha='left', va='center', transform=ax.transAxes)
    plt.text(0.7, 0.05,'p value: ' + "{:10.4f}".format(p_value), ha='left', va='center', transform=ax.transAxes)
    plt.savefig('PLOTS_BOM/Plot_temp_trend_' + str(ID) + '_mhws.png')
    plt.show()    
        
    ## Run MHW
    import marineHeatWaves_AS as mhw    
    mhws, clim = mhw.detect(t, sst, climatologyPeriod=ClimatologyPeriod,MHWPeriod=MHWPeriod,smoothPercentileWidth=31)
    # Make sure there is no interpolation or replacement with climatology...
    sst[clim['missing']]=np.nan
    
    mhws['n_events']
    mhwname = 'MHWS'

    ## write in file
    list_mhws.append(mhws)
    list_clim.append(clim)    
    list_sst.append(sst)
    list_sst_time.append(t)
    IDs.append(ID)
    
    ############################
    ###### Save the data in file
    import shelve
    d = shelve.open("SSAVE_BOMwater_" + ID)  # open -- file may get suffix added by low-level
    d['BOMwaterTEMP_time'] = BOMwaterTEMP_time              # store data at key (overwrites old data if    
    d['BOMwaterTEMP'] = sst              # store data at key (overwrites old data if    
    d['BOMwater_mhws'] = mhws              # store data at key (overwrites old data if    
    d['BOMwater_clim'] = clim              # store data at key (overwrites old data if    
    d['BOMwater_t'] = t              # store data at key (overwrites old data if    
    #data = d['list']              # retrieve a COPY of data at key (raise KeyError
    d.close()                  # close it


## Save the data in file
import shelve
d = shelve.open('SSAVE_BOM_mhws')  # open -- file may get suffix added by low-level                          # library
d['list_mhws'] = list_mhws              # store data at key (overwrites old data if    
d['list_clim'] = list_clim              # store data at key (overwrites old data if    
d['list_sst'] = list_sst              # store data at key (overwrites old data if    
d['list_sst_time'] = list_sst_time              # store data at key (overwrites old data if    
d['t'] = t              # store data at key (overwrites old data if    
d['IDs'] = IDs              # store data at key (overwrites old data if    

#data = d['list']              # retrieve a COPY of data at key (raise KeyError
d.close()                  # close it



####################################
####################################
####################################
###### Plot events for all depths
ts = date(1992,1,1).toordinal()
te = date(2019,1,1).toordinal()
L = len(list_mhws)
plt.figure(figsize=(25,5))
#N_events = np.zeros(L)
for d in range(L):
    ax=plt.subplot(L,1,d+1)
#    plt.bar(list_mhws[d]['date_peak'], 1+np.zeros(len(list_mhws[d]['date_peak'])), width=10, color=(0.7,0.7,0.7))
    plt.bar(list_mhws[d]['date_start'], 1+np.zeros(len(list_mhws[d]['date_start'])), width=list_mhws[d]['duration'], facecolor='steelblue', edgecolor='steelblue')
    plt.xlim(ts, te)
    plt.ylabel('BOM ' + str(IDs[d]))
    ax.set_yticklabels([])
    years = mdates.YearLocator()   # every year
    if d+1 < L:
        ax.set_xticklabels([])
    if d+1 == L:
        plt.xlabel('Dates')        
        ax.xaxis.set_major_locator(years)
    if d == 0:
        plt.title('BOM, MHWs events', size=15)
plt.savefig('PLOTS_BOM/Plot_events_ALLdepths_mhw.png')
plt.show()


####################################
YEARS = np.arange(1992,2019)    # TO CHANGE!!!
sst_nan = []

###### Compute Nb events per years / + Nb missing data per year
#L = len(list_mhws)
t_MHW_time = grid = [[0] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: nan except where days MHS, then value day
t_MHW_time_year = grid = [[0] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: nan except where days MHS, then value day
t_MHW_time_bool = grid = [[0] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: True and False for each day for MHW
t_year = np.zeros(len(t))   # array of years
sst_nan =  [[np.nan] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: nan except where days MHS, then value day
sst_nan_year =  [[0] * len(t) for _ in range(L)]  # list of 0, same size as list_sst: nan except where days MHS, then value day
YEARS_MHWS_nb = np.zeros(shape=(L,len(YEARS)))  # array of nb events per year
YEARS_NaN_nb = np.zeros(shape=(L,len(YEARS)))   # array of nb missing data per year

for i in range(len(t)):    # loop time
    t_year[i] =  date.fromordinal(t[i]).year     
for d in range(L):
    sst_nan_year[d] = sst_nan[d]*t_year

for d in range(L):
    t0 = t*0*np.NaN
    t00 = t*0*np.NaN
    for i in range(list_mhws[d]['n_events']):    # loop events
        aaa = ((t >= list_mhws[d]['time_start'][i]) & (t <= list_mhws[d]['time_end'][i]+1))
        t0[aaa] = t[aaa]
        t00[aaa] = t_year[aaa]        
    t_MHW_time[d] = t0
    t_MHW_time_bool[d] = ~np.isnan(t_MHW_time[d])
    t_MHW_time_year[d] = t00
    sst_nan_year[d] = sst_nan[d]*t_year                         
    for y in range(len(YEARS)):            
        YEARS_MHWS_nb[d,y] = (np.array(t_MHW_time_year[d]) == YEARS[y]).sum()
        YEARS_NaN_nb[d,y] = (np.array(sst_nan_year[d]) == YEARS[y]).sum()


# Percentage MHW days over the non gap days
YEARS_MHWS_nb_percNoNan = YEARS_MHWS_nb /(367 - YEARS_NaN_nb) *100
plt.pcolor(YEARS,range(3),YEARS_MHWS_nb_percNoNan)
plt.colorbar()
plt.xticks(YEARS,rotation='vertical')
plt.show()
#
trend = np.array(range(L),dtype = 'float')
trend_p_value = np.array(range(L),dtype = 'float')
trend_r_value = np.array(range(L),dtype = 'float')
from scipy import stats
for d in range(L):
#    depth = DEPTHS[d]
    y =YEARS_MHWS_nb_percNoNan[d,:]
    nogaps = (~np.isnan(YEARS) & ~np.isnan(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(YEARS[nogaps],y[nogaps]) 
    yhat = intercept + slope*YEARS    
    trend[d] = slope
    trend_p_value[d] = p_value
    trend_r_value[d] = r_value
    
    plt.figure(figsize=(15,5))
    ax = plt.subplot(1,1,1)
    plt.plot(YEARS,y,'bo-')
    plt.plot(YEARS,yhat,'r-',linewidth=2)
    plt.title('BOM: % MHW days / year, ' + str(IDs[d]), size=20)
    plt.ylabel('[%]')
    plt.text(0.1, 0.9,'slope: ' + "{:10.2f}".format(slope) + '[%/ year]', ha='left', va='center', transform=ax.transAxes)
    plt.text(0.5, 0.9,'p value: ' + "{:10.5f}".format(p_value), ha='left', va='center', transform=ax.transAxes)
    plt.savefig('PLOTS_BOM/Plot_MHWs_trends' + str(IDs[d]) + '.png')
    plt.show()
###### Plot
plt.figure(figsize=(5,8))
ax = plt.subplot(1,1,1)
plt.plot(trend,range(3),'o-',color = 'steelblue')    # Trend by century
plt.xlabel('Trend [% / year]')
plt.ylabel('Depth [m]')
plt.gca().invert_yaxis()
plt.title('BOM: trends MHW days / year, ' + str(IDs[d]), size=20)
for d in range(L):
    plt.text(-0.1, d,'p value: ' + "{:5.2f}".format(trend_p_value[d]))
plt.savefig('PLOTS_BOM/Plot_MHWs_trends_all.png')
plt.show()


###### Plot events for all depths YEARS
plt.figure(figsize=(8,13))
#N_events = np.zeros(L)
for d in range(L):
    ax=plt.subplot(L,1,d+1)
    plt.plot(YEARS,YEARS_MHWS_nb[d,:],'x-', lw=2, color='steelblue')
    ax.fill_between(YEARS,YEARS_MHWS_nb[d,:], YEARS_MHWS_nb[d,:]+YEARS_NaN_nb[d,:], facecolor='steelblue', alpha=0.1)
    plt.xlim(1991, 2019)
    plt.ylim(0,150)    
    plt.ylabel('z=' + str((IDs[d])) + 'm')
    plt.yticks([50,100,150], size=8)
    ax.yaxis.tick_right()
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.1)
    plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.1)
    plt.xticks(np.arange(1992,2019,1),rotation='vertical')
    if d+1 < L:
        ax.set_xticklabels([])
    if d+1 == L:
        plt.xlabel('Years') 
    if d == 0:
        plt.title('BOM, MHWs days / years' + str(IDs[d]), size=15)
plt.savefig('PLOTS_BOM/Plot_days_ALLstations_years_MHWs.png')
plt.show()
