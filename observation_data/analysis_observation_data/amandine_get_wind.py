# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:36:50 2016

@author: amandine
"""
#%reset -f
#%matplotlib qt 
#%matplotlib inline 

import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt

###### Get wind data
#### BOM wind from Kurnell
#FILE = '/home/nfs/z3340777/hdrive/My_documents/AUSTRALIE/METEO/WIND/BOM_1990_jan2017/HM01X_Data_066043_45574879379602.txt'
#### BOM wind from Sydney airport: 1929 - Jan 2017 .... careful local time!!!
#FILE = '/home/nfs/z3340777/hdrive/My_documents/AUSTRALIE/METEO/WIND/BOM_1990_jan2017/HM01X_Data_066037_45574879379602.txt'
FILE = '/home/nfs/z3340777/hdrive/My_documents/AUSTRALIE/METEO/WIND/BOM_1990_2018_30min/HM01X_Data_066037_999999999503749.txt'
# df = pd.read_csv("snow.txt", sep='\s*', names=["year", "month", "day", "snow_depth"])
df = pd.read_csv(FILE)    
df1 = df.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! grrrr
print(df1.head())

# Time
Wind_year = df1['Year Month Day Hour Minutes in YYYY'][:]
Wind_month = df1['MM'][:]
Wind_day = df1['DD'][:]
Wind_hour = df1['HH24'][:]   # Local time

Wind_time = np.zeros(len(Wind_year))  
Wind_time_UTC = np.zeros(len(Wind_year))  
for l in range(len(Wind_year)):
    aa = datetime.date(Wind_year[l],Wind_month[l],Wind_day[l])
    Wind_time[l] = aa.toordinal() + Wind_hour[l]/24     # Local time
Wind_time_UTC = Wind_time - 10/24     # Local time
    
# Wind
Wind_speed_kmh = np.array(df1['Wind speed in km/h'].astype('float64') )
Wind_speed_kmh [Wind_speed_kmh > 80] = np.NaN
Wind_dir_deg = np.array(df1['Wind direction in degrees true'].astype('float64')) 
Wind_MSLP = np.array(df1['Mean sea level pressure in hPa'][:])  
Air_temp = np.array(df1['Air Temperature in degrees C'][:]) 
## Plot
plt.figure(1,figsize=(15,5))
plt.plot_date(Wind_time_UTC,Wind_speed_kmh, fmt='b-', tz=None, xdate=True,ydate=False)
plt.xlabel('Time')
plt.ylabel('Wind speed [km h$^{-1}$]')
plt.title('Sydney airport wind speed', size=20)
#plt.savefig('mhw_stats_' + Name_platform + '/Plot_sst' + str(depth) + '.png')
plt.figure(1,figsize=(15,5))
plt.plot_date(Wind_time_UTC,Air_temp, fmt='b-', tz=None, xdate=True,ydate=False)
plt.xlabel('Time')
plt.ylabel('Air temp]')
plt.title('Sydney airport air temp', size=20)
plt.show()


###### Conversions
Wind_speed_ms = Wind_speed_kmh * 1000 / 3600
#Wind_dir_deg2 = (90 - Wind_dir_deg + 180);
#Wind_dir_deg2 [Wind_dir_deg2 < 0] = Wind_dir_deg2[Wind_dir_deg2 < 0]+360;
#Wind_dir_rad = Wind_dir_deg2*np.pi/180;
#[uwd,vwd]=pol2cart(Wind_speed_ms, Wind_dir_rad);
Wind_u = - Wind_speed_ms * np.sin(np.pi / 180 * Wind_dir_deg)
Wind_v = - Wind_speed_ms * np.cos(np.pi / 180 * Wind_dir_deg)

## Plot
plt.figure(1,figsize=(15,5))
plt.plot_date(Wind_time_UTC,Wind_u, fmt='b-', tz=None, xdate=True,ydate=False)
#plt.plot_date(Wind_time_UTC,uwd, fmt='r-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_v, fmt='r-', tz=None, xdate=True,ydate=False)
#plt.plot_date(Wind_time_UTC,vwd, fmt='r-', tz=None, xdate=True,ydate=False)
plt.xlabel('Time')
plt.ylabel('Wind components [m s$^{-1}$]')
plt.title('Sydney airport wind speed', size=20)
plt.legend(['Wind_u','Wind_v'],loc=4)
#plt.savefig('mhw_stats_' + Name_platform + '/Plot_sst' + str(depth) + '.png')
plt.show()


###### Wind stress
[u, angle] = cart2pol(Wind_u,Wind_v);
rho=1.3;	# density of air
cd = (0.61 + 0.063*np.abs(u))*1e-3;
cd[np.abs(u) < 6] = 1.1e-3;
tau = cd * rho * np.abs(u) * u;
Wind_tau_u = tau*np.sin(angle);
Wind_tau_v = tau*np.cos(angle);
## Plot
plt.figure(figsize=(15,5))
plt.plot_date(Wind_time_UTC,Wind_tau_u, fmt='b-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_tau_v, fmt='r-', tz=None, xdate=True,ydate=False)
plt.xlabel('Time')
plt.ylabel('Wind speed [N m$^{-2}$]')
plt.title('Sydney airport wind stress', size=20)
plt.legend(['Tau_x','Tau_y'],loc=4)
#plt.savefig('mhw_stats_' + Name_platform + '/Plot_sst' + str(depth) + '.png')
plt.show()


###### Rotation along-across shelf
rot_deg_angle = - 25
Wind_u_rot = np.cos(rot_deg_angle * np.pi / 180) * Wind_u + np.sin(rot_deg_angle * np.pi / 180) * Wind_v;  #  across-shelf 
Wind_v_rot = - np.sin(rot_deg_angle * np.pi / 180) * Wind_u + np.cos(rot_deg_angle * np.pi / 180) * Wind_v;  #  along -shelf 
Wind_tau_u_rot = np.cos(rot_deg_angle * np.pi / 180) * Wind_tau_u + np.sin(rot_deg_angle * np.pi / 180) * Wind_tau_v;  #  across-shelf 
Wind_tau_v_rot = - np.sin(rot_deg_angle * np.pi / 180) * Wind_tau_u + np.cos(rot_deg_angle * np.pi / 180) * Wind_tau_v;  #  along -shelf 
## Plot
plt.figure(1,figsize=(15,5))
plt.plot_date(Wind_time_UTC,Wind_u_rot, fmt='b-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_v_rot, fmt='r-', tz=None, xdate=True,ydate=False)
plt.xlabel('Time')
plt.ylabel('Wind speed [m s$^{-1}$]')
plt.title('Sydney airport wind', size=20)
plt.legend(['Wind_u_rot','Wind_v_rot'],loc=4)
#plt.savefig('mhw_stats_' + Name_platform + '/Plot_sst' + str(depth) + '.png')
plt.show()
plt.figure(figsize=(15,5))
plt.plot_date(Wind_time_UTC,Wind_tau_u_rot, fmt='b-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_tau_v_rot, fmt='r-', tz=None, xdate=True,ydate=False)
plt.xlabel('Time')
plt.ylabel('Wind speed [N m$^{-2}$]')
plt.title('BOM wind stress', size=20)
plt.legend(['Wind_tau_u_rot','Wind_tau_v_rot'],loc=4)
#plt.savefig('mhw_stats_' + Name_platform + '/Plot_sst' + str(depth) + '.png')
plt.show()


###### Filter
#Wind_tau_u_rot_fill = Wind_tau_u_rot.fillna(Wind_tau_u_rot.mean())  # fillnans #AMANDINE USED TO BE 9/08/18 
#Wind_tau_v_rot_fill = Wind_tau_v_rot.fillna(Wind_tau_v_rot.mean())  # fillnans #AMANDINE USED TO BE 9/08/18 
Wind_tau_u_rot_fill = pad(Wind_tau_u_rot, maxPadLength=False)  # fillnans
Wind_tau_v_rot_fill = pad(Wind_tau_v_rot, maxPadLength=False)  # fillnans
Wind_MSLP_fill = pad(Wind_MSLP, maxPadLength=False)  # fillnans
Air_temp_fill = pad(Air_temp, maxPadLength=False)  # fillnans
import scipy.signal as signal
Wn = 2/(24*2)     # for 24 hour low pass Wn is a fraction of the Nyquist frequency (half the sampling frequency).
b, a = signal.butter(2, Wn, 'lowpass')  # Butterworth filter
Wind_tau_u_rot_filt = signal.filtfilt(b, a, Wind_tau_u_rot_fill)
Wind_tau_v_rot_filt = signal.filtfilt(b, a, Wind_tau_v_rot_fill)
Wind_MSLP_filt = signal.filtfilt(b, a, Wind_MSLP_fill)
Air_temp_filt = signal.filtfilt(b, a, Air_temp_fill)
Wind_tau_u_rot_filt [np.isnan(Wind_tau_u_rot)] = np.nan #AMANDINE USED TO BE ~np.nan 9/08/18 
Wind_tau_v_rot_filt [np.isnan(Wind_tau_v_rot)] = np.nan #AMANDINE USED TO BE ~np.nan 9/08/18 
Wind_MSLP_filt [np.isnan(Wind_MSLP)] = np.nan #AMANDINE USED TO BE ~np.nan 9/08/18 
Air_temp_filt [np.isnan(Air_temp)] = np.nan #AMANDINE USED TO BE ~np.nan 9/08/18 
plt.plot_date(Wind_time_UTC,Wind_tau_v_rot, fmt='b-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_tau_v_rot_filt, fmt='r-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_tau_u_rot, fmt='b-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_tau_u_rot_filt, fmt='r-', tz=None, xdate=True,ydate=False)
#plt.xlim(datetime.date(2013,3,1).toordinal(), datetime.date(2013,3,15).toordinal())
plt.plot_date(Wind_time_UTC,Wind_MSLP, fmt='g-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_MSLP_fill, fmt='b-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Wind_MSLP_filt, fmt='r-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Air_temp, fmt='g-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Air_temp_fill, fmt='b-', tz=None, xdate=True,ydate=False)
plt.plot_date(Wind_time_UTC,Air_temp_filt, fmt='r-', tz=None, xdate=True,ydate=False)

############################
###### Save the data in file
import shelve
#d = shelve.open("SAVE_wind_Kurnell")  # open -- file may get suffix added by low-level
d = shelve.open("SSAVE_wind_Sydney_airport")  # open -- file may get suffix added by low-level
d['Wind_time_UTC'] = Wind_time_UTC              # store data at key (overwrites old data if    
d['Wind_speed_ms'] = Wind_speed_ms              # store data at key (overwrites old data if    
d['Wind_dir_deg'] = Wind_dir_deg              # store data at key (overwrites old data if    
d['Wind_u'] = Wind_u              # store data at key (overwrites old data if    
d['Wind_v'] = Wind_v              # store data at key (overwrites old data if    
d['Wind_tau_u'] = Wind_tau_u              # store data at key (overwrites old data if    
d['Wind_tau_v'] = Wind_tau_v              # store data at key (overwrites old data if    
d['Wind_u_rot'] = Wind_u_rot              # store data at key (overwrites old data if    
d['Wind_v_rot'] = Wind_v_rot              # store data at key (overwrites old data if    
d['Wind_tau_u_rot'] = Wind_tau_u_rot              # store data at key (overwrites old data if    
d['Wind_tau_v_rot'] = Wind_tau_v_rot              # store data at key (overwrites old data if    
d['Wind_tau_u_rot_filt'] = Wind_tau_u_rot_filt              # store data at key (overwrites old data if    
d['Wind_tau_v_rot_filt'] = Wind_tau_v_rot_filt              # store data at key (overwrites old data if    
d['Wind_MSLP'] = Wind_MSLP              # store data at key (overwrites old data if    
d['Wind_MSLP_filt'] = Wind_MSLP_filt              # store data at key (overwrites old data if    
d['Air_temp'] = Air_temp              # store data at key (overwrites old data if    
d['Air_temp_filt'] = Air_temp_filt              # store data at key (overwrites old data if    
#data = d['list']              # retrieve a COPY of data at key (raise KeyError
d.close()                  # close it




#################################
# Functions

def pad(data, maxPadLength=False):
    import scipy.ndimage as ndimage
    '''

    Linearly interpolate over missing data (NaNs) in a time series.

    Inputs:

      data	     Time series [1D numpy array]
      maxPadLength   Specifies the maximum length over which to interpolate,
                     i.e., any consecutive blocks of NaNs with length greater
                     than maxPadLength will be left as NaN. Set as an integer.
                     maxPadLength=False (default) interpolates over all NaNs.

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Jun 2015

    '''
    data_padded = data.copy()
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data_padded[bad_indexes] = interpolated
    if maxPadLength:
        blocks, n_blocks = ndimage.label(np.isnan(data))
        for bl in range(1, n_blocks+1):
            if (blocks==bl).sum() > maxPadLength:
                data_padded[blocks==bl] = np.nan

    return data_padded


def nonans(array):
    '''
    Return input array [1D numpy array] with
    all nan values removed
    '''
    return array[~np.isnan(array)]



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
    
    