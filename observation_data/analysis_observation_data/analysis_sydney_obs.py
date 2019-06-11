 # -*- coding: utf-8 -*-
"""
Created on Thu June 06 12:10:20 2019

@author : Natacha 
"""
import scipy.stats as stats
import datetime
#import colorlover as cl
#import plotly.graph_objs as go
import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import glob 
from windrose import plot_windrose

class time:
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year
        
    def jan_to_01(self):
        list_name=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        list_numbers=['1','2','3','4','5','6','7','8','9','10','11','12']
        for i in range(len(list_name)):
            if self.month==list_name[i]:
                self.month=list_numbers[i]
               

def GetDateSomeLikelyNone(beach_nb,bluebottle_nb):
    date_number = []
    for j in range(len(date[beach_nb])):
        if bluebottles[beach_nb][j]==bluebottle_nb:
            date_number.append(date[beach_nb][j])
    return date_number
                
def DayEqual(object1, object2):
    if object1.day==object2.day and object1.month==object2.month and object1.year==object2.year:
        return True
    else:
        return False
    
    

def BoxPlot(nb,date_plot,BOMdaily):   
    """
    Box plot pour la plage numero nb de wind direction pour les 3 cas : none likely observed
    """
    location=['Clovelly','Coogee','Maroubra']
    wind_direction_box0=[]
    wind_direction_box1=[]
    wind_direction_box2=[]
    
    for i in range(len(date_box[nb][0])):
        for j in range(len(date_plot)):
            if date_box[nb][0][i]==date_plot[j]:
                if np.isnan(BOMdaily[j-1])==False:
                    wind_direction_box0.append(BOMdaily[j-1])
    
    for i in range(len(date_box[nb][1])):
        for j in range(len(date_plot)):
            if date_box[nb][1][i]==date_plot[j]:
                if np.isnan(BOMdaily[j-1])==False:
                    wind_direction_box1.append(BOMdaily[j-1])
                
    for i in range(len(date_box[nb][2])):
        for j in range(len(date_plot)):
            if date_box[nb][2][i]==date_plot[j]:
                if np.isnan(BOMdaily[j-1])==False:
                    wind_direction_box2.append(BOMdaily[j-1])
                    
   
    x=[wind_direction_box0, wind_direction_box1, wind_direction_box2]
    fig = plt.figure(figsize=(12,9))
    plt.title(location[nb])
    plt.ylabel('Wind direction (degrees)')
    plt.boxplot(x,whis=[5,95])
    plt.xticks([1,2,3],['None','Likely','Some'])
    plt.show()
    
def GetVariables(filename):
    """
    Return date, water temp, #of bluebottles of a file
    """
    date, datee, water_temp, bluebottles, description = [], [], [], [], []
    for i in range(0,len(filename)):
        day=''
        month=''
        year=''
        date.append(str(filename.Name[i][:-12]))
        for j in range(0,2):
            if(date[i][j]!='/'):
                day+=date[i][j]
        for j in range(2,len(date[i])-4):
            if(date[i][j]!='/'):
                month+=date[i][j]
        for j in range(len(date[i])-4,len(date[i])):
            if(date[i][j]!='/'):
                year+=date[i][j] 
        
        if filename.Water_temp[i]!=14: #dont take values for water_temp=14C
            datee.append(time(str(day),str(month),str(year)))
            water_temp.append(filename.Water_temp[i])
            description.append(filename.Description[i])
            if filename.Bluebottles[i]=='none':
                bluebottles.append(0.)
            elif filename.Bluebottles[i]=='some' or filename.Bluebottles[i]=='many':
                bluebottles.append(1.)
            elif filename.Bluebottles[i]=='likely':
                bluebottles.append(0.5)

    middle_date = []
    final_date, final_water_temp, final_bluebottles, final_description = [], [], [], []
    for l in range(len(datee)):
        middle_date.append(datetime.date(int(datee[l].year), int(datee[l].month), int(datee[l].day)))
    
    final_date.append(middle_date[0])
    final_water_temp.append(water_temp[0])
    final_bluebottles.append(bluebottles[0])
    final_description.append(description[0])
    
    for l in range(1,len(middle_date)):  
        if middle_date[l]!=middle_date[l-1]: #to only have one value per day
            final_date.append(middle_date[l])
            final_water_temp.append(water_temp[l])
            final_bluebottles.append(bluebottles[l])
            final_description.append(description[l])
            
    
    return final_date, final_water_temp, final_bluebottles, final_description

files_name = glob.glob('../raw_observation_data/bluebottle_lifeguard_reports/*2.xlsx') #0Clovelly 1Coogee 2Maroubra

beach=[]
date=[0,1,2]
water_temp=[0,1,2]
bluebottles=[0,1,2]
description=[0,1,2]
date_box=[0,1,2]

for i in range(0,len(files_name)):
    beach.append(pd.read_excel(files_name[i]))

for i in range(0,len(water_temp)):
    date[i], water_temp[i], bluebottles[i], description[i] = GetVariables(beach[i])
    date_box[i]=[GetDateSomeLikelyNone(i,0.),GetDateSomeLikelyNone(i,0.5),GetDateSomeLikelyNone(i,1.)]

def nonans(array):
    '''
    author : Dr. Schaeffer
    Return input array [1D numpy array] with
    all nan values removed
    '''
    return array[~np.isnan(array)]


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def GetData(file):
    day=np.zeros(len(file))
    month=np.zeros(len(file))
    year=np.zeros(len(file))
    hours=np.zeros(len(file))
    minutes=np.zeros(len(file))
    time=np.zeros(len(file))
    direction=np.zeros(len(file))
    speed=np.zeros(len(file))
    u=np.zeros(len(file))
    v=np.zeros(len(file))
    date=[]
    
    def GetU(speed,direction):
        wind_dir_deg=(90-direction+180)
        wind_u = - speed * np.sin(np.pi / 180 * wind_dir_deg) 
        return wind_u
    
    def GetV(speed,direction):
        wind_dir_deg=(90-direction+180)
        wind_v = - speed * np.cos(np.pi / 180 * wind_dir_deg)
        return wind_v
    
    for i in range(len(file)):
        minutes[i]=file.MI_local_time[i]
        hours[i]=file.HH24[i]
        day[i]=file.DD[i]
        month[i]=file.MM[i]
        year[i]=file.YYYY[i]
        speed[i]=file.Wind_speed_ms[i]
        direction[i]=file.Wind_direction_degrees[i]
        #u[i],v[i]=pol2cart(speed[i],direction[i])
        u[i]=GetU(speed[i],direction[i])
        v[i]=GetV(speed[i],direction[i])

    for i in range(len(file)):
        date.append(datetime.date(int(year[i]),int(month[i]),int(day[i])))
        time[i] = date[i].toordinal() + hours[i]/24 + minutes[i]/(24*60)

    return date, time, speed, direction, u, v


def RosePlot(beachnb,bluebnb):
    """
    returns a rose plot of the wind for the past day for the beach beachnb
    and for the bluebottle scenario bluenb (0:none, 1:likely, 2:some) 
    """
    location=['Clovelly','Coogee','Maroubra']
    blueb=['none','likely','some']
    wind_speed=[]
    wind_direction=[]
    one_day = datetime.timedelta(days=1)
    for i in range(len(date_obs)):
        for j in range(len(date_box[beachnb][bluebnb])):
                if (date_obs[i]+one_day)==date_box[beachnb][bluebnb][j]:
                    wind_speed.append(speed_obs[i])
                    wind_direction.append(direction_obs[i])

    df = pd.DataFrame({"speed": wind_speed, "direction": wind_direction})
    bins = np.arange(0.01, 24, 4)
    kind = "bar"
    plot_windrose(df, kind=kind, normed=True, opening=0.8, edgecolor="white",bins=bins)
  #  plt.set_title(str(location[beachnb])+" "+str(blueb[bluebnb]))
    plt.savefig("../outputs_observation_data/rose"+str(location[beachnb])+"_"+str(blueb[bluebnb])+"_pastday.png")


file_name = '../raw_observation_data/wind_kurnell_sydney_observatory/wind_66037_local_time.csv'
filename=pd.read_csv(file_name)
df = filename.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! grrrr
date_obs, time_obs, speed_obs, direction_obs, u_obs, v_obs=GetData(df)
date_obs=date_obs[450500:]
time_obs=time_obs[450500:]
speed_obs=speed_obs[450500:]
direction_obs=direction_obs[450500:]
u_obs=u_obs[450500:]
v_obs=v_obs[450500:]

t=time_obs.astype('int')

wind_direction_daily = np.zeros((len(t)))+np.nan
wind_speed_daily = np.zeros((len(t)))+np.nan
LENN = np.zeros((len(t)))

for i in range(len(t)):
    tt0=np.where(time_obs==t[i])[0]
    LENN[i]=sum(np.isfinite(direction_obs[tt0.astype(int)]))
    if LENN[i]>0:
        wind_direction_daily[i]=np.mean(nonans(direction_obs[tt0.astype(int)]))
        wind_speed_daily[i]=np.mean(nonans(speed_obs[tt0.astype(int)]))

#BoxPlot(1,date_obs,wind_direction_daily)
RosePlot(0,0)
RosePlot(0,1)
RosePlot(0,2)

RosePlot(1,0)
RosePlot(1,1)
RosePlot(1,2)

RosePlot(2,0)
RosePlot(2,1)
RosePlot(2,2)
"""
timeseries plot
import cmocean
fig=plt.figure()
ax=plt.axes()
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator(range(0, 12), interval=2) # every 2month
years_fmt = mdates.DateFormatter('%Y')
month_fmt = mdates.DateFormatter('%m')
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)
ax.xaxis.set_minor_formatter(month_fmt)
fig.autofmt_xdate()
try_date=[]
try_speed=[]
try_direction=[]
ax.set_title('Dots for BB at Coogee')
ax.set_ylabel('Wind speed in m/s')
for i in range(len(date_obs)-50000,len(date_obs)-1):
    for j in range(len(date[2])):
        if date_obs[i+1]==date[2][j]:
            if bluebottles[2][j]==1:
                ax.scatter(date_obs[i],20,marker='+',s=10,c='skyblue')
                try_date.append(date_obs[i])
                try_speed.append(speed_obs[i])
                try_direction.append(direction_obs[i])
                
sc=ax.scatter(try_date[:],try_speed[:],c=try_direction[:],marker='+',cmap=cmocean.cm.phase)
cbar=plt.colorbar(sc)
cbar.set_label('Wind direction in degree')
fig.savefig("../outputs_observation_data/sydney_obs/timeseries_bb_only_maroubra_past.png",dpi=300)
"""
