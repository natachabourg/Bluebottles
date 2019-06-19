 # -*- coding: utf-8 -*-
"""
Created on Thu June 06 12:10:20 2019

@author : Natacha 
"""
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
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

def GetU(speed,direction):
    wind_dir_deg=(90-direction+180)
    wind_u = - speed * np.sin(np.pi / 180 * wind_dir_deg) 
    return wind_u
    
def GetV(speed,direction):
    wind_dir_deg=(90-direction+180)
    wind_v = - speed * np.cos(np.pi / 180 * wind_dir_deg)
    return wind_v

def GetData(file):
    day=np.zeros(len(file))
    month=np.zeros(len(file))
    year=np.zeros(len(file))
    hours=np.zeros(len(file))
    minutes=np.zeros(len(file))
    time=np.zeros(len(file))
    direction=np.zeros(len(file))
    speed=np.zeros(len(file))
    gust_speed=np.zeros(len(file))
    date=[]
    
    for i in range(len(file)):
        minutes[i]=file.MI_local_time[i]
        hours[i]=file.HH24[i]
        day[i]=file.DD[i]
        month[i]=file.MM[i]
        year[i]=file.YYYY[i]
        speed[i]=file.Wind_speed_ms[i]
        direction[i]=file.Wind_direction_degrees[i]
        gust_speed[i]=file.Windgust_speed_ms[i]

    for i in range(len(file)):
        date.append(datetime.date(int(year[i]),int(month[i]),int(day[i])))
        time[i] = date[i].toordinal() + hours[i]/24 + minutes[i]/(24*60)
        
    return date, time, speed, direction, gust_speed


def PolarPlot(nb,direction,speed):
    blueb=[]
    daily=[]
    markersize=[]
    marker=[]
    list_marker=["s","s","D","D","D","o","o","o","^","^","^","s"]

    fig=plt.figure(figsize=(12,9))
    location=['Clovelly','Coogee','Maroubra']
    for i in range(len(direction)): #start in 2017
        markersize.append(speed[i]*speed[i])
        marker.append(list_marker[int(date_obs[i].month-1)])

        for j in range(len(date[nb])):
            if (date_obs[i]+datetime.timedelta(days=1))==date[nb][j]:
                daily.append(direction[i]*np.pi/180)
                if bluebottles[nb][j]==0.:
                    blueb.append('hotpink')
                elif bluebottles[nb][j]==0.5:
                    blueb.append('palegreen')
                elif bluebottles[nb][j]==1.:
                    blueb.append('dodgerblue')
    ax = plt.subplot(111, projection='polar')
    theta = daily
    r=8.*np.random.rand(len(daily))+1
    colors = blueb
    markz=marker
    size=markersize
    for i in range(len(theta)):
        ax.scatter(theta[i], r[i], c=colors[i],  cmap='hsv', alpha=0.75,s=size[i],marker=markz[i])
    ax.set_rorigin(-2.5)
    ax.set_theta_zero_location('W', offset=10)
    plt.title("Daily averaged wind direction at "+str(location[nb]))
    legend_elements = [Line2D([0],[0],marker='s',label='Summer', color='w',markerfacecolor='dodgerblue', markersize=10),
                       Line2D([0],[0],marker='D',label='Autumn', color='w',markerfacecolor='dodgerblue', markersize=10),
                       Line2D([0],[0],marker='o',label='Winter', color='w',markerfacecolor='dodgerblue', markersize=10),
                       Line2D([0],[0],marker='^',label='Spring', color='w',markerfacecolor='dodgerblue', markersize=10)]
    
    legend_elements_two = [Patch(facecolor='hotpink', edgecolor='hotpink',label='None'),
                           Patch(facecolor='palegreen', edgecolor='palegreen',label='Likely'),
                           Patch(facecolor='dodgerblue', edgecolor='dodgerblue',label='Observed')]
    legend1=plt.legend(handles=legend_elements, loc='lower right')
    legend2=plt.legend(handles=legend_elements_two, loc='upper right')
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    plt.show()
    fig.savefig("../outputs_observation_data/sydney_obs/polar_plot_"+str(location[nb])+".png",dpi=300)

def RosePlot(beachnb,bluebnb,date_obs,direction_obs,speed_obs):
    """
    returns a rose plot of the wind for the past day for the beach beachnb
    and for the bluebottle scenario bluenb (0:none, 1:likely, 2:some) 
    """
    location=['Clovelly','Coogee','Maroubra']
    blueb=['none','likely','some']
    wind_speed=[]
    wind_direction=[]
    one_day = datetime.timedelta(days=2)
    for i in range(len(date_obs)):
        for j in range(len(date_box[beachnb][bluebnb])):
                if (date_obs[i]+one_day)==date_box[beachnb][bluebnb][j]:
                    wind_speed.append(speed_obs[i])
                    wind_direction.append(direction_obs[i])

    df = pd.DataFrame({"speed": wind_speed, "direction": wind_direction})
    bins = np.arange(0.01, 24, 4)
    bins_new=np.arange(11.25,371.25,22.5)
    kind = "bar"
    fig=plt.figure()
    plot_windrose(df, kind=kind, normed=True, opening=0.8, edgecolor="white",bins=bins)
    plt.title("Daily averaged wind direction at "+str(location[beachnb])+" "+str(blueb[bluebnb]))
 #   fig2=plt.figure()
  #  plt.hist(wind_direction,bins_new)
    
    fig.savefig("../outputs_observation_data/sydney_obs/rose_plots/2_days/rose_two_"+str(location[beachnb])+"_"+str(blueb[bluebnb])+".png",dpi=300)


def TimeSeriesPlot():
    color=np.zeros(len(date_obs))
    for j in range(len(date_box[2][2])):
        for i in range(len(date_obs)):  
            if date_obs[i]==date_box[2][2][j]:   
                color[i]=1
    fig=plt.figure()
    plt.subplot(511)
    plt.plot(date_obs,color)
    plt.ylabel('1 : Bluebottles')
    plt.subplot(512)
    plt.plot(date_obs,wind_direction_daily)
    plt.ylabel('daily averaged direction')
    plt.subplot(513)
    plt.plot(date_obs,wind_speed_daily)
    plt.ylabel('daily averaged speed')
    plt.subplot(514)
    plt.plot(date_obs,u_daily)
    plt.ylabel('daily averaged U')
    plt.subplot(515)
    plt.plot(date_obs,v_daily)
    plt.ylabel('daily averaged V')
    plt.show()
  #  fig.savefig("../outputs_observation_data/sydney_obs/timeseries_5.png",dpi=300)


file_name = '../raw_observation_data/wind_kurnell_sydney_observatory/wind_66037_local_time.csv'
filename=pd.read_csv(file_name)
df = filename.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! grrrr
date_obs_full, time_obs, speed_obs, direction_obs, gust_speed=GetData(df)
date_obs_full=date_obs_full[450500:] #take data from 2017
date_obs=list(dict.fromkeys(date_obs_full)) #remove repetition 
time_obs=time_obs[450500:]
speed_obs=speed_obs[450500:]
direction_obs=direction_obs[450500:]
gust_speed=gust_speed[450500:]

u_obs=GetU(speed_obs,direction_obs)
v_obs=GetV(speed_obs,direction_obs)



t=[]
for i in range(len(time_obs)):
    t.append(time_obs[i].astype('int')) #list of days in time format
t=list(dict.fromkeys(t[:])) #remove repetition

wind_direction_daily = np.zeros((len(t)))
wind_speed_daily = np.zeros((len(t)))
LENN = np.zeros((len(t)))
time_new=[]
for i in range (len(time_obs)):
    time_new.append(int(time_obs[i]))
for i in range(len(t)):
    tt0=np.where(time_new==t[i]) #find all items from the same day
    LENN[i]=sum(np.isfinite(direction_obs[tt0]))
    if LENN[i]>0:
        wind_direction_daily[i]=np.mean(nonans(direction_obs[tt0])) #daily mean of wind direction
        wind_speed_daily[i]=np.mean(nonans(speed_obs[tt0])) #daily mean of wind speed
        
u_daily=GetU(wind_speed_daily, wind_direction_daily)
v_daily=GetV(wind_speed_daily, wind_direction_daily)

#TimeSeriesPlot()
#PolarPlot(0, wind_direction_daily, wind_speed_daily)
#PolarPlot(1, wind_direction_daily, wind_speed_daily)
#PolarPlot(2, wind_direction_daily, wind_speed_daily)

#BoxPlot(1,date_obs,wind_direction_daily)


RosePlot(0,0,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(0,1,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(0,2,date_obs,wind_direction_daily,wind_speed_daily)

RosePlot(1,0,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(1,1,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(1,2,date_obs,wind_direction_daily,wind_speed_daily)

RosePlot(2,0,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(2,1,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(2,2,date_obs,wind_direction_daily,wind_speed_daily)

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
