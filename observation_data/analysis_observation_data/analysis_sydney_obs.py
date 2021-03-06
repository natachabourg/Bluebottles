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
import math
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
    one_day = datetime.timedelta(days=1)
    for i in range(len(date_box[nb][0])):
        for j in range(len(date_plot)):
            if date_box[nb][0][i]==(date_plot[j]+one_day):
                if np.isnan(BOMdaily[j])==False:
                    wind_direction_box0.append(BOMdaily[j])

    for i in range(len(date_box[nb][1])):
        for j in range(len(date_plot)):
            if date_box[nb][1][i]==(date_plot[j]+one_day):
                if np.isnan(BOMdaily[j])==False:
                    wind_direction_box1.append(BOMdaily[j])

    for i in range(len(date_box[nb][2])):
        for j in range(len(date_plot)):
            if date_box[nb][2][i]==(date_plot[j]+one_day):
                if np.isnan(BOMdaily[j])==False:
                    wind_direction_box2.append(BOMdaily[j])


    x=[wind_direction_box0, wind_direction_box1, wind_direction_box2]
    fig = plt.figure(figsize=(12,9))
    plt.title(location[nb]+" 1 day before")
    plt.ylabel('Wind direction (degrees)')
    plt.boxplot(x,whis=[5,95])
    plt.xticks([1,2,3],['None','Likely','Some'])
   # fig.savefig("../outputs_observation_data/sydney_obs/box_plots/oneday_"+str(location[nb])+".png",dpi=300)
    
def GetVariables(filename):
    """
    Return date, water temp, #of bluebottles of a file
    """
    date, datee, water_temp, bluebottles, description, wave_height = [], [], [], [], [], []
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
            wave_height.append(filename.Wave_height[i])
            if filename.Bluebottles[i]=='none':
                bluebottles.append(0.)
            elif filename.Bluebottles[i]=='some' or  filename.Bluebottles[i]=='many':
                bluebottles.append(1.)
    #        elif filename.Bluebottles[i]=='many':
    #            bluebottles.append(2.)
            elif filename.Bluebottles[i]=='likely':
                bluebottles.append(0.5)

    middle_date = []
    final_date, final_water_temp, final_bluebottles, final_description, final_wave_height = [], [], [], [], []
    for l in range(len(datee)):
        middle_date.append(datetime.date(int(datee[l].year), int(datee[l].month), int(datee[l].day)))
    
    final_date.append(middle_date[0])
    final_water_temp.append(water_temp[0])
    final_bluebottles.append(bluebottles[0])
    final_description.append(description[0])
    final_wave_height.append(wave_height[0])
    
    for l in range(1,len(middle_date)):  
        if middle_date[l]!=middle_date[l-1]: #to only have one value per day
            final_date.append(middle_date[l])
            final_water_temp.append(water_temp[l])
            final_bluebottles.append(bluebottles[l])
            final_description.append(description[l])
            final_wave_height.append(wave_height[l])
            
    
    return final_date, final_water_temp, final_bluebottles, final_description, final_wave_height

files_name = glob.glob('../raw_observation_data/bluebottle_lifeguard_reports/*2.xlsx') #0Clovelly 1Coogee 2Maroubra

beach=[]
date_bb=[0,1,2]
date=[0,1,2]
water_temp=[0,1,2]
bluebottles=[0,1,2]
description=[0,1,2]
wave_height=[0,1,2]
date_box=[0,1,2]

for i in range(0,len(files_name)):
    beach.append(pd.read_excel(files_name[i]))

for i in range(0,len(water_temp)):
    date_bb[i], water_temp[i], bluebottles[i], description[i], wave_height[i] = GetVariables(beach[i])
    
date[0]=date_bb[0]
date[1]=date_bb[1][:1036] #delete data before 05/2016
date[2]=date_bb[2][:1025] #delete data before 05/2016


water_temp[1]=water_temp[1][:1036]
water_temp[2]=water_temp[2][:1025] #delete data before 05/2016

bluebottles[1]=bluebottles[1][:1036]
bluebottles[2]=bluebottles[2][:1025] 

description[1]=description[1][:1036]
description[2]=description[2][:1025] 

wave_height[1]=wave_height[1][:1036]
wave_height[2]=wave_height[2][:1025] 




for i in range(0,len(water_temp)):    
    date_box[i]=[GetDateSomeLikelyNone(i,0.),GetDateSomeLikelyNone(i,0.5),GetDateSomeLikelyNone(i,1.)]


for beachnb in (0,1,2):
    for i in range(len(wave_height[beachnb])):
        if wave_height[beachnb][i][0]=='b':
            wave_height[beachnb][i]=0
        elif wave_height[beachnb][i][1]=='o':
            wave_height[beachnb][i]=0.5
        elif wave_height[beachnb][i][-1]=='l':
            wave_height[beachnb][i]=1
        elif wave_height[beachnb][i][4]=='p':
            wave_height[beachnb][i]=1.5
        elif wave_height[beachnb][i][0]=='t':
            wave_height[beachnb][i]=2
        elif wave_height[beachnb][i][1]=='l':
            wave_height[beachnb][i]=3



def nonans(array):
    '''
    author : Dr. Schaeffer
    Return input array [1D numpy array] with
    all nan values removed
    '''
    return array[~np.isnan(array)]


def pol2cart(rho, phi):
    """
    author : Dr. Schaeffer
    """
    
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
def cart2pol(x, y):
    """
    author : Dr. Schaeffer
    """
    
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def GetU(speed,direction):
    wind_dir_deg=(90-direction+180)
    wind_u = - speed * np.sin(np.pi / 180 * wind_dir_deg) 
    return wind_u
    
def GetV(speed,direction):
    wind_dir_deg=(90-direction+180)
    wind_v = - speed * np.cos(np.pi / 180 * wind_dir_deg)
    return wind_v


def GetData(file):

    date=[]
    time =[]
    
    minutes=file.MI_local_time
    hours=file.HH24
    day=file.DD
    month=file.MM
    year=file.YYYY
    speed=file.Wind_speed_ms
    direction=file.Wind_direction_degrees
    gust_speed=file.Windgust_speed_ms


    for i in range(len(file)):
        date.append(datetime.date(int(year[i]),int(month[i]),int(day[i])))
        time.append(date[i].toordinal() + hours[i]/24 + minutes[i]/(24*60))
        
    return np.asarray(date), np.asarray(time), np.asarray(speed), np.asarray(direction), np.asarray(gust_speed)


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
  #  fig.savefig("../outputs_observation_data/sydney_obs/daily_averaged/polar_plot_"+str(location[nb])+"_pastday.png",dpi=300)

def RosePlot(beachnb,bluebnb,date_obs,direction_obs,speed_obs):
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
  #  fig=plt.figure()
    plot_windrose(df, kind=kind, normed=True, opening=0.8, edgecolor="white",bins=bins)
    plt.title("Daily averaged wind direction 1 day before at "+str(location[beachnb])+" "+str(blueb[bluebnb]))
    plt.legend('wind speed (m/s)')
 #   fig2=plt.figure()
  #  plt.hist(wind_direction,bins_new)
    
    plt.savefig("../outputs_observation_data/sydney_obs/daily_averaged/rose"+str(location[beachnb])+"_"+str(blueb[bluebnb])+"_pastday.png",dpi=300)


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

file_name = '../raw_observation_data/wind_kurnell_sydney_observatory/wind_66043_local_time.csv'
filename=pd.read_csv(file_name)
df = filename.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! grrrr
date_obs_full, time_obs, speed_obs, direction_obs, gust_speed=GetData(df)
date_obs_full=date_obs_full[276838:] #take data from 2016
date_obs=list(dict.fromkeys(date_obs_full)) #remove repetition 
time_obs=time_obs[276838:]
speed_obs=speed_obs[276838:]
direction_obs=direction_obs[276838:]
gust_speed=gust_speed[276838:]#[450500:] = forsydney obs

u_obs=GetU(speed_obs,direction_obs)
v_obs=GetV(speed_obs,direction_obs)

def ToOceano(meteo_direction):
    oceano_direction=np.zeros(len(meteo_direction))
    for i in range(0,len(meteo_direction)):
        if meteo_direction[i]<=270:
            oceano_direction[i]=270-meteo_direction[i]
        else:
           oceano_direction[i]=360+270-meteo_direction[i]
    return oceano_direction

def ToMeteo(oceano_direction):
    meteo_direction=np.zeros(len(oceano_direction))
    for i in range(0,len(oceano_direction)):
        if oceano_direction[i]<=270:
            meteo_direction[i]=270-oceano_direction[i]
        else:
            meteo_direction[i]=360+270-oceano_direction[i]
    return meteo_direction

def ToNormal(from_u_direction):
    """
    from -180;+180 to 0;360
    """
    
    normal_direction=np.zeros(len(from_u_direction))
    for i in range(0,len(from_u_direction)):
        if from_u_direction[i]<0:
            normal_direction[i]=360+from_u_direction[i]
        else:
            normal_direction[i]=from_u_direction[i]
    return normal_direction

"""
day from midnight to 9
"""
time_obs=np.asarray(time_obs)-0.375

direction_obs_new=ToOceano(direction_obs)
u_all, v_all = pol2cart(speed_obs,direction_obs_new*np.pi/180) #seem correct
t=[]
for i in range(len(time_obs)):
    t.append(time_obs[i].astype('int')) #list of days in time format
t=list(dict.fromkeys(t[:]))#remove repetition
#t=t[:-1] #remove last day bc not in date_obs
u_daily = np.zeros((len(t)))
v_daily = np.zeros((len(t)))
LENN = np.zeros((len(t)))

time_new=[]
for i in range (len(time_obs)):
    time_new.append(int(time_obs[i]))

for i in range(len(t)):
    tt0 = np.where(time_new==t[i]) #find all items from the same day
    LENN[i] = sum(np.isfinite(u_all[tt0]))
    if LENN[i]>0:
        u_daily[i] = np.mean(nonans(u_all[tt0])) #daily mean of wind direction
        v_daily[i] = np.mean(nonans(v_all[tt0])) #daily mean of wind speed


wind_speed_daily, direction_daily_o=cart2pol(u_daily,v_daily)
direction_daily_o=direction_daily_o*180/np.pi #rad to deg
direction_daily_step=ToNormal(direction_daily_o)
wind_direction_daily=ToMeteo(direction_daily_step)
#TimeSeriesPlot()


def GetMonthIndex(monthnb, beachnb):
    """
    return the index of the month nbmonth(1,..,12) in a date dataset
    """
    index = [date[beachnb].index(d) for d in date[beachnb] if d.month==monthnb]
    
    blueb_month = np.asarray(bluebottles[beachnb])[index]
    
    none = [b for b in blueb_month if b==0]
    likely = [b for b in blueb_month if b==0.5] 
    observed = [b for b in blueb_month if b==1] 
    print(str(beachnb)+"  "+str(monthnb)+"None : "+str(len(none))+", Likely : "+str(len(likely))+", Observed : "+str(len(observed)))


""""
Histogram plots for each season


"""

date_obs_array=np.asarray(date_obs)
summer=[d for d in date_obs if d.month == 12 or d.month == 1 or d.month == 2]
autumn=[d for d in date_obs if d.month == 3 or d.month == 4 or d.month == 5]
winter=[d for d in date_obs if d.month == 6 or d.month == 7 or d.month == 8]
spring=[d for d in date_obs if d.month == 9 or d.month == 10 or d.month == 11]
location=['Clovelly','Coogee','Maroubra']
    
both_summer=set(date_obs_array).intersection(summer)
index_summer = [date_obs.index(x) for x in both_summer]
direction_daily_summer=wind_direction_daily[index_summer]
    
both_autumn=set(date_obs_array).intersection(autumn)
index_autumn = [date_obs.index(x) for x in both_autumn]
direction_daily_autumn=wind_direction_daily[index_autumn]
    
both_winter=set(date_obs_array).intersection(winter)
index_winter = [date_obs.index(x) for x in both_winter]
direction_daily_winter=wind_direction_daily[index_winter]
    
both_spring=set(date_obs_array).intersection(spring)
index_spring = [date_obs.index(x) for x in both_spring]
direction_daily_spring=wind_direction_daily[index_spring]

direction_season=[direction_daily_spring,direction_daily_summer,direction_daily_autumn,direction_daily_winter]
index_season=[index_spring,index_summer,index_autumn,index_winter]

def ColorHist(nb,seas):
    direction_daily=direction_season[seas]
    index=index_season[seas]
    
    date_obs_new=date_obs_array[index]
    
    NE=np.where(np.logical_and(direction_daily>11.25, direction_daily<=101.25))
    SE=np.where(np.logical_and(direction_daily>101.25, direction_daily<=191.25))
    SW=np.where(np.logical_and(direction_daily>191.25, direction_daily<=281.25))
    NW=np.where(np.logical_or(direction_daily>281.25, direction_daily<=11.25))    

    season=['spring', 'summer', 'autumn', 'winter']
    
    date=np.asarray(date_obs_new)
    date=[date[NE],date[SE],date[SW],date[NW]]
    observed_list, none_list = [], []
    for l in range(len(date)):
        observed=0
        none=0
        for i in range(len(date[l])):
            for j in range(len(date_box[nb][2])): 
                if date[l][i]==date_box[nb][2][j]:
                    observed+=1
    
        for i in range(len(date[l])):
            for j in range(len(date_box[nb][0])):
                if date[l][i]==date_box[nb][0][j]:
                    none+=1
        observed_list.append(observed/(observed+none))
        none_list.append(none/(observed+none))
    
    ind = np.arange(4)
    width=0.2
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(ind, ('NE','SE','SW','NW'))
    ax.bar(ind-width/2, none_list, width=width, color='lightgrey', align='center',label='None')
    ax.bar(ind+width/2, observed_list, width=width, color='dodgerblue', align='center',label='Observed')
    plt.legend()
    plt.title(location[nb]+' '+str(season[seas]))
    plt.show()
    fig.savefig('../outputs_observation_data/kurnell/histograms_observation/seasonal_histograms/direction_'+str(location[nb])+'_'+str(season[seas])+'.png',dpi=300)
 

def Sth(nb,seas):
    direction_daily=direction_season[seas]
    index=index_season[seas]
    
    date_obs_new=date_obs_array[index]
    
    NE=np.where(np.logical_and(direction_daily>11.25, direction_daily<=101.25))
    SE=np.where(np.logical_and(direction_daily>101.25, direction_daily<=191.25))
    SW=np.where(np.logical_and(direction_daily>191.25, direction_daily<=281.25))
    NW=np.where(np.logical_or(direction_daily>281.25, direction_daily<=11.25))    

    season=['spring', 'summer', 'autumn', 'winter']
    
    date=np.asarray(date_obs_new)
    date=[date[NE],date[SE],date[SW],date[NW]]
    NE_list, SE_list, SW_list, NW_list =[0,0], [0,0], [0,0], [0,0]
    liste=[NE_list, SE_list, SW_list, NW_list]
    sum_none=0.
    sum_observed=0.
    for l in range(len(date)):
        observed=0
        none=0
        for i in range(len(date[l])):
            for j in range(len(date_box[nb][2])): #nb
                if date[l][i]==date_box[nb][2][j]:
                    observed+=1
    
        for i in range(len(date[l])):
            for j in range(len(date_box[nb][0])):
                if date[l][i]==date_box[nb][0][j]:
                    none+=1
        liste[l][0]=none
        liste[l][1]=observed
        sum_none+=none
        sum_observed+=observed
    for l in range(len(date)):
        liste[l][0]=liste[l][0]/sum_none
        liste[l][1]=liste[l][1]/sum_observed
    
    
    xbar=np.arange(2)
    width=0.2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(xbar, ('None', 'Some'))
    ax.bar(xbar-3*width/2, liste[0], width=0.2, color='olivedrab', align='center',label='NE')
    ax.bar(xbar-width/2, liste[1], width=0.2, color='skyblue', align='center',label='SE')
    ax.bar(xbar+width/2, liste[2], width=0.2, color='plum', align='center',label='SW')
    ax.bar(xbar+3*width/2, liste[3], width=0.2, color='orange', align='center',label='NW')
    plt.legend()
    plt.title(location[nb]+' '+str(season[seas]))
    plt.show()
    fig.savefig('../outputs_observation_data/kurnell/histograms_observation/seasonal_histograms/situation_'+str(location[nb])+'_'+str(season[seas])+'.png',dpi=300)


def WaveHeightPlot():
    we=[0,1,2]
    we_0=[0,1,2]
    for beachnb in (0,1,2):
        we[beachnb]=[]
        we_0[beachnb]=[]
        for i in range(len(date[beachnb])):
            for j in range(len(date_box[beachnb][2])):
                    if (date[beachnb][i])==date_box[beachnb][2][j]:
                        we[beachnb].append(wave_height[beachnb][i])
                    if (date[beachnb][i])==date_box[beachnb][0][j]:
                        we_0[beachnb].append(wave_height[beachnb][i])
    
    fig=plt.figure(figsize=(15,7))
    plt.suptitle('Wave height')
    plt.subplot(2,2,3)
    plt.hist(we[0],alpha=0.3,label='obs',normed=True)
    plt.hist(we_0[0],alpha=0.3,label='none',normed=True)
    plt.title('Clovelly')
    plt.grid()
    plt.legend()
    plt.subplot(2,2,2)
    plt.hist(we[1],alpha=0.3,label='obs',normed=True)
    plt.hist(we_0[1],alpha=0.3,label='none',normed=True)
    plt.title('Coogee')
    plt.grid()
    plt.legend()
    plt.subplot(2,2,1)
    plt.hist(we[2],alpha=0.3,label='obs',normed=True)
    plt.hist(we_0[2],alpha=0.3,label='none',normed=True)
    plt.title('Maroubra')
    plt.grid()
    plt.legend()

def ToRotateShelf(Wind_u, Wind_v):
    rot_deg_angle = - 25
    Wind_u_rot = np.cos(rot_deg_angle * np.pi / 180) * Wind_u + np.sin(rot_deg_angle * np.pi / 180) * Wind_v;  #  across-shelf 
    Wind_v_rot = - np.sin(rot_deg_angle * np.pi / 180) * Wind_u + np.cos(rot_deg_angle * np.pi / 180) * Wind_v;  #  along -shelf 
    return Wind_u_rot, Wind_v_rot












"""

UV data


file=pd.read_csv('../raw_observation_data/file_adcp_SYD_2016_2019.csv')
uv_datetime=[datetime.datetime.strptime(day, '%Y-%m-%d') for day in file.DATE]
u_syd_int=file['UCUR_ROT_int'].values.astype('float')
v_syd_int=file['VCUR_ROT_int'].values.astype('float')
u_syd_17=file['UCURrot_17m'].values.astype('float')
v_syd_17=file['VCURrot_17m'].values.astype('float')
uv_date=[datetime.date() for datetime in uv_datetime]
speed_current, direction_current_weird = cart2pol(u_syd_int, v_syd_int)
direction_current_oceano = ToNormal(direction_current_weird*180/np.pi)


#Get the index of observed BB days at maroubra
index=[]
for i in range(0,len(uv_date)):
    if np.any(np.asarray(uv_date[i])==np.asarray(date_box[2][2])):
        index.append(i)

uv_date_mar_obs=np.asarray(uv_date)[index]



def PlotHistUV(beachnb):
    index=[]
    index0=[]
    for i in range(0,len(uv_date)):
        if np.any(np.asarray(uv_date[i])==np.asarray(date_box[beachnb][2])):
            index.append(i)
        if np.any(np.asarray(uv_date[i])==np.asarray(date_box[beachnb][0])):
            index0.append(i)

    bins=np.linspace(-0.15,0.15,40)
    fig=plt.figure(figsize=(15,7))
    plt.suptitle(str(location[beachnb]))
    plt.subplot(2,2,1)
    plt.hist(u_syd_17[index],alpha=0.3,label='obs',bins=bins,normed=True)
    plt.hist(u_syd_17[index0],alpha=0.3,label='none',bins=bins,normed=True)
    plt.title('u ors 17')
    plt.grid()
    plt.legend()
    plt.subplot(2,2,2)
    plt.hist(u_syd_int[index],alpha=0.3,label='obs',bins=bins,normed=True)
    plt.hist(u_syd_int[index0],alpha=0.3,label='none',bins=bins,normed=True)
    plt.title('u ors int')
    plt.grid()
    plt.legend()
    plt.subplot(2,2,3)
    bins_v=np.linspace(-1,1,40)
    plt.hist(v_syd_17[index],alpha=0.3,label='obs',bins=bins_v,normed=True)
    plt.hist(v_syd_17[index0],alpha=0.3,label='none',bins=bins_v,normed=True)
    plt.title('v ors 17')
    plt.grid()
    plt.legend()
    plt.subplot(2,2,4)
    plt.hist(v_syd_int[index],alpha=0.3,label='obs',bins=bins_v,normed=True)
    plt.hist(v_syd_int[index0],alpha=0.3,label='none',bins=bins_v,normed=True)
    plt.title('v ors int')
    plt.grid()
    plt.legend()
    fig.savefig('u_v_current_hist'+str(location[beachnb])+'.png',dpi=300)
    
    


def RosePlotCurrent():
    df = pd.DataFrame({"speed": speed_current, "direction": ToMeteo(direction_current_oceano)})
    bins = np.arange(0.01, 1, 0.2)

    kind = "bar"
#  fig=plt.figure()
    plot_windrose(df, kind=kind, normed=True, opening=0.8, edgecolor="white",bins=bins,blowto=True)
    plt.title('Daily averaged current SYD100, oceano')
RosePlotCurrent()
"""


"""

NE=np.where(np.logical_and(wind_direction_daily>11.25, wind_direction_daily<=101.25))
SE=np.where(np.logical_and(wind_direction_daily>101.25, wind_direction_daily<=191.25))
SW=np.where(np.logical_and(wind_direction_daily>191.25, wind_direction_daily<=281.25))
NW=np.where(np.logical_or(wind_direction_daily>281.25, wind_direction_daily<=11.25))

date=np.asarray(date_obs)
date=[date[NE],date[SE],date[SW],date[NW]]
observed_list, none_list = [], []
for l in range(len(date)):
    observed=0
    none=0
    for i in range(len(date[l])):
        for j in range(len(date_box[1][2])): #Coogee
            if date[l][i]==date_box[1][2][j]:
                observed+=1

    for i in range(len(date[l])):
        for j in range(len(date_box[1][0])):
            if date[l][i]==date_box[1][0][j]:
                none+=1
    observed_list.append(observed/(observed+none))
    none_list.append(none/(observed+none))

ind = np.arange(4)
width=0.2
plt.xticks(ind, ('NE','SE','SW','NW'))
ax = plt.subplot(111)
ax.bar(ind-width/2, none_list, width=width, color='lightgrey', align='center',label='None')
ax.bar(ind+width/2, observed_list, width=width, color='dodgerblue', align='center',label='Observed')
plt.legend()
plt.title('Coogee')
plt.show()
fig.savefig('../outputs_observation_data/kurnell/histograms_observation/direction_coogee.png',dpi=300)


date=np.asarray(date_obs)
date=[date[NE],date[SE],date[SW],date[NW]]
NE_list, SE_list, SW_list, NW_list =[0,0], [0,0], [0,0], [0,0]
liste=[NE_list, SE_list, SW_list, NW_list]
sum_none=0.
sum_observed=0.
for l in range(len(date)):
    observed=0
    none=0
    for i in range(len(date[l])):
        for j in range(len(date_box[1][2])): #Coogee
            if date[l][i]==date_box[1][2][j]:
                observed+=1

    for i in range(len(date[l])):
        for j in range(len(date_box[1][0])):
            if date[l][i]==date_box[1][0][j]:
                none+=1
    liste[l][0]=none
    liste[l][1]=observed
    sum_none+=none
    sum_observed+=observed
for l in range(len(date)):
    liste[l][0]=liste[l][0]/sum_none
    liste[l][1]=liste[l][1]/sum_observed


xbar=np.arange(2)
ax = plt.subplot(111)
plt.xticks(xbar, ('None', 'Some'))
ax.bar(xbar-3*width/2, liste[0], width=0.2, color='olivedrab', align='center',label='NE')
ax.bar(xbar-width/2, liste[1], width=0.2, color='skyblue', align='center',label='SE')
ax.bar(xbar+width/2, liste[2], width=0.2, color='plum', align='center',label='SW')
ax.bar(xbar+3*width/2, liste[3], width=0.2, color='orange', align='center',label='NW')
plt.legend()
plt.title('Coogee')
plt.show()
fig.savefig('../outputs_observation_data/kurnell/histograms_observation/situation_coogee.png',dpi=300)



TimeSeriesPlot()
BoxPlot(0,date_obs,wind_direction_daily)
BoxPlot(1,date_obs,wind_direction_daily)
BoxPlot(2,date_obs,wind_direction_daily)

PolarPlot(0, wind_direction_daily, wind_speed_daily)
PolarPlot(1, wind_direction_daily, wind_speed_daily)
PolarPlot(2, wind_direction_daily, wind_speed_daily)


BoxPlot(0,date_obs,wind_direction_daily)
BoxPlot(1,date_obs,wind_direction_daily)
BoxPlot(2,date_obs,wind_direction_daily)



RosePlot(0,0,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(0,1,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(0,2,date_obs,wind_direction_daily,wind_speed_daily)

RosePlot(1,0,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(1,1,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(1,2,date_obs,wind_direction_daily,wind_speed_daily)

RosePlot(2,0,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(2,1,date_obs,wind_direction_daily,wind_speed_daily)
RosePlot(2,2,date_obs,wind_direction_daily,wind_speed_daily)


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
