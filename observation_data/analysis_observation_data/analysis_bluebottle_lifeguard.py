# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:14:40 2019

@author : Natacha 
"""
from matplotlib.lines import Line2D
import datetime
import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import glob 
from astropy.io import ascii
import matplotlib.dates as mdates
from astropy.table import Table, Column

"""
Lifeguard reports data
"""

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
                
def DayEqual(object1, object2):
    if object1.day==object2.day and object1.month==object2.month and object1.year==object2.year:
        return True
    else:
        return False


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
            if filename.Bluebottles[i]=='none' or filename.Bluebottles[i]=='likely':
                bluebottles.append(0.)
            elif filename.Bluebottles[i]=='some':
                bluebottles.append(1.)
            elif filename.Bluebottles[i]=='many':
                bluebottles.append(2.)

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


def TableDiff(date1,date2,file1,file2):
    """
    Returns a table showing every element file1 and file2 when they are different at the same date
    Last row is the number of days with the same nb of bluebottles out of all the days
    """
    equal=0
    diff=0
    date=[]
    first_beach=[]
    second_beach=[]
    print('First beach :')
    first = input()
    print('2nd beach :')
    second = input()
    for i in range(len(date1)):
        for j in range(len(date2)):
            if (date1[i]==date2[j]):
                if int(file1[i])==int(file2[j]):
                    equal+=1
                else:
                    diff+=1
                    date.append(date1[i])
                    first_beach.append(file1[i])
                    second_beach.append(file2[j])
    t=Table([date,first_beach,second_beach], names=('date',first,second))
    total=equal+diff
    t.add_row((0, equal, total))
#    ascii.write(t, '../outputs_observation_data/diff'+first+second+'.csv', format='csv', fast_writer=False, overwrite=True)  

def PlotTemp():
    """
    Save fig of number of bluebottles depending on time and water temperature
    """
    location=['Clovelly','Coogee','Maroubra']
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator(range(0, 12), interval=2) # every 2month
    years_fmt = mdates.DateFormatter('%Y')
    month_fmt = mdates.DateFormatter('%m')

    for i in range(len(bluebottles)):
        fig=plt.figure(figsize=(12,9))
        for j in range(len(bluebottles[i])):
            if bluebottles[i][j]==1:
                somemany=plt.scatter(date[i][j], water_temp[i][j], color='dodgerblue',s=16,marker='o')
     #       elif bluebottles[i][j]==0.5:
     #           likely=plt.scatter(date[i][j], water_temp[i][j], color='lightskyblue',s=12,alpha=0, marker='o')
            else:
                none=plt.scatter(date[i][j], water_temp[i][j], color='lightgrey', marker='o',s=16,alpha=0.3)
        ax=plt.axes()
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(month_fmt)
        ax.tick_params(labelsize=12)

        fig.autofmt_xdate()
        plt.ylabel('Water temperature (celsius)', fontsize=16)
        plt.axvline(x=datetime.date(2017,6,1))
        plt.axvline(x=datetime.date(2017,9,1))
        plt.axvline(x=datetime.date(2018,6,1))
        plt.axvline(x=datetime.date(2018,9,1))
        plt.title('Bluebottles observations : '+str(location[i]), fontsize=16)
        plt.legend((somemany,  none),#likely,
                    ('observed','none'),# 'likely',
                    scatterpoints=1,
                    loc='upper left',
                    ncol=3,
                    fontsize=16)
        plt.show()
        fig.savefig("../../writing/plot_temp"+str(location[i])+".png",dpi=300)


def TableMonthBeach():
    """
    save csv files for each month and for each beach, with the nb of 
    some, many, likely, none and the percentage of none
    """
    observed_month=[0,0,0]
    likely_month=[0,0,0]
    none_month=[0,0,0]
    percentage_none_month=[0,0,0] 
    
    location=['Clovelly','Coogee','Maroubra']
    yearr=[2016,2017,2018,2019]
    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthh=[1,2,3,4,5,6,7,8,9,10,11,12]
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharey=True, sharex=True)
    ax=[ax1, ax2, ax3, ax4]
    xbar=np.arange(0,12)
    width=0.2
    plt.xticks(xbar, ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))
    for y in range(len(yearr)):
        for i in range(len(date)):
            observed_month[i]=[0 for i in range(0,12)]
            likely_month[i]=[0 for i in range(0,12)]
            none_month[i]=[0 for i in range(0,12)]
            percentage_none_month[i]=[0 for i in range(0,12)]
        for i in range(len(date)):
            for j in range(len(date[i])): 
                for m in range(len(monthh)):
                    if date[i][j].year==yearr[y]:
                        if date[i][j].month==monthh[m]:
                            if bluebottles[i][j]==1.:
                                observed_month[i][m]+=1
                            elif bluebottles[i][j]==2:
                                observed_month[i][m]+=2
                            elif bluebottles[i][j]==0.5:
                                likely_month[i][m]+=1
                            elif bluebottles[i][j]==0.: 
                                none_month[i][m]+=1
                            percentage_none_month[i][m]=np.divide(100.*none_month[i][m],observed_month[i][m]+likely_month[i][m]+none_month[i][m])
            month_beach=Table([month,observed_month[i][:12],likely_month[i][:12],none_month[i][:12], percentage_none_month[i][:12]],names=('Month','Observed','Likely','Noone','% of None'))
            ascii.write(month_beach, '../outputs_observation_data/new_monthly_bluebottles_'+str(yearr[y])+'_'+location[i]+'.csv', format='csv', fast_writer=False, overwrite=True)  
        ax[y].set_ylabel('# observations')
        ax[y].bar(xbar-width/2, observed_month[0], width=0.2, color='dodgerblue', align='center',label='observed')
        ax[y].bar(xbar+width/2, none_month[0], width=0.2, color='hotpink', align='center',label='none')
        plt.legend()
        ax[y].set_title("Clovellu "+str(yearr[y]))
    plt.show()

    

def GetDateSomeLikelyNone(beach_nb,bluebottle_nb):
    date_number = []
    for j in range(len(date[beach_nb])):
        if bluebottles[beach_nb][j]==bluebottle_nb:
            date_number.append(date[beach_nb][j])
    return date_number


def CalcHist(file):
    observedd=[]
    likelyy=[]
    nonee=[]
    for i in range(len(file)):
        nonee.append(file.Noone[i])
        observedd.append(file.Observed[i])
        likelyy.append(file.Likely[i])
    return observedd,likelyy, nonee

def PlotHist():
    f_monthly=[]
    filesmonthly = glob.glob('../outputs_observation_data/monthly*.csv')
    obs=[0 for i in range(12)]
    lik=[0 for i in range(12)]
    non=[0 for i in range(12)]
    month=np.arange(12)
    
    for i in range(len(filesmonthly)):
        f_monthly.append(pd.read_csv(filesmonthly[i]))
        obs[i], lik[i], non[i]=CalcHist(f_monthly[i])
    for i in range(12):
        obbs=np.mean(obs[:])
        liik=np.mean(lik[:])
        noon=np.mean(non[:])
    ax = plt.subplot(111)
    bins=np.arange(1,14)
    ax.set_xticks(bins[:-1])
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.bar(month-0.2,obbs,width=0.2,color='lightskyblue',align='center',label='observed')
    ax.bar(month,liik,width=0.2,color='lightpink',align='center',label='likely')
    ax.bar(month+0.2,noon,width=0.2,color='grey',align='center',label='none')
    
    plt.legend()

files_name = glob.glob('../raw_observation_data/bluebottle_lifeguard_reports/*2.xlsx') #0Clovelly 1Coogee 2Maroubra
beach=[]
date_bb=[0,1,2]
date=[0,1,2]
water_temp=[0,1,2]
bluebottles=[0,1,2]
description=[0,1,2]
date_box=[0,1,2]

for i in range(0,len(files_name)):
    beach.append(pd.read_excel(files_name[i]))

for i in range(0,len(water_temp)):
    date_bb[i], water_temp[i], bluebottles[i], description[i] = GetVariables(beach[i])
    
date[0]=date_bb[0]
date[1]=date_bb[1][:1036] #delete data before 05/2016
date[2]=date_bb[2][:1025] #delete data before 05/2016


water_temp[1]=water_temp[1][:1036]
water_temp[2]=water_temp[2][:1025] #delete data before 05/2016

bluebottles[1]=bluebottles[1][:1036]
bluebottles[2]=bluebottles[2][:1025] 

description[1]=description[1][:1036]
description[2]=description[2][:1025] 

for i in range(0,len(water_temp)):    
    date_box[i]=[GetDateSomeLikelyNone(i,0.),GetDateSomeLikelyNone(i,0.5),GetDateSomeLikelyNone(i,1.)]
#PlotHist()
#TableDiff(date[0],date[1],bluebottles[0],bluebottles[1])
#TableDiff(date[0],date[2],bluebottles[0],bluebottles[2])
#TableDiff(date[1],date[2],bluebottles[1],bluebottles[2])

#PlotTemp()
#TableMonthBeach()



"""
BOM data
"""

def GetBOMVariables(filename):
    """
    Return date, water temp, #of bluebottles of a file
    """
    hour, datee, date, water_temp, wind_direction, wind_speed= [], [], [], [], [], []
    for i in range(len(filename)):
        if filename.Water_Temperature[i]>0:
            water_temp.append(filename.Water_Temperature[i])
            datee.append(filename.Date_UTC_Time[i][:11])
            date.append(time(str(filename.Date_UTC_Time[i][:2]),str(filename.Date_UTC_Time[i][3:6]),str(filename.Date_UTC_Time[i][7:11])))
            hour.append(int(filename.Date_UTC_Time[i][12:14]))
            wind_direction.append(filename.Wind_Direction[i])
            wind_speed.append(filename.Wind_Speed[i])
    for i in range(len(date)):   
        date[i].jan_to_01()        
    
    BOMdate = []
    BOMtime_UTC = np.zeros(len(date))
    BOMtime = np.zeros(len(date))

    for l in range(len(date)):
        BOMdate.append(datetime.date(int(date[l].year), int(date[l].month), int(date[l].day)))
        BOMtime_UTC[l] = BOMdate[l].toordinal() + hour[l]/24     #UTC
    BOMtime = BOMtime_UTC + 10/24

    return BOMtime, BOMdate, water_temp, wind_direction, wind_speed
    

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
#    fig.savefig("../outputs_observation_data/box_plot_past_"+str(location[nb])+".png",dpi=300)

def DailyAverage():
    t=[]
    BOMwind_direction_daily=[]
    for i in range(0,len(BOMdate)-1):
        if BOMdate[i]!=BOMdate[i+1]:
            t.append(BOMdate[i])
    
    for j in range(len(t)):
        for i in range(len(BOMwind_direction)):
            if t[j]==BOMdate[i]:
                if t[j]!=BOMdate[i+1]:
                    BOMwind_direction_daily.append(np.mean(BOMwind_direction[i-23:i+1]))

    return BOMwind_direction_daily, t

def WindDirectionTime(nb, date_plot, BOMdaily):
    
    fig=plt.figure(figsize=(12,9))
    bluebottlesoupas=[]
    location=['Clovelly','Coogee','Maroubra']
    for i in range(len(date_plot)):
        for j in range(len(date[nb])):
            if date[nb][j]==date_plot[i]:
                bluebottlesoupas.append(bluebottles[nb][j]) 
    ax=plt.axes()
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator(range(0, 12), interval=2) # every 2month
    years_fmt = mdates.DateFormatter('%Y')
    month_fmt = mdates.DateFormatter('%m')
    for i in range(len(bluebottlesoupas)):
        if bluebottlesoupas[i]==1.0:
            observed=plt.scatter(date_plot[i],BOMdaily[i-1],color='dodgerblue',s=12)
        elif bluebottlesoupas[i]==0.5:
            likely=plt.scatter(date_plot[i],BOMdaily[i-1],color='palegreen',s=12)
        elif bluebottlesoupas[i]==0.:
            none=plt.scatter(date_plot[i],BOMdaily[i-1],color='hotpink',alpha=0.3, marker='+',s=10)
            
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(month_fmt)

    fig.autofmt_xdate()
    plt.ylabel('Wind Direction')
    plt.title(location[nb])
    plt.legend((observed, likely, none),
                ('observed','likely','none'),
                scatterpoints=1,
                loc='lower right',
                ncol=3,
                fontsize=8)
    plt.show()
  #  fig.savefig("../outputs_observation_data/gust_direction_past_"+str(location[nb])+".png",dpi=300)


#file_name = '../raw_observation_data/bom_port_kembla/all_IDO.csv'
#f=pd.read_csv(file_name)

#BOMtime, BOMdate, BOMwater_temp, BOMwind_direction, BOMwind_speed = GetBOMVariables(f)
#BOMdaily,date_plot=DailyAverage()

#plt.hist(x,bins=30)
#plt.ylabel('proba')



def GetKurnellData(file):
    day=np.zeros(len(file))
    month=np.zeros(len(file))
    year=np.zeros(len(file))
    hours=np.zeros(len(file))
    minutes=np.zeros(len(file))
    date=[]

    def GetU(daily_speed,daily_direction):
        wind_u = - daily_speed/3.6 * np.sin(np.pi / 180 * daily_direction) 
        return wind_u
    
    def GetV(daily_speed,daily_direction):
        wind_v = - daily_speed/3.6 * np.cos(np.pi / 180 * daily_direction)
        return wind_v
    
    for i in range(len(file)):
        minutes[i]=file.MI_local_time[i]
        hours[i]=file.HH24[i]
        day[i]=file.DD[i]
        month[i]=file.MM[i]
        year[i]=file.YY[i]

    
    for i in range(len(file)):
        date.append(datetime.date(int(year[i]),int(month[i]),int(day[i])))

    return date, daily_direction, daily_speed, wind_u, wind_v, max_direction, max_speed


"""
for i in range(len(t)):
    tt0 = np.where(day_w == t[i])[0] #prend lindice de quand c egal
    Wind_speed_ms_Daily[i] = np.mean(nonans(Wind_speed_ms[tt0.astype(int)]))
"""


def PolarPlot(nb,direction):
    blueb=[]
    daily=[]
    fig=plt.figure(figsize=(12,9))
    location=['Clovelly','Coogee','Maroubra']
    for i in range(len(direction)):
        for j in range(len(date[nb])):
            if date_plot[i]+datetime.timedelta(days=1)==date[nb][j]: #date_kurnell
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
    legend_elements = [Line2D([0],[0],marker='o',label='None', color='w',markerfacecolor='hotpink', markersize=10),
                       Line2D([0],[0],marker='o',label='Likely', color='w',markerfacecolor='palegreen', markersize=10),
                       Line2D([0],[0],marker='o',label='Observed', color='w',markerfacecolor='dodgerblue', markersize=10)]

    legend1=plt.legend(handles=legend_elements, loc='lower right')
    ax.add_artist(legend1)
    ax.scatter(theta, r, c=colors,  cmap='hsv', alpha=0.75)
    ax.set_rorigin(-2.5)
    ax.set_theta_zero_location('W', offset=10)
    plt.title("Daily averaged wind direction (day before) at "+str(location[nb]))
    plt.show()
   # fig.savefig("../outputs_observation_data/with_BOMdata/polar_plot_"+str(location[nb])+".png",dpi=300)

#file_name_kurnell = '../raw_observation_data/wind_kurnell_sydney_observatory/Kurnell_Data.csv'
#file=pd.read_csv(file_name_kurnell)
#df = file.apply(pd.to_numeric, args=('coerce',)) # inserts NaNs where empty cell!!! grrrr

#date_kurnell, direction_kurnell, speed_kurnell, u_kurnell, v_kurnell, max_direction, max_speed=GetKurnellData(df)

def UVplot():
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator(range(0, 12), interval=2) # every 2month
    years_fmt = mdates.DateFormatter('%Y')
    month_fmt = mdates.DateFormatter('%m')
    fig=plt.figure()
    ax=plt.subplot(2,1,1)
    plt.plot(date_kurnell,v_kurnell)
    plt.ylabel('V')
    ax1=plt.subplot(2,1,2)
    plt.plot(date_kurnell,u_kurnell)
    plt.ylabel('U')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(month_fmt)
    ax1.xaxis.set_major_locator(years)
    ax1.xaxis.set_major_formatter(years_fmt)
    ax1.xaxis.set_minor_locator(months)
    ax1.xaxis.set_minor_formatter(month_fmt)
    plt.show()


blueb1_2017, date1_2017 = [], []
blueb1_2018, date1_2018 = [], []
for i in range(len(date[1])):
    if date[1][i].year==2017:
        blueb1_2017.append(bluebottles[1][i])
        date1_2017.append(date[1][i])
    elif date[1][i].year==2018:
        blueb1_2018.append(bluebottles[1][i])
        date1_2018.append(date[1][i])
                
blueb1_2017=pd.Series(blueb1_2017)
blueb1_2018=pd.Series(blueb1_2018)
date1_2017=pd.Series(np.asarray(date1_2017).astype('datetime64'))
date1_2018=pd.Series(np.asarray(date1_2018).astype('datetime64'))
        
      #  blueb1_2018.groupby(date1_2018.dt.week).sum().plot(kind='bar')
        
blueb2_2017, date2_2017 = [], []
blueb2_2018, date2_2018 = [], []
for i in range(len(date[2])):
    if date[2][i].year==2017:
       blueb2_2017.append(bluebottles[2][i])
       date2_2017.append(date[2][i])
    elif date[2][i].year==2018:
       blueb2_2018.append(bluebottles[2][i])
       date2_2018.append(date[2][i])
        
blueb2_2017=pd.Series(blueb2_2017)
blueb2_2018=pd.Series(blueb2_2018)
date2_2017=pd.Series(np.asarray(date2_2017).astype('datetime64'))
date2_2018=pd.Series(np.asarray(date2_2018).astype('datetime64'))
       
blueb_2018=(blueb1_2018.groupby(date1_2018.dt.week).sum())+(blueb2_2018.groupby(date2_2018.dt.week).sum())
blueb_2017=(blueb1_2017.groupby(date1_2017.dt.week).sum())+(blueb2_2017.groupby(date2_2017.dt.week).sum())
     

blueb_day_2018=(blueb1_2018.groupby(date1_2018.dt.day).sum())+(blueb2_2018.groupby(date2_2018.dt.day).sum())
blueb_day_2017=(blueb1_2017.groupby(date1_2017.dt.day).sum())+(blueb2_2017.groupby(date2_2017.dt.day).sum())
 
blueb_month_2018=(blueb1_2018.groupby(date1_2018.dt.month).sum())+(blueb2_2018.groupby(date2_2018.dt.month).sum())
blueb_month_2017=(blueb1_2017.groupby(date1_2017.dt.month).sum())+(blueb2_2017.groupby(date2_2017.dt.month).sum())

def Normalise(data):
    average = np.mean(data)
    std=np.std(data)
    normalized_data = [np.divide((d-average),std) for d in data]
    return normalized_data
    
   # plt.figure()
  #  blueb_2018.plot(kind='bar')


import pandas as pd
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt 
import glob

filenames18_raw = glob.glob('../../imos_current_data/analysis_imos_current_data/results_beaching/18*_konsole.csv')
filenames17_raw = glob.glob('../../imos_current_data/analysis_imos_current_data/results_beaching/17*_konsole.csv')


filenames18 = [f[71:] for f in filenames18_raw]
filenames17 = [f[71:] for f in filenames17_raw]

file18 = []
blueb_weekly18, blueb_monthly18 = [], []
total_beaching18 = []
average18, std18 = [], []

file17 = []
blueb_weekly17, blueb_monthly17 = [], []
total_beaching17 = []
average18, std17 = [], []
filen, filenn = [],[]
for i in range(len(filenames18_raw)):
    for j in range(len(filenames17_raw)):
        if filenames18[i]==filenames17[j]:
            file18.append(pd.read_csv(filenames18_raw[i]))
            file17.append(pd.read_csv(filenames17_raw[j]))
            filen.append(filenames18[i])

for i in range(len(file18)):
    file18[i]['time']=file18[i]['time'].astype("datetime64")
    blueb_weekly18.append(file18[i]['time'].groupby(file18[i]["time"].dt.week).count())
    blueb_monthly18.append(file18[i]['time'].groupby(file18[i]["time"].dt.month).count())
    total_beaching18.append(np.sum(blueb_weekly18[i]))
    
for j in range(len(file17)):
    file17[j]['time']=file17[j]['time'].astype("datetime64")
    blueb_weekly17.append(file17[j]['time'].groupby(file17[j]["time"].dt.week).count())
    blueb_monthly17.append(file17[j]['time'].groupby(file17[j]["time"].dt.month).count())
    total_beaching17.append(np.sum(blueb_weekly17[j]))

norm_data_17 = [Normalise(blueb_data) for blueb_data in blueb_weekly17]
norm_data_18 = [Normalise(blueb_data) for blueb_data in blueb_weekly18]

norm_obs_17 = Normalise(blueb_2017)
norm_obs_18 = Normalise(blueb_2018)

norm_month_data_17 = [Normalise(blueb_data) for blueb_data in blueb_monthly17]
norm_month_data_18 = [Normalise(blueb_data) for blueb_data in blueb_monthly18]

norm_month_obs_17 = Normalise(blueb_month_2017)
norm_month_obs_18 = Normalise(blueb_month_2018)

#norm_month_obs = np.hstack((norm_month_obs_17,norm_month_obs_18))
#norm_obs = np.hstack((norm_obs_17,norm_obs_18))
#norm_month_data = np.hstack((norm_month_data_17,norm_month_data_18))
#norm_data = np.hstack((norm_data_17,norm_data_18))
#filenames = np.hstack((filenames17,filenames18))
#total_beaching = np.hstack((total_beaching17,total_beaching18))

def Difference(obs,sim):
    diff=0.
    div=0.
    for i in range(len(obs)):
        for j in range(len(sim)):
            if i==j:
                div+=1
                diff+=(obs[i]-sim[j])**2
    return diff/div

diff_week_18 = [Difference(norm_obs_18,norm_d) for norm_d in norm_data_18]
diff_month_18 = [Difference(norm_month_obs_18,norm_d) for norm_d in norm_month_data_18]


diff_week_17 = [Difference(norm_obs_17,norm_d) for norm_d in norm_data_17]
diff_month_17 = [Difference(norm_month_obs_17,norm_d) for norm_d in norm_month_data_17]

diff_week = np.asarray(diff_week_17) + np.asarray(diff_week_18)
diff_month = np.asarray(diff_month_17) + np.asarray(diff_month_18)

total = np.asarray(total_beaching17) + np.asarray(total_beaching18)



def sort_list(list1,list2):
    zipped_pairs = zip(list2,list1)
    
    z = [x for _,x in sorted(zipped_pairs)]
    return z

name_week = sort_list(filen,diff_week)
total_week = sort_list(total/800, diff_week)
difference_week = sorted(diff_week/2)

name_month = sort_list(filen,diff_month)
total_month = sort_list(total/800, diff_month)
difference_month = sorted(diff_month/2)


table_week = Table([name_week, total_week, difference_week], names = ('simul','total obs','diff/week'))
ascii.write(table_week, 'week.csv', format='csv', fast_writer=False, overwrite=True)  

table_month = Table([name_month, total_month, difference_month], names = ('simul','total obs','diff/month'))
ascii.write(table_month, 'month.csv', format='csv', fast_writer=False, overwrite=True)  


#   fig.savefig("../outputs_observation_data/U_V_plot.png",dpi=300)

#for i in range(len(beach)):
 #   WindDirectionTime(i,date_kurnell,max_direction)
    
#for i in range(3):
 #   PolarPlot(i,BOMdaily)

#UVplot()
    
#for i in range(3):
 #  BoxPlot(i, date_kurnell, max_direction)
  #  plt.figure()
   # blueb_2017.plot(kind='bar')


"""
    from scipy.optimize import curve_fit
    def f(x,a,b,c,d,e,f,g,h,i,j,k):
          return a*x**10+b*x**9+h*x**8+i*x**7+j*x**6+k*x**5+c*x**4+d*x**3+e*x**2+f*x+g
    weeks = np.linspace(1,52,52)
    popt17, popc17 = curve_fit(f,weeks,blueb_2017)
    popt18, popc18 = curve_fit(f,weeks,blueb_2018)
    
    x_axis=np.linspace(weeks[0],weeks[-1],100)
    plt.figure()
    plt.plot(x_axis,f(x_axis,*popt18),'r--',label='Fitted curve')
    plt.bar(weeks, blueb_2018,label='Observations')
    plt.legend()
    plt.title('BB obs/week 2018',fontsize=12)
    plt.figure()
    plt.plot(x_axis,f(x_axis,*popt17),'r--',label='Fitted curve')
    plt.bar(weeks, blueb_2017,label='Observations')
    plt.legend()
    plt.title('BB obs/week 2018',fontsize=12)
    
    
        from scipy.optimize import curve_fit
    def f(x,a,b,c,d,e,f,g,h,i):
          return h*x**8+i*x**7+a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g
    weeks = np.linspace(1,52,52)
    popt18, popc18 = curve_fit(f,weeks,blueb_2018)
    popt17, popc17 = curve_fit(f,weeks,blueb_2017)
    
    x_axis=np.linspace(weeks[0],weeks[-1],100)
    plt.figure()
    plt.plot(x_axis,f(x_axis,*popt18),'r--',label='Fitted curve')
    plt.bar(weeks, blueb_2018,label='Observations')
    plt.legend()
    plt.title('BB obs/week 2018',fontsize=12)
    plt.figure()
    plt.plot(x_axis,f(x_axis,*popt17),'r--',label='Fitted curve')
    plt.bar(weeks, blueb_2017,label='Observations')
    plt.legend()
    plt.title('BB obs/week 2017',fontsize=12)
    plt.show()

    
"""