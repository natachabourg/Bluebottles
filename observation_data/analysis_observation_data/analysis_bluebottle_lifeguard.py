# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:14:40 2019

@author : Natacha 
"""
import scipy.stats as stats
import datetime
import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import glob 
import matplotlib.ticker as ticker
from astropy.table import Table, Column
from astropy.io import ascii
from dateutil.parser import parse
import matplotlib.dates as mdates

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
    ascii.write(t, '../outputs_observation_data/diff'+first+second+'.csv', format='csv', fast_writer=False, overwrite=True)  

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
                somemany=plt.scatter(date[i][j], water_temp[i][j], color='dodgerblue',s=12)
            elif bluebottles[i][j]==0.5:
                likely=plt.scatter(date[i][j], water_temp[i][j], color='lightskyblue',s=12)
            else:
                none=plt.scatter(date[i][j], water_temp[i][j], color='hotpink',alpha=0.2, marker='+',s=10)
        ax=plt.axes()
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(month_fmt)

        fig.autofmt_xdate()
        plt.ylabel('Water temperature (celsius)')
        plt.title('Bluebottles at '+str(location[i])+' beach')
        plt.legend((somemany, likely, none),
                    ('observed','likely','none'),
                    scatterpoints=1,
                    loc='upper left',
                    ncol=3,
                    fontsize=8)
    
        plt.show()
        fig.savefig("../outputs_observation_data/plot_temp"+str(location[i])+".png",dpi=300)


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
                            elif bluebottles[i][j]==0.5:
                                likely_month[i][m]+=1
                            elif bluebottles[i][j]==0.: 
                                none_month[i][m]+=1
                            percentage_none_month[i][m]=np.divide(100.*none_month[i][m],observed_month[i][m]+likely_month[i][m]+none_month[i][m])
            month_beach=Table([month,observed_month[i][:12],likely_month[i][:12],none_month[i][:12], percentage_none_month[i][:12]],names=('Month','Observed','Likely','Noone','% of None'))
            ascii.write(month_beach, '../outputs_observation_data/monthly_bluebottles_'+str(yearr[y])+'_'+location[i]+'.csv', format='csv', fast_writer=False, overwrite=True)  

    """
    ax=plt.axes()
    x=[1,2,3,4,5,6,7,8,9,10,11,12]
    width=0.25
    observed0=[observed_month[1][i] for i in range(12)]
    likely0=[likely_month[1][i] for i in range(12)]
    none0=[none_month[1][i] for i in range(12)]
    ax.bar(x,observed0,width)
    ax.bar(x+width,likely0,width,color='C2')
    ax.bar(x+2*width,none0,width,color='C3')
    ax.set_xticks(x+width)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

    ax.hist()
    plt.show()"""

    

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
        obbs[i]=np.mean(obs[:])
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
    

def BoxPlot(nb):   
    """
    Box plot pour 0Clovelly ici de wind direction pour les 3 cas : none likely observed
    """
    location=['Clovelly','Coogee','Maroubra']
    wind_direction_box0=[]
    wind_direction_box1=[]
    wind_direction_box2=[]
    for i in range(len(date_box[nb][0])):
        for j in range(len(date_plot)):
            if date_box[nb][0][i]==date_plot[j]:
                wind_direction_box0.append(BOMdaily[j])
    
    for i in range(len(date_box[nb][1])):
        for j in range(len(date_plot)):
            if date_box[nb][1][i]==date_plot[j]:
                wind_direction_box1.append(BOMdaily[j])
    
    for i in range(len(date_box[nb][2])):
        for j in range(len(date_plot)):
            if date_box[nb][2][i]==date_plot[j]:
                wind_direction_box2.append(BOMdaily[j])
    
    x=[wind_direction_box0,wind_direction_box1,wind_direction_box2]
    fig = plt.figure(figsize=(12,9))
    plt.title(location[nb])
    plt.ylabel('Wind direction (degrees)')
    plt.boxplot(x)
    plt.xticks([1,2,3],['None','Likely','Some'])
    plt.show()
    fig.savefig("../outputs_observation_data/box_plot_"+str(location[nb])+".png",dpi=300)

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

def WindDirectionTime(nb):
    
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
            observed=plt.scatter(date_plot[i],BOMdaily[i],color='dodgerblue',s=12)
        elif bluebottlesoupas[i]==0.5:
            likely=plt.scatter(date_plot[i],BOMdaily[i],color='lightskyblue',s=12)
        elif bluebottlesoupas[i]==0.:
            none=plt.scatter(date_plot[i],BOMdaily[i],color='hotpink',alpha=0.3, marker='+',s=10)
            
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
    #plt.show()
    fig.savefig("../outputs_observation_data/plot_wind_direction"+str(location[nb])+".png",dpi=300)


file_name = '../raw_observation_data/bom_port_kembla/all_IDO.csv'
f=pd.read_csv(file_name)

BOMtime, BOMdate, BOMwater_temp, BOMwind_direction, BOMwind_speed = GetBOMVariables(f)
BOMdaily,date_plot=DailyAverage()

for i in range(3):
    BoxPlot(i)

#for i in range(len(beach)):
#    WindDirectionTime(i)

#plt.hist(x,bins=30)
#plt.ylabel('proba')


