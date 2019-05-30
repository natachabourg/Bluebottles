# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:14:40 2019

@author : Natacha 
"""

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
        
    return date, datee, water_temp, bluebottles, description


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
            if (DayEqual(date1[i],date2[j])):
                if int(file1[i])==int(file2[j]):
                    equal+=1
                else:
                    diff+=1
                    date.append(date1[i].day+"/"+date1[i].month+"/"+date1[i].year)
                    first_beach.append(file1[i])
                    second_beach.append(file2[j])
    t=Table([date,first_beach,second_beach], names=('date',first,second))
    total=equal+diff
    t.add_row((0, equal, total))
    ascii.write(t, '../outputs_observation_data/diff'+first+second+'.csv', format='csv', fast_writer=False, overwrite=True)  

def PlotFirstTry():
    """
    Save fig of number of bluebottles depending on time and water temperature
    """
    
    for i in range(len(bluebottles)):
        for j in range(400,700):
            if bluebottles[i][j]==1:
                somemany=plt.scatter(bitchdate[i][j], water_temp[i][j], color='dodgerblue')
            elif bluebottles[i][j]==0.5:
                likely=plt.scatter(bitchdate[i][j], water_temp[i][j], color='lightskyblue')
            else:
                none=plt.scatter(bitchdate[i][j], water_temp[i][j], color='hotpink',alpha=0.8)
    ax=plt.axes()
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))

#    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
  #  years=date[0][:300].year
    # format the coords message box
  #  ax.xaxis.set_major_locator(years)

    plt.ylabel('Water temperature (celsius)')
    plt.title('Bluebottles at Coogee Beach')
    plt.legend((somemany, likely, none),
                ('many some','likely','none'),
                scatterpoints=1,
                loc='upper left',
                ncol=3,
                fontsize=8)

    plt.show()
   # plt.savefig('new_coogee_first')


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
    yearr=['2016','2017','2018','2019']
    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthh=['1','2','3','4','5','6','7','8','9','10','11','12']
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
            month_beach=Table([month,observed_month[i][:12],likely_month[i][:12],none_month[i][:12], percentage_none_month[i][:12]],names=('Month','Observed','Likely','None','% of None'))
            ascii.write(month_beach, '../outputs_observation_data/Bluebottles per month for'+yearr[y]+' at'+location[i]+'.csv', format='csv', fast_writer=False, overwrite=True)  
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

TableMonthBeach()
            

def GetDateSomeLikelyNone(number):
    date_number = [] #for coogee
  #  for i in range(len(date)):
    for j in range(len(date[1])):
        if bluebottles[1][j]==number:
            date_number.append(date[1][j])
    return date_number


files_name = glob.glob('../raw_observation_data/bluebottle_lifeguard_reports/*2.xlsx') #0Clovelly 1Coogee 2Maroubra

beach=[]
bitchdate=[0,1,2]
date=[0,1,2]
water_temp=[0,1,2]
bluebottles=[0,1,2]
description=[0,1,2]

for i in range(0,len(files_name)):
    beach.append(pd.read_excel(files_name[i]))

for i in range(0,len(water_temp)):
    bitchdate[i], date[i], water_temp[i], bluebottles[i], description[i] = GetVariables(beach[i])

date_box=[GetDateSomeLikelyNone(0.),GetDateSomeLikelyNone(0.5),GetDateSomeLikelyNone(1.)]
#TableDiff(date[0],date[1],bluebottles[0],bluebottles[1])
#TableDiff(date[0],date[2],bluebottles[0],bluebottles[2])
#TableDiff(date[1],date[2],bluebottles[1],bluebottles[2])

#PlotFirstTry()

"""
BOM data
"""

def GetBOMVariables(filename):
    """
    Return date, water temp, #of bluebottles of a file
    """
    datee, date, water_temp, wind_direction, wind_speed = [], [], [], [], []
    for i in range(len(filename)):
        if filename.Water_Temperature[i]>0:
            water_temp.append(filename.Water_Temperature[i])
            datee.append(filename.Date_UTC_Time[i][:11])
            date.append(time(str(filename.Date_UTC_Time[i][:2]),str(filename.Date_UTC_Time[i][3:6]),str(filename.Date_UTC_Time[i][7:11])))
            wind_direction.append(filename.Wind_Direction[i])
            wind_speed.append(filename.Wind_Speed[i])
    for i in range(len(date)):   
        date[i].jan_to_01()

    return datee, date, water_temp, wind_direction, wind_speed


def JoinBomData():
    for i in range(len(BOMdate)):
        for j in range(len(BOMdate[i])):
            BOMtimeNew.append(BOMtime[i][j])
            BOMdateNew.append(BOMdate[i][j])
            BOMwater_tempNew.append(BOMwater_temp[i][j])
            BOMwind_directionNew.append(BOMwind_direction[i][j])
            BOMwind_speedNew.append(BOMwind_speed[i][j])
    

def BoxPlot(nb):   
    """
    Box plot pour 0Clovelly ici de wind direction pour les 3 cas : none likely observed
    """
    location=['Clovelly','Coogee','Maroubra']
    wind_direction_box0=[]
    wind_direction_box1=[]
    wind_direction_box2=[]
    for i in range(len(date_box[0])):
        for j in range(len(BOMtime[nb])):
            if DayEqual(date_box[0][i],BOMtime[nb][j]):
                wind_direction_box0.append(BOMwind_direction[nb][j])
    
    for i in range(len(date_box[1])):
        for j in range(len(BOMtime[nb])):
            if DayEqual(date_box[1][i],BOMtime[nb][j]):
                wind_direction_box1.append(BOMwind_direction[nb][j])
    
    for i in range(len(date_box[2])):
        for j in range(len(BOMtime[nb])):
            if DayEqual(date_box[2][i],BOMtime[nb][j]):
                wind_direction_box2.append(BOMwind_direction[nb][j])
    
    x=[wind_direction_box0,wind_direction_box1,wind_direction_box2]
    fig = plt.figure()
    plt.title(location[nb])
    plt.ylabel('Wind direction (degrees)')
    plt.boxplot(x)
    plt.xticks([1,2,3],['None','Likely','Some'])
    plt.show()
    fig.savefig("../outputs_observation_data/box_plot_"+str(location[nb])+".png")


def WindDirectionTime(nb):
    mean=0.
    meann=[]
    date_plot=[]
    bluebottlesoupas=[]
    location=['Clovelly','Coogee','Maroubra']
    for i in range(len(BOMtimeNew)-1):
        for j in range(len(date[nb])):
            if DayEqual(date[nb][j],BOMtimeNew[i]):
                if DayEqual(BOMtimeNew[i],BOMtimeNew[i+1])==False:
                    bluebottlesoupas.append(bluebottles[nb][j]) 
                    mean=np.mean(BOMwind_directionNew[:i+1])
                    meann.append(mean)
                    date_plot.append(BOMdateNew[i])
    ax=plt.axes()
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    for i in range(len(bluebottlesoupas)):
        if bluebottlesoupas[i]==1.0:
            observed=plt.scatter(date_plot[i],meann[i],alpha=0.8,color='deepskyblue')
        elif bluebottlesoupas[i]==0.5:
            likely=plt.scatter(date_plot[i],meann[i],alpha=0.8,color='lightpink')
        elif bluebottlesoupas[i]==0.:
            none=plt.scatter(date_plot[i],meann[i],alpha=0.8,color='lightgrey')
    plt.ylabel('Wind Direction')
    plt.title(location[nb])
    plt.legend((observed, likely, none),
                ('observed','likely','none'),
                scatterpoints=1,
                loc='lower right',
                ncol=3,
                fontsize=8)
    plt.show()


files_name = glob.glob('../raw_observation_data/bom_port_kembla/IDO*.csv')
f=[]
BOMdate=[0,1,2]
BOMtime=[0,1,2]
BOMwater_temp=[0,1,2]
BOMwind_direction=[0,1,2]
BOMwind_speed=[0,1,2]
daily_wind_direction=[0,1,2]
for i in range(len(files_name)):
    f.append(pd.read_csv(files_name[i]))
for i in range(len(BOMdate)):
    BOMdate[i], BOMtime[i], BOMwater_temp[i], BOMwind_direction[i], BOMwind_speed[i] = GetBOMVariables(f[i])
    
BOMtimeNew=[]
BOMdateNew=[]
BOMwater_tempNew=[]
BOMwind_directionNew=[]
BOMwind_speedNew=[]
JoinBomData()

for i in range(3):
    BoxPlot(i)


#for i in range(len(beach)):
 #   WindDirectionTime(i)

