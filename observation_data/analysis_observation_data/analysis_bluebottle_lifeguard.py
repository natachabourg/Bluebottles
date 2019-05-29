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
            elif filename.Bluebottles[i]=='some':
                bluebottles.append(1.)
            elif filename.Bluebottles[i]=='many':
                bluebottles.append(2.)
            elif filename.Bluebottles[i]=='likely':
                bluebottles.append(0.5)
        
    return datee, water_temp, bluebottles, description


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
            if ((date1[i].day==date2[j].day) and (date1[i].month==date2[j].month) and (date1[i].year==date2[j].year)):
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
    ascii.write(t, 'diff'+first+second+'.csv', format='csv', fast_writer=False, overwrite=True)  

def PlotFirstTry():
    """
    Save fig of number of bluebottles depending on time and water temperature
    """
    for i in range(len(bluebottles[1])):
        if bluebottles[1][i]==2:
            r=plt.scatter(date[1][i].month, water_temp[1][i], color='r')
        elif bluebottles[1][i]==1:
            hotpink=plt.scatter(date[1][i].month, water_temp[1][i], color='hotpink')
        elif bluebottles[1][i]==0.5:
            pink=plt.scatter(date[1][i].month, water_temp[1][i], color='pink')
        else:
            blue=plt.scatter(date[1][i].month, water_temp[1][i], color='b', alpha=0.0)
    ax=plt.axes()
    plt.xlabl=('year')
    plt.ylabel('Water temperature (celsius)')
    plt.title('Bluebottles at Coogee Beach')
    plt.legend((r, hotpink, pink,blue),
                ('many','some','likely','none'),
                scatterpoints=1,
                loc='upper left',
                ncol=3,
                fontsize=8)
  #  ax.xaxis.set_major_locator(ticker.MultipleLocator(150))
    plt.show()
   # plt.savefig('new_coogee_first')


files_name = glob.glob('*2.xlsx') #0Clovelly 1Coogee 2Maroubra

beach=[]
date=[0,1,2]
water_temp=[0,1,2]
bluebottles=[0,1,2]
description=[0,1,2]

for i in range(0,len(files_name)):
    beach.append(pd.read_excel(files_name[i]))

for i in range(0,len(water_temp)):
    date[i], water_temp[i], bluebottles[i], description[i] = GetVariables(beach[i])

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
    for i in range(0,len(filename)):
        water_temp.append(filename.Water_Temperature[i])
        datee.append(filename.Date_UTC_Time[i][:11])
        date.append(time(str(filename.Date_UTC_Time[i][:2]),str(filename.Date_UTC_Time[i][3:6]),str(filename.Date_UTC_Time[i][7:11])))
        date[i].jan_to_01()
        wind_direction.append(filename.Wind_Direction[i])
        wind_speed.append(filename.Wind_Speed[i])

    return datee, date, water_temp, wind_direction, wind_speed


def TableMonthBeach():
    """
    save csv files for each month and for each beach, with the nb of 
    some, many, likely, none and the percentage of none
    """
    many_month=[0,0,0]
    some_month=[0,0,0]
    likely_month=[0,0,0]
    none_month=[0,0,0]
    percentage_none_month=[0,0,0] 
    
    location=['Clovelly','Coogee','Maroubra']
    yearr=['2016','2017','2018','2019']
    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthh=['1','2','3','4','5','6','7','8','9','10','11','12']
    for y in range(len(yearr)):
        for i in range(len(date)):
            many_month[i]=[0 for i in range(0,12)]
            some_month[i]=[0 for i in range(0,12)]
            likely_month[i]=[0 for i in range(0,12)]
            none_month[i]=[0 for i in range(0,12)]
            percentage_none_month[i]=[0 for i in range(0,12)]
        for i in range(len(date)):
            for j in range(len(date[i])): 
                for m in range(len(monthh)):
                    if date[i][j].year==yearr[y]:
                        if date[i][j].month==monthh[m]:
                            if bluebottles[i][j]==2.:
                                many_month[i][m]+=1
                            elif bluebottles[i][j]==1.:
                                some_month[i][m]+=1
                            elif bluebottles[i][j]==0.5:
                                likely_month[i][m]+=1
                            elif bluebottles[i][j]==0.: 
                                none_month[i][m]+=1
                            percentage_none_month[i][m]=np.divide(100.*none_month[i][m],many_month[i][m]+some_month[i][m]+likely_month[i][m]+none_month[i][m])
            month_beach=Table([month,many_month[i][:12],some_month[i][:12],likely_month[i][:12],none_month[i][:12], percentage_none_month[i][:12]],names=('Month','Many','Some','Likely','None','% of None'))
            ascii.write(month_beach, 'Bluebottles per month for'+yearr[y]+' at'+location[i]+'.csv', format='csv', fast_writer=False, overwrite=True)  


#TableMonthBeach()


files_name = glob.glob('../TIDE_Port_Kembla/IDO*.csv')
f=[]
BOMdate=[0,1,2,3]
BOMtime=[0,1,2,3]
BOMwater_temp=[0,1,2,3]
BOMwind_direction=[0,1,2,3]
BOMwind_speed=[0,1,2,3]
for i in range(len(files_name)):
    f.append(pd.read_csv(files_name[i]))
for i in range(len(BOMdate)):
    BOMdate[i], BOMtime[i], BOMwater_temp[i], BOMwind_direction[i], BOMwind_speed[i] = GetBOMVariables(f[i])

check_date=[]#list of dates where water_temp=14C
for i in range(0,len(beach)):
    for j in range(len(water_temp[i])):
        if water_temp[i][j]=='14':
            check_date.append(date[i][j])
            








