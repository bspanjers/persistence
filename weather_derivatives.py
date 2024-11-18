#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:21:19 2024

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from QAR import QAR_temperature
from scipy.interpolate import interp1d
from meteostat import Point, Daily
import pandas as pd

yearstart, yearend = 1990, 2020
# import data and set index

def HDD_year_i(vY):
    return np.maximum(0, (18 + 1/3) - vY)

def cum_HDD(vY):
    return np.sum(HDD_year_i(vY))


vY = pd.read_csv('./data_persistence/mB_new_for_coverage_test.csv')
vY = vY.iloc[:5000, 1:]
vY = vY.T
vY.index = pd.DatetimeIndex(vY.index)

m_cum_hdd = np.zeros((vY.shape[1], 39))
l_realization = []
l_U = []
for iYear in np.arange(1980, 2019):
    winter_dates = pd.date_range(start=f'{iYear}-11-01', end=f'{iYear + 1}-03-31', freq='D')

    real_Y = test.new.loc[test.new.index.isin(winter_dates)]
    l_realization.append(cum_HDD(real_Y))
    
    vY_year = vY.loc[vY.index.isin(winter_dates), :]
    l_cum_hdd = []
    for i in range(vY_year.shape[1]):
        l_cum_hdd.append(cum_HDD(vY_year.iloc[:,i]))
    
    l_U.append(np.mean(l_cum_hdd < np.repeat(cum_HDD(real_Y), len(l_cum_hdd))))
    m_cum_hdd[:, iYear-1980] = np.sort(l_cum_hdd)
    
df_cum_hdd = pd.DataFrame(m_cum_hdd, columns=np.arange(1981, 2020))

# Define the location and date range
chicago = Point(41.96017, -87.93164, 204.8)  # chicago coordinates and elevation
atlanta = Point(33.62972, -84.44224, 308.2)  # atlanta coordinates and elevation
lasvegas = Point(36.0719, -115.16343, 662.8)  # atlanta coordinates and elevation
philadelphia = Point(39.87326, -75.22681, 2.1)  # atlanta coordinates and elevation

station = chicago

data = Daily(station, start=pd.Timestamp('1980-01-01'), end=pd.Timestamp('2019-12-31'))
data = data.fetch().tavg 
data = pd.DataFrame(data)
data.index.name='Date'
data.columns=['Temp']
data = data[~((data.index.month == 2) & (data.index.day == 29))] #delete leap days
data = data.dropna()
test.new = data

# Define the location and date range
data = Daily(station, start=pd.Timestamp('1950-01-01'), end=pd.Timestamp('1979-12-31'))
data = data.fetch().tavg 
data = pd.DataFrame(data)
data.index.name='Date'
data.columns=['Temp']
data = data[~((data.index.month == 2) & (data.index.day == 29))] #delete leap days
data = data.dropna()
test.old = data