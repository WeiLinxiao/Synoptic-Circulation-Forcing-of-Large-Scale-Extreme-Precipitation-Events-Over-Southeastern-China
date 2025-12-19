#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:19:03 2023

@author: dai
"""


import numpy as np
import xarray as xr
import pandas as pd 
import glob

from datetime import datetime
from datetime import date
from datetime import timedelta


loc = '/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/'
#%%
name=sorted(glob.glob(loc+'*'))

#var_number = [0,1,2,3,4,5]
#name=[name[i] for i in var_number]  # loc for variables
name_var= [ x[107:len(x)] for x in name ]

#%%
print(datetime.now())
for i in range(len(name)):
    print(i)
    a = xr.open_dataarray(name[i]).sel(time=slice('1979-01-01','2019-12-31'))
    time= a.time
    smooth=a.rolling(time=5, center=True, min_periods=1).mean()
    dates_grouped = pd.to_datetime(a.time.values).strftime('%m%d') 
    smooth=smooth.assign_coords({'time':dates_grouped})
    Climatology = smooth.groupby('time').mean() ## 每年的该日平均
    Anomalies = smooth.groupby('time') - Climatology #
    Anomalies=Anomalies.assign_coords({'time':time})
    Anomalies.to_netcdf(loc+'smoothed_anomaly_1979_2019_'+name_var[i])
print(datetime.now())

