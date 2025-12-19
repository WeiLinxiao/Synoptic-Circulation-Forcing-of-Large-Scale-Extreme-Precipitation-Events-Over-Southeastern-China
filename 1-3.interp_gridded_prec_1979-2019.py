#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:13:18 2023

@author: dai
"""
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from global_land_mask import globe
from datetime import datetime
from datetime import date
from datetime import timedelta
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages # for multipages pdf
from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib
import matplotlib.lines as mlines
import pymannkendall as mk
import cc3d
#%%
'''
print(datetime.now())
for y in range(1979,2020):
    print(y)
    a = xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_'+str(y)+'.nc')
    interval=0.5
    a1 = a.interp(lon=np.arange(73.,135.,interval),lat=np.arange(53.,18.,-interval))
    a1.to_netcdf('/media/dai/DATA1/China/1km/China_1km_prep_'+str(y)+'_0.5.nc')
print(datetime.now())


prec = []
for y in range(1979,2020):
    print(y)
    a = xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_'+str(y)+'_0.5.nc')
    prec.append(a)
prec = xr.concat(prec, dim='time')
prec.to_netcdf('/media/dai/DATA1/China/1km/China_1km_prep_0.5_1979-2019.nc')
'''
#%%

print(datetime.now())
for y in range(1979,2020):
    print(y)
    a = xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_'+str(y)+'.nc')
    interval=0.25
    a1 = a.interp(lon=np.arange(73.,135.,interval),lat=np.arange(53.,18.,-interval))
    a1.to_netcdf('/media/dai/DATA1/China/1km/China_1km_prep_'+str(y)+'_0.25.nc')
print(datetime.now())


prec = []
for y in range(1979,2020):
    print(y)
    a = xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_'+str(y)+'_0.25.nc')
    prec.append(a)
prec = xr.concat(prec, dim='time')
prec.to_netcdf('/media/dai/DATA1/China/1km/China_1km_prep_0.25_1979-2019.nc')

#%%











'''

b=a.values
b1 = a1.values

labels_in = np.ones((512, 512, 512), dtype=np.int32)
labels_out = cc3d.connected_components(labels_in) # 26-connected

c = (b>10).astype(int)
labels_out = cc3d.connected_components(c) # 26-connected

d=c.sum((1,2))
e = labels_out[2,:,:]

a2 = (a1.values>10).astype(int)
labels_out1 = cc3d.connected_components(a2)
#%%
fig = plt.figure(figsize = (17.5/2.54, 17.5/2.54)) # 宽、高
ax = plt.axes([0,0.7,0.32,0.22])
contf = ax.contourf(a1['lon'].values, a1['lat'].values,a1.mean('time'), # plot contourf for SLP anomalies
                    #transform=ccrs.PlateCarree(), #levels=Colors_limits,
                     cmap=Colors_used) 



prec = xr.open_dataset('/media/dai/DATA1/research/6.southeastern-china-EOF/data/china_daily_precipitation_1961_2018.nc')
date(1978,12,31)-date(1961,1,1)

prec=prec.sel(
    #longitude=slice(110,120),
    #latitude=slice(20,30),
    time=slice(6575,21184) )
prec1 = prec['precipitation'][:365,:,:]

#%%
fig = plt.figure(figsize = (17.5/2.54, 17.5/2.54)) # 宽、高
ax = plt.axes([0,0.7,0.32,0.22])
contf = ax.contourf(a1['lon'].values, a1['lat'].values,a1.mean('time'), # plot contourf for SLP anomalies
                    #transform=ccrs.PlateCarree(), #levels=Colors_limits,
                     cmap=Colors_used) 

ax = plt.axes([0.3,0.7,0.32,0.22])
contf = ax.contourf(prec1['longitude'].values, prec1['latitude'].values,prec1.mean('time'), # plot contourf for SLP anomalies
                    #transform=ccrs.PlateCarree(), #levels=Colors_limits,
                     cmap=Colors_used) 


p1 = a1.mean('time').values
p2 = prec1.mean('time').values


labels_out2 = cc3d.connected_components((prec1.values>10).astype(int))

#%%
fig = plt.figure(figsize = (17.5/2.54, 17.5/2.54)) # 宽、高
ax = plt.axes([0,0.7,0.32,0.22])
contf = ax.contourf(a1['lon'].values, a1['lat'].values,labels_out1[194,:,:], # plot contourf for SLP anomalies
                    #transform=ccrs.PlateCarree(), #levels=Colors_limits,
                     cmap=Colors_used) 

ax = plt.axes([0.4,0.7,0.32,0.22])
contf = ax.contourf(prec1['longitude'].values, prec1['latitude'].values,labels_out2[194,:,:], # plot contourf for SLP anomalies
                    #transform=ccrs.PlateCarree(), #levels=Colors_limits,
                     cmap=Colors_used) 
'''