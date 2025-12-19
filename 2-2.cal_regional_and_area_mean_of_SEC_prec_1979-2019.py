#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:37:34 2023

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

#%% 1. cal daily regional prec  (~25 mins)
print(datetime.now() )
p_daily = []
for y in range(1979,2020):
    print(y)
    p =xr.open_dataarray ('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_'+str(y)+'.nc')
    p_d = p.weighted(np.cos(np.deg2rad(p.lat))).mean(('lon','lat'))
    p_daily.append(p_d)

prec_daily = xr.concat(p_daily, dim='time')
prec_daily.to_netcdf('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/regional_grid_daily_prec.nc')
print(datetime.now() )
#%% 2. get exceed days  (~100 mins) 

print(datetime.now() )
for q in [0.8]:
    print(q)
    ex_th = prec_daily.quantile(q=q,dim='time')
    if_ex = (prec_daily>ex_th).astype(int).values
    day_times = prec_daily['time']
    ex_days = day_times[np.where(if_ex==1)]
    
    print(datetime.now() )
    
    ex_prec = xr.open_dataarray ('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_'+str(1979)+'.nc').sel(time=ex_days[ex_days['time.year']==1979])
    for y in range(1980,2020):
        print(y)
        print(datetime.now() )
        p =xr.open_dataarray ('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_'+str(y)+'.nc').sel(time=ex_days[ex_days['time.year']==y])
        ex_prec = xr.concat([ex_prec,p], dim='time')
    
    ex_prec.to_netcdf('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/grid_daily_ex_prec_'+str(q*100)+'.nc')
print(datetime.now() )
#%% 3. get extreme threshold of each grid (~50 mins) 
for q in [0.9,0.95,0.99]:
    prec = xr.open_mfdataset('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_*',parallel=True).sel(lon=slice(108,115))
    prec_threshold = prec.chunk(dict(time=-1)).quantile([q],dim='time',).compute()
    
    prec = xr.open_mfdataset('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_*',parallel=True).sel(lon=slice(115,123))
    prec_threshold_right = prec.chunk(dict(time=-1)).quantile([q],dim='time',).compute()
    
    prec_threshold_all = xr.concat([prec_threshold,prec_threshold_right], dim='lon')
    
    prec_threshold_all.to_netcdf('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/grid_daily_ex_prec_clim_threshold_'+str(q*100)+'.nc')

#%%
#  3. area of >1mm (~20 mins)
print(datetime.now() )
area_daily = []
for y in range(1979,2020):
    print(y)
    p =xr.open_dataarray ('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_'+str(y)+'.nc')
    p1 = p.where(p>1)
    if_prec = (p1>=1).astype(int).weighted(np.cos(np.deg2rad(p.lat))).sum(('lon','lat'))
    area_daily.append(if_prec)
prec_daily = xr.concat(area_daily, dim='time')
prec_daily.to_netcdf('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/regional_grid_daily_prec_area.nc')
print(datetime.now() )

#%%  5. area of ex prec (~50 mins)

print(datetime.now() )
for q in [0.9,0.95,0.99]:
    prec_threshold = xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/grid_daily_ex_prec_clim_threshold_'+str(q*100)+'.nc')

    th = prec_threshold.sel(quantile=q).values
    
    area_daily = []
    for y in range(1979,2020):
        print(y)
        p =xr.open_dataarray ('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_'+str(y)+'.nc')
        if_ex = (p>th).astype(int)
        b = if_ex.weighted(np.cos(np.deg2rad(p.lat))).sum(('lon','lat'))
        area_daily.append(b)
        
        
    prec_daily = xr.concat(area_daily, dim='time')
    prec_daily.to_netcdf('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/regional_grid_daily_ex_prec_area_'+str(q*100)+'.nc')
    print(datetime.now() )
    
    #%%
'''


#%%
















# 草稿
prec_daily = xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/regional_grid_daily_prec.nc')
area_ex_daily = xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/regional_grid_daily_ex_prec_area_'+str(0.9)+'.nc')
area_daily = xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/regional_grid_daily_prec_area.nc')

a = (prec_daily>prec_daily.quantile(q=0.9,dim='time')).astype(int).values
b = (area_ex_daily>area_ex_daily.quantile(q=0.9,dim='time')).astype(int).values
c = (area_daily>area_daily.quantile(q=0.9,dim='time')).astype(int).values


f =pd.concat([pd.Series(a),pd.Series(b),pd.Series(c)],axis=1)

f.sum(axis=0)

(c+a==2).sum()/1498

#%%
p =xr.open_dataarray ('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_'+str(1979)+'.nc')

p1=p[100,:,:].values

plt.contourf(p.lon,p.lat,p1)
plt.colorbar()
#%%
prec_daily = xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/regional_grid_daily_prec.nc')
area_ex_daily = xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/regional_grid_daily_ex_prec_area_'+str(0.9)+'.nc')


a = area_ex_daily.values
a1= (area_ex_daily>area_ex_daily.quantile(q=0.9,dim='time')).astype(int).values
b = (prec_daily>prec_daily.quantile(q=0.9,dim='time')).astype(int).values


c =pd.concat([pd.Series(a),pd.Series(a1),pd.Series(b),pd.Series(prec_daily.values)],axis=1)
d=p_d.values
#%


a=prec_daily.values


if_ex_prec = (ex_prec>th).astype(int)

c=b
a=p[1,:,:]
d=c/(a>=0).astype(int).weighted(np.cos(np.deg2rad(p.lat))).sum(('lon','lat')).values


b=if_ex_prec[2000,:,:].values
c=p[107,:,:].values

plt.contourf(p.lon,p.lat,c)
plt.contour(a.lon,a.lat,c)

#%%
100    
    
    
    
    
q=0.8    
ex_prec = xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/grid_daily_ex_prec_'+str(q)+'.nc')
q = 0.9
ex_th = prec_daily.quantile(q=q,dim='time')
if_ex = (prec_daily>ex_th).astype(int).values
day_times = prec_daily['time']
ex_days = day_times[np.where(if_ex==1)]
exp=ex_prec.sel(time = ex_days)

a = exp[1000,:,:].values
np.nanmax(a)
b = (a>25).astype(int)

c = pd.DataFrame(b).iloc[np.where(b==1)[0][0]:np.where(b==1)[0][-1],np.where(b==1)[1][0]:np.where(b==1)[1][-1]]
c1 = np.array(c)


plt.contourf(exp.lon,exp.lat,a)
plt.contour(exp.lon,exp.lat,b)