#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:11:11 2023

@author: Xinxin Wu
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

#%%
prec_climatology={}

#%% ------------------------------ parameters -----------------------------

lon_w=108
lon_e=123
lat_n=34
lat_s=18


p_annual = xr.open_dataarray(('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_annual_prec_1979_2019.nc'))

#%% 1. regional annual mean prec of gridded dataset


regional_p_annual = p_annual.weighted(np.cos(np.deg2rad(p_annual.lat))).mean(('lon','lat'))
prec_climatology.update({"regional_annual_prec_grid":np.array(regional_p_annual)}) 

annual_prec_grid = p_annual.mean('year')
prec_climatology.update({"climatology_annual_prec_all_grid_of_SEC":annual_prec_grid}) 




#%% 2. regional annual mean prec of station data

station=pd.read_csv('/media/dai/disk2/suk/Data/China_697_stations/location.csv')
lat_weighted=np.cos(np.deg2rad(station['V3'] ))

prec_station_annual_mean=pd.DataFrame()
for y in range(1979,2020):
    print(y)
    prec_station=pd.read_csv('/media/dai/disk2/suk/Data/China_697_stations/'+str(y)+'.csv')
    prec_station=prec_station.drop(columns=prec_station.columns.values[0])
    p_y=prec_station.sum(axis=1)
    prec_station_annual_mean=pd.concat([prec_station_annual_mean,p_y],axis=1) #un-weighted
    

station_SEC=station[ (station.V2>=lon_w) & (station.V2<=lon_e) & 
                    (station.V3>=lat_s) & (station.V3<=lat_n)]

##
prec_station_annual_SEC=prec_station_annual_mean[ (station.V2>=lon_w) & (station.V2<=lon_e) &
                                                 (station.V3>=lat_s) & (station.V3<=lat_n)]


X,Y=np.meshgrid(np.cos(np.deg2rad(station_SEC.V3)),np.arange(1,prec_station_annual_SEC.shape[1]+1,1))
wei=X.T

a=prec_station_annual_SEC*wei
prec_station_annual_SEC_region_mean=a.sum(axis=0)/wei.sum(axis=0)

prec_climatology.update({"regional_annual_prec_station":np.array(prec_station_annual_SEC_region_mean)})   
prec_climatology.update({"climatology_annual_prec_all_station":prec_station_annual_mean.mean(axis=1)})   
prec_climatology.update({"station_location":station})   
prec_climatology.update({"annual_prec_all_station":prec_station_annual_mean})   
prec_climatology.update({"annual_prec_SEC":prec_station_annual_SEC})   


#%% 3. find annual prec of nearest grid point 

lon = p_annual.lon.values
lat = p_annual.lat.values   
    
station_SEC=station[ (station.V2>=lon_w) & (station.V2<=lon_e) & 
                    (station.V3>=lat_s) & (station.V3<=lat_n)].iloc[:,2:4]


grid_station_nearest=pd.DataFrame()
for i in range(0,len(station_SEC)):
    slon = lon[np.argmin(np.abs((station_SEC.iloc[i,0]-lon)))]
    slat = lat[np.argmin(np.abs((station_SEC.iloc[i,1]-lat)))]
    grid_station_nearest = pd.concat( [grid_station_nearest, pd.Series([slon,slat])],axis=1)
    
prec_nearest_grid_annual_mean= pd.DataFrame()
for i in range(0,len(station_SEC)):
    p1 = p_annual.sel(lon = grid_station_nearest.iloc[0,i],lat=grid_station_nearest.iloc[1,i]).values
    prec_nearest_grid_annual_mean=pd.concat([prec_nearest_grid_annual_mean,pd.Series(p1)],axis=1)


prec_climatology.update({"annual_prec_nearest_grid":prec_nearest_grid_annual_mean.T})   

#%%

prec_climatology
np.save('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/2.SEC_prec_climatology_and_trend/prec_climatology.npy',prec_climatology)
