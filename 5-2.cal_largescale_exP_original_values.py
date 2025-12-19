#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:44:05 2023

@author: dai
"""


import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from global_land_mask import globe
from datetime import datetime
from datetime import date
from datetime import timedelta
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages # for multipages pdf
from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib
import matplotlib.lines as mlines
import pymannkendall as mk
#import cc3d
from math import radians,sin,cos,asin,sqrt,degrees,atan,atan2,tan,acos
#import multiprocessing
#import tqdm
import shapely.geometry as sgeom
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import tqdm
import multiprocessing

import os
os.chdir('/media/dai/DATA1/research/6.southeastern-china-EOF/precip_extremes_scaling-master/precip_extremes_scaling-master')
from precip_extremes_scaling import scaling
#%%
basic_dir ='/media/dai/DATA1/'

prec_annual = xr.open_dataarray(basic_dir+'China/1km/China_1km_prep_0.25_1979-2019.nc')
prec=prec_annual[(prec_annual['time.month']>=4) & (prec_annual['time.month']<=9),:,:]
prec = prec.where(prec>0.1) ##有雨日
prec_annual = prec_annual.where(prec_annual>0.1) ##有雨日

a=(prec.sum('time')/prec_annual.sum('time')).sel(lon=slice(108,123),lat=slice(34,18))
a=a.weighted(np.cos(np.deg2rad(a.lat))).mean(('lon','lat'))

#%%

for qu in [0.99]:
    for ar in [0.95,0.97,0.99]:
        d3_events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
        event_sum_original = d3_events_stats.get('event_sum')
        
        large_scale_prec = np.full(shape=prec.shape,fill_value=np.nan )
        for i in range(len(event_sum_original)):
            print(i)
            a= event_sum_original[i]
            for j in  range(len(a[0])):
                large_scale_prec [ a[0][j],a[1][j],a[2][j] ] = prec [ a[0][j],a[1][j],a[2][j] ].values
        
        large_scale_exP = prec.copy(deep=True)
        large_scale_exP.values = large_scale_prec
        a = large_scale_exP.mean('time')
        a.max()
        b=(large_scale_exP>0).sum('time')
        b=b.where(b>0)
        b.max()
        
        
        ####发生频率
        large_scale_freq = np.full(shape=prec.shape,fill_value=np.nan )
        for i in range(len(event_sum_original)):
            print(i)
            a= event_sum_original[i]
            for j in  range(len(a[0])):
                large_scale_freq [ a[0][j],a[1][j],a[2][j] ] = i
        
        large_scale_exp_freq = prec.copy(deep=True)
        large_scale_exp_freq.values = large_scale_freq
        
        
        
        #%%
        '''
        Colors_used = sns.color_palette('Blues', n_colors=20)#[2:22]
        #Colors_used= ['white'] +Colors_used
        #Colors_used = ListedColormap(Colors_used)
        
        #%
        #Colors_limits = np.concatenate([np.array([-20]),np.arange(1,10,1),np.array([40])])
        Colors_limits = np.arange(0,200,10)
        #Colors_limits = np.arange(-100,101,10)
        fig = plt.figure(figsize = (17.5/2.54, 7/2.54)) # 宽、高
        
        ax = plt.axes([0,0,0.25,1],projection=ccrs.PlateCarree() )
        cf=ax.contourf(a.lon,a.lat,a ,
                       levels=Colors_limits,
                       colors=Colors_used,
                       extend='both',
                       #cmap='RdBu_r',
                       projection=ccrs.PlateCarree() )
        
        ax.coastlines(resolution='50m', linewidth=0.6)
        '''
        #%%
        
        large_scale_exP.to_netcdf('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/large_scale_exP_values_'+str(qu)+'_'+str(ar)+'.nc')
        
        large_scale_exp_freq.to_netcdf('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/large_scale_exP_frequency_'+str(qu)+'_'+str(ar)+'.nc')
