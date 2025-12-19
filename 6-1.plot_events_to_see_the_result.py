#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 22:44:08 2023

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

#%%


prec = xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_0.25_1979-2019.nc')
prec=prec[(prec['time.month']>=4) & (prec['time.month']<=9),:,:]
prec = prec.where(prec>0.1) ##有雨日

p_lat = prec.lat
p_lon = prec.lon

Clustering=np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/3.EOFs/Clustering_1deg.npy',allow_pickle=True).tolist()

for r in [3]:
    for vv in ['SLP~Z500']:
        for f in [6]:
            pattern=Clustering.get('region'+str(r)+'_'+vv)['Clusters_'+str(f)]  ##找到结果
            sequence_f = pattern.value_counts().index  #最新的顺序
            count_f = pattern.value_counts()*100/len(pattern)
            
      
                
#%%
# 看每种类型的降水中心集中在什么位置
qu=0.97
ar=0.95
basic_dir ='/media/dai/DATA1/'
d3_events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
event_sum_original = d3_events_stats.get('event_sum')
#Summary_Stats = d3_events_stats.get('area_Summary_Stats')

events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
event_all_stats = events_stats.get('event_all_stats')
event_basic_stats = events_stats.get('event_basic_stats')


j=5
aa= event_basic_stats[event_basic_stats['circulation_type']==sequence_f[j]]
aa= event_basic_stats[event_basic_stats['lifespan']>=8]
#%%
for i in aa.index:
    #%
    
    print(i)
    a= pd.concat([pd.Series(event_sum_original[i][0]), 
                  pd.Series(event_sum_original[i][1]), 
                  pd.Series(event_sum_original[i][2])],axis=1)
    a.columns=['day','ilat','ilon']
    #%
    fig = plt.figure(figsize = (10.5/2.54, 10/2.54)) # 宽、高
    

    days = np.unique(a.iloc[:,0])
    for j in range(len(np.unique(a.iloc[:,0]))):
        ax = plt.axes([0.25*j,0,0.25,0.25])
        
        p = prec[days[j],:,:]
        p1=p.sel(lon=slice(100,130),lat=slice(40,18))
        ax.contourf(p1.lon,p1.lat, p1,cmap='Blues')
        
    
        '''
        #简单的权重中心
        center_lat = p_lat[a[a['day']==days[j]]['ilat'].values].mean()
        center_lon = p_lon[a[a['day']==days[j]]['ilon'].values].mean()
        ax.scatter(center_lon,center_lat,c='red',s=2,zorder=500) # 1.简单的权重中心
        '''
        p_event =np.full((140,248),fill_value=0) #圈出范围
        s= a[a['day']==days[j]]
        prec_max = [] #找出最大值
        for k in range(len(a[a['day']==days[j]])):
            p_event[s.iloc[k,1],s.iloc[k,2]]=1
            prec_max.append(p[s.iloc[k,1],s.iloc[k,2]].values)
            if np.isnan(p[s.iloc[k,1],s.iloc[k,2]].values):
                print(k)
                ax.scatter(p_lon[s.iloc[k,2]],p_lat[s.iloc[k,1]],c='red',s=0.2,zorder=500) 
        ax.contour(p.lon,p.lat, p_event,color='black',levels=[0,1,2],linewidth=0.2)     
            
            #%
        prec_max=np.array(prec_max)
        p_max_where = np.where(prec_max==prec_max.max())[0][0]
        #最大值
        ax.scatter(p_lon[s.iloc[p_max_where,2]],p_lat[s.iloc[p_max_where,1]],c='pink',s=1,zorder=500) #降水位置最大的地方
        
        #ax.contourf(p1.lon,p1.lat, p1,cmap='Blues')

        
        #第三种方式，降水量和经纬度的权重
        
        
        ax.scatter((p_lon[s['ilon'].values]*prec_max).sum()/prec_max.sum(),
                   (p_lat[s['ilat'].values]*prec_max).sum()/prec_max.sum(),
                   c='yellow',s=2,zorder=500) #降水位置最大的地方
        
        
        ax.set_xlim((100,125))
        ax.set_ylim((18,35))
        
        
        
    fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/6.plot_events/_event_"+str(i)+".png",dpi=300, bbox_inches='tight')
