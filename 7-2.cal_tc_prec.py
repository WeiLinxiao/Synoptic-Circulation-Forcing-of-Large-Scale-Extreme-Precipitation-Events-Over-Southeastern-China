#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:01:59 2023

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
def getDistance(latA, lonA, latB, lonB):
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = (ra - rb) / ra  # Partial rate of the earth
    # change angle to radians
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
 
    pA = atan(rb / ra * tan(radLatA))
    pB = atan(rb / ra * tan(radLatB))
    x = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(radLonA - radLonB))
    c1 = (sin(x) - x) * (sin(pA) + sin(pB))**2 / cos(x / 2)**2
    c2 = (sin(x) + x) * (sin(pA) - sin(pB))**2 / sin(x / 2)**2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (x + dr)
    return distance/1000

#%%
basic_dir ='/media/dai/DATA1/'

Clustering=np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/3.EOFs/Clustering_1deg.npy',allow_pickle=True).tolist()


                        #%%
for qu in [0.9,0.95,0.97,0.99]:
    for ar in [0.8,0.9,0.95,0.97,0.99]:

        d3_events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
        event_sum_original = d3_events_stats.get('event_sum')
        #Summary_Stats = d3_events_stats.get('area_Summary_Stats')
        
        events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
        event_all_stats = events_stats.get('event_all_stats')
        event_basic_stats = events_stats.get('event_basic_stats')
        
        
        for r in [3]:
            for vv in ['SLP~Z500']:
                for f in [6]:
                    pattern=Clustering.get('region'+str(r)+'_'+vv)['Clusters_'+str(f)]  ##找到结果
                    sequence_f = pattern.value_counts().index  #最新的顺序
                    count_f = pattern.value_counts()*100/len(pattern)
                    
                    
                    percent_event = pd.DataFrame()
                    for i in range(len(event_sum_original)):
                        #percent.extend(pattern[np.unique(event_sum[i][0])].values)
                        p1= (pattern[np.unique(event_sum_original[i][0])])
                        p2 = pd.Series(np.repeat(p1.value_counts().index[0],len(p1)),index=p1.index)
                        percent_event = pd.concat([percent_event,   p1  ])
                                
        
        
        
        #%%
        track = np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/7.tc_track/tc_track_valid.npy',allow_pickle=True)
        track_number = np.unique(track[:,3])
        track =pd.DataFrame(track)
        track_time = [pd.Timestamp(datetime.strptime(track.iloc[i,4]+track.iloc[i,5]+track.iloc[i,6],'%Y%m%d')) for i in range(len(track))]
        track  = track.astype(float)
        track['time'] = track_time
        #%%
        
        tcp_events=pd.DataFrame()
        for i in range(len(event_basic_stats)):
            
            event_lat = event_all_stats[event_all_stats['event']==i]['lat'].astype(float)
            event_lon = event_all_stats[event_all_stats['event']==i]['lon'].astype(float)
            event_time = percent_event.index[np.where(event_all_stats['event']==i)[0]]
            event_circ = percent_event.iloc[np.where(event_all_stats['event']==i)[0]]
        
            
            if_in_time = list(map(lambda x:x in event_time,track_time)) #找到这一场事件发生时，同时有出现的气旋轨迹时间
            if sum(if_in_time)>0: #如果同期存在热带气旋
                #print(i)
                tc_n = np.unique(track.iloc[np.where(np.array(if_in_time)==True)].iloc[:,3])
                for j in tc_n:
                    tc_tk = track[track.iloc[:,3]==j] # 找到这个编号的轨迹
                    
                    tc_tk['time']
                    #将这个轨迹赋给这个
                    
                    
                    
                    
                    if_prec_tc = list(map(lambda x:x in event_time,tc_tk['time']))##tc事件中有降水的index
                    if_tc_prec = list(map(lambda x:x in tc_tk['time'].tolist(),event_time)) ##降水事件中有tc的index
                    tc_first = np.where(np.array(if_prec_tc)>0)[0][0]
                    prec_first = np.where(np.array(if_tc_prec)>0)[0][0]
                    
                    dist = []
                    for k in range(sum(if_prec_tc)):
                        dist.append(getDistance(tc_tk.iloc[tc_first+k,0],tc_tk.iloc[tc_first+k,1],event_lat.iloc[prec_first+k],event_lon.iloc[prec_first+k]))
                        
                    #where_nearest = np.argmin(np.array(dist))
                    if sum(np.array(dist)<500)>0 :
                        print('wow'+str(i))
                        
                        tcp = pd.concat([tc_tk,
                                               pd.Series(np.repeat(i,len(tc_tk)),index=tc_tk.index,name = 'tc_event'),
                                               pd.Series(np.repeat(event_circ.iloc[prec_first].values,len(tc_tk)),index=tc_tk.index,name='circulation'),
                                               ],axis=1)
                        
                        tcp_events = pd.concat([tcp_events,tcp],axis=0)
                        
                        print(event_time)
        tcp_events.iloc[:,2].groupby(tcp_events['tc_event']).mean().max()
        tcp_events.iloc[:,2].groupby(tcp_events['tc_event']).mean().min()
        
        
        #np.save('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/7.tc_track/tcp_events.npy',tcp_events)        
        tcp_events.to_csv('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/7.tc_track/tcp_events_'+str(qu)+'_'+str(ar)+'.csv')
               #%% 
                
'''               
大范围降水事件中最频繁出现的环流模式
p2 = pd.Series(np.repeat(p1.value_counts().index[0],len(p1)),index=p1.index) #将其变成那几天最主导的环流(暂时先不管这个！用p1)






    
track_in_circ = track[if_in_time]
track_in_circ_number,count  = np.unique(track_in_circ.iloc[:,3],return_counts=True)


#%%

fig = plt.figure(figsize = (17.5/2.54, 10/2.54)) # 宽、高
ax = plt.axes([0,0,1,1],projection=ccrs.PlateCarree() )
ax.coastlines(resolution='50m', linewidth=1,color='grey')
ax.set_extent((100, 150, 0, 40), ccrs.PlateCarree())
ax.set_xlim((100, 150))
ax.set_aspect('auto') ##Non-fixed aspect
for i in range(len(track_in_circ_number)):
    print(i)
    t=track_in_circ[track_in_circ.iloc[:,3]==track_in_circ_number[i]]
    if count[i]==1:
        ax.scatter(t[1],t[0])
    else:
        ax.plot(t.iloc[:,1],t.iloc[:,0])
'''