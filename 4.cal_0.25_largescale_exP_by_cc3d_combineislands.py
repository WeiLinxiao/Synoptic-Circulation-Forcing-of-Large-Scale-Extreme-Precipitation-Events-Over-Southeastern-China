#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:14:50 2023

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
from math import radians,sin,cos,asin
import multiprocessing
import tqdm

#%% 
'''
Clustering=np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/3.EOFs/Clustering_1deg.npy',allow_pickle=True).tolist()

#%%

mf_v=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_moisture_flux_integraded_v_component.nc').sel(
    longitude=slice(90,140),
    latitude=slice(50,0))

mf_u=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_moisture_flux_integraded_u_component.nc').sel(
   longitude=slice(90,140),
   latitude=slice(50,0))

z500=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_Z500.nc').sel(
   longitude=slice(90,140),
   latitude=slice(50,0))/9.806


z500_anomaly=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_Z500.nc').sel(
   longitude=slice(90,140),
   latitude=slice(50,0))/9.806

slp_anomaly=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_slp.nc').sel(
   longitude=slice(90,140),
   latitude=slice(50,0))/100
'''
#%%
Colors_used = sns.color_palette('RdBu_r', n_colors=25)
#Colors_used= Colors_used[1:6]+Colors_used[7:(7+n_max)]
Colors_used = ListedColormap(Colors_used)
Colors_limits=np.arange(-12,13,1)

#%% 1.得到雨季降水（注意极端降水阈值）

prec = xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_0.25_1979-2019.nc')
prec=prec[(prec['time.month']>=4) & (prec['time.month']<=9),:,:]


#%%
## 调整一下台湾的位置
a=prec[0,:,:].values
taiwan_loc = prec[0,111:125,189:196]
taiwan_lat = taiwan_loc.lat[np.where(taiwan_loc.values>=0)[0]]
taiwan_lon = taiwan_loc.lon[np.where(taiwan_loc.values>=0)[1]]

joint_tw_lat = taiwan_lat+2.75
joint_tw_lon = taiwan_lon-0.5

for i in range(len(joint_tw_lat)):
    print(i)
    prec[:,np.where(prec.lat==joint_tw_lat[i].values)[0][0] , np.where(prec.lon==joint_tw_lon[i].values)[0][0] ] = prec.sel(lon=taiwan_lon[i].values,lat=taiwan_lat[i].values).values
    prec[:,np.where(prec.lat==taiwan_lat[i].values)[0][0] , np.where(prec.lon==taiwan_lon[i].values)[0][0] ] = np.nan

b=prec[0,:,:].values

#%% 把海南跟大陆连起来


prec[:,131,148:151 ]= prec[:,132,149:152].values
#%%

prec = prec.where(prec>0.1) ##有雨日
#%%

for qu in [0.9,0.95,0.97,0.99]:
    
    quan = prec.quantile(q=qu,dim='time') #的极端降水
    '''
    calender  = pd.to_datetime(prec.time.values).strftime('%m%d')   
    prec1 = prec.assign_coords({'time':calender})              
    prec_clim = prec1.groupby('time').mean()
    prec_anomaly = prec1.groupby('time')-prec_clim
    prec_anomaly = prec_anomaly.assign_coords({'time':prec.time})   
    p_std = prec_anomaly.std('time')
    '''
    
    #%%2.不同纬度的每个网格的面积（注意分辨率）
    
    lat = prec.lat.values
    lon = prec.lon.values
    area_lat = []
    for i in range(len(lat)):
        lon1,lat1,lon2,lat2=map(radians,[110,lat[i]+0.125,111,lat[i]-0.125])
        r=6731.393
        area_lat.append(abs(r**2 * (lon2-lon1)*(sin(lat2)-sin(lat1))))
    
    #%% 3.采用联通性方法求得大范围降水事件
    
    
    prec_if = (prec>quan).astype(int).values
    labels_out = cc3d.connected_components(prec_if) # 26-connected
    uniq,count = np.unique(labels_out,return_counts=True)
    count = count[1:] #每个事件的数量
    uniq = uniq[1:] #每个事件
    
    #%% 4.找到质心在SEC地区的事件，并计算每个事件的降水面积（注意质心范围）
    def cal_SEC_area(i):
        global labels_out,uniq,count,lon,lat
        a = np.where(labels_out == uniq[i])
        #a[1]
        b = pd.concat([pd.Series(a[0]),pd.Series(lat[a[1]]),pd.Series(lon[a[2]])],axis=1)
        #b.columns=['a','b','c']
        c= b.groupby(by=b.iloc[:,0]).mean()
        
        #print(c[0])
        #print(lon[a[1]].mean())
        if (c[2].mean()>108) &(c[2].mean()<123) & (c[1].mean()>18) &(c[1].mean()<34):
            ar = [area_lat[j] for j in np.where(labels_out == uniq[i])[1]]
            return {'event':uniq[i],'area':np.array(ar).sum()}
        #else:
        #    return {'event':np.nan,'area':np.nan}
        
        
    print(datetime.now())
    pool = multiprocessing.Pool(processes = 48) # object for multiprocessing
    Summary_Stats = list(tqdm.tqdm(pool.imap( cal_SEC_area , range(len(uniq))), 
                                   total=len(uniq), position=0, leave=True))
    pool.close()    
    del(pool)
    print(datetime.now())
    
    #%%  5.计算大范围降水事件（注意阈值）
    sec_events=[]
    sec_event_area=[]
    for i in range(len(uniq)):
        if str(type(Summary_Stats[i]))!= "<class 'NoneType'>":
            sec_events.append(Summary_Stats[i].get('event'))
            sec_event_area.append(Summary_Stats[i].get('area'))
    
    for ar in [0.8,0.9,0.95,0.97,0.99]:
        area_quan = np.quantile(np.array(sec_event_area),q=ar)
        sec_la_event = np.array(sec_events)[np.where(np.array(sec_event_area)>area_quan)]  #筛选出来的事件
        
        #%%# 6.先统计一下每一场降水的基本情况
        
        event_sum= []  
        for i in range(len(sec_la_event)):
            print(i)
            a = np.where(labels_out == sec_la_event[i])
            
            event_sum.append(a)
            
            
            
        #%%
        
        event_sum_new= []  
        #需要整理时间的位置，即放回原来的台湾位置，以及将海南那三个点删掉。
        for i in range(len(event_sum)):
            #print(i)
            a =event_sum[i]
            a_lat = a[1]
            a_lon = a[2]
            ##删除掉海南衔接的部分
            locs = np.where( (a_lat==131) & (a_lon>=148) & (a_lon<=151) )[0]
            if (len(locs)>0):
                #print(i)
                a_days1= np.delete(a[0],locs)
                a_lat1 = np.delete(a_lat,locs)
                a_lon1 = np.delete(a_lon,locs)
                a1 = (a_days1, a_lat1, a_lon1) 
            else:
                a1=a
            ##将台湾衔接的部分移回原处
            
            locs_tw=[]
            for j in range(len(joint_tw_lat)):
                lo = np.where(  (a1[1]==  np.where(prec.lat == joint_tw_lat[j].values)[0][0]) 
                              & (a1[2]==  np.where(prec.lon == joint_tw_lon[j].values)[0][0])   )[0]
                if (len(lo)>0):
                    locs_tw.extend(lo)
            if (len(locs_tw)>0):
                print(i)     
                 
                a1[1][locs_tw] = a1[1][locs_tw]+ (2.75/0.25)
                a1[2][locs_tw] = a1[2][locs_tw]+ (0.5/0.25)
                
                
            event_sum_new.append(a1)
                
                
        #%%        
        a = event_sum_new[4][1] - event_sum[4][1]
                
        
        
        #%%
        
        
            
        d3_events_stats={}
        d3_events_stats.update({'event_sum':event_sum_new})
        d3_events_stats.update({'area_Summary_Stats':Summary_Stats})
        #np.save('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/4.3d_event_indentify/d3_events_stats.npy',d3_events_stats)
        np.save('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',d3_events_stats)
        
        
    
    



#%%  需要设置一些筛选机制？比如？得看从事件哪里来到哪里去
'''
#################################################暂时还没有完成！！############################################
for i in range(len(sec_la_event)):
    #print(i)
    a = np.where(labels_out == sec_la_event[i])
    b = pd.concat([ pd.Series(a[0]),pd.Series(a[1]),pd.Series(a[2]) ],axis=1)
    b.columns = ['day','lat','lon']
    #b= b.groupby('day').mean()
    
    uni_day = np.unique(a[0])
    prec_days = pd.DataFrame()
    for ud in uni_day:
        a1=b[b['day']==ud]
        p_day = prec_anomaly[ud,:,:]
        p_s=[]
        for loc in range(len(a1)):
            p_s.append(p_day[a1.iloc[loc,1],a1.iloc[loc,2]].values)
        arg = np.argmax(np.array(p_s))
        p_s = pd.Series([ lat[a1.iloc[arg,1]],
                    lon[a1.iloc[arg,2]],
                    np.max(np.array(p_s))],index=['lat','lon','prec_ano'])
        prec_days= pd.concat([prec_days,p_s],axis=1)
    
    b=prec_days.T
    b=(b['lat']>18) & ((b['lat']<34+10)) & (b['lon']>108-10) & ((b['lon']<123))
    if (b==False).sum()>0 :
        print(i)
        print(b)
  
for i in range(len(event_sum)):
    #percent.extend(pattern[np.unique(event_sum[i][0])].values)
    
    a = pd.concat([ pd.Series(event_sum[i][0]),pd.Series(lat[event_sum[i][1]]),pd.Series(lon[event_sum[i][2]]) ],axis=1)
    a.columns = ['day','lat','lon']
    a=a.groupby('day').mean()
    b=(a['lat']>18) & ((a['lat']<34+5)) & (a['lon']>108-5) & ((a['lon']<123))
    if (b==False).sum()>0 :
        print(i)
        print(b)
        plt.plot(a['lon'],a['lat'])

  

#%%  7.查看不同环流聚类下，降水的环流一致率（用于判断哪种环流聚类效果最优的一个标准？）

types = {}
for r in range(1,5):
    for vv in ['SLP~T850~Z500','SLP~Z500']:
        for f in [4,6,8,9]:
            pattern=Clustering.get('region'+str(r)+'_'+vv)['Clusters_'+str(f)]  ##找到结果
            
            event_circ=[]
            for i in range(len(event_sum)):
               if len(np.unique(event_sum[i][0]))>0:
                   print(i)

                   a= pd.concat([pd.Series(np.repeat(i,len(pattern[np.unique(event_sum[i][0])])),index = pattern[np.unique(event_sum[i][0])].index), 
                              pattern[np.unique(event_sum[i][0])]],axis=1)
                   a.columns=['event','cluster']
                   event_circ.append(a.groupby('event').value_counts().max()/len(a))
            
            types.update({str(r)+'~'+vv+'~'+str(f):np.array(event_circ).mean()})
            
#%% 8.看下降水事件都有没有重复？即判断一日是否只有一场大范围极端降水
s = []
for i in range(len(event_sum)):
    s.extend(np.unique(event_sum[i][0]))

a,b = np.unique(np.array(s),return_counts=True)
len(np.where(b>1)[0])/len(b)  ##只有2.5%的日子出现了两场降水，所以事件基本都是独立的
''''''
##选一个例子看一下
p = prec[a[np.where(b==2)[0][20]],:,:]
plt.contourf(p.lon,p.lat,p.values)
''''''
'''
