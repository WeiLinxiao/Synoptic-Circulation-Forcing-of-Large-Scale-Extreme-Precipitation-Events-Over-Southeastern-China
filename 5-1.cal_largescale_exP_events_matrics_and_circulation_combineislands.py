#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:52:54 2023

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
import cc3d
from math import radians,sin,cos,asin,sqrt,degrees,atan,atan2,tan,acos
import multiprocessing
import tqdm
import shapely.geometry as sgeom
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


#%%
def getDegree(latA, lonA, latB, lonB): 
    """
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """ 
    radLatA = radians(latA) 
    radLonA = radians(lonA) 
    radLatB = radians(latB) 
    radLonB = radians(lonB) 
    dLon = radLonB - radLonA 
    y = sin(dLon) * cos(radLatB) 
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon) 
    brng = degrees(atan2(y, x)) 
    brng = (brng + 360) % 360 
    return brng  


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



for qu in [0.99]:
    for ar in [0.8,0.9,0.95,0.97,0.99]:
        d3_events_stats = np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
        event_sum_original = d3_events_stats.get('event_sum')
        Summary_Stats = d3_events_stats.get('area_Summary_Stats')
        #
        #%% 1.得到雨季降水（注意极端降水阈值）
        
        prec = xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_0.25_1979-2019.nc')
        prec=prec[(prec['time.month']>=4) & (prec['time.month']<=9),:,:]
        prec = prec.where(prec>0.1) ##有雨日
        quan = prec.quantile(q=0.95,dim='time') #的极端降水
        
        calender  = pd.to_datetime(prec.time.values).strftime('%m%d')   
        prec1 = prec.assign_coords({'time':calender})              
        prec_clim = prec1.groupby('time').mean()
        prec_anomaly = prec1.groupby('time')-prec_clim
        prec_anomaly = prec_anomaly.assign_coords({'time':prec.time})   
        p_std = prec_anomaly.std('time')
        
        p_lat = prec.lat
        p_lon = prec.lon
        
        a=prec_anomaly.values
        #%%
        '''
        mf_v=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_moisture_flux_integraded_v_component.nc').sel(
            longitude=slice(90,140),
            latitude=slice(50,0))
        
        mf_u=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_moisture_flux_integraded_u_component.nc').sel(
           longitude=slice(90,140),
           latitude=slice(50,0))
        
        
        ivt = np.sqrt( mf_v**2 + mf_u**2 )
        
        z500=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_Z500.nc').sel(
           longitude=slice(90,140),
           latitude=slice(50,0))/9.806
        
        slp=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_slp.nc').sel(
           longitude=slice(90,140),
           latitude=slice(50,0))/100
        
        
        z500_anomaly=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_Z500.nc').sel(
           longitude=slice(90,140),
           latitude=slice(50,0))/9.806
        
        
        t850_anomaly=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_T850.nc').sel(
           longitude=slice(90,140),
           latitude=slice(50,0))
        
        
        slp_anomaly=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_slp.nc').sel(
           longitude=slice(90,140),
           latitude=slice(50,0))/100
        
        '''
        #%%
        Clustering=np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/3.EOFs/Clustering_1deg.npy',allow_pickle=True).tolist()
        
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
                        percent_event = pd.concat([percent_event,   p2  ])
                        
        #%%
        
        '''
        # 看每种类型的降水中心集中在什么位置，选出最合适的质心方案
        for i in range(len(event_sum_original)):
            
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
                
                p = prec_anomaly[days[j],:,:]
                
                
                #%
                #简单的权重中心
                center_lat = p_lat[a[a['day']==days[j]]['ilat'].values].mean()
                center_lon = p_lon[a[a['day']==days[j]]['ilon'].values].mean()
                
                p_event =np.full((140,248),fill_value=0) #圈出范围
                s= a[a['day']==days[j]]
                prec_max = [] #找出最大值
                for k in range(len(a[a['day']==days[j]])):
                    p_event[s.iloc[k,1],s.iloc[k,2]]=1
                    prec_max.append(p[s.iloc[k,1],s.iloc[k,2]].values)
                prec_max=np.array(prec_max)
                
                p1=p.sel(lon=slice(100,130),lat=slice(40,18))
                ax.contourf(p1.lon,p1.lat, p1,cmap='Blues')
                ax.contour(p.lon,p.lat, p_event,color='black',levels=[0,1,2],linewidth=0.2)
                
                
                ax.scatter(center_lon,center_lat,c='red',s=2,zorder=500) # 1.简单的权重中心
                
                p_max_where = np.where(prec_max==prec_max.max())[0][0]
                #2 最大值
                ax.scatter(p_lon[s.iloc[p_max_where,2]],p_lat[s.iloc[p_max_where,1]],c='pink',s=2,zorder=500) #降水位置最大的地方
                  
                #3 降水量和经纬度的权重
        
                ax.scatter((p_lon[s['ilon'].values]*prec_max).sum()/prec_max.sum(),
                           (p_lat[s['ilat'].values]*prec_max).sum()/prec_max.sum(),
                           c='yellow',s=2,zorder=500) #降水位置最大的地方
        
                ax.set_xlim((100,125))
                ax.set_ylim((18,35))
        
            fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/drafting/events/event_"+str(i)+".png",dpi=300, bbox_inches='tight')
        '''
        #%%#%%2.不同纬度的每个网格的面积（注意分辨率）
        lat = prec.lat.values
        lon = prec.lon.values
        area_lat = []
        for i in range(len(lat)):
            lon1,lat1,lon2,lat2=map(radians,[110,lat[i]+0.125,111,lat[i]-0.125])
            r=6731.393
            area_lat.append(abs(r**2 * (lon2-lon1)*(sin(lat2)-sin(lat1))))
            
            
        p_area = np.full((140,248),fill_value=0) #每个网格点的面积
        for i in range(len(area_lat)):
            p_area[i,:]=area_lat[i]
            
        #%%
        #%% 3.采用联通性方法求得大范围降水事件
        '''
        prec_if = (prec>quan).astype(int).values
        labels_out = cc3d.connected_components(prec_if) # 26-connected
        uniq,count = np.unique(labels_out,return_counts=True)
        count = count[1:] #每个事件的数量
        uniq = uniq[1:] #每个事件
        
        #%%  5.计算大范围降水事件（注意阈值）
        
        sec_events=[]
        sec_event_area=[]
        for i in range(len(uniq)):
            if type(Summary_Stats[i])!= type(Summary_Stats[0]):
                sec_events.append(Summary_Stats[i].get('event'))
                sec_event_area.append(Summary_Stats[i].get('area'))
        
        
        area_quan = np.quantile(np.array(sec_event_area),q=0.95)
        sec_la_event = np.array(sec_events)[np.where(np.array(sec_event_area)>area_quan)]  #筛选出来的事件
        
        sec_la_event_area = [Summary_Stats[i-1].get('area') for i in sec_la_event] #累计面积
        '''
        #%%
        #%%经过测试，带降水值权重的结果可能更好
        ##现在诊断每场降水的基本统计要素
        
        ##累计面积、持续天数、投影面积、总降水量（面积x降水量 km2*mm）、降水强度（总降水量/总降水面积 mm/day）、移动距离、移动方向(前段时间和后段时间的质心的方向角)、移动速度（移动距离/天数 km/day）
        #
        event_all_stats = pd.DataFrame()
        for i in range(len(event_sum_original)):
            
            print(i)
            a= pd.concat([pd.Series(event_sum_original[i][0]), 
                          pd.Series(event_sum_original[i][1]), 
                          pd.Series(event_sum_original[i][2])],axis=1)
            a.columns=['day','ilat','ilon'] #得到这一场事件的所有经纬度网格
            #%
            # 1.该事件的投影面积
            proj_area = np.full((140,248),fill_value=0) #圈出范围
            for k in range(len(a)):
                proj_area[a.iloc[k,1],a.iloc[k,2]]=1
            proj_area = np.array([area_lat[np.where(proj_area==1)[0][k]] for k in range(len(np.where(proj_area==1)[0]))]).sum()
            
            
        
            days = np.unique(a.iloc[:,0]) #查看分别是哪几天
           
            for j in range(len(np.unique(a.iloc[:,0]))): 
                
                p = prec[days[j],:,:] #得到这一天的降水数据
                p_event =np.full((140,248),fill_value=0) #圈出范围
                s= a[a['day']==days[j]] #找出这一天的自己
                prec_max = [] #找出网格内的所有降水值
                
                for k in range(len(a[a['day']==days[j]])):
                    p_event[s.iloc[k,1],s.iloc[k,2]]=1 #将这个范围内的降水mask出来
                    prec_max.append(p[s.iloc[k,1],s.iloc[k,2]].values)
                prec_max=np.array(prec_max) 
                #2.这一天的降水面积、降水量 （km2*mm）
                area = (p_event* p_area ).sum()
                amounts = (p_event* p_area * p).sum().values
                # 3. 找到质心
                lon_1 = (p_lon[s['ilon'].values]*prec_max).sum()/prec_max.sum() #centriod lon
                lat_1 = (p_lat[s['ilat'].values]*prec_max).sum()/prec_max.sum() ##centriod lat
                
            
                event_all_stats = pd.concat([event_all_stats, pd.Series([i,days[j],lat_1.values,lon_1.values,proj_area,area,amounts]) ],axis=1)
                
        
        
        
        event_all_stats = event_all_stats.T
        event_all_stats.columns = ['event','day','lat','lon','projection_area','daily_area','daily_amounts']
        
        event_all_stats=pd.concat([event_all_stats,pd.Series(percent_event[0].values,index= event_all_stats.index,name='circulation_type')],axis=1)
        #以上是基本结果，可以导出大量需要的结果
        #%%
        
        event_basic_stats = pd.DataFrame()
        for i in range(len(event_sum_original)):
            a = np.unique(event_all_stats[event_all_stats['event']==i]['day'])
            if len(a) > 1:
                a = event_all_stats[event_all_stats['event']==i]
                if (len(a)%2)!=0:
                    split= int(np.floor(len(a)/2))
                    la1 =a.iloc[:split+1,2].mean()
                    la2 =a.iloc[split:len(a),2].mean()
                    lo1 =a.iloc[:split+1,3].mean()
                    lo2 =a.iloc[split:len(a),3].mean()
                    direction = getDegree(la1,lo1,la2,lo2)
                else:
                    split= int(np.floor(len(a)/2))
                    la1 =a.iloc[:split,2].mean()
                    la2 =a.iloc[split:len(a),2].mean()
                    lo1 =a.iloc[:split,3].mean()
                    lo2 =a.iloc[split:len(a),3].mean()
                    direction = getDegree(la1,lo1,la2,lo2)
                distance = []    
                for j in range(len(a)-1):
                    distance.append(getDistance(a.iloc[j,2], a.iloc[j,3], a.iloc[j+1,2], a.iloc[j+1,3]))
                distance=np.array(distance).sum()
                speed = distance/len(a)
                lifespan = len(a)
            else:
                direction = np.nan
                distance = np.nan
                speed = np.nan
                lifespan = 1
            
            event_basic_stats=pd.concat([event_basic_stats, pd.Series([lifespan,direction,distance,speed]) ] ,axis=1)
        
        event_basic_stats=event_basic_stats.T
        event_basic_stats.columns=['lifespan','direction','distance','speed']
        
        prec_amount = event_all_stats.iloc[:,:7].groupby('event').sum()['daily_amounts']
        accu_area = event_all_stats.iloc[:,:7].groupby('event').sum()['daily_area']
        proj_area = event_all_stats.iloc[:,:7].groupby('event').mean()['projection_area'] ##注意，这个是平均,因为每天的值都是一样的
        intensity = prec_amount/accu_area
        circulation =event_all_stats.groupby('event')['circulation_type'].mean()
        event_basic_stats.index=range(len(event_basic_stats))
        event_basic_stats= pd.concat([event_basic_stats,prec_amount,accu_area,proj_area,intensity,circulation],axis=1)
        
        #%%
        
        
        events_stats={}
        events_stats.update({'event_all_stats':event_all_stats})
        events_stats.update({'event_basic_stats':event_basic_stats})
            
        np.save('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',events_stats)
        
        



#%%



