#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:27:09 2023

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
basic_dir ='/media/dai/DATA1/'
#basic_dir ='E:'
Clustering=np.load(basic_dir+'research/6.Southeastern_China_Clustering/code2/3.EOFs/Clustering_1deg.npy',allow_pickle=True).tolist()

#%%
mf_v=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_moisture_flux_integraded_v_component.nc').sel(
    longitude=slice(100,140),
    latitude=slice(40,10))

mf_u=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_moisture_flux_integraded_u_component.nc').sel(
   longitude=slice(100,140),
   latitude=slice(40,10))

z500=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_Z500.nc').sel(
   longitude=slice(100,140),
   latitude=slice(40,10))/9.806

Area_used = {'region1': [40, 95, 10, 135], 
             'region2': [40, 100, 10, 140], 
             'region3': [40, 100, 10, 130],
             'region4': [40, 90, 10, 140],}

#%%




prec = xr.open_dataarray(basic_dir+'China/1km/China_1km_prep_0.25_1979-2019.nc')
prec=prec[(prec['time.month']>=4) & (prec['time.month']<=9),:,:]
prec = prec.where(prec>0.1) ##有雨日

#quan = prec.quantile(q=0.95,dim='time') #的极端降水

calender  = pd.to_datetime(prec.time.values).strftime('%m%d')   
prec1 = prec.assign_coords({'time':calender})              
prec_clim = prec1.groupby('time').mean()
prec_anomaly = prec1.groupby('time')-prec_clim
prec_anomaly = prec_anomaly.assign_coords({'time':prec.time})   
p_std = prec_anomaly.std('time')

prec_sec = prec.sel(lon=slice(108,124),lat=slice(34,18))
prec_sec = prec_sec.weighted(np.cos(np.deg2rad(prec_sec.lat))).mean(('lon','lat'))
prec_sec_quan = prec_sec.quantile(q=0.95,dim='time')
prec_sec = (prec_sec>prec_sec_quan).astype(float)


#%%
for qu in [0.99]:
    for ar in [0.95]:

        
        d3_events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
        event_sum_original = d3_events_stats.get('event_sum')
        #Summary_Stats = d3_events_stats.get('area_Summary_Stats')
        
        events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
        event_all_stats = events_stats.get('event_all_stats')
        event_basic_stats = events_stats.get('event_basic_stats')
         
        tcp_events = pd.read_csv('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/7.tc_track/tcp_events_'+str(qu)+'_'+str(ar)+'.csv',index_col=0)
        '''
        len(np.unique(tcp_events['tc_event']))
        17/48
        '''
        #%%
        
        for r in [3]:
            for vv in ['SLP~Z500']:
                for f in [6]:
                    pattern=Clustering.get('region'+str(r)+'_'+vv)['Clusters_'+str(f)]  ##找到结果
                    
                    sequence_f = pattern.value_counts().index  #最新的顺序
                    count_f = pattern.value_counts()*100/len(pattern)
                    
                    
                    percent_event = pd.DataFrame()
                    for i in range(len(event_sum_original)):
                        #percent.extend(pattern[np.unique(event_sum[i][0])].values)
                        p1= (pattern[np.unique(event_sum_original[i][0])])  #这个事件里面那几天的环流
                        p2 = pd.Series(np.repeat(p1.value_counts().index[0],len(p1)),index=p1.index) #将其变成那几天最主导的环流(暂时先不管这个！用p1)
                        percent_event = pd.concat([percent_event,   p1  ]) #每个事件统一一种环流类型
                    
                    
                    count_events=[]  #计算每种环流模式里面一共有多少天有这种大尺度降水
                    for j in range(f):     
                        a=len(percent_event.iloc[np.where(percent_event.values==j)[0]])*100/len(pattern[pattern==j])
                        count_events.append(a)
                        
                    
                        
                    '''
                    for j in range(f): #所有事件的时间里面
                        time = percent_event.iloc[np.where(percent_event.values==sequence_f[j])[0]].index  #事件发生的时间
                        print(sequence_f[j])
                        
                        # 1. 查看每种环流型发生月份，
                        times = pd.DataFrame()  
                        for y in range(1979,2020):
                            for m in range(4,10):
                                t = time[(time.year == y)&(time.month==m)]
                                times = pd.concat([times,pd.Series([y,m,len(t)])],axis=1) 
                        
                        times = times.T
                        times.columns=['year','month','times']
                        
                        times.groupby('month').sum()
                        
                        
                        
                    ex_event = pd.concat([pattern,pd.Series(prec_sec,index=pattern.index,name='if_ex')],axis=1)
                    ex_event = ex_event[ex_event['if_ex'] ==1]
                    '''
        #%%
        Colors_used = sns.color_palette('Spectral_r', n_colors=20)#[2:22]
        #Colors_used= Colors_used[:10] +['white']+['white'] +Colors_used[10:]
        Colors_used = ListedColormap(Colors_used)
        
        colors_month = ['#7d58ad','#5a82af','#ecc400','#e3770c','#b9484e','#ae5041']
        
        
        Colors_slp = sns.color_palette('Blues', n_colors=11)#[2:22]
        #Colors_used= Colors_used[:10] +['white']+['white'] +Colors_used[10:]
        #Colors_slp = ListedColormap(Colors_slp)
        
        #slp_max = tcp_events.iloc[:,2].max()
        #slp_min = tcp_events.iloc[:,2].min()
        
        slp_max = tcp_events.iloc[:,2].groupby(tcp_events['tc_event']).mean().max()
        slp_min = tcp_events.iloc[:,2].groupby(tcp_events['tc_event']).mean().min()
        #%% 
        f=6
        
        y_loc = [0.5 if l<3 else 0 for l in range(6)]
        
        
        x_loc =[]
        for l in range(6):
            if l in [0,3]:
                x_loc.append(0)
            elif l in [1,4]:
                x_loc.append(0.36)
            else:
                x_loc.append(0.72)
        
        title_aux = list(map(chr, range(97, 123)))[:8]
        
        x_length = 0.34
        y_length = 0.45
        
        #%%
        fig = plt.figure(figsize = (17.5/2.54, 10/2.54)) # 宽、高
        
        Colors_limits = np.concatenate([np.arange(-4,15,1)])
        import warnings
        warnings.filterwarnings('ignore')
        for j in range(f):
           
            ax = plt.axes([x_loc[ j  ],y_loc[  j  ], x_length, y_length],projection=ccrs.PlateCarree() )
            #print(sequence_f[j])
            time = percent_event.iloc[np.where(percent_event.values==sequence_f[j])[0]].index
            #time = ex_event.iloc[np.where(ex_event['Clusters_6'].values==sequence_f[j])[0]].index
            
        
            
            a1,b1=np.unique(time.month.values,return_counts=True)
            p1 =  prec_anomaly.sel(lon=slice(100,130),lat=slice(40,15)).sel(time =time).mean('time')
            print(p1.max())
            #p1= p1.where( (p1>p_std) & (p1<-p_std) )
            mf_u_mean = mf_u.sel(longitude=slice(100,130),latitude=slice(40,15)).sel(time =time).mean('time')
            mf_v_mean = mf_v.sel(longitude=slice(100,130),latitude=slice(40,15)).sel(time =time).mean('time')
            z500_mean = z500.sel(longitude=slice(100,130),latitude=slice(40,15)).sel(time =time).mean('time')
            z500_mean = z500_mean.interp(longitude=np.arange(100,131,1),latitude = np.arange(40,14,-1))
            
            
            ivt = np.sqrt((mf_v_mean)**2 + (mf_u_mean)**2).values
            ivt = (ivt>50).astype(float)
            ivt[ivt==0]=np.nan    
            mf_u_mean = mf_u_mean *ivt
            mf_v_mean = mf_v_mean *ivt
            
            #
            #ccrs.AlbersEqualArea(central_longitude=lon1+(lon2-lon1)/2, central_latitude=lat1+(lat2-lat1)/2, standard_parallels=(30, 40))
            #a=(z500_mean>5880).astype(float).values
            #a[a==0]=np.nan
            #b=a.values
            ivtq=ax.quiver(  mf_u_mean['longitude'].values, mf_u_mean['latitude'].values, 
                         mf_u_mean.values,mf_v_mean.values,
                         transform = ccrs.PlateCarree(),
                         headwidth=4,
                         color='black',zorder=100,scale_units = 'inches',
                         angles = 'uv', regrid_shape = 80,scale=900)
            cf=ax.contourf(p1.lon, p1.lat,p1.values,cmap=Colors_used,
                        transform=ccrs.PlateCarree(),
                        levels=Colors_limits,
                        extend='both'
                        )
            ax.contour(z500_mean['longitude'].values, z500_mean['latitude'].values, z500_mean,colors='black',
                       levels=[5880]
                       )
            '''
            for i in range(len(event_track_type)):
                if event_track_type[i]==j:
                    
                    
                    print(i)
                    if (event_track[i]['lat'].values[0]>40):
                        print(event_track[i]['lat'].values[0])
                    ax.plot(event_track[i]['lon'].values,event_track[i]['lat'].values,linewidth=0.1)
                    
                    if len(event_track[i])>1:
                        print(i)
                        ax.arrow(event_track[i]['lon'].values[0],event_track[i]['lat'].values[0],
                                   event_track[i]['lon'].values[-1]-event_track[i]['lon'].values[0],event_track[i]['lat'].values[-1]-event_track[i]['lat'].values[0],
                                   transform = ccrs.PlateCarree(),
                                   linewidth=0.1,
                                   #length_includes_head=True,
                                   color='red')
            '''
            
          
            tc_e = np.unique(tcp_events[tcp_events['circulation']==sequence_f[j]]['tc_event'])
            for k in tc_e:
                tc_lat = tcp_events[tcp_events['tc_event']==k].iloc[:,0]
                tc_lon = tcp_events[tcp_events['tc_event']==k].iloc[:,1]
                
                rank_slp = np.floor((tcp_events[tcp_events['tc_event']==k].iloc[:,2].mean() - slp_min)*10/(slp_max - slp_min)).astype(int)
                c_slp  = Colors_slp [ rank_slp  ] 
                ax.plot(tc_lon,tc_lat,c = c_slp,linestyle='--',linewidth=1,label='TC track')
                #print(np.floor((tcp_events[tcp_events['tc_event']==k].iloc[:,2].mean() - slp_min)*10/(slp_max - slp_min)).astype(int)  )
            print(len(tc_e))
                    
            # 4. 研究区域shp和边框E:/research/6.Southeastern_China_Clustering
            china = shpreader.Reader(basic_dir + '/research/6.Southeastern_China_Clustering/data/shp/china-shapefiles-master/china-shapefiles-master/china_country.shp').geometries()
            ax.add_geometries(china, ccrs.PlateCarree(), 
                               facecolor='none', edgecolor='k', linewidth=0.5, zorder=10) # 添加中国边界
            #ax.stock_img()
            ax.coastlines(resolution='50m', linewidth=0.5,color='grey')
            
          
            x2=[108,123,123,108,108]
            y2=[18,18,34,34,18]
            plot_geometries= sgeom.LineString(zip(x2,y2))
            ax.add_geometries([plot_geometries],ccrs.PlateCarree(),facecolor='none', edgecolor='#5a82af',lw=1)#add_geometries绘制方法
            
            
            rectangle = Area_used.get('region'+str(r))
            x2=[rectangle[1],rectangle[3],rectangle[3],rectangle[1],rectangle[1]]
            y2=[rectangle[2],rectangle[2],rectangle[0],rectangle[0],rectangle[2]]
            plot_geometries1= sgeom.LineString(zip(x2,y2))
            ax.add_geometries([plot_geometries1],ccrs.PlateCarree(),facecolor='none', edgecolor='#5a82af',lw=1,ls=":")
            
            
            
            
            ax.coastlines(resolution='50m', linewidth=0.6)
            ax.set_extent((100, 130, 15, 40), ccrs.PlateCarree())
            ax.set_xlim((100, 130))
            ax.set_aspect('auto') ##Non-fixed aspect
        
            ax.set_title(r"$\bf{("+title_aux[j]+")}$ Cl. "+ str(j+1)+" ("+str(np.round(count_events[ sequence_f[j] ],1))+"%)", 
                             pad=4, size=8.5, loc='left')
        
        
            
        
        
        
            ax1 = plt.axes([x_loc[ j  ],y_loc[  j  ],0.03,y_length])
            #计算每个月发生的次数
            month = time.month
            e_0,e_count_0=np.unique(month,return_counts=True)
            e=[]
            e_count=[]
            for i in [4,5,6,7,8,9]:
                if i in e_0:
                    e_count.append(e_count_0[np.where(e_0==i)][0])
                    e.append(e_0[np.where(e_0==i)][0])
                else:
                    e_count.append(0)
                    e.append(i)
                
            
            labels=['Apr','May','Jun','Jul','Aug','Sept']
            sum_e_count =0
            where_max = np.argmax(e_count) #发生最多的是在哪俩个月
            where_max2 =np.where(e_count==sorted(e_count)[-2])[0][0]
           
            for i in range(6): #堆积图
                ax1.bar(0, e_count[i],bottom=sum_e_count ,color = colors_month[i],alpha=0.5,edgecolor= colors_month[i],label=labels[i])
                #找到发生最频繁的那个月，标记label
                if i == where_max:
                    loc_count = ((sum(e_count[:where_max])+ e_count[where_max]/2) )/sum(e_count) #把位置放在那个月中间
                    
                    if e_count[i]>100:
                        loc_x=0.1
                    else:
                        loc_x=0.17
                    ax1.annotate(e_count[i], xy=(loc_x, loc_count-0.02), xycoords="axes fraction",c='black',size=7) #
                    
                #找到发生最频繁的第二个月，标记label
                if i == where_max2:
                    loc_count = ((sum(e_count[:where_max2])+ e_count[where_max2]/2) )/sum(e_count ) #把位置放在那个月中间
                    if e_count[i]>100:
                        loc_x=0.1
                    else:
                        loc_x=0.17
                    ax1.annotate(e_count[i], xy=(loc_x, loc_count-0.02), xycoords="axes fraction",c='black',size=7) #
                sum_e_count=sum_e_count+e_count[i]
                
                
            ax1.margins(x=0,y=0)
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            if j==5:
                plt.legend( bbox_to_anchor=(12, -0.06),ncol=6, framealpha=0,fontsize=7,columnspacing=0.4, labelspacing=0.3,)
        
            if j==5:
                a=ax.get_position()
                pad=0.06
                height=0.03
                ax_f = fig.add_axes([  x_loc[0],  a.ymin - pad,  0.5 , height ]) #长宽高
                cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
                cb.set_ticklabels(Colors_limits,size=6.5)
                cb.outline.set_linewidth(0.8)
                ax_f.tick_params(length=1,pad=0.2)
                cb.set_label( label = "Anomaly Prec (mm)",fontdict={'size':7},labelpad=1)
                
                qk = ax.quiverkey(ivtq,  0.57,a.ymin - 0.015, 100, r'100 $kg·m^{-1} s^{-1}$', labelpos='S',
                      coordinates='figure',zorder=200,
                      fontproperties={'size':7.5})
                     
            ## 画核密度    
            ax2 = plt.axes([x_loc[j]+x_length*((108-100)/(130-100)),
                            y_loc[j]+((34-15)/(40-15))*y_length,
                            x_length*((123-108)/(130-100)),
                            0.05])
            event_days = event_all_stats.iloc[np.where(percent_event.values==sequence_f[j])[0]]['lon']
            sns.kdeplot(event_days.astype(float),shade=True,color='orange')
            ax2.set_xlim(108,123)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)   
            ax2.spines['left'].set_visible(False)  
            ax2.set_facecolor('none')
            ax2.xaxis.set_ticks([])
            ax2.xaxis.set_ticklabels([])
        
            ax2.yaxis.set_ticks([])
            ax2.yaxis.set_ticklabels([])
            
            ax2.set_ylabel(' ')
            ax2.set_xlabel(' ')
        
            #ax2.set_extent((100, 130, 15, 40), ccrs.PlateCarree())
            
            ax3 = plt.axes([x_loc[j] + x_length*((123-100)/(130-100)),
                            y_loc[j] + y_length*((18-15)/(40-15)),
                            0.05,
                            y_length*((34-18)/(40-15)),])
            event_days = event_all_stats.iloc[np.where(percent_event.values==sequence_f[j])[0]]['lat']
            sns.kdeplot(event_days.astype(float),vertical=True,shade=True,color='orange')
            ax3.set_ylim(18,34)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)   
            ax3.spines['bottom'].set_visible(False)  
            ax3.set_facecolor('none')
            ax3.xaxis.set_ticks([])
            ax3.xaxis.set_label([])
            ax3.yaxis.set_ticks([])
            
            ax3.set_ylabel(' ')
            ax3.set_xlabel(' ')
        fig.savefig('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/Figures/Figure.3_'+str(qu)+'_'+str(ar)+'.png',dpi=1500, bbox_inches='tight')