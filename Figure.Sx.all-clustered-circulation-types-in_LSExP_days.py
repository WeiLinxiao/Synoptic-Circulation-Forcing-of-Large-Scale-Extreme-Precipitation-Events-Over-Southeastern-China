#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:49:28 2023

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
from math import radians,sin,cos,asin,sqrt
import multiprocessing
import tqdm
import shapely.geometry as sgeom
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
basic_dir ='/media/dai/DATA1/'
#basic_dir ='E:'
#%% 
Area_used = {'region1': [40, 95, 10, 135], 
             'region2': [40, 100, 10, 140], 
             'region3': [40, 100, 10, 130],
             'region4': [40, 90, 10, 140],}

lonl=108
lonr=123
latt=34
latb=18


Colors_used = sns.color_palette('RdBu_r', n_colors=24)[2:22]
Colors_used= Colors_used[:10] +['white']+['white'] +Colors_used[10:]
Colors_used_slp = ListedColormap(Colors_used)
Colors_limits_1=[-5]+[i for i in np.arange(-3.6,3.7,.4)+0.001 ]+[5]
Colors_limits_slp = [np.round(i,1) for i in Colors_limits_1]



Colors_used = sns.color_palette('PuOr', n_colors=24)[2:22]
#Colors_used= Colors_used[:10] +['white']+['white'] +Colors_used[10:]
Colors_used_regional = ListedColormap(Colors_used)

colors = ['#7d58ad','#016041','#ecc400','#e3770c','#b9484e','#ae5041','#5a82af']
colors = ['#7d58ad','#5a82af','#ecc400','#e3770c','#b9484e','#ae5041']


title_aux = list(map(chr, range(97, 123)))[:8]
'''
Colors_used_z500 = sns.color_palette('RdBu_r', n_colors=12)
Colors_used_z500= Colors_used_z500[:6] +['white'] +Colors_used_z500[6:]
Colors_used_z500 = ListedColormap(Colors_used_z500)
'''
Colors_used_z500 = sns.color_palette('RdBu_r', n_colors=12)
Colors_used_z500= Colors_used_z500[:4] +['white'] +Colors_used_z500[8:]
Colors_used_z500 = ListedColormap(Colors_used_z500)


Colors_used_prec = sns.color_palette('OrRd', n_colors=16)
Colors_used_prec= Colors_used_prec[2:]
Colors_used_prec = ListedColormap(Colors_used_prec)

#%% 
Clustering=np.load(basic_dir+'research/6.Southeastern_China_Clustering/code/3.EOFs/Clustering_1deg.npy',allow_pickle=True).tolist()

#%%
mf_v=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_moisture_flux_integraded_v_component.nc')
mf_u=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_moisture_flux_integraded_u_component.nc')

z500=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_Z500.nc')/9.806

slp=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_slp.nc')/100


z500_anomaly=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_Z500.nc')/9.806


t850_anomaly=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_T850.nc').sel(
   longitude=slice(90,140),
   latitude=slice(50,0))


slp_anomaly=xr.open_dataarray(basic_dir+'research/6.Southeastern_China_Clustering/data/ERA5_variables/smoothed_anomaly_1979_2019_slp.nc')/100

#ivt = sqrt(mf_u**2 + mf_v**2)
#%% 1.得到雨季降水（注意极端降水阈值）

prec = xr.open_dataarray(basic_dir+'China/1km/China_1km_prep_0.25_1979-2019.nc')
prec=prec[(prec['time.month']>=4) & (prec['time.month']<=9),:,:]
prec = prec.where(prec>0.1) ##有雨日
quan = prec.quantile(q=0.95,dim='time') #的极端降水

calender  = pd.to_datetime(prec.time.values).strftime('%m%d')   
prec1 = prec.assign_coords({'time':calender})              
prec_clim = prec1.groupby('time').mean()
prec_anomaly = prec1.groupby('time')-prec_clim
prec_anomaly = prec_anomaly.assign_coords({'time':prec.time})   
p_std = prec_anomaly.std('time')

#%%
def cal_everage_condition(time):
    
    t850_mean = t850_anomaly.sel(time =time).sel(longitude=slice(lonl,lonr),latitude=slice(latt,latb))
    lats = t850_mean.latitude
    t850_mean = t850_mean.weighted(np.cos(np.deg2rad(lats))).mean(('longitude','latitude'))
    mf_u_mean = mf_u.sel(time =time).sel(longitude=slice(lonl,lonr),latitude=slice(latt,latb)).weighted(np.cos(np.deg2rad(lats))).mean(('longitude','latitude'))
    mf_v_mean = mf_v.sel(time =time).sel(longitude=slice(lonl,lonr),latitude=slice(latt,latb)).weighted(np.cos(np.deg2rad(lats))).mean(('longitude','latitude'))
    z500_mean = z500_anomaly.sel(time =time).sel(longitude=slice(lonl,lonr),latitude=slice(latt,latb)).weighted(np.cos(np.deg2rad(lats))).mean(('longitude','latitude'))
    slp_mean = slp_anomaly.sel(time =time).sel(longitude=slice(lonl,lonr),latitude=slice(latt,latb)).weighted(np.cos(np.deg2rad(lats))).mean(('longitude','latitude'))
    prec_box=prec_anomaly.sel(lon=slice(108,123),lat=slice(34,18))
    prec_box=prec_box.weighted(np.cos(np.deg2rad(prec_box.lat))).mean(('lon','lat')).sel(time=time)
    

    a = pd.concat([pd.Series(z500_mean),
                   pd.Series(slp_mean),
                   pd.Series(t850_mean),
                   pd.Series( np.sqrt( mf_u_mean **2 + mf_v_mean **2  ) ),
                  
                   pd.Series(prec_box)
                   ], axis=1)
    a.columns=['z500','slp','t850','ivt','prec']
    a.index = time

   
    t850_interval = np.arange(-7,7,0.5)
    slp_interval = np.arange(-9.,9,0.5)
    z500_interval = np.arange(-62,63,5)

    sz500 =  [z500_interval[np.where((a.iloc[i,0]>z500_interval)==1)[0][-1]] for i in range(len(a))]
    sslp =  [np.round(slp_interval[np.where((a.iloc[i,1]>slp_interval)==1)[0][-1]],1) for i in range(len(a))]
    st850 =  [np.round(t850_interval[np.where((a.iloc[i,2]>t850_interval)==1)[0][-1]],1) for i in range(len(a))]
  
    b=  pd.concat([pd.Series(sz500), 
                   pd.Series(sslp), 
                   pd.Series(st850), 
                   pd.Series(t850_mean), 
                   pd.Series(prec_box)],axis=1)

    b.columns=['z500','slp','t850','original_t850','prec']
    
    intervals = pd.DataFrame()
    for i in np.unique(sz500):
        for s in np.unique(sslp):
            p_value = b.loc[ (b['z500']==i) & (b['slp']==s)].mean()['original_t850']
            intervals=pd.concat([intervals,  pd.Series([i,s,p_value]) ],axis=1)
    intervals = intervals.T    
    intervals.columns=['z500','slp','t850']  

    intervals_prec = pd.DataFrame()
    for i in np.unique(sz500):
        for s in np.unique(sslp):
            p_value = b.loc[ (b['z500']==i) & (b['slp']==s)].mean()['prec']
            intervals_prec=pd.concat([intervals_prec,  pd.Series([i,s,p_value]) ],axis=1)
    intervals_prec = intervals_prec.T    
    intervals_prec.columns=['z500','slp','prec']  

    intervals_prec = intervals_prec[intervals_prec['prec']>0]
    return intervals,intervals_prec

#%%

y_loc = [0.5 if l<3 else 0 for l in range(6)]


x_loc =[]
for l in range(6):
    if l in [0,3]:
        x_loc.append(0)
    elif l in [1,4]:
        x_loc.append(0.36)
    else:
        x_loc.append(0.72)
        
#%%

qu=0.99
ar=0.95
d3_events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
event_sum_original = d3_events_stats.get('event_sum')
#Summary_Stats = d3_events_stats.get('area_Summary_Stats')

events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
event_all_stats = events_stats.get('event_all_stats')
event_basic_stats = events_stats.get('event_basic_stats')
 
tcp_events = pd.read_csv('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/7.tc_track/tcp_events_'+str(qu)+'_'+str(ar)+'.csv',index_col=0)

#%%

for r in [3]:
    for vv in ['SLP~Z500']:
        for f in [6]:
            pattern=Clustering.get('region'+str(r)+'_'+vv)['Clusters_'+str(f)]  ##找到结果
            
            percent_event = pd.DataFrame()
            for i in range(len(event_sum_original)):
                #percent.extend(pattern[np.unique(event_sum[i][0])].values)
                p1= (pattern[np.unique(event_sum_original[i][0])])
                p2 = pd.Series(np.repeat(p1.value_counts().index[0],len(p1)),index=p1.index)
                
                percent_event = pd.concat([percent_event,   p1  ])
            
            
            sequence_f = pattern.value_counts().index  #最新的顺序
            count_f = pattern.value_counts()*100/len(pattern)
            
            count_events=[]  #计算每种环流模式里面一共有多少天有这种大尺度降水
            for j in range(f):     
                a=len(percent_event.iloc[np.where(percent_event.values==j)[0]])*100/len(pattern[pattern==j])
                count_events.append(a)
                

#%%

fig = plt.figure(figsize = (17.5/2.54, 10/2.54)) # 宽、高

for j in [0,1,2,3,4,5]:

    ax = plt.axes([x_loc[ j  ],y_loc[  j  ],0.34,0.42],projection=ccrs.PlateCarree() )
    time = percent_event.iloc[np.where(percent_event.values==sequence_f[j])[0]].index  ##？？法
    #time = pattern.iloc[np.where(pattern.values==sequence_f[j])[0]].index  #circu发生的事件
    print(sequence_f[j])
    
    # 1. 查看每种环流型发生月份，
    times = pd.DataFrame()  
    for y in range(1979,2020):
        for m in range(4,10):
            t = time[(time.year == y)&(time.month==m)]
            times = pd.concat([times,pd.Series([y,m,len(t)])],axis=1) 
    
    times = times.T
    times.columns=['year','month','times']
    points_month = np.tile([4,5,6,7,8,9],41)
    arr=points_month
    stdev = .025 * (max(points_month) - min(points_month))
    arr1 = points_month + np.random.randn(len(points_month)) * stdev
    
    # 2. 每种环流型的水汽和大气状态
    mf_u_mean = mf_u.sel(time =time).mean('time')
    mf_v_mean = mf_v.sel(time =time).mean('time')
    z500_mean = z500_anomaly.sel(time =time).mean('time')
    slp_mean = slp_anomaly.sel(time =time).mean('time')
    t850_mean = t850_anomaly.sel(time =time).mean('time')
    
    z500_ori_mean = z500.sel(time =time).mean('time')
    
    ivt = np.sqrt((mf_v_mean)**2 + (mf_u_mean)**2).values
    ivt = (ivt>5).astype(float)
    ivt[ivt==0]=np.nan    
    mf_u_mean = mf_u_mean *ivt
    mf_v_mean = mf_v_mean *ivt
    
    # 3. 水汽大气状态
    ivtq = ax.quiver(  mf_u['longitude'].values, mf_u['latitude'].values, 
                 mf_u_mean.values,mf_v_mean.values,
                 transform = ccrs.PlateCarree(),
                 headwidth=4,
                 color='black',zorder=100,scale_units = 'inches',
                 angles = 'uv', regrid_shape = 70,scale=1200)
    
    
    cf = ax.contourf(slp_mean['longitude'].values, slp_mean['latitude'].values, 
                slp_mean.values,cmap=Colors_used_slp,
                transform=ccrs.PlateCarree(),
                levels=Colors_limits_slp)
    
    cont = ax.contour(z500_mean['longitude'].values, z500_mean['latitude'].values, 
               z500_mean.values,
               extend='both',
               cmap=Colors_used_z500,
               transform=ccrs.PlateCarree(),
               levels=[-40,-30,-20,-10,0,10,20,30,40],#extend='both'
               #levels=[5800,5840,5860],
               #levels=[-30,-25,-20,-15,-10,-2,2,10,15,20,25,30],#extend='both'
               )
    
    
    
    
    '''
    ax.clabel(cont, inline=1, fontsize=6, fmt='%d')
    for line, lvl in zip(cont.collections, cont.levels):
        #print(lvl)
        if lvl == 0:
            line.set_linestyle(':')
    '''        
    #副高线
    cont1=ax.contour(z500_ori_mean['longitude'].values, z500_ori_mean['latitude'].values,
           (z500_ori_mean).values,
           colors='black',
           levels=[5880],
           )        
    clabels = ax.clabel(cont1, inline=True, fontsize=6, fmt='%d',
                        colors='black',
                        #inline_spacing=5,use_clabeltext=True,manual=False,rightside_up=True,
                        #bbox=dict(facecolor='white',edgecolor='none',pad=0)
                        #inline_bgedge='#ffffff'
                        )
   
    # 4. 研究区域shp和边框
    china = shpreader.Reader(basic_dir + 'research/6.Southeastern_China_Clustering/data/shp/china-shapefiles-master/china-shapefiles-master/china_country.shp').geometries()
    ax.add_geometries(china, ccrs.PlateCarree(), 
                       facecolor='none', edgecolor='k', linewidth=0.5, zorder=10) # 添加中国边界
    ax.coastlines(resolution='50m', linewidth=0.5)
    
  
    x2=[108,123,123,108,108]
    y2=[18,18,34,34,18]
    plot_geometries= sgeom.LineString(zip(x2,y2))
    ax.add_geometries([plot_geometries],ccrs.PlateCarree(),facecolor='none', edgecolor='#5a82af',lw=1)#add_geometries绘制方法
    
    
    rectangle = Area_used.get('region'+str(r))
    x2=[rectangle[1],rectangle[3],rectangle[3],rectangle[1],rectangle[1]]
    y2=[rectangle[2],rectangle[2],rectangle[0],rectangle[0],rectangle[2]]
    plot_geometries1= sgeom.LineString(zip(x2,y2))
    ax.add_geometries([plot_geometries1],ccrs.PlateCarree(),facecolor='none', edgecolor='#5a82af',lw=1,ls=":")
    
    
    '''
    # 5. 月份分布小图
    ax1 = plt.axes([x_loc[j],y_loc[j]+0.27,0.15,0.18] )
    labels=['Apr','May','Jun','Jul','Aug','Sept']
    for l in range(6):
        ax1.bar(4+l,times.groupby('month').mean()['times'].iloc[l],color = colors[l],alpha=0.2,edgecolor=colors[l],label=labels[l])
    if j ==5:
        #lines, labels = fig.axes[-2].get_legend_handles_labels()                
        plt.legend( bbox_to_anchor=(2.28, -1.56),ncol=6, framealpha=0,fontsize=7,columnspacing=0.4, labelspacing=0.3,)
    ax1.scatter(arr1,times['times'],color = np.tile(colors,41),alpha=np.repeat(np.arange(0,41)/41,6),s=2)
    
    ax1.hlines(xmin=3.5,xmax=9.5,y=times.mean()['times'],color='grey',linestyle='--',linewidth=0.6 )
    ax1.set_xlim((3.5,9.5))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)    
    ax1.xaxis.set_ticks([])
    ax1.xaxis.set_ticklabels([])
    
    ax1.yaxis.set_ticks([0,10,20])
    ax1.yaxis.set_ticklabels([0,10,20],fontsize=7)
    ax1.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1)
    
    ax1.annotate('(day)', xy=(0.03, 0.9), xycoords="axes fraction",c='black',size=6)
    
    
    
    # 6. 这些日内的平均环流情况

    #
    size_list = [23,17,14,15,19,11]
    intervals,intervals_prec = cal_everage_condition(time)
    
    ax2 = plt.axes([x_loc[j]+0.165,y_loc[j]+0.27,0.175,0.18] )
    
    ax2.axvline(0, color='grey',linestyle='--',linewidth=0.6)
    ax2.axhline(0, color='grey',linestyle='--',linewidth=0.6)
    
    norm = matplotlib.colors.Normalize(vmin=-5.5, vmax=5.5)
    scat=ax2.scatter(intervals['z500'],intervals['slp'],c=intervals['t850'],s=size_list[j],marker='s',cmap=Colors_used_regional,norm=norm)
    
    norm_prec = matplotlib.colors.Normalize(vmin=0, vmax=14)
    scat2 = ax2.scatter(intervals_prec['z500'],intervals_prec['slp'],c=intervals_prec['prec'],
               s=(size_list[j]-3)*intervals_prec['prec']/intervals_prec['prec'].max(),norm=norm_prec,
               marker='^',cmap=Colors_used_prec )
    
    print(intervals_prec['prec'].max())
    aaa = 5*intervals_prec['prec']/intervals_prec['prec'].max()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)    

    ax2.xaxis.set_major_locator(MultipleLocator(50) )
    ax2.xaxis.set_minor_locator( MultipleLocator(25) )
    ax2.yaxis.set_major_locator(MultipleLocator(5) )
    ax2.yaxis.set_minor_locator(MultipleLocator(2.5))
    #xmajorFormatter = FormatStrFormatter('%d') 
    #ymajorFormatter = FormatStrFormatter('%1.1f') 
    ax2.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1)
    ax2.tick_params(axis='x', labelsize=6.5,pad=0.5,length=1)
    ax2.annotate('(m)', xy=(0.9, 0.03), xycoords="axes fraction",c='black',size=6)
    ax2.annotate('(hPa)', xy=(0.03, 0.9), xycoords="axes fraction",c='black',size=6)

    '''
    # 7. 一些基本设置
    ax.set_extent((70, 140, 0, 60), ccrs.PlateCarree())
    ax.set_aspect('auto') ##Non-fixed aspect
    
    ax.set_title(r"$\bf{("+title_aux[j]+")}$ Cl. "+ str(j+1)+" ("+str(np.round(count_events[sequence_f[j]],1))+"%)",
                     pad=4, size=8.5, loc='left')
    
    #ax1.set_ticks([1000,1500,2000,2500,3000,3500],fontsize=size_tick_label )
    
    # 8.图例
    if j==5:
        a=ax.get_position()
        pad=0.04
        height=0.02
        ax_f = fig.add_axes([  x_loc[0],  a.ymin - pad,  0.5 , height ]) #长宽高
        cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
        cb.set_ticklabels(Colors_limits_slp,size=6.5)
        cb.outline.set_linewidth(0.8)
        ax_f.tick_params(length=1,pad=0.2)
        cb.set_label( label = "Anomaly SLP (hPa)",fontdict={'size':7},labelpad=1)
        
        qk = ax.quiverkey(ivtq,  0.57,a.ymin - 0.015, 100, r'100 $kg·m^{-1} s^{-1}$', labelpos='S',
              coordinates='figure',zorder=200,
              fontproperties={'size':7.5})
    '''   
    if j==1:    
        ax_f1 = fig.add_axes([  1.08,  0.5,  0.015 , 0.45 ])
        cb = fig.colorbar(scat, orientation='vertical',cax=ax_f1)
        cb.set_ticks ( ticks=[-4,-2,0,2,4],labelsize=6.5)
        cb.set_ticklabels([-4,-2,0,2,4],size=6.5)
        ax_f1.tick_params(length=1,pad=0.1)
        cb.outline.set_linewidth(0.8)
        cb.set_label( label = "Anomaly T (°K)",fontdict={'size':7},labelpad=1)
        
        ax_f2 = fig.add_axes([  1.08,  0.,  0.015 , 0.45 ])
        cb = fig.colorbar(scat2, orientation='vertical',cax=ax_f2)
        cb.set_ticks ( ticks=np.linspace(0,12,6),labelsize=6.5)
        cb.set_ticklabels( np.round(np.linspace(0,12,6),1),size=6.5)
        ax_f2.tick_params(length=1,pad=0.1)
        cb.outline.set_linewidth(0.8)
        cb.set_label( label = "Anomaly Prec (mm)",fontdict={'size':7},labelpad=1)
    '''
fig.savefig('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/Figures/Figure.Sx.circulation_type_in_LSExP_days_'+str(qu)+'_'+str(ar)+'.png',dpi=1500, bbox_inches='tight')


#fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/Figures/Figure.xx.png",dpi=1500, bbox_inches='tight')
    #fig.savefig(basic_dir+"code_whiplash/5-2.Figures/Fig.S16.png",dpi=1500, bbox_inches='tight')
    
    
    