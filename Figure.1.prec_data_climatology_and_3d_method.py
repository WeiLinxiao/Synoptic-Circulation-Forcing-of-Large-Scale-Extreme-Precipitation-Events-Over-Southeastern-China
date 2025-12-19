#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:10:08 2023

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
import pymannkendall as mk
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs
from matplotlib.collections import LineCollection, PolyCollection
import cartopy.feature as cf
from cartopy.mpl.patch import geos_to_path
import itertools 
#%% ----------------------------------------------------function------------------------------------


def cal_trend_and_p(c_all)  :  
    model = LinearRegression()
    x = np.arange(1979,2020).reshape(-1,1)
    y = c_all
    model.fit(x,y)
    #print(model.coef_[0])

    # 泰尔-森
    mk_p = mk.original_test(y).p
    predicted = model.predict(x)
    print(mk_p)
    return predicted,mk_p,model.coef_[0]

#%%
basic_dir = '/media/dai/DATA1/research/6.Southeastern_China_Clustering/'

prec_climatology = np.load(basic_dir + 'code/2.SEC_prec_climatology_and_trend/prec_climatology.npy',allow_pickle=True).tolist()

#%% 站点数据

station = prec_climatology.get('station_location')
prec_station_annual_mean =  prec_climatology.get('annual_prec_all_station')
prec_station_climatology = prec_climatology.get('climatology_annual_prec_all_station')
prec_grid_climatology = prec_climatology.get('climatology_annual_prec_all_grid_of_SEC')

#%%
prec = xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_0.25_1979-2019.nc')
prec=prec[(prec['time.month']>=4) & (prec['time.month']<=9),:,:]
prec = prec.where(prec>0.1) ##有雨日


qu=0.99
ar=0.95
#
basic_dir1 ='/media/dai/DATA1/'
d3_events_stats = np.load(basic_dir1+'/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
event_sum_original = d3_events_stats.get('event_sum')
#Summary_Stats = d3_events_stats.get('area_Summary_Stats')

events_stats = np.load(basic_dir1+'/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
event_all_stats = events_stats.get('event_all_stats')
event_basic_stats = events_stats.get('event_basic_stats')

event_basic_stats['distance'].max()
#%%
prec_sec = prec.sel(lon=slice(108,124),lat=slice(34,18))
prec_sec_regional = prec_sec.weighted(np.cos(np.deg2rad(prec_sec.lat))).mean(('lon','lat')) #海洋地方已经是nan
prec_sec_regional_amount = prec_sec_regional.groupby('time.year').sum()  #雨季降水总量

#%%


#%% 极端降水日出现的频率 以日来统计
'''
ex_days = np.where((prec_sec_regional > prec_sec_regional.quantile(q=0.95,dim='time')).astype(int)>0)
ex_intensity = prec_sec_regional[ex_days].groupby('time.year').mean()

freq_ex_day = [(t[ex_days[0]]==i).sum() for i in range(1979,2020)]
'''
#%%
#极端降水总量
prec_sec_quan = prec_sec.quantile(q=0.95,dim='time')
prec_sec_ex = (prec_sec>prec_sec_quan).astype(float) #是否是极端降水
prec_sec_ex = prec_sec*prec_sec_ex #非极端降水都变成0
prec_sec_ex_amount = prec_sec_ex.groupby('time.year').sum() #极端降水总量
prec_sec_ex_amount =  prec_sec_ex_amount.where(prec_sec_ex_amount>0) #总量为0的地方为海洋，设置为nan
prec_sec_regional_ex_amount =prec_sec_ex_amount.weighted(np.cos(np.deg2rad(prec_sec.lat))).mean(('lon','lat')) #区域平均极端降水总量
#%强度
prec_sec_ex = prec_sec_ex.where(prec_sec_ex>0) #将非极端降水都去掉
prec_sec_regional_ex =prec_sec_ex.weighted(np.cos(np.deg2rad(prec_sec.lat))).mean(('lon','lat')) #求区域平均
ex_intensity = prec_sec_regional_ex.groupby('time.year').mean() #此时可以求出强度（完全没有极端降水的日子为nan，就相当于年总量/频率了）

#频率，这里得出的频率是，存在极端降水的天数就算
freq_ex_day = [(prec_sec_regional_ex[prec_sec_regional_ex>0]['time.year']==i).sum().values for i in range(1979,2020)]

#
#%%大尺度极端降水总量

days =np.floor(event_all_stats.iloc[:,0:2].groupby('event').mean()).values[:,0].astype(int)
t = prec['time.year'].values
event_years  = t[days] #417个事件发生的年份

large_scale_amount=[]
for i in range(1979,2020):
    large_scale_amount.append( event_basic_stats.iloc[:,4][ event_years==i].sum())
    
##大尺度极端降水面积    
large_scale_area=[]
for i in range(1979,2020):
    large_scale_area.append( event_basic_stats.iloc[:,5][ event_years==i].sum())    
    
##大尺度极端降水历时
large_scale_span=[]
for i in range(1979,2020):
    large_scale_span.append( event_basic_stats.iloc[:,0][ event_years==i].mean())      
    
##大尺度极端降水频率
large_scale_frequency=[]
for i in range(1979,2020):
    large_scale_frequency.append( len(event_basic_stats.iloc[:,4][ event_years==i]))

##大尺度极端降水强度
large_scale_intensity=[]
for i in range(1979,2020):
    large_scale_intensity.append( event_basic_stats.iloc[:,7][ event_years==i].mean())
    
large_scale_exP =xr.open_dataarray ('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/large_scale_exP_values_'+str(qu)+'_'+str(ar)+'.nc')
large_scale_freq =xr.open_dataarray ('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/large_scale_exP_frequency_'+str(qu)+'_'+str(ar)+'.nc')
a=large_scale_exP.values
np.nanmin(a)
LSExP = large_scale_exP.groupby('time.year').sum('time').mean('year')
LSExP=LSExP.where(LSExP>0)


a =large_scale_freq.values
LSExP_freq = np.full((a.shape[1],a.shape[2]),np.nan)
for i in range(a.shape[1]):
    for j in range(a.shape[2]):
        uniq  = np.unique(a[:,i,j])
        if len(uniq)>1:
            LSExP_freq[i,j]=len(uniq)-1

k_list = []
for i in range(len(event_sum_original)):
    if len(np.unique(event_sum_original[i][0]))==3:
        k_list.append(i)
#%%
z500=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_Z500.nc').sel(time=slice('1979-01-01','2019-12-31'))
z500 = z500[(z500['time.month']>=4)&(z500['time.month']<=9)]
z500_rainy = z500.mean('time')

ivt_v=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_moisture_flux_integraded_v_component.nc').sel(time=slice('1979-01-01','2019-12-31'))
ivt_v = ivt_v[(ivt_v['time.month']>=4)&(ivt_v['time.month']<=9)]
ivt_v_rainy = ivt_v.mean('time')

ivt_u=xr.open_dataarray('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/ERA5_19790101_20201231_daily_moisture_flux_integraded_u_component.nc').sel(time=slice('1979-01-01','2019-12-31'))
ivt_u = ivt_u[(ivt_u['time.month']>=4)&(ivt_u['time.month']<=9)]
ivt_u_rainy = ivt_u.mean('time')

#%%  plotting parameters 
size_tick_label = 7
size_tick_title = 7
size_legend = 6
size_cbar_label = 6
size_cbar_title = 7
size_title = 8
#%%
########################################   画图 ######################################
china = shpreader.Reader(basic_dir + 'data/shp/china-shapefiles-master/china-shapefiles-master/china_country.shp').geometries()
region=shpreader.Reader(basic_dir + 'data/shp/region/region.shp').geometries()

fig = plt.figure(figsize = (17/2.54, 10/2.54)) # 宽、高 ，单位为英寸 1英寸的2.5cm

##########################################  a  站点-多年平均降水 ##########################################
#mapcrs = ccrs.AlbersEqualArea(central_longitude=115, central_latitude=10, standard_parallels=(10, 30))

ax1 = fig.add_axes([0, 0.6, 0.32, 0.4], projection = ccrs.PlateCarree())
colors_used =  sns.color_palette('Spectral_r', n_colors=12) 
colors_used = ListedColormap(colors_used)
cf1=ax1.scatter(station['V2'],station['V3'], #散点图
                c=prec_station_climatology,
                s=1,
                cmap=colors_used,#transform = ccrs.PlateCarree()
                )
ax1.coastlines(resolution='10m', linewidth=0.8)
#ax1.add_feature(cfeature.RIVERS,linewidth=0.5)
ax1.stock_img()
ax1.set_extent((70, 138, 15, 55), ccrs.PlateCarree())
##XY轴
plt.yticks([20,30,40,50],fontsize=size_tick_label)
plt.xticks([75,90,105,120,135],fontsize=size_tick_label)
ax1.tick_params(length=1.5,pad=1)
ax1.set_ylabel('Latitude (°N)',fontsize=size_tick_title,labelpad=1)
ax1.set_xlabel('Longitude (°E)',fontsize=size_tick_title,labelpad=1)
ax1.set_xlim(72,138)
ax1.set_aspect('auto') ##设置不固定长宽
ax1.set_title(r"$\bf{(a)}$ Annual precipitation of Stations",
                 pad=4, size=8.5, loc='left')

##地区图
ax1.add_geometries(region, ccrs.PlateCarree(), 
                   facecolor='none', edgecolor='k', linewidth=1.2, zorder=10) # 添加中国边界
ax1.add_geometries(china, ccrs.PlateCarree(), 
                   facecolor='none', edgecolor='k', linewidth=0.8, zorder=200) # 添加中国边界

##标签位置
'''''
a=ax1.get_position()
pad=0.015
width=0.015
ax1_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
cb=fig.colorbar(cf1, cax=ax1_f,)
'''''

a=ax1.get_position()
pad=0.1
#ax_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
#cb=fig.colorbar(cf,cax=ax_f)
ax1_f = fig.add_axes([  a.xmin,  a.ymin - pad,  a.xmax-a.xmin , 0.025]) #长宽高
cb=fig.colorbar(cf1, orientation='horizontal',cax=ax1_f)

#标签题目
cb.set_label( label = "Annual precipitation (mm)",fontdict={'size':size_cbar_title},labelpad=1.5)
#标签刻度位置
ax1_f.yaxis.set_ticks_position('right')
ax1_f.tick_params(length=1,pad=0.2)
#标签刻度值
#ax1_f.set_xticklabels([0, 400,800,1200,1600,2000, 2400],rotation=-40,size=size_cbar_label)
cb.set_ticks([0, 400,800,1200,1600,2000, 2400],size=size_cbar_title)
#ax1.set_title('total_loss')
#
cb.ax.tick_params(labelsize=size_cbar_title)
##########################################  b  grid SEC 多年平均降水 （可能要改为所有大范围极端降水）##########################################

colors_used1 =  sns.color_palette('Spectral_r', n_colors=10)#'mako_r'

#colors_used1 = ListedColormap(colors_used1)


X,Y=np.meshgrid(prec_grid_climatology['lon'].values,
                prec_grid_climatology['lat'].values)

mask=globe.is_land(Y, X)
mask=mask.astype(int).astype(float)
mask[mask==0]=np.nan
#prec_grid_annual_land=prec_grid_annual*mask   


#prec_grid_annual_land=prec_grid_annual_land.values
#prec_grid_annual_land[prec_grid_annual_land==0]=np.nan

#p_max=np.nanmax(prec_grid_annual_land)
#p_min=np.nanmin(prec_grid_annual_land)
#ct_breaks = np.arange(p_min, p_max, 50)


#fig = plt.figure(figsize = (10, 7)) # 宽、高
ax2 = fig.add_axes([0.37, 0.6, 0.25, 0.4],projection = ccrs.PlateCarree() )
cf = ax2.contourf(prec_grid_climatology.lon,
                 prec_grid_climatology.lat, 
                 prec_grid_climatology, 
                 levels= np.arange(400,2600,200),
                 colors = colors_used1, extend='both',
                 
                 #marker='s',
                 #s=28,
                 #vmin=0,
                  transform = ccrs.PlateCarree() )
ax2.set_extent((108, 123, 18, 34), ccrs.PlateCarree())
ax2.stock_img()
ax2.gridlines(linestyle='--')

##XY轴刻度值
plt.yticks([20,25,30],fontsize=6)
plt.xticks([110,115,120],fontsize=6)
##XY轴标题
ax2.set_ylabel('Latitude (°N)',fontsize=size_tick_title,labelpad=1)
ax2.set_xlabel('Longitude (°E)',fontsize=size_tick_title,labelpad=1)
ax2.tick_params(length=1.5,pad=1)
ax2.set_aspect('auto') ##设置不固定长宽
#ax1.set_xlim(110,120)
ax2.set_title(r"$\bf{(b)}$ Annual gridded precipitation ",
                 pad=4, size=8.5, loc='left')


##地区图
china = shpreader.Reader(basic_dir + 'data/shp/china-shapefiles-master/china-shapefiles-master/china_country.shp').geometries()
#ax2.add_geometries(region, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1.2, zorder=10) # 添加中国边界
ax2.add_geometries(china, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.8, zorder=200) # 添加中国边界

'''''
a=ax2.get_position()
pad=0.015
width=0.015
ax2_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
cb2=fig.colorbar(cf, cax=ax2_f)
'''''
a=ax2.get_position()
pad=0.1
#ax_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
#cb=fig.colorbar(cf,cax=ax_f)
ax2_f = fig.add_axes([  a.xmin,  a.ymin - pad,  a.xmax-a.xmin , 0.025]) #长宽高
cb2=fig.colorbar(cf, orientation='horizontal',cax=ax2_f)

cb2.set_label( label = "Annual precipitation (mm)",fontdict={'size':size_cbar_title},labelpad=1)
#标签刻度位置
ax2_f.yaxis.set_ticks_position('right')
#标签刻度值

#ax2_f.set_yticklabels([800,1200,1600,2000,2400,2800],rotation=-40,size=6)
cb2.set_ticks([800,1200,1600,2000,2400],size=size_cbar_title)
cb2.ax.tick_params(labelsize=size_cbar_title)
ax2_f.tick_params(length=1,pad=0.2)


###########################################  c #########################################
ax= fig.add_axes([0.69, 0.6, 0.3, 0.4],projection=ccrs.PlateCarree())
Colors_used = sns.color_palette('RdYlBu_r', n_colors=15)[2:]
Colors_line = sns.color_palette('viridis_r', n_colors=7)
Colors_limits=np.arange(0,360,30)

ivt = np.sqrt((ivt_v_rainy)**2 +(ivt_u_rainy)**2)
    

    
ivtq=ax.quiver(  ivt_v_rainy['longitude'].values, ivt_v_rainy['latitude'].values, 
             ivt_u_rainy.values,ivt_v_rainy.values,
             transform = ccrs.PlateCarree(),
             headwidth=6,
             color='black',zorder=100,scale_units = 'inches',
             angles = 'uv', regrid_shape = 80,scale=1600)

cf=ax.contourf( ivt_v_rainy['longitude'].values, ivt_v_rainy['latitude'].values, 
               ivt.values,
            transform=ccrs.PlateCarree(),
            colors=Colors_used,
            levels=Colors_limits,
            extend='both'
            )

cont=ax.contour(z500_rainy['longitude'].values, z500_rainy['latitude'].values,
           (z500_rainy/9.806).values,
           colors=Colors_line,
           levels=[5620,5680,5740,5800,5840,5860,5880],
           )
    
clabels = ax.clabel(cont, inline=True, fontsize=8, fmt='%d',
                    colors='black',
                    #inline_spacing=5,use_clabeltext=True,manual=False,rightside_up=True,
                    #bbox=dict(facecolor='white',edgecolor='none',pad=0)
                    #inline_bgedge='#ffffff'
                    )
china = shpreader.Reader(basic_dir + 'data/shp/china-shapefiles-master/china-shapefiles-master/china_country.shp').geometries()
ax.add_geometries(china, ccrs.PlateCarree(), 
                   facecolor='none', edgecolor='k', linewidth=0.5, zorder=100) # 添加中国边界

x2=[108,123,123,108,108]
y2=[18,18,34,34,18]
plot_geometries= sgeom.LineString(zip(x2,y2))
ax.add_geometries([plot_geometries],ccrs.PlateCarree(),facecolor='none', edgecolor='black',lw=1)#add_geometries绘制方法

ax.set_aspect('auto') ##设置不固定长宽
#ax.stock_img()
ax.coastlines(resolution='50m', linewidth=0.5,color='k')    
ax.set_xlim(80,140)
ax.set_ylim(10,55)


##XY轴刻度值
plt.yticks([10,25,40,55],fontsize=6)
plt.xticks([80,100,120,140],fontsize=6)
##XY轴标题
ax.set_ylabel('Latitude (°N)',fontsize=size_tick_title,labelpad=1)
ax.set_xlabel('Longitude (°E)',fontsize=size_tick_title,labelpad=1)
ax.tick_params(length=1.5,pad=1)
ax.set_aspect('auto') ##设置不固定长宽


ax.set_title(r"$\bf{(c)}$ Large scale circulation",
                 pad=4, size=8.5, loc='left')

a=ax.get_position()
pad=0.1
#ax_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
#cb=fig.colorbar(cf,cax=ax_f)
ax_f = fig.add_axes([  a.xmin,  a.ymin - pad,  a.xmax-a.xmin , 0.025]) #长宽高
cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)

cb.set_ticklabels(Colors_limits,size=6.5)
cb.outline.set_linewidth(0.8)
ax_f.tick_params(length=1,pad=0.2)

#标签刻度位置
#ax_f.yaxis.set_ticks_position('right')
cb.set_label( label = "Column-integrated moisture flux\n("+r'kg·m$^{-1}$ s$^{-1}$'+")",fontdict={'size':7},labelpad=1)

qk = ax.quiverkey(ivtq,  a.xmax + pad+0.03,a.ymin - 0.02, 200, r'100 kg·m$^{-1}$ s$^{-1}$', labelpos='S',
      coordinates='figure',zorder=200,
      fontproperties={'size':7.5})

#

############################################# e ################################################
china = shpreader.Reader(basic_dir + 'data/shp/china-shapefiles-master/china-shapefiles-master/china_country.shp').geometries()
region=shpreader.Reader(basic_dir + 'data/shp/region/region.shp').geometries()

ax1 = fig.add_axes([0.35, -0.05, 0.28, 0.4], projection = ccrs.PlateCarree())

colors_used =  sns.color_palette('Spectral_r', n_colors=15)[1:] 
Colors_limits = np.arange(5,66,5)
ax1.stock_img()

cf1=ax1.contourf(LSExP.lon,LSExP.lat,LSExP ,
               levels=Colors_limits,
               colors=colors_used,
               extend='both',
               #cmap='RdBu_r',
               projection=ccrs.PlateCarree() )


ax1.coastlines(resolution='10m', linewidth=0.8)

ax1.set_extent((80, 135, 15, 50), ccrs.PlateCarree())
##XY轴
plt.yticks([20,30,40,50],fontsize=size_tick_label)
plt.xticks([85,100,115,130],fontsize=size_tick_label)
ax1.tick_params(length=1.5,pad=1)
ax1.set_ylabel('Latitude (°N)',fontsize=size_tick_title,labelpad=1)
ax1.set_xlabel('Longitude (°E)',fontsize=size_tick_title,labelpad=1)
ax1.set_xlim(85,130)
ax1.set_aspect('auto') ##设置不固定长宽
ax1.set_title(r"$\bf{(e)}$ Annual LSExP amounts",
                 pad=4, size=8.5, loc='left')

##地区图
ax1.add_geometries(region, ccrs.PlateCarree(), 
                   facecolor='none', edgecolor='k', linewidth=1.2, zorder=10) # 添加中国边界
ax1.add_geometries(china, ccrs.PlateCarree(), 
                   facecolor='none', edgecolor='k', linewidth=0.8, zorder=200) # 添加中国边界

##标签位置
'''
a=ax1.get_position()
pad=0.015
width=0.015
ax1_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
cb=fig.colorbar(cf1, cax=ax1_f,)
'''

a=ax1.get_position()
pad=0.1
#ax_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
#cb=fig.colorbar(cf,cax=ax_f)
ax1_f = fig.add_axes([  a.xmin,  a.ymin - pad,  a.xmax-a.xmin , 0.025]) #长宽高
cb=fig.colorbar(cf1, orientation='horizontal',cax=ax1_f)

#标签题目
cb.set_label( label = "Precipitation (mm)",fontdict={'size':size_cbar_title},labelpad=1.5)
#标签刻度位置

ax1_f.tick_params(length=1,pad=0.2)
#标签刻度值
#ax1_f.set_xticklabels([0, 400,800,1200,1600,2000, 2400],rotation=-40,size=size_cbar_label)
cb.set_ticks(Colors_limits[np.arange(0,len(Colors_limits),2)],size=size_cbar_title)
#ax1.set_title('total_loss')
#
cb.ax.tick_params(labelsize=size_cbar_title)

#%
############################################# f ################################################
china = shpreader.Reader(basic_dir + 'data/shp/china-shapefiles-master/china-shapefiles-master/china_country.shp').geometries()
region=shpreader.Reader(basic_dir + 'data/shp/region/region.shp').geometries()

ax1 = fig.add_axes([0.69, -0.05, 0.3, 0.4], projection = ccrs.PlateCarree())

colors_used =  sns.color_palette('Spectral_r', n_colors=18)[2:] 
Colors_limits = np.arange(2,31,2)
ax1.stock_img()

cf1=ax1.contourf(LSExP.lon,LSExP.lat,LSExP_freq ,
               levels=Colors_limits,
               colors=colors_used,
               extend='both',
               #cmap='RdBu_r',
               projection=ccrs.PlateCarree() )


ax1.coastlines(resolution='10m', linewidth=0.8)

ax1.set_extent((80, 135, 15, 50), ccrs.PlateCarree())
##XY轴
plt.yticks([20,30,40,50],fontsize=size_tick_label)
plt.xticks([85,100,115,130],fontsize=size_tick_label)
ax1.tick_params(length=1.5,pad=1)
ax1.set_ylabel('Latitude (°N)',fontsize=size_tick_title,labelpad=1)
ax1.set_xlabel('Longitude (°E)',fontsize=size_tick_title,labelpad=1)
ax1.set_xlim(85,130)
ax1.set_aspect('auto') ##设置不固定长宽
ax1.set_title(r"$\bf{(f)}$ LSExP frequency",
                 pad=4, size=8.5, loc='left')


##地区图
ax1.add_geometries(region, ccrs.PlateCarree(), 
                   facecolor='none', edgecolor='k', linewidth=1.2, zorder=10) # 添加中国边界
ax1.add_geometries(china, ccrs.PlateCarree(), 
                   facecolor='none', edgecolor='k', linewidth=0.8, zorder=200) # 添加中国边界

##标签位置
'''
a=ax1.get_position()
pad=0.015
width=0.015
ax1_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
cb=fig.colorbar(cf1, cax=ax1_f,)
'''

a=ax1.get_position()
pad=0.1
#ax_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])
#cb=fig.colorbar(cf,cax=ax_f)
ax1_f = fig.add_axes([  a.xmin,  a.ymin - pad,  a.xmax-a.xmin , 0.025]) #长宽高
cb=fig.colorbar(cf1, orientation='horizontal',cax=ax1_f)

#标签题目
cb.set_label( label = "Frequency (times)",fontdict={'size':size_cbar_title},labelpad=1.5)
#标签刻度位置

ax1_f.tick_params(length=1,pad=0.2)
#标签刻度值
#ax1_f.set_xticklabels([0, 400,800,1200,1600,2000, 2400],rotation=-40,size=size_cbar_label)
cb.set_ticks(Colors_limits[np.arange(0,len(Colors_limits),2)],size=size_cbar_title)
#ax1.set_title('total_loss')
#
cb.ax.tick_params(labelsize=size_cbar_title)


#################################################### d   ##########################33
import cartopy.feature as cf
from cartopy.mpl.patch import geos_to_path
k=20
a  =pd.concat([pd.Series(event_sum_original[k][0]),pd.Series(event_sum_original[k][1]),pd.Series(event_sum_original[k][2])],axis=1)
a.columns=['day','lat','lon']
prec_event =  np.full(shape=(3,140,248),fill_value=np.nan)
for i in range(3):
    b=a[a['day']== np.unique(event_sum_original[k][0])[i] ]
    for j in range(len(b)):
        prec_event[ i ,  b['lat'].iloc[j],   b['lon'].iloc[j]] = prec[np.unique(event_sum_original[k][0])[i],b['lat'].iloc[j],   b['lon'].iloc[j] ]

days = prec.time[np.unique(event_sum_original[k][0])].values
##得到基本的降水统计

#%
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#负号


ax = Axes3D(fig,[-0.16, -0.19, 0.6, 0.6], xlim=[106, 123], ylim=[20, 30])
ax.set_zlim(0,0.5)
ax.set_facecolor('none')
proj_ax=plt.figure().add_subplot(111,projection=ccrs.PlateCarree())
proj_ax.set_xlim(ax.get_xlim())#使地图投影获得当前3d投影一样的绘图范围
proj_ax.set_ylim(ax.get_ylim())
concat = lambda iterable: list(itertools.chain.from_iterable(iterable))
target_projection=proj_ax.projection
feature=cf.NaturalEarthFeature('physical', 'land', '50m')
geoms=feature.geometries()

#geoms = shpreader.Reader('/media/dai/disk2/suk/research/4.East_Asia/data/shapefile/china-shapefile/china.shp').geometries()

boundary=proj_ax._get_extent_geom()
geoms = [target_projection.project_geometry(geom, feature.crs)
         for geom in geoms]
geoms2=[]
for i in range(len(geoms)):
    if geoms[i].is_valid:
        geoms2.append(geoms[i])
geoms=geoms2

geoms=[boundary.intersection(geom)for geom in geoms]
paths = concat(geos_to_path(geom) for geom in geoms)
polys = concat(path.to_polygons() for path in paths)
lc = PolyCollection(polys, 
                    edgecolor='black',
                    alpha=0.5,
                    facecolor='#DDDDDD', 
                    closed=False,
                    linewidth=0.5)
ax.add_collection3d(lc)

#ax.gridlines(linestyle='--')
proj_ax.spines['geo'].set_visible(False)#解除掉用于确定地图的子图
ax.set_xlabel('Lon')
ax.set_ylabel('Lat')
#ax.set_zlabel('Day')
#ax.bar3d(prec.lon[w[1]],prec.lat[w[0]],0.,0.05,0.05,0.25,color='red')

##第一层
w = np.where(prec_event[0,:,:]==np.nanmax(prec_event[0,:,:])) 
#ax.scatter(prec.lon[w[1]].values[0],prec.lat[w[0]].values[0],0,s=30,color='red') #第一层的点

ax.plot3D([prec.lon[w[1]].values[0],prec.lon[w[1]].values[0]], 
          [prec.lat[w[0]].values[0],prec.lat[w[0]].values[0]],
          [0,0.25],color='red',zorder=10) 

#ax.scatter(prec.lon[w[1]].values[0],prec.lat[w[0]].values[0],0.25,s=20,color='red') #第二层初始

ax.contourf(prec.lon,prec.lat,prec_event[0,:,:],zdir='z',offset=0,zorder=20)

##第二层
w1 = np.where(prec_event[1,:,:]==np.nanmax(prec_event[1,:,:]))
#ax.scatter(prec.lon[w1[1]].values[0],prec.lat[w1[0]].values[0],0.25,s=20,color='red')

ax.plot([prec.lon[w[1]].values[0],prec.lon[w1[1]].values[0]],
        [prec.lat[w[0]].values[0],prec.lat[w1[0]].values[0]]  ,0.25,color='red',zorder=70)

ax.plot3D([prec.lon[w1[1]].values[0],prec.lon[w1[1]].values[0]], 
          [prec.lat[w1[0]].values[0],prec.lat[w1[0]].values[0]],
          [0.25,0.5],color='red',zorder=20)

#ax.scatter(prec.lon[w1[1]].values[0],prec.lat[w1[0]].values[0],0.5,s=20,color='red')
##第三层
w2 = np.where(prec_event[2,:,:]==np.nanmax(prec_event[2,:,:]))
#ax.scatter(prec.lon[w2[1]].values[0],prec.lat[w2[0]].values[0],0.5,s=20,color='red')


ax.plot([prec.lon[w2[1]].values[0],prec.lon[w1[1]].values[0]],
        [prec.lat[w2[0]].values[0],prec.lat[w1[0]].values[0]]  ,0.5,color='red',zorder=80)


lc2 = PolyCollection(polys, 
                    edgecolor='black',
                    alpha=0.5,
                    facecolor='#DDDDDD', 
                    closed=False,
                    linewidth=0.5)
ax.add_collection3d(lc2,zs=0.25)


lc3 = PolyCollection(polys, 
                    edgecolor='black',
                    alpha=0.5,
                    facecolor='#DDDDDD', 
                    closed=False,
                    linewidth=0.5)
ax.add_collection3d(lc3,zs=0.5)


ax.contourf(prec.lon,prec.lat,prec_event[1,:,:],zdir='z',offset=0.25,zorder=40)
ax.contourf(prec.lon,prec.lat,prec_event[2,:,:],zdir='z',offset=0.5,zorder=70)
ax.view_init(elev=20, #仰角
              azim=-75 #方位角
)


ax.set_ylabel('Latitude (°N)',fontsize=size_tick_title,labelpad=-10)
ax.set_xlabel('Longitude (°E)',fontsize=size_tick_title,labelpad=-10)
ax.tick_params(axis="both",length=1.5,pad=-5)
#ax.set_yticks([30,35,40],fontsize=size_tick_label)
#ax.set_xticks([105,115,125],fontsize=size_tick_label)
ax.yaxis.set_ticks([20,25,30],labelpad=-5)
ax.yaxis.set_ticklabels([20,25,30],fontsize=7.5)
ax.xaxis.set_ticks([110,115,120],labelpad=-20)
ax.xaxis.set_ticklabels([110,115,120],fontsize=7.5)
ax.zaxis.set_ticks([0,0.25,0.5],labelpad=-5)
ax.zaxis.set_ticklabels(['',
                         '',
                         ''],fontsize=7.5)
ax.annotate(r"$\bf{(d)}$ A 3-dimentional LSExP event",xy=(0.05,0.6), size=8.5, xycoords='figure fraction')
#%
'''
ax.zaxis.set_ticklabels([np.datetime_as_string(days[0],unit='D'),
                         np.datetime_as_string(days[1],unit='D'),
                         np.datetime_as_string(days[2],unit='D')],fontsize=7.5)
'''
ax.annotate(np.datetime_as_string(days[2],unit='D'),xy=(0.62,0.75), size=7.5, xycoords='axes fraction',rotation=-8)
ax.annotate(np.datetime_as_string(days[1],unit='D'),xy=(0.62,0.53), size=7.5, xycoords='axes fraction',rotation=-8)
ax.annotate(np.datetime_as_string(days[0],unit='D'),xy=(0.60,0.32), size=7.5, xycoords='axes fraction',rotation=-8)

fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/Figures/Figure.1.png",dpi=1500, bbox_inches='tight')
fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/Figures/Figure.1.pdf",dpi=1500, bbox_inches='tight')
#%%

