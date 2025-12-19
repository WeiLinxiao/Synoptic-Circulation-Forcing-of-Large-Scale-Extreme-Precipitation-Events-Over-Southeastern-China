#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 21:08:38 2023

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
import shapely.geometry as sgeom
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

basic_dir ='/media/dai/DATA1/'
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
labels=['April','May','June','July','August','September']
#%%

import warnings
warnings.filterwarnings('ignore')

Colors_used = sns.color_palette('RdYlBu_r', n_colors=20)[3:17]
Colors_line = sns.color_palette('viridis_r', n_colors=7)
Colors_limits=np.arange(0,390,30)
fig = plt.figure(figsize = (17.5/2.54, 10/2.54)) # 宽、高
for j in range(6):

    
    z500_rainy = z500[(ivt_v['time.month']==j+4)].mean('time')
    ivt_v_rainy = ivt_v[(ivt_v['time.month']==j+4)].mean('time')
    ivt_u_rainy = ivt_u[(ivt_v['time.month']==j+4)].mean('time')
    ivt = np.sqrt((ivt_v_rainy)**2 +(ivt_u_rainy)**2)
    
    print(j+4)
    
    
    
       
    ax = plt.axes([x_loc[ j  ],y_loc[  j  ], x_length, y_length],projection=ccrs.PlateCarree() )
        #print(sequence_f[j])
        
        
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
    
    #[txt.set_backgroundcolor('white') for txt in clabels.labelTexts]
    #[txt.set_bbox(dict(facecolor='white',edgecolor='none',pad=0)) for txt in clabels]
    '''
    for txt in clabels:
        print(txt)
        txt.set_bbox(dict(facecolor='white',edgecolor='none',pad=0.5))
    
    for line, lvl in zip(cont.collections, cont.levels):
        #print(lvl)
        if lvl == 0:
            line.set_linestyle(':')
    '''
    
    china = shpreader.Reader(basic_dir + '/research/6.Southeastern_China_Clustering/data/shp/china-shapefiles-master/china-shapefiles-master/china_country.shp').geometries()
    ax.add_geometries(china, ccrs.PlateCarree(), 
                       facecolor='none', edgecolor='k', linewidth=1, zorder=100) # 添加中国边界
    
    x2=[108,123,123,108,108]
    y2=[18,18,34,34,18]
    plot_geometries= sgeom.LineString(zip(x2,y2))
    ax.add_geometries([plot_geometries],ccrs.PlateCarree(),facecolor='none', edgecolor='red',lw=1)#add_geometries绘制方法
    
    
    #ax.stock_img()
    ax.coastlines(resolution='50m', linewidth=1,color='k')    
    ax.set_xlim(80,140)
    ax.set_ylim(10,55)
    
    
    ax.set_title(r"$\bf{("+title_aux[j]+")}$ "+labels[j], 
                     pad=4, size=8.5, loc='left')
    if j==5:
        a=ax.get_position()
        pad=0.06
        height=0.03
        ax_f = fig.add_axes([  x_loc[0]+0.25,  a.ymin - pad,  0.5 , height ]) #长宽高
        cb=fig.colorbar(cf, orientation='horizontal',cax=ax_f)
        cb.set_ticklabels(Colors_limits,size=6.5)
        cb.outline.set_linewidth(0.8)
        ax_f.tick_params(length=1,pad=0.2)
        cb.set_label( label = "Column-integrated moisture flux ("+r'kg·m$^{-1}$ s$^{-1}$'+")",fontdict={'size':7},labelpad=1)
        
        qk = ax.quiverkey(ivtq,  0.57+0.25,a.ymin - 0.015, 200, r'100 kg·m$^{-1}$ s$^{-1}$', labelpos='S',
              coordinates='figure',zorder=200,
              fontproperties={'size':7.5})
        
fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/Figures/Figure.Sx.monthly_Z500_and_IVT.png",dpi=1500, bbox_inches='tight')
    #%%
