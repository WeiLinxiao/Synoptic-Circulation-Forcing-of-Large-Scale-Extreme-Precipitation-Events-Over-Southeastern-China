#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 20:55:54 2023

@author: dai
"""



import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from global_land_mask import globe
from datetime import datetime
from datetime import date
from datetime import timedelta
import cartopy.io.shapereader as shpreader
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages # for multipages pdf
from matplotlib.colors import ListedColormap, BoundaryNorm # for user-defined colorbars on matplotlib
import matplotlib.lines as mlines
import pymannkendall as mk

#%%

basic_dir = '/media/dai/DATA1/research/6.Southeastern_China_Clustering/'

prec_climatology = np.load(basic_dir + 'code/2.SEC_prec_climatology_and_trend/prec_climatology.npy',allow_pickle=True).tolist()


#%%

station = prec_climatology.get('station_location')
prec_station_annual_mean =  prec_climatology.get('annual_prec_all_station')
prec_station_climatology = prec_climatology.get('climatology_annual_prec_all_station')
prec_grid_climatology = prec_climatology.get('climatology_annual_prec_all_grid_of_SEC')


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

fig = plt.figure(figsize = (17/2.54, 9/2.54)) # 宽、高 ，单位为英寸 1英寸的2.5cm


##########################################  a  grid station SEC 多年平均降水 长期趋势##########################################
#%
#区域平均趋势对比图

grid=prec_climatology.get('regional_annual_prec_grid')
sta=prec_climatology.get('regional_annual_prec_station')

grid =grid-grid.mean()
sta =sta-sta.mean()
slope_grid,inter=np.polyfit(range(1979,2020),grid, 1)
abline_grid=[slope_grid*i + inter for i in range(1979,2020)]

slope_sta,inter=np.polyfit(range(1979,2020),sta, 1)
abline_sta=[slope_sta*i + inter for i in range(1979,2020)]


#mk.original_test(sta) 不显著


#fig = plt.figure(figsize = (10, 7))
ax= fig.add_axes([0.08, 0.0, 0.42, 0.4])

##格点散点
color_grid='#436BAA'
color_obs='#D36054'

#站点散点   
scatter2=ax.scatter(range(1979,2020),sta,
                    s=8,
           c=color_obs)
ax.plot(range(1979,2020),abline_sta,linestyle = "dashed",
        c=color_obs)

ax.plot(range(1979,2020),sta,
        linewidth=1,
        c=color_obs)

scatter1=ax.scatter(range(1979,2020),grid,
                    s=8,
           c=color_grid)
ax.plot(range(1979,2020),abline_grid,
        c=color_grid, linestyle = "dashed")
ax.plot(range(1979,2020),grid,
        linewidth=1,
        c=color_grid)


ax.hlines(0, 1979, 2020, linestyle = "dotted", color = "gray", zorder = 10, linewidth = 1.5)

plt.yticks([-200,0,200,400,600],fontsize=size_tick_label)
plt.xticks([1980,1990,2000,2010,2020],fontsize=size_tick_label)

ax.set_xlabel('year',fontsize=size_tick_title,labelpad=1)
ax.set_ylabel('prec anomaly (mm)',fontsize=size_tick_title,labelpad=-1)
ax.tick_params(length=1.5,pad=1)

#title_aux = list(map(chr, range(97, 123)))[0]

ax.set_title('(a)',  fontweight='bold',
                 pad=4, size=size_title, loc='left')

ax.margins(x=0)
#handles1, _ = scatter1.legend_elements()
#handles2, _ = scatter2.legend_elements()

one_more1 = mlines.Line2D([], [], color=color_grid, marker='o', linestyle='None', markersize = 5)
one_more2 = mlines.Line2D([], [], color=color_obs, marker='o', linestyle='None', markersize = 5)

#labels = sorted([f'{item}: {count}' for item, count in Counter(y).items()])
labels1=['Gridded slope: '+str(round(slope_grid,2)),'Obs slope: '+str(round(slope_sta,2))]



ax.legend([one_more1] + [one_more2], labels1, loc = "upper left",frameon=False,fontsize=size_legend) 

ab = np.array([sta, grid])
np.corrcoef(ab)

#ax.legend('1', loc = "lower left",frameon=False,fontsize=6) 
ax.text(0.7, 0.87, s = 'CC > '+str(round(np.corrcoef(ab)[0,1],2)) , 
        size = size_legend,  ha = 'right', transform = ax.transAxes)



##########################################  d  grid SEC performance ##########################################

#%fig = plt.figure(figsize = (10, 7)) # 宽、高
ax = fig.add_axes([0.6, 0.00, 0.3, 0.4]) 

x = [0,4000]
y = [0,4000]
a=np.array(prec_climatology.get('annual_prec_nearest_grid')).flatten()
b=np.array(prec_climatology.get('annual_prec_SEC')).flatten()

a=a[np.where(np.isnan(a)==False)[0]]
b=b[np.where(np.isnan(a)==False)[0]]
####

x2=0
for i in range(len(a)):
    x2=x2+(a[i]-b[i])**2
RMSE=np.sqrt(x2/len(a))   

x1=0
for i in range(len(a)):
    x1=x1+(a[i]-b[i])*100
Bias=x1/(len(a)*b.mean() )




ax.scatter(a, b, 
           color = (68/255, 2/255, 86/255), s = 1.5, zorder = 0)

plt.yticks([1000,1500,2000,2500,3000,3500],fontsize=size_tick_label )
plt.xticks([1000,1500,2000,2500,3000,3500],fontsize=size_tick_label )

ax.set_xlabel('Obs annual prec (mm)',fontsize=size_tick_title ,labelpad=1)
ax.set_ylabel('Gridded annual prec (mm)',fontsize=size_tick_title,labelpad=1)
ax.tick_params(length=1.5,pad=1)

p = ax.hist2d(a, b, bins = 35, cmin =5, 
          #norm = colors.LogNorm(),  #将颜色按对数坐标
          zorder = 5, cmap = "twilight_shifted" )
plt.plot(x, y, '--', zorder = 10)
ax.set_title('(b)',  fontweight='bold',
                 pad=4, size=size_title, loc='left')

ax.text(0.55, 0.07, s = 'RMSE = '+str(round(RMSE,2)) +' mm', 
        size = size_legend ,  ha = 'left', transform = ax.transAxes)
ax.text(0.55, 0.17, s = 'Bias = '+str(round(Bias,2)) +' %', 
        size = size_legend,  ha = 'left', transform = ax.transAxes)

a=ax.get_position()
pad=0.015
width=0.015
ax3_f = fig.add_axes([a.xmax + pad, a.ymin, width, (a.ymax-a.ymin) ])

cbar = plt.colorbar(p[3],cax=ax3_f)

cbar.set_label(  label = "number of days in bin",fontdict={'size':size_cbar_label },labelpad=1.5)
#标签刻度位置
#ax3_f.yaxis.set_ticks_position('right')
ax3_f.tick_params(length=1,pad=0.2)
#标签刻度值
ax3_f.set_yticklabels([10,60,110,160],size=size_cbar_label)
cbar.set_ticks([10,60,110,160])

#
fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/Figures/Figure.Sx.compare_station_and_grids.png",dpi=1500, bbox_inches='tight')