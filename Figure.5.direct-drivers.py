#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:21:08 2023

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
#%
#%%
var_list =['ivt','ivt_anomaly','iwv','iwv_anomaly','omega','omega_anomaly','slp','slp_anomaly','z500','z500_anomaly','t850','t850_anomaly']
var_list =['ivt','iwv','omega','slp','slp_anomaly','z500','z500_anomaly','t850','t850_anomaly']

var_quantile_list = np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/9.contribution/var_quantile_list.npy',allow_pickle=True).tolist()    
    
var_quantile_list_all_day = np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/9.contribution/var_quantile_list_all_day.npy',allow_pickle=True).tolist()       

area_mean_stats = np.load('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/9.contribution/area_mean_stats.npy',allow_pickle=True).tolist()       
#%%



fig = plt.figure(figsize = (16/2.54, 10/2.54)) # 宽、高

ax1 = plt.axes([0.1,0.5, 0.8, 0.3] )
ax = plt.axes([0.1,0.1, 0.8, 0.3] )
ax.grid(axis="y", linestyle="--", alpha=0.5,zorder=0)
ax1.grid(axis="y", linestyle="--", alpha=0.5,zorder=0)
for j in range(6):
    
   
    '''
    time = percent_event.iloc[np.where(percent_event.values==sequence_f[j])[0]].index
    ###cl下的exP
    LSExP_cl = large_scale_exP.sel(time=time)
    LSExP_cl_mask = LSExP_cl.copy()
    ##exp位置
    LSExP_cl_mask = (LSExP_cl_mask>0).astype(float).where(LSExP_cl_mask>0)
    
    stats = pd.DataFrame(    pd.Series(     LSExP_cl.mean(('lon','lat'))   ) ) #平均降水量
    for i in var_list:
        print(i)
        ''''''
        a = vars()[i].sel(
            longitude=slice(108,123),
            latitude=slice(34,18.1))
        b=vars()[i+'_quan'].sel(
            longitude=slice(108,123),
            latitude=slice(34,18.1))
        var_cl = cal_variable_days_quantile(j,a,b)
        ''''''
        var_cl = var_quantile_list.get(i+'_'+str(j))
        
        stats = pd.concat([ stats,
                   pd.Series((var_cl*LSExP_cl_mask).mean(('lon','lat')).values).astype(float) ],axis=1)
    #%
    #stats.columns = ['prec','ivt','ivt_anomaly','iwv','iwv_anomaly','omega','omega_anomaly','slp','slp_anomaly','z500','z500_anomaly','t850','t850_anomaly']
    stats.columns = ['prec','ivt','iwv','omega','slp','slp_anomaly','z500','z500_anomaly','t850','t850_anomaly']
    #%
    stats2=stats.iloc[:,[2,3,1]]
    #plt.bar(stats2.columns,(stats2>=90).sum()*100/len(stats2))
    
    #plt.scatter(stats['prec'],stats['t850_anomaly'])
    '''
    '''
    box1=ax.boxplot( area_mean_stats.get(str(j)+'vars')['slp_anomaly'],positions=[j-0.25])
    box1=ax.boxplot( area_mean_stats.get(str(j)+'vars')['z500_anomaly'],positions=[j])
    box1=ax.boxplot( area_mean_stats.get(str(j)+'vars')['t850_anomaly'],positions=[j+0.25])
    '''
    '''
    vio1 = ax.violinplot(  area_mean_stats.get(str(j)+'vars')['iwv'],positions=[j-0.25],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
    vio2 = ax.violinplot(area_mean_stats.get(str(j)+'vars')['omega'],positions=[j],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
    vio3 = ax.violinplot(area_mean_stats.get(str(j)+'vars')['ivt'].astype(float),positions=[j+0.25],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
    
    for vp in vio1['bodies']:
        vp.set_facecolor(cmap1[3])
        vp.set_edgecolor(cmap1[3])
        vp.set_linewidth(1)
        vp.set_alpha(0.8)
    # Make all the violin statistics marks red:
    vp = vio1['cmedians']
    vp.set_edgecolor(cmap1[1])
    vp.set_facecolor(cmap1[1])
    vp.set_linewidth(2)
    vp.set_alpha(1)
    vp.set_linestyle('--')
    
        
    for vp in vio2['bodies']:
        vp.set_facecolor(cmap1[6])
        vp.set_edgecolor(cmap1[6])
        vp.set_linewidth(1)
        vp.set_alpha(0.8)
    # Make all the violin statistics marks red:
    vp = vio2['cmedians']
    vp.set_edgecolor(cmap1[4])
    vp.set_facecolor(cmap1[4])
    vp.set_linewidth(2)
    vp.set_alpha(1)
    vp.set_linestyle('--')
    
    for vp in vio3['bodies']:
        vp.set_facecolor(cmap1[18])
        vp.set_edgecolor(cmap1[18])
        vp.set_linewidth(1)
        vp.set_alpha(1)
    # Make all the violin statistics marks red:
    vp = vio3['cmedians']
    vp.set_edgecolor(cmap1[16])
    vp.set_facecolor(cmap1[16])
    vp.set_linewidth(2)
    vp.set_alpha(1)
    vp.set_linestyle('--')

    '''
    a= area_mean_stats.get(str(j)+'vars')
    
    ax1.bar([j-0.2],((a>=99).sum()*100/len(a))['iwv'],width=0.2,color=[cmap1[3]],label='Integrated Water Vapor',edgecolor='black')
    ax1.bar([j],((a>=99).sum()*100/len(a))['omega'],width=0.2,color=[cmap1[6]],label='Vertival Velocity',edgecolor='black')
    ax1.bar([j+0.2],((a>=99).sum()*100/len(a))['ivt'],width=0.2,color=[cmap1[18]],label='Integrated Water Vapor Transport',edgecolor='black')
    print(((a>=99).sum()*100/len(a))['iwv'])
    if j ==0:
        #lines, labels = fig.axes[-2].get_legend_handles_labels()                
        ax1.legend( bbox_to_anchor=(0.8, -1.5),ncol=4, framealpha=0,fontsize=7.5,columnspacing=0.4, labelspacing=0.3,)

    ax.bar([j-0.2],((a<=5).sum()*100/len(a))['slp_anomaly'],width=0.2,color=[cmap1[3]],label='Anomaly SLP',edgecolor='black')
    ax.bar([j],((a<=5).sum()*100/len(a))['z500_anomaly'],width=0.2,color=[cmap1[6]],label='Anomaly Z500',edgecolor='black')
    ax.bar([j+0.2],((a<=5).sum()*100/len(a))['t850_anomaly'],width=0.2,color=[cmap1[18]],label='Anomaly T850',edgecolor='black')
    

    #ax.bar()
    if j ==0:
        #lines, labels = fig.axes[-2].get_legend_handles_labels()                
        ax.legend( bbox_to_anchor=(0.8, -1.3),ncol=4, framealpha=0,fontsize=7.5,columnspacing=0.4, labelspacing=0.3,)
        
        
    ax.xaxis.set_ticks(np.arange(0,6,1))
    ax.xaxis.set_ticklabels(['Cl. '+str(j+1) for j in range(6)],fontsize=6.5)
    ax.yaxis.set_ticks(np.arange(0,51,10))
    ax.yaxis.set_ticklabels(np.arange(0,51,10),fontsize=7.5)
    ax.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
    #ax.set_ylabel(  "Relative change (%)",fontdict={'size':7.5},labelpad=2)
    #b=b.where(b>keep_value)    
    #plt.contourf(mask.longitude,mask.latitude, b)
    
    ax1.xaxis.set_ticks([])
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticks(np.arange(0,61,10))
    ax1.yaxis.set_ticklabels(np.arange(0,61,10),fontsize=7.5)
    ax1.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
    #ax1.set_ylabel(  "Relative change (%)",fontdict={'size':7.5},labelpad=2)
    
    ax1.set_title(r"$\bf{(a)}$ Percentage of extreme (≥99) variable days in LSExP event days (%)",
                     pad=4, size=8, loc='left')
    ax.set_title(r"$\bf{(b)}$ Percentage of extreme (≤5) variable days in LSExP event days (%)",
                     pad=4, size=8, loc='left')
    fig.savefig('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/Figures/Figure.5.png',dpi=1500, bbox_inches='tight')
    fig.savefig('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/Figures/Figure.5.pdf',dpi=1500, bbox_inches='tight')

#%%

j=5

    
time = percent_event.iloc[np.where(percent_event.values==sequence_f[j])[0]].index
###cl下的exP
LSExP_cl = large_scale_exP.sel(time=time)
LSExP_cl_mask = LSExP_cl.copy()
##exp位置
LSExP_cl_mask = (LSExP_cl_mask>0).astype(float).where(LSExP_cl_mask>0)
p = LSExP_cl.mean(('lon','lat'))
    

p_stats=[]
for i in range(len(LSExP_cl_mask)):
    if np.isnan(p[i].values)==False: 
        p_stats.append(p[i].values.astype(float))

a= area_mean_stats.get(str(j)+'vars')


b=((a['omega']>=90)&(a['ivt']>=90)).astype(int)
p_stats_omega_iwv = pd.Series(p_stats,index=b.index)[b==1]
#b=b[b==1]

plt.scatter(p_stats,a['ivt'],)
plt.scatter(p_stats,a['omega'],)

plt.scatter(p_stats_omega_iwv,a['omega'][b==1],color='red')

#%%

j=1

    
time = percent_event.iloc[np.where(percent_event.values==sequence_f[j])[0]].index
###cl下的exP
LSExP_cl = large_scale_exP.sel(time=time)
LSExP_cl_mask = LSExP_cl.copy()
##exp位置
LSExP_cl_mask = (LSExP_cl_mask>0).astype(float).where(LSExP_cl_mask>0)
p = LSExP_cl.mean(('lon','lat'))
    

p_stats=[]
for i in range(len(LSExP_cl_mask)):
    if np.isnan(p[i].values)==False: 
        p_stats.append(p[i].values.astype(float))

a= area_mean_stats.get(str(j)+'vars')

b=pd.concat([pd.Series(p_stats,index=a.index,name='prec'),a],axis=1)

b=b.sort_values(by='prec',ascending=False)

plt.scatter(range(len(b)),np.repeat(1,len(b)),alpha=b['iwv']*0.9/b['iwv'].max())



b=((a['omega']>=90)&(a['ivt']>=90)).astype(int)
p_stats_omega_ivt = pd.Series(p_stats,index=b.index)[b==1]

b=((a['omega']>=90)&(a['iwv']>=90)).astype(int)
p_stats_omega_iwv = pd.Series(p_stats,index=b.index)[b==1]

b=((a['ivt']>=90)&(a['iwv']>=90)).astype(int)
p_stats_ivt_iwv = pd.Series(p_stats,index=b.index)[b==1]

b=((a['omega']>=90)&(a['iwv']>=90)&(a['ivt']>=90)).astype(int)
p_stats_omega_iwv_ivt = pd.Series(p_stats,index=b.index)[b==1]

plt.scatter(p_stats,p_stats,s=2)
plt.scatter(p_stats_omega_ivt,p_stats_omega_ivt,s=2)
plt.scatter(p_stats_omega_iwv,p_stats_omega_iwv,s=2)
plt.scatter(p_stats_ivt_iwv,p_stats_ivt_iwv,s=2)
plt.scatter(p_stats_omega_iwv_ivt,p_stats_omega_iwv_ivt,s=2)