#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:30:23 2023

@author: dai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:10:59 2023

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

qu=0.99
ar=0.95


#%%

#%%

colors = plt.get_cmap('tab20b')
colors2 = plt.get_cmap('tab20c')

cmap1 = [colors(i) for i in range(12,16)]
cmap1.extend([colors2(i) for i in range(4,8)])
cmap1.extend([colors(i) for i in range(8,12)])
cmap1.extend([colors2(i) for i in range(8,12)])
cmap1.extend([colors2(i) for i in range(0,4)])
cmap1.extend([colors2(i) for i in range(12,16)])
Colors_used = ListedColormap(cmap1)

#%%


#%%
f=6
title_aux = list(map(chr, range(97, 123)))[:8]
j=0

for qu in [0.99]:
    for ar in [0.95]:
        basic_dir ='/media/dai/DATA1/'
        d3_events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/4.3d_event_indentify/d3_events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
        event_sum_original = d3_events_stats.get('event_sum')
        #Summary_Stats = d3_events_stats.get('area_Summary_Stats')
        
        events_stats = np.load(basic_dir+'/research/6.Southeastern_China_Clustering/code2/5.largescale_exP_events_circulation_stats/events_stats_combineislands_'+str(qu)+'_'+str(ar)+'.npy',allow_pickle=True).tolist()
        event_all_stats = events_stats.get('event_all_stats')
        event_basic_stats = events_stats.get('event_basic_stats')
        Clustering=np.load(basic_dir+'research/6.Southeastern_China_Clustering/code2/3.EOFs/Clustering_1deg.npy',allow_pickle=True).tolist()
        for r in [3]:
            for vv in ['SLP~Z500']:
                for f in [6]:
                    pattern=Clustering.get('region'+str(r)+'_'+vv)['Clusters_'+str(f)]  ##找到结果
                    
                    sequence_f = pattern.value_counts().index  #最新的顺序
                    count_f = pattern.value_counts()*100/len(pattern)
        #%%
        fig = plt.figure(figsize = (16/2.54, 10/2.54)) # 宽、高
        
        ax = plt.axes([0.1,0.1, 0.8, 0.3] )
        ax2 = ax.twinx()
        ax4 = plt.axes([0.1,0.53, 0.8, 0.26] )
        ax5 = ax4.twinx()
        
        ax6 = plt.axes([0.1,0.81, 0.8, 0.26] )
        ax7 = ax6.twinx()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax4.grid(axis="y", linestyle="--", alpha=0.5)
        ax6.grid(axis="y", linestyle="--", alpha=0.5)
        for j in range(6):
            print(j+1)
            a= event_basic_stats[event_basic_stats['circulation_type']==sequence_f[j]]
            type_name = a.columns.values
            type_name[7]='areamean_intensity'
            
            
            ##### 面积相关的降水特性
            ax.bar(j-0.12, a.mean()[5],color =cmap1[2],width=0.16,alpha=0.7,edgecolor= cmap1[0],label='Accumulated Area')
            ax.errorbar(j-0.12, a.mean()[5], yerr=a.std()[5], capsize=3, capthick=0.5,elinewidth=1,color=cmap1[0])
            print('area '+str(a.mean()[5]))
            ax.bar(j+0.12, a.mean()[6],color =cmap1[6],width=0.16,alpha=0.7,edgecolor= cmap1[4] ,label='Projected Area')
            ax.errorbar(j+0.12, a.mean()[6], yerr=a.std()[6], capsize=3, capthick=0.5,elinewidth=1,color=cmap1[4])
            print('p_area '+str(a.mean()[6]))
            ax2.bar(j+0.36, a.mean()[4],color =cmap1[18],width=0.16,alpha=0.7,edgecolor= cmap1[16] ,label='Precipitatipn Amounts')
            ax2.errorbar(j+0.36, a.mean()[4], yerr=a.std()[4], capsize=3, capthick=0.5,elinewidth=1,color=cmap1[16])
            ax2.set_ylim((0,1.7*(10**7) ))
            print('amounts '+str(a.mean()[4]))
            #print(4*j+1)
            '''
            if j==5:
                plt.legend( bbox_to_anchor=(-1, -5),ncol=6, framealpha=0,fontsize=7,columnspacing=0.4, labelspacing=0.3,)
            '''
            
            if j ==0:
                #lines, labels = fig.axes[-2].get_legend_handles_labels()                
                ax.legend( bbox_to_anchor=(0.28, 0.8),ncol=2, framealpha=0,fontsize=7.5,columnspacing=0.4, labelspacing=0.3,)
                ax2.legend( bbox_to_anchor=(0.28, 0.72),ncol=1, framealpha=0,fontsize=7.5,columnspacing=0.4, labelspacing=0.3,)

            
            
            #风向相关的特性
            ax3 = plt.axes([0.12+0.13333*j,0.4,0.1,0.1],projection='polar' )
            b = a['direction']
            c=b.copy()
            c.index= range(len(c))
            for i in range(len(c)):
                c.iloc[i]= 30*np.arange(0,12,1)[np.argmin(np.abs(b.iloc[i]-30*np.arange(0,12,1)))]
                if b.iloc[i]>345:
                    c.iloc[i]=0
            uni,count = np.unique(c,return_counts=True)
            count = 100*count/len(b)        
                    #
        
            ax3.set_theta_zero_location('N')
            ax3.set_theta_direction(-1)
            ax3.bar((uni/360)*2*np.pi,count,alpha=0.9,width=0.51,color=cmap1[3],edgecolor='black')
            ax3.xaxis.set_visible(False)
            ax3.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
            #ax3.set_xlabel([0,180])
            #print(4*j+1)
            
            #%面积和历时相关的特性
            a.columns=type_name
            vio1 = ax4.violinplot(dataset=a['areamean_intensity'].astype(float),positions=[j-0.15],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
            vio2 = ax5.violinplot(dataset=a['lifespan'].astype(float),positions=[j+0.15],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
            print('intensity '+str(a['areamean_intensity'].astype(float).median() ))
            print('lifespan '+str(a['lifespan'].astype(float).median() ))
            #vio1 = ax.violinplot(dataset=a[~np.isnan(a['intensity'])]['intensity'].astype(float),positions=[j-0.15],showmeans=False,showextrema=False,showmedians=True,widths=0.8,vert=False)
        #
            # Make the violin body blue with a red border:
            for vp in vio1['bodies']:
                vp.set_facecolor(cmap1[2])
                vp.set_edgecolor(cmap1[2])
                vp.set_linewidth(1)
                vp.set_alpha(0.5)
            # Make all the violin statistics marks red:
            vp = vio1['cmedians']
            vp.set_edgecolor(cmap1[1])
            vp.set_facecolor(cmap1[1])
            vp.set_linewidth(2)
            vp.set_alpha(1)
            
            # Make the violin body blue with a red border:
            for vp in vio2['bodies']:
                vp.set_facecolor(cmap1[17])
                vp.set_edgecolor(cmap1[17] )
                vp.set_linewidth(1)
                vp.set_alpha(0.5)
            # Make all the violin statistics marks red:
            vp = vio2['cmedians']
            vp.set_edgecolor(cmap1[16])
            vp.set_facecolor(cmap1[16])
            vp.set_linewidth(2)
            #vp.set_linestyle('--')
            vp.set_alpha(1)
            
            
            #
            #%面积和历时相关的特性
            a.columns=type_name
            #vio1 = ax4.violinplot(dataset=a['speed'].astype(float),positions=[j-0.15],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
            #vio2 = ax5.violinplot(dataset=a['lifespan'].astype(float),positions=[j+0.15],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
        #
            vio1 = ax6.violinplot(dataset=a[~np.isnan(a['speed'])]['speed'].astype(float),positions=[j-0.15],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
            vio2 = ax7.violinplot(dataset=a[~np.isnan(a['distance'])]['distance'].astype(float),positions=[j+0.15],showmeans=False,showextrema=False,showmedians=True,widths=0.3,vert=True)
            print('speed '+str(a[~np.isnan(a['speed'])]['speed'].astype(float).median() ))
            print('distance '+str(a[~np.isnan(a['distance'])]['distance'].astype(float).median() ))
            # Make the violin body blue with a red border:
            for vp in vio1['bodies']:
                vp.set_facecolor(cmap1[2])
                vp.set_edgecolor(cmap1[2])
                vp.set_linewidth(1)
                vp.set_alpha(0.5)
            # Make all the violin statistics marks red:
            vp = vio1['cmedians']
            vp.set_edgecolor(cmap1[1])
            vp.set_facecolor(cmap1[1])
            vp.set_linewidth(2)
            vp.set_alpha(1)
            
            # Make the violin body blue with a red border:
            for vp in vio2['bodies']:
                vp.set_facecolor(cmap1[17])
                vp.set_edgecolor(cmap1[17] )
                vp.set_linewidth(1)
                vp.set_alpha(0.5)
            # Make all the violin statistics marks red:
            vp = vio2['cmedians']
            vp.set_edgecolor(cmap1[16])
            vp.set_facecolor(cmap1[16])
            vp.set_linewidth(2)
            #vp.set_linestyle('--')
            vp.set_alpha(1)
            
            
            
            
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.yaxis.set_ticks(np.arange(0,2.1,0.5)*(10**6))
        ax.yaxis.set_ticklabels(np.arange(0,2.1,0.5),fontsize=7.5)
        ax.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
        ax.set_ylabel(  "Area \n(× 10 $^6$ km$^2$)",fontdict={'size':7.5},labelpad=2)
        
        ax2.yaxis.set_ticks(np.arange(0,2.1,0.5)*(10**8))
        ax2.yaxis.set_ticklabels(np.arange(0,2.1,0.5),fontsize=7.5)
        ax2.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
        ax2.set_ylabel(  "Precipitation Amounts \n(× 10$^7$ km$^2$ × mm)",fontdict={'size':7.5},labelpad=2)
        
        ax.xaxis.set_ticks(np.arange(0,6,1))
        ax.xaxis.set_ticklabels(['Cl. '+str(j+1) for j in range(6)],fontsize=7.5)
        
        '''
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        '''
        
        ax4.spines['bottom'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        #ax4.yaxis.set_ticks(np.arange(30,71,10))
        #ax4.yaxis.set_ticklabels(np.arange(30,71,10),fontsize=7.5)
        ax4.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
        ax4.set_ylabel(  "Intensity (mm)",fontdict={'size':7.5},labelpad=2)
        #ax5.yaxis.set_ticks(np.arange(0,21,5))
        #ax5.yaxis.set_ticklabels(np.arange(0,21,5),fontsize=7.5)
        ax5.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
        ax5.set_ylabel(  "Lifespan (day)",fontdict={'size':7.5},labelpad=2)
        ax4.xaxis.set_visible(False)
        ax5.xaxis.set_visible(False)
        
        ax6.spines['bottom'].set_visible(False)
        ax7.spines['bottom'].set_visible(False)
        ax6.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
        ax7.tick_params(axis='y', labelsize=6.5,pad=0.5,length=1) 
        ax6.xaxis.set_visible(False)
        ax7.xaxis.set_visible(False)
        ax6.set_ylabel(  "Movement speed (km)",fontdict={'size':7.5},labelpad=2)
        ax7.set_ylabel(  "Movement distance (km)",fontdict={'size':7.5},labelpad=2)
           
        ax6.set_title('LSExP stats of different circulation patterns',fontsize=8.5,loc='left')
        
        ax6.annotate('Movement\nDirection',xy=(0.03,0.41), size=7.5, xycoords='figure fraction')
        
        fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/Figures/Figure.4_"+str(qu)+'_'+str(ar)+".png",dpi=1500, bbox_inches='tight')
        fig.savefig("/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/Figures/Figure.4_"+str(qu)+'_'+str(ar)+".pdf",dpi=1500, bbox_inches='tight')