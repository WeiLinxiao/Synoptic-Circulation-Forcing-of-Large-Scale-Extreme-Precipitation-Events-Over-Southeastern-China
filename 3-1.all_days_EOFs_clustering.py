#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:45:07 2023

@author: dai
"""

import multiprocessing # parallel processing
import tqdm # timing
from datetime import datetime # timing
from pathlib import Path # creation of dictionaries
import warnings # for suppressing RuntimeWarning
import glob
# basic libraries for data analysis
import numpy as np 
import pandas as pd 
import xarray as xr
import os
from itertools import combinations, product
# specialized libraries
from eofs.xarray import Eof # EOF analysis
from sklearn.cluster import KMeans # K-means clustering
from scipy.stats import binom # binomial distribution for significance testing of extremes and large-scale patterns


#%%
Area_used = {'region1': [40, 95, 10, 135], 
             'region2': [40, 100, 10, 140], 
             'region3': [40, 100, 10, 130],
             'region4': [40, 90, 10, 140],}

variables_used = ['SLP', 'T850', 'Z500']

Var_ex = 90 # define the minimum total variance [0-100] that the subset of kept EOFs should explain
Clusters_used = [4,6,8,9] # clusters for Kmeans (4 for also analysing the common EuroAtlantic regimes)
interval=1.

name=glob.glob('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/ERA5_variables/smooth*')
name=sorted(name)

Anomalies = {variables_used[0]: 0, 
             variables_used[1] : 1, 
             variables_used[2] : 2,
             }
#%%

T850=xr.open_dataarray(name[1]).interp(longitude=np.arange(70.,141.,interval),latitude=np.arange(60.,-1,-interval))
Z500=xr.open_dataarray(name[2]).interp(longitude=np.arange(70.,141.,interval),latitude=np.arange(60.,-1,-interval))
SLP=xr.open_dataarray(name[5]).interp(longitude=np.arange(70.,141.,interval),latitude=np.arange(60.,-1,-interval))


T850 = T850[(T850['time.month']>=4) & (T850['time.month']<=9),:,:]
Z500 = Z500[(Z500['time.month']>=4) & (Z500['time.month']<=9),:,:]
SLP = SLP[(SLP['time.month']>=4) & (SLP['time.month']<=9),:,:]

for var in variables_used:
    Anomalies [var] = vars()[var]

#%%


def eof_analysis(input_data):
    
    area_subset = input_data[0] # name of area_used (based on the keys of the "Area_used" dictionary)
    area_subset = Area_used[area_subset] # list with the cordinates of the boundary box of the selected area
    variable = input_data[1] # name of variable used (based on the keys of the "Anomalies" dictionary)
    
    dataset_used = Anomalies[variable].copy(deep=True) # dataset to be used for the analysis
    
    # subset area of interest 
    dataset_used = dataset_used.sel(latitude=slice(area_subset[0], area_subset[2]), 
                                    longitude=slice(area_subset[1], area_subset[3]))    
    
    
    coslats = np.cos(np.deg2rad(dataset_used.latitude.values)).clip(0, 1) # coslat for weights on EOF
    wgts = np.sqrt(coslats)[..., np.newaxis] # calculation of weights
    solver = Eof(dataset_used, weights=wgts) # EOF analysis of the subset
    
    N_eofs = int(np.searchsorted(np.cumsum(solver.varianceFraction().values), Var_ex/100)) # n. of EOFs needed
    N_eofs += 1 # add 1 since python does not include the last index of a range
    print(input_data[0])
    print(input_data[1])
    print(N_eofs)
    EOFS = solver.eofs(neofs=N_eofs)
    PCS = pd.DataFrame(solver.pcs(npcs=N_eofs).values, index=dataset_used.time.values)
    VARS = solver.varianceFraction(neigs=N_eofs).values*100
    NOR = solver.northTest(neigs=N_eofs, vfscaled=True).values*100
    
    return {'EOFS': EOFS, 'PCS': PCS, 'VARS': VARS,'NOR':NOR}



#%%

#pool = multiprocessing.Pool(processes = 12)
Combs_used = list(product(Area_used.keys(), variables_used)) # generate all combinations of area and variable
#EOF_analysis = list(tqdm.tqdm(pool.imap(eof_analysis, Combs_used), total=len(Combs_used), position=0, leave=True))
#pool.close()

EOF_analysis=[]
for i in Combs_used:
    print(i)
    a=eof_analysis(i)
    EOF_analysis.append(a)

Combs_used = ['_'.join(i) for i in Combs_used]
EOF_analysis = {Combs_used[i_c]: i_eof for i_c, i_eof in enumerate(EOF_analysis)}

#%%
np.save('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/3.EOFs/EOF_analysis.npy',EOF_analysis) 

#%%
def PC_norm(var_used):
    
    ' Normalize PCs based on standard deviation and weight them based on the % of explained variance'
    
    PCs = EOF_analysis[var_used]['PCS'] # extract PCs
    Stand = PCs/PCs.std() # standardize PCs
    
    # normalize per sqrt of variance so K-means distance is weighted based on the importance of each PC to expl. var.
    variance = EOF_analysis[var_used]['VARS']
    
    return Stand*np.sqrt(variance)

def combo_clusters(input_data):
    
    area_used = input_data[0] # area_subset used
    var_used = input_data[1] # variable used
    
    var_used = list( np.array(var_used).flatten() ) # consistent format; always list
    var_used = [area_used+'_'+i_var for i_var in var_used] # add info about area, so it is same name as EOF keys
    
    """ 
    Get the PCs of interest. If only 1 variable, then use the actual PCs, and if more than 1, then use normalized PCs.
    In fact the results with actual data or normalized for only 1 variable are practically the same (~99% similarity),
    but the actual ones are prefered for reducing additional computations that potentially lead to rounding errors.
    """
    if len(var_used) == 1:
        PCs = EOF_analysis[var_used[0]]['PCS']
    else:
        all_PCs = [PC_norm(i) for i in var_used] # make a list with the PCs of all variables of interest
        PCs = pd.concat(all_PCs, axis=1) # concatenate all the PCs to the final DF
    PCs.fillna(0,inplace = True)
    
    Col_names = ['Clusters_'+str(i_c) for i_c in Clusters_used]
    Labels_all = pd.DataFrame(np.nan, columns=Col_names, index=PCs.index)
    
    for i_c, clusters_used in enumerate(Clusters_used):
        
        KM_cluster = KMeans(n_clusters=clusters_used)
        np.random.seed(10) # set always the same seed for reproducibility
        KM_cluster.fit(PCs)
        Labels_all.iloc[:, i_c] = KM_cluster.labels_
        
    return Labels_all


def comb_lists(r):
    
    ' Use the combinations function to generate all combs of used variables from only 1, up to all of them together '
    
    data = combinations(variables_used, r)
    
    return [list(i) for i in data]

#%%%


All_combs = [comb_lists(i) for i in range(1, len(variables_used)+1)] # create list with all combinations of variables
All_combs = [j for i in All_combs for j in i] # concat the sublists
All_combs = [ All_combs[i] for i in [3,4,5,6] ]
All_combs = list(product(Area_used.keys(), All_combs)) # final list with all combinations of variables and areas


#pool = multiprocessing.Pool() # object for multiprocessing
#Clustering = list(tqdm.tqdm(pool.imap(combo_clusters, All_combs), total=len(All_combs), position=0, leave=True))
#pool.close()

Clustering=[]
for i in All_combs:
    a=combo_clusters(i)
    Clustering.append(a)
    print(i)

#%%
# change format so it can be used as dictionary key; format: <area_used>_<var1>~<var2>~<varN>, var2, ..N if avail.
All_combs = [[i, '~'.join(j)]  for i, j in All_combs]
All_combs = ['_'.join(i) for i in All_combs]

Clustering = {All_combs[i_c]: i_clustering for i_c, i_clustering in enumerate(Clustering)}

# save data used for plots (identified from the results of the last part of this script; Check Script4 Aux. Figure)

np.save('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code/3.EOFs/Clustering_1deg',Clustering)