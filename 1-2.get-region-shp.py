# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:54:21 2022

@author: daisukiiiii

"""
# %%
import os
from pathlib import Path
import numpy as np
#import pandas as pd
#import xarray as xr
#import matplotlib.pyplot as plt
import geopandas as gpd
#import regionmask
#import cartopy.io.shapereader as sr
from shapely.geometry import Polygon

#%%

path="/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/shp/region/"
Path(path).mkdir(parents=True, exist_ok=True)




region_shp_file =path+ "region.shp"


region_lat = slice(18,34)
region_lon = slice(108, 123)
region_box_y = [region_lat.start, region_lat.start, region_lat.stop, region_lat.stop, region_lat.start]
region_box_x = [region_lon.start, region_lon.stop, region_lon.stop, region_lon.start, region_lon.start]


#import numpy as np
rx = np.interp( np.arange(0,100), [0, 25, 50, 75, 100], region_box_x) ## x coordinates

ry = np.interp(np.arange(0,100), [0, 25, 50, 75, 100], region_box_y) ## y coordinates

#np.linspace(1,20,20)
#import geopandas as gpd


region_shp = gpd.GeoDataFrame(index=[0], crs='epsg:4326', 
                          geometry=[Polygon(zip(rx, ry))])

region_shp.to_file(filename=region_shp_file, driver="ESRI Shapefile")

#region_mask = regionmask.mask_geopandas(region_shp, pr_prism)

a=1