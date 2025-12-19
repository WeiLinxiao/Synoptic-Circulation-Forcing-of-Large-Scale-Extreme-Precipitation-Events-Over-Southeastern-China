#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:39:39 2023

@author: dai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:11:11 2023

@author: Xinxin Wu
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


#%% ------------------------------ parameters -----------------------------

lon_w=108
lon_e=123
lat_n=34
lat_s=18


#%% 1. get southeastern China prec  (~80 mins)
for y in range(1979,2020):
    print(y)
    p=xr.open_dataarray('/media/dai/DATA1/China/1km/China_1km_prep_'+str(y)+'.nc').sel(
        lon=slice(lon_w,lon_e),
        lat=slice(lat_n,lat_s))
    p.to_netcdf('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_'+str(y)+'.nc')
    

#%% 2. distribution of annual mean prec from 1979-2019 (~40 mins)

p_annual = []
for y in range(1979,2020):
    print(y)
    p = xr.open_dataarray ('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_prep_'+str(y)+'.nc').sum('time')
    p = p.where(p>0)
    p_annual.append(p)
    
p_annual_1 = xr.concat(p_annual, dim=pd.Index(range(1979,2020),name='year' ) )

p_annual_1.to_netcdf('/media/dai/disk2/suk/Data/China_1km/China_1km_SEC_annual_prec_1979_2019.nc')

