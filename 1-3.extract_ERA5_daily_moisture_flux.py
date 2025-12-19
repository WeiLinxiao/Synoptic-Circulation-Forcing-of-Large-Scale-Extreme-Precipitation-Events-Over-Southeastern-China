#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:34:44 2022

@author: dai
"""


import numpy as np 
import pandas as pd 
import xarray as xr
import glob 
from datetime import datetime

name=glob.glob('/home/dai/disk_1/data/ERA5_East_Asia/daily_integraded_moisture_flux/daily*')
name=sorted(name)
a=xr.open_mfdataset(name,
                    concat_dim='time',combine='nested')


####time 
year=[x[87:91] for x in name]
month=[x[91:93] for x in name]
date=[x[93:95] for x in name]
d=str(year)+'-'+str(month)


time=[str(year[i])+'-'+str(month[i])+'-'+str(date[i]) for i in range(0,15249)]
time=[datetime.strptime(i,'%Y-%m-%d') for i in time]



##################################################


q=xr.open_dataset('/home/dai/disk_1/data/ERA5_East_Asia/20200930-20201231/adaptor.mars.internal-1654009895.9951189-20895-3-bc24e626-0e42-4ce7-8529-8096c0504048.nc')
q=q.sel(level=slice(300,1000)).groupby('time.date').mean('time')
q


u=xr.open_dataset('/home/dai/disk_1/data/ERA5_East_Asia/20200930-20201231/adaptor.mars.internal-1654504581.2462623-31363-1-04f2077a-ea3e-4e20-83cf-dfcdbdcf21ba.nc')
u=u.sel(level=slice(300,1000)).groupby('time.date').mean('time')
u

v=xr.open_dataset('/home/dai/disk_1/data/ERA5_East_Asia/20200930-20201231/adaptor.mars.internal-1654506991.528387-12235-9-91130c25-d25d-4f03-8bab-9c2c1eeb42f3.nc')
v=v.sel(level=slice(300,1000)).groupby('time.date').mean('time')
v

moisture_flux2 = xr.Dataset()
#mf_u=100*(uwind['u']*spehu['q']*1/9.80666).integrate(coord = 'level')
#mf_u=100*(uwind['u']*spehu['q']*1/9.80666).integrate(coord = 'level')
moisture_flux2['u_dir'] = 100*(u['u']*q['q']*1/9.80666).integrate(coord = 'level')
moisture_flux2['v_dir'] = 100*(v['v']*q['q']*1/9.80666).integrate(coord = 'level')
moisture_flux2['mag'] = np.sqrt((moisture_flux2['u_dir']**2) + (moisture_flux2['v_dir']**2))



time=[]
for i in range(len(moisture_flux2['date'].values)):
    t=np.datetime64(moisture_flux2['date'].values[i]) 
    time.append(t)
time=np.array(time)    

moisture_flux2=moisture_flux2.rename({'date':'time'})
moisture_flux2=moisture_flux2.assign_coords({'time':time})

moisture_flux2.to_netcdf("/home/dai/disk_1/data/ERA5_East_Asia/20200930-20201231/daily_moisture_flux_20201001-20201231.nc")




###################################################

moisture_flux2=xr.open_dataset("/home/dai/disk_1/data/ERA5_East_Asia/20200930-20201231/daily_moisture_flux_20201001-20201231.nc")

b1=moisture_flux2['u_dir']
b=a['u_dir']
b=b.assign_coords({'time':time})
b2=xr.concat([b,b1], dim='time')
b2
b2.to_netcdf('/home/dai/suk/southeastern-china-EOF/data/ERA5_19790101_20201231_daily_moisture_flux_integraded_u_component.nc')

b1=moisture_flux2['v_dir']
b=a['v_dir']
b=b.assign_coords({'time':time})
b2=xr.concat([b,b1], dim='time')
b2
b2.to_netcdf('/home/dai/suk/southeastern-china-EOF/data/ERA5_19790101_20201231_daily_moisture_flux_integraded_v_component.nc')

b1=moisture_flux2['mag']
b=a['mag']
b=b.assign_coords({'time':time})
b2=xr.concat([b,b1], dim='time')
b2
b2.to_netcdf('/home/dai/suk/southeastern-china-EOF/data/ERA5_19790101_20201231_daily_moisture_flux_integraded_magnitude.nc')












