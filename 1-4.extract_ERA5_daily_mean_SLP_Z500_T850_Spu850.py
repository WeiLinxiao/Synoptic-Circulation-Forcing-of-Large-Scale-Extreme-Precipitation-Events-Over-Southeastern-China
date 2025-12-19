# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:57:06 2022

@author: daisukiiiii
"""

import numpy as np 
import pandas as pd 
import xarray as xr
import multiprocessing # 这个要替换掉原来的。
import tqdm # timing
import glob
import os
from datetime import datetime

os.chdir('/media/dai/My Passport1/MSLP/')
#读取列表下所有文件
date=os.listdir() #注意顺序
#找到所有原始文件
date1=[]
for x in date: 
    if os.path.isfile(x) and 'era5.mslp.' in x: 
        date1.append(x)
        
##读取其中的日期
date=np.array([ x[10:18] for x in date1 ])
date=np.sort(date)

##comb函数：将小时数据处理为daily数据
def comb(i):
   
    import xarray as xr
    import pandas as pd 
    grb_file_name = 'era5.mslp.'+str(i)+'.nc' 
    prec=xr.open_dataarray(grb_file_name) 
    ##选择华南区域
    prec=prec.sel(longitude=slice(70,140),latitude=slice(60,0) )
    ##提取该xarray的时间这一维度,
    time=prec.time.values
    #转换为pd包的时间格式
    dates_all = pd.to_datetime(prec.time.values) 
    ##转换为只精确到date的结果
    dates_all = pd.to_datetime(dates_all.strftime('%Y%m%d')) 
    ###此时将转变后的时间重新插入到原始数据的时间维度去，
    prec = prec.assign_coords({'time': dates_all}) 
    ##get sum of variable
    prec=prec.mean(dim='time')
    return prec

##并行运算
if __name__=='__main__':
   pool = multiprocessing.Pool(48) # object for multiprocessing
   
pp = list(tqdm.tqdm(pool.imap(comb, date), 
                           total=len(date), position=0, leave=True))
pool.close()
##重新调整其维度，将list重塑为时间维度
prec = xr.concat(pp, dim=pd.Index(date, name='time') ) 

#a1=prec['time'].values

prec.to_netcdf('~/suk/southeastern-china-EOF/data/ERA5_19790101_20200930_daily_mslp.nc') 



##################################################################################################



os.chdir('/media/dai/My Passport1/geopotential/')
#读取列表下所有文件
date=os.listdir() #注意顺序
#找到所有原始文件
date1=[]
for x in date: 
    if os.path.isfile(x) and 'era5.geopotential.' in x: 
        date1.append(x)
        
##读取其中的日期
date=np.array([ x[18:26] for x in date1 ])
date=np.sort(date)

##comb函数：将小时数据处理为daily数据
def comb(i):
    import xarray as xr
    import pandas as pd 
    grb_file_name = 'era5.geopotential.'+str(i)+'.nc' 
    prec=xr.open_dataarray(grb_file_name) 
    ##选择华南区域
    prec=prec.sel(longitude=slice(70,140),latitude=slice(60,0),level=500 )
    ##提取该xarray的时间这一维度,
    #time=prec.time.values
    #转换为pd包的时间格式
    dates_all = pd.to_datetime(prec.time.values) 
    ##转换为只精确到date的结果
    dates_all = pd.to_datetime(dates_all.strftime('%Y%m%d')) 
    ###此时将转变后的时间重新插入到原始数据的时间维度去，
    prec = prec.assign_coords({'time': dates_all}) 
    ##求日降水之和
    prec=prec.mean(dim='time')
    return prec

##并行运算
if __name__=='__main__':
   pool = multiprocessing.Pool(48) # object for multiprocessing
   
pp = list(tqdm.tqdm(pool.imap(comb, date), 
                           total=len(date), position=0, leave=True))
pool.close()
##重新调整其维度，将list重塑为时间维度
prec = xr.concat(pp, dim=pd.Index(date, name='time') ) 

#a1=prec['time'].values

prec.to_netcdf('~/suk/southeastern-china-EOF/data/ERA5_19790101_20200930_daily_Z500.nc') 


############################################### Temp 850 ###################################################



os.chdir('/media/dai/My Passport1/temperature/')
#读取列表下所有文件
date=os.listdir() #注意顺序
#找到所有原始文件
date1=[]
for x in date: 
    if os.path.isfile(x) and 'era5.temperature.' in x: 
        date1.append(x)
        
##读取其中的日期
date=np.array([ x[17:25] for x in date1 ])
date=np.sort(date)

##comb函数：将小时数据处理为daily数据
def comb(i):
    import xarray as xr
    import pandas as pd 
    grb_file_name = 'era5.temperature.'+str(i)+'.nc' 
    prec=xr.open_dataarray(grb_file_name) 
    ##选择华南区域
    prec=prec.sel(longitude=slice(70,140),latitude=slice(60,0),level=850 )
    ##提取该xarray的时间这一维度,
    #time=prec.time.values
    #转换为pd包的时间格式
    dates_all = pd.to_datetime(prec.time.values) 
    ##转换为只精确到date的结果
    dates_all = pd.to_datetime(dates_all.strftime('%Y%m%d')) 
    ###此时将转变后的时间重新插入到原始数据的时间维度去，
    prec = prec.assign_coords({'time': dates_all}) 
    ##求日降水之和
    prec=prec.mean(dim='time')
    return prec

##并行运算
if __name__=='__main__':
   pool = multiprocessing.Pool(48) # object for multiprocessing
   
pp = list(tqdm.tqdm(pool.imap(comb, date), 
                           total=len(date), position=0, leave=True))
pool.close()
##重新调整其维度，将list重塑为时间维度
prec = xr.concat(pp, dim=pd.Index(date, name='time') ) 

#a1=prec['time'].values

prec.to_netcdf('~/suk/southeastern-china-EOF/data/ERA5_19790101_20200930_daily_T850.nc') 




############################################### specific_humidity 850 ###################################################



os.chdir('/media/dai/My Passport/specific_humidity/')
#读取列表下所有文件
date=os.listdir() #注意顺序
#找到所有原始文件
date1=[]
for x in date: 
    if os.path.isfile(x) and 'era5.specific_humidity.' in x: 
        date1.append(x)
        
##读取其中的日期
date=np.array([ x[23:31] for x in date1 ])
date=np.sort(date)

##comb函数：将小时数据处理为daily数据
def comb(i):
    import xarray as xr
    import pandas as pd 
    grb_file_name = 'era5.specific_humidity.'+str(i)+'.nc' 
    prec=xr.open_dataarray(grb_file_name) 
    ##选择华南区域
    prec=prec.sel(longitude=slice(70,140),latitude=slice(60,0),level=850 )
    ##提取该xarray的时间这一维度,
    #time=prec.time.values
    #转换为pd包的时间格式
    dates_all = pd.to_datetime(prec.time.values) 
    ##转换为只精确到date的结果
    dates_all = pd.to_datetime(dates_all.strftime('%Y%m%d')) 
    ###此时将转变后的时间重新插入到原始数据的时间维度去，
    prec = prec.assign_coords({'time': dates_all}) 
    ##求日降水之和
    prec=prec.mean(dim='time')
    return prec

##并行运算
if __name__=='__main__':
   pool = multiprocessing.Pool(48) # object for multiprocessing
   
pp = list(tqdm.tqdm(pool.imap(comb, date), 
                           total=len(date), position=0, leave=True))
pool.close()
##重新调整其维度，将list重塑为时间维度
prec = xr.concat(pp, dim=pd.Index(date, name='time') ) 

#a1=prec['time'].values

prec.to_netcdf('~/suk/southeastern-china-EOF/data/ERA5_19790101_20200930_daily_SpeHu850.nc') 




'''
b=xr.open_dataarray('~/suk/southeastern-china-EOF/data/ERA5_19790101_20200930_daily_Z500.nc')
time1=[[i[0:4] for i in b['time'].values] [j]  +  
       '-' + [i[4:6] for i in b['time'].values][j] +
       '-' + [i[6:8] for i in b['time'].values][j] 
       for j in range(0,15249)]

time2=[datetime.strptime(time1[i],'%Y-%m-%d') for i in range(0,15249)]



a=xr.open_dataarray("/media/dai/My Passport/adaptor.mars.internal-1654005980.1430705-15423-9-70d6d800-ad87-464d-a65b-d3cbd938479b.nc")
a=a.groupby('time.date').mean(dim='time')
time=a['date'].values
datetime.strftime(time[0], '%Y-%m-%d')
time_later=[time[i] for i in range(len(time))]



t=pd.Index(time2,name='time')
t2=pd.Index(time_later,name='time')
pd.concat([t,t2])


time2[15200:15300]


#time3=datetime.strptime(b['time'].values[0],'%y%m%d')
'''

