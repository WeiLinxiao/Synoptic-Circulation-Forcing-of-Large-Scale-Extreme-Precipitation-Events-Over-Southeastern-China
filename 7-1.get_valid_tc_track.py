# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:20:50 2023

@author: daisukiiiii
"""
import numpy as np
import pandas as pd

#%%
tc_valid = pd.DataFrame()
s=1
for y in range(1979,2020):
    print(y)
    a =pd.read_csv('/media/dai/DATA1/research/6.Southeastern_China_Clustering/data/CMABSTdata/CH'+str(y)+'BST.txt',sep='\s+',header=None)
    tc_number = np.where(a.iloc[:,0]==66666)[0]
    tc_number = np.concatenate([tc_number,[len(a)]])
    #%
    
    
    for j in range(len(tc_number)-1):
        
        a1 = a.iloc[tc_number[j]+1:tc_number[j+1],:] #原始的整个时间
        a1_date = [str(a1[0].iloc[i])[0:8] for i in range(len(a1))] #找出其每日时间
        a1_uni_valid = np.unique(a1_date,return_counts=True)[0][np.unique(a1_date,return_counts=True)[1]>=4] #找出每日至少有2个轨迹的事件
        tc_event_valid = a1.iloc[[i for i in range(len(a1_date)) if (a1_date[i] in a1_uni_valid) ],:] #该事件的有效日期
        if len(tc_event_valid)>4:
            #
            tc_event_valid.iloc[:,0] = [str(tc_event_valid[0].iloc[i])[0:8] for i in range(len(tc_event_valid))]
            #求日平均大气压，经纬度
            tc_event_valid.iloc[:,4]=tc_event_valid.iloc[:,4].astype(float)
            tc_stats = tc_event_valid.groupby(tc_event_valid.iloc[:,0]).mean()
            
            tc_stats = pd.concat([tc_stats.iloc[:,1:4],pd.Series(np.repeat(j,len(tc_stats)),index=tc_stats.index)],axis=1)
            tc_stats.iloc[:,0:2] = tc_stats.iloc[:,0:2]/10 ##经纬度
            tc_month = [int(tc_stats.index[i][4:6]) for i in range(len(tc_stats))]
            tc_month_rainy = np.array([ (tc_month[i] in [4,5,6,7,8,9]  ) for i in range(len(tc_month))]).sum() #属于雨季
            if tc_month_rainy>0:
                #if ((tc_stats.iloc[:,0].max()>=18) & (tc_stats.iloc[:,1].min()<=127)):
                if ((tc_stats.iloc[:,0].max()>=18) & (tc_stats.iloc[np.argmax(tc_stats.iloc[:,0] == tc_stats.iloc[:,0].max()),1]<=128)):    
                    tc_stats.iloc[:,3] = s
                    s=s+1
                    tc_valid = pd.concat([tc_valid,tc_stats],axis=0)
            
tc_valid.columns = ['lat','lon','slp','number']        
tc_valid['year'] = [tc_valid.index[i][:4] for i in range(len(tc_valid))]
tc_valid['month'] = [tc_valid.index[i][4:6] for i in range(len(tc_valid))]
tc_valid['day'] = [tc_valid.index[i][6:8] for i in range(len(tc_valid))]
np.save('/media/dai/DATA1/research/6.Southeastern_China_Clustering/code2/7.tc_track/tc_track_valid.npy',tc_valid)        



