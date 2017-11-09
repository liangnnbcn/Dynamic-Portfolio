# Dynamic-Portfolio





# -*- coding: utf-8 -*-

"""

Created on Tue Nov  7
16:05:43 2017

 

@author: 44091781

"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime

import urllib,time

import scipy.optimize as opt

 

'''HSI-48 Kick out five new added-in stocks from 2014-2017
backtesting'''

 

#%% (1) 清洗数据

#35 stock + 1 benchmark

path_HSI='/Users/44091781/Desktop/Python Test/Smart Beta -
HHI - HSI/HSI-47.xlsx'

df= pd.read_excel(path_HSI)

 

list = df.iloc[0,:-1]

data_stock = df.iloc[:,:-1]

data_hsi = df.iloc[:,-1:]

 

plt.figure(2,figsize=(10, 6)) 

lines = plt.plot(data_stock.index,data_stock,label='stock
close price')

plt.grid(True)  

 

plt.figure(2,figsize=(10, 6)) 

lines = plt.plot(data_hsi.index,data_hsi,label='hsi close
price')

plt.grid(True)  

 

data_final = data_hsi

 

#%% (2) M-V model 总体组合和画图

T =251

length=len(list)

price = data_stock

cdata = price/price.shift(1)

cdata = cdata.fillna(1)

log_r = np.log(cdata) 


 

mean = log_r.mean()*252

cov = log_r.cov()*252

wts = np.random.random(length) 

wts /= np.sum(wts)

rf = 0.01

 

def stats(wts):

    port_r =
np.sum(mean*wts)

    port_var =
np.sqrt(np.dot(wts.T, np.dot(cov, wts)))

    return
np.array([port_r, port_var, (port_r-rf)/port_var])

 

port_r = []

port_var = []

for p in range(30000):

    wts =
np.random.random(len(log_r.columns))

    wts /= np.sum(wts)

   
port_r.append(np.sum(mean*wts))

   
port_var.append(np.sqrt(np.dot(wts.T, np.dot(cov, wts))))

port_r = np.array(port_r)

port_var = np.array(port_var)

 

plt.figure(figsize=(8, 4)) 


plt.scatter(port_var, port_r, c=(port_r-rf)/port_var,
marker='o')

plt.grid(True)  # 网格

plt.xlabel('excepted volatility')

plt.ylabel('expected return')

plt.colorbar(label='Sharpe ratio')

 

#补充 sharp那条线和 图

 

#%% (3)等权配置

 

w = np.linspace (1/length,1/length,length)

 

data_final['equal return'] = 0.000

for i in range(len(log_r)):

   
data_final.iloc[i,1] = np.sum(log_r.iloc[i,:].values * w.T)

data_final['equal_index'] = 
(data_final['equal return']+1).cumprod()

data_final['hi_index'] = data_final['恒生指数']/data_final['恒生指数'][0]

 

plt.figure(2,figsize=(10, 6)) 

lines =
plt.plot(data_final.index,data_final['hsi_index'],label='hsi')

lines =
plt.plot(data_final.index,data_final['equal_index'],label='equal index ')

lines =
plt.plot(data_final.index,data_final['equal_index']-data_final['hsi_index'],label='equal-hsi
')

plt.grid(True)  

plt.legend( loc = "upright")

 

#%% （4） 静态 最小方差+ 最大sharp ratio 对比

# T=1 日调仓。考虑增加 周调仓

#(4.1)MVP 最小方差静态全数据组合 和 对比

def min_var(wts):

    return
stats(wts)[1]

 

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})

bnds = tuple((0, 1) for x in range(len(log_r.columns)))

lens = len(log_r.columns)

opts = opt.minimize(

        min_var,
lens*[1./lens], method='SLSQP',

        bounds=bnds,
constraints=cons

)

opts

data_final['mvp_return']=0.0000

for i in range(len(data_final)):

   
data_final['mvp_return'][i] = np.sum(log_r.iloc[i,:] * opts['x'])

data_final['mvp_index'] =
(data_final['mvp_return']+1).cumprod() 

 

#(4.2)MAX 最大sharp-ratio 组合

def min_sharpe(wts):

    return
-stats(wts)[2]

cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})

bnds = tuple((0, 1) for x in range(len(log_r.columns)))

lens = len(log_r.columns)

opts = opt.minimize(

    min_sharpe,
lens*[1./lens], method='SLSQP',

    bounds=bnds,
constraints=cons

)

opts

data_final['maxsp_return']=0.0000

for i in range(len(data_final)):

   
data_final['maxsp_return'][i] = np.sum(log_r.iloc[i,:] * opts['x'])

data_final['maxsp_index'] =
(data_final['maxsp_return']+1).cumprod()

 

#(4.3) 三条线

plt.figure(2,figsize=(10, 6)) 

line1 =
plt.plot(data_final.index,data_final['hsi_index'],label='hsi_index')

line2 =
plt.plot(data_final.index,data_final['equal_index'],label='equal_index ')

line3 =
plt.plot(data_final.index,data_final['mvp_index'],label='mvp_index ')

line4 =
plt.plot(data_final.index,data_final['maxsp_index'],label='maxsp_index ')

plt.grid(True)  

plt.legend( loc = "upright")

#补充， 周调仓， log
复合

 

# %% (5) 最小方差 MVP 动态组合

wts = np.random.random(length)

wts /= np.sum(wts)

print (wts)

rf = 0.01

 

def min_var(wts):

    return
stats(wts)[1]

 

T = 120 #半年数据 trainning

D = 21 # 一个月调整一次 run

data_final['dy_mvp_return'] = 0.000

for i in range (int((len(log_r)-T-1)/D)):

    start = i*D

    end = start+T

    train_data =
log_r.iloc[start:end,:]

    mean =
train_data.mean()*252

    cov =
train_data.cov()*252   

    cons = ({'type':
'eq', 'fun': lambda x: np.sum(x)-1})

    bnds = tuple((0,
1) for x in range(len(train_data.columns)))

    lens =
len(train_data.columns)

    opts =
opt.minimize(

            min_var, lens*[1./lens], method='SLSQP',

           
bounds=bnds, constraints=cons

    )

    opts

    for k in range(D):

       
data_final['dy_mvp_return'][end+1+k]=np.sum(log_r.iloc[end+1+k]*opts['x'])

data_final['dy_mvp_index'] = (data_final['dy_mvp_return']
+1).cumprod()

 

data_dy_final = data_final.iloc[T:,:]

data_dy_final['adjust_dy_mvp_index'] =
data_dy_final['dy_mvp_index'] * data_dy_final['hsi_index'][0]

 

plt.figure(2,figsize=(10, 6)) 

line1 = plt.plot(data_dy_final.index,data_dy_final['hsi_index'],label='hsi_index')

line2 =
plt.plot(data_dy_final.index,data_dy_final['equal_index'],label='equal index ')

line3 =
plt.plot(data_dy_final.index,data_dy_final['mvp_index'],label='mvp_index ')

line4 =
plt.plot(data_dy_final.index,data_dy_final['maxsp_index'],label='maxsp_index ')

line5 =
plt.plot(data_dy_final.index,data_dy_final['dy_mvp_index'],label='dy_mvp_index
')

plt.grid(True)  

plt.legend( loc = "upright")

 

# %% (6) 最大sharpe ratio 动态组合 

wts = np.random.random(length)

wts /= np.sum(wts)

print (wts)

rf = 0.01

 

def min_sharpe(wts):

    return
-stats(wts)[2]

 

T = 120 #半年数据 trainning

D = 21 # 一个月调整一次 run

data_final['dy_maxsp_return'] = 0.000

for i in range (int((len(log_r)-T-1)/D)):

    start = i*D

    end = start+T

    train_data =
log_r.iloc[start:end,:]

    mean =
train_data.mean()*252

    cov =
train_data.cov()*252   

    cons = ({'type':
'eq', 'fun': lambda x: np.sum(x)-1})

    bnds = tuple((0,
1) for x in range(len(train_data.columns)))

    lens = len(train_data.columns)

    opts =
opt.minimize(

           
min_sharpe, lens*[1./lens], method='SLSQP',

           
bounds=bnds, constraints=cons

    )

    opts

    for k in range(D):

       
data_final['dy_maxsp_return'][end+1+k]=np.sum(log_r.iloc[end+1+k]*opts['x'])

data_final['dy_maxsp_index'] =
(data_final['dy_maxsp_return'] +1).cumprod()

data_dy_final = data_final.iloc[T:,:]

 

data_dy_final['adjust_dy_maxsp_index'] = data_dy_final['dy_maxsp_index']
* data_dy_final['hsi_index'][0]

 

plt.figure(2,figsize=(10, 6)) 

line1 =
plt.plot(data_dy_final.index,data_dy_final['hsi_index'],label='hsi_index')

line2 =
plt.plot(data_dy_final.index,data_dy_final['equal_index'],label='equal index ')

line3 =
plt.plot(data_dy_final.index,data_dy_final['mvp_index'],label='mvp_index ')

line4 =
plt.plot(data_dy_final.index,data_dy_final['maxsp_index'],label='maxsp_index ')

line5 =
plt.plot(data_dy_final.index,data_dy_final['dy_mvp_index'],label='dy_mvp_index
')

line6 =
plt.plot(data_dy_final.index,data_dy_final['dy_maxsp_index'],label='dy_maxsp_index
')

plt.grid(True)  

plt.legend( loc = "upright")

 

#%%

'''

修改 ：

第一，设置了月 调整，可以设置统计各种参数进行周度/月度/季度/等等 的表现值

第二，log return是复合收益，不是cumprod，设置算法

第三，没有完整50个连续股票，因为经常 季度调出和调入股票

第四，无风险利率的设置 [0.01]  

第五，sharp-ratio的图设置

总结：

第一，港股hsi近期波动非常小，采用mvp发现表现比benchmark更强。主要可能类似腾讯这种高收益

低波动的反常行为导致仓位加大，其次整个hsi低波动的股反而涨幅更客观。在风险配置上等权设置

发现表现也是比原来更好。（观网也是等权强）

第二，采用动态调整mvp发现整个收益和原来差不多，主要是低波动确实长期来看仓位调整不大，感觉

可以做这个策略，节省手续费能够追求未来（全样本） 的效果，失真不大 。

第三，sharp-ratio 追求最大化表现最优，为了预测未来，采用动态最大化sharp ratio发现表现

中等，波动同时有点大。

第四，可以记录每一次w，查看差别，观测是否需要增加限制。或者归因分析，因为hhi的动态最大化夏普

比率是非常失败的，可能是hsi以腾讯为核心，hhi？，最后hsi可能面临这种风险。

'''

