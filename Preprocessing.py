#!/usr/bin/env python
# coding: utf-8

# In[181]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader
from pandas_datareader.data import DataReader
from datetime import date
import datetime
import statsmodels.tsa.x12 as x12


# In[223]:


# Import SP500 Data
sp500 = pd.read_csv('S&P.csv',index_col='Date',parse_dates=True)
sp500.drop(['PE10','Real Price','Real Dividend','Earnings','Consumer Price Index','Long Interest Rate','Real Earnings'],axis=1,inplace=True)


# In[129]:


# Import Treasury Bill Rates
T1 = pd.read_csv('FF3.csv',index_col='Date',parse_dates=True)
T1.drop(['Mkt-RF','SMB','HML'],axis=1,inplace=True)
T3 = DataReader('TB3MS', 'fred', start = date(1900,1,1))
T6 = DataReader('TB6MS', 'fred', start = date(1900,1,1))
T12 = DataReader('TB1YR', 'fred', start = date(1900,1,1))
T60 = DataReader('GS5', 'fred', start = date(1900,1,1))
T120 = DataReader('GS10', 'fred', start = date(1900,1,1))


# In[140]:


# Import CD Rates
CD1 = DataReader('CD1M', 'fred', start = date(1900,1,1))
CD3 = DataReader('CD3M', 'fred', start = date(1900,1,1))
CD6 = DataReader('CD6M', 'fred', start = date(1900,1,1))


# In[132]:


# Import Corporate Bond Rates
AAA = DataReader('AAA', 'fred', start = date(1900,1,1))
BAA = DataReader('BAA', 'fred', start = date(1900,1,1))


# In[172]:


# Import Price and Industrial Indicies
PP = DataReader('WPUFD49207', 'fred', start = date(1900,1,1)).shift(-1)
IP = DataReader('INDPRO', 'fred', start = date(1900,1,1)).shift(-1)
CP = DataReader('CPIAUCSL', 'fred', start = date(1900,1,1)).shift(-1)


# In[171]:


# Import Monetary Supply
M1 = DataReader('M1NS', 'fred', start = date(1900,1,1)).shift(-1)


# In[224]:


# Monthly Return
sp500['R'] = sp500['SP500'].pct_change()

# Dividend Yield
sp500['DY'] = sp500['Dividend']/sp500['SP500']
sp500.dropna(inplace=True)

# Bring everything existing into one and filter to desire dates
data = sp500.join([M1,CP,IP,PP,AAA,BAA,CD6,CD3,CD1,T120,T60,T12,T6,T3,T1],how='outer')
data.columns = ['SP','DIV','R','DY','M1','CP','IP','PP','AAA','BAA','CD6','CD3','CD1','T120','T60','T12','T6','T3','T1']


# In[227]:


# Create spreads and other calculated features
data['T1H'] = data.T1.div(12)
data['ER'] = data['R']- data['T1H'].shift(-1)
data['ER'] = data['ER'].shift(1)

data['TE1'] = data['T120'] - data['T1']
data['TE2'] = data['T120'] - data['T3']
data['TE3'] = data['T120'] - data['T6']
data['TE4'] = data['T120'] - data['T12']
data['TE5'] = data['T3'] - data['T1']
data['TE6'] = data['T6'] - data['T1']

data['DE1'] = data['BAA'] - data['AAA']
data['DE2'] = data['BAA'] - data['T120']
data['DE3'] = data['BAA'] - data['T12']
data['DE4'] = data['BAA'] - data['T6']
data['DE5'] = data['BAA'] - data['T3']
data['DE6'] = data['BAA'] - data['T1']
data['DE7'] = data['CD6'] - data['T6']

data.info()


# In[228]:


data = data.truncate('1976-03-01','1999-12-01')


# In[233]:


# Split data into 4 training/test sets

X_train1 = data.truncate('1976-03-01','1992-10-01').drop(['T1H','R','ER'],axis=1)
Y_train1 = np.sign(data.truncate('1976-03-01','1992-10-01')['ER'])

X_test1 = data.truncate('1992-11-01','1994-08-01').drop(['T1H','R','ER'],axis=1)
Y_test1 = np.sign(data.truncate('1992-11-01','1994-08-01')['ER'])

X_train2 = data.truncate('1978-01-01','1994-08-01').drop(['T1H','R','ER'],axis=1)
Y_train2 = np.sign(data.truncate('1978-01-01','1994-08-01')['ER'])

X_test2 = data.truncate('1992-11-01','1994-08-01').drop(['T1H','R','ER'],axis=1)
Y_test2 = np.sign(data.truncate('1994-09-01','1996-06-01')['ER'])

X_train3 = data.truncate('1979-11-01','1996-06-01').drop(['T1H','R','ER'],axis=1)
Y_train3 = np.sign(data.truncate('1979-11-01','1996-06-01')['ER'])

X_test3 = data.truncate('1996-07-01','1998-04-01').drop(['T1H','R','ER'],axis=1)
Y_test3 = np.sign(data.truncate('1996-07-01','1998-04-01')['ER'])

X_train4 = data.truncate('1981-09-01','1998-04-01').drop(['T1H','R','ER'],axis=1)
Y_train4 = np.sign(data.truncate('1981-09-01','1998-04-01')['ER'])

X_test4 = data.truncate('1998-05-01','1999-12-01').drop(['T1H','R','ER'],axis=1)
Y_test4 = np.sign(data.truncate('1998-05-01','1999-12-01')['ER'])


# In[240]:



# Sliding window 1 information
print("\n SLIDING WINDOW 1 TRAIN \n")
print(X_train1.info())
print(X_train1.head())
print(Y_train1)
print("\n SLIDING WINDOW 1 TEST \n")
print(X_test1.info())
print(X_test1.head())
print(Y_test1)

# Sliding window 2 information
print("\n SLIDING WINDOW 2 TRAIN \n")
print(X_train2.info())
print(X_train2.head())
print(Y_train2)
print("\n SLIDING WINDOW 2 TEST \n")
print(X_test2.info())
print(X_test2.head())
print(Y_test2)

# Sliding window 3 information
print("\n SLIDING WINDOW 3 TRAIN \n")
print(X_train3.info())
print(X_train3.head())
print(Y_train3)
print("\n SLIDING WINDOW 3 TEST \n")
print(X_test3.info())
print(X_test3.head())
print(Y_test3)

# Sliding window 4 information
print("\n SLIDING WINDOW 4 TRAIN \n")
print(X_train4.info())
print(X_train4.head())
print(Y_train4)
print("\n SLIDING WINDOW 4 TEST \n")
print(X_test4.info())
print(X_test4.head())
print(Y_test4)


# In[ ]:




