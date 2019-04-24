#!/usr/bin/env python
# coding: utf-8

# In[432]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader
from pandas_datareader.data import DataReader
from datetime import date
import datetime
import statsmodels.tsa.x13 as x13
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import export_graphviz


# In[397]:


# Import SP500 Data
sp500 = pd.read_csv('S&P.csv',index_col='Date',parse_dates=True)
sp500.drop(['PE10','Real Price','Real Dividend','Earnings','Consumer Price Index','Long Interest Rate','Real Earnings'],axis=1,inplace=True)


# In[398]:


# Import Treasury Bill Rates
T1 = pd.read_csv('FF3.csv',index_col='Date',parse_dates=True)
T1.drop(['Mkt-RF','SMB','HML'],axis=1,inplace=True)
T3 = DataReader('TB3MS', 'fred', start = date(1900,1,1))
T6 = DataReader('TB6MS', 'fred', start = date(1900,1,1))
T12 = DataReader('TB1YR', 'fred', start = date(1900,1,1))
T60 = DataReader('GS5', 'fred', start = date(1900,1,1))
T120 = DataReader('GS10', 'fred', start = date(1900,1,1))


# In[399]:


# Import CD Rates
CD1 = DataReader('CD1M', 'fred', start = date(1900,1,1))
CD3 = DataReader('CD3M', 'fred', start = date(1900,1,1))
CD6 = DataReader('CD6M', 'fred', start = date(1900,1,1))


# In[400]:


# Import Corporate Bond Rates
AAA = DataReader('AAA', 'fred', start = date(1900,1,1))
BAA = DataReader('BAA', 'fred', start = date(1900,1,1))


# In[401]:


# Import Price and Industrial Indicies
PP = DataReader('WPUFD49207', 'fred', start = date(1900,1,1)).shift(-1)
IP = DataReader('INDPRO', 'fred', start = date(1900,1,1)).shift(-1)
CP = DataReader('CPIAUCSL', 'fred', start = date(1900,1,1)).shift(-1)

# Import Monetary Supply
M1 = DataReader('M1NS', 'fred', start = date(1900,1,1)).shift(-1)


# In[402]:


# Monthly Return
sp500['R'] = sp500['SP500'].pct_change()

# Dividend Yield
sp500['DY'] = sp500['Dividend']/sp500['SP500']
sp500.dropna(inplace=True)

# Bring everything existing into one and filter to desire dates
data = sp500.join([M1,CP,IP,PP,AAA,BAA,CD6,CD3,CD1,T120,T60,T12,T6,T3,T1],how='outer')
data.columns = ['SP','DIV','R','DY','M1','CP','IP','PP','AAA','BAA','CD6','CD3','CD1','T120','T60','T12','T6','T3','T1']


# In[403]:


data = data.truncate('1976-02-01','1999-12-01')

# Seasonally adjust desired columns
x12path = '/Users/andrewpalmer/Downloads/x13assrc_V1.1_B39/x13as'
for column in data.drop(['DIV','T1','SP','DY'],axis=1).columns:
    data[column] = x13.x13_arima_analysis(data[column],x12path = x12path).seasadj
    data[column].plot(title=column)
    plt.show()


# In[406]:


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


# In[407]:


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


# In[408]:



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


# In[409]:


# Feature Importance 1
tree_model = DecisionTreeClassifier(criterion='entropy')
tree_model.fit(X_train1,Y_train1)
features1 = pd.DataFrame(dict(zip(X_train2.columns,tree_model.feature_importances_)),index=[0]).T
features1 = features1.loc[(features1 > 0.001).any(axis=1),:].sort_values(0,ascending=False)
features1.plot(title='Sliding Window 1 Feature Importance',kind='bar',legend=False)
plt.xticks(rotation=45)
plt.show()

# Feature Importance 2
tree_model = DecisionTreeClassifier(criterion='entropy')
tree_model.fit(X_train2,Y_train2)
features2 = pd.DataFrame(dict(zip(X_train2.columns,tree_model.feature_importances_)),index=[0]).T
features2 = features2.loc[(features2 > 0.001).any(axis=1),:].sort_values(0,ascending=False)
features2.plot(title='Sliding Window 2 Feature Importance',kind='bar',legend=False)
plt.xticks(rotation=45)
plt.show()

# Feature Importance 3
tree_model = DecisionTreeClassifier(criterion='entropy')
tree_model.fit(X_train3,Y_train3)
features3 = pd.DataFrame(dict(zip(X_train3.columns,tree_model.feature_importances_)),index=[0]).T
features3 = features3.loc[(features3 > 0.001).any(axis=1),:].sort_values(0,ascending=False)
features3.plot(title='Sliding Window 3 Feature Importance',kind='bar',legend=False)
plt.xticks(rotation=45)
plt.show()

# Feature Importance 4
tree_model = DecisionTreeClassifier(criterion='entropy')
tree_model.fit(X_train4,Y_train4)
features4 = pd.DataFrame(dict(zip(X_train4.columns,tree_model.feature_importances_)),index=[0]).T
features4 = features4.loc[(features4 > 0.001).any(axis=1),:].sort_values(0,ascending=False)
features4.plot(title='Sliding Window 4 Feature Importance',kind='bar',legend=False)
plt.xticks(rotation=45)
plt.show()


# In[420]:


# Make difference and normalize for neural net input
X_train1 = data.diff().truncate('1976-03-01','1992-10-01').drop(['T1H','R','ER'],axis=1)
X_test1 = data.diff().truncate('1992-11-01','1994-08-01').drop(['T1H','R','ER'],axis=1)
X_train1 = (X_train1 - X_train1.mean()) / (X_train1.max() - X_train1.min())
X_test1 = (X_test1 - X_test1.mean()) / (X_test1.max() - X_test1.min())

X_train2 = data.diff().truncate('1978-01-01','1994-08-01').drop(['T1H','R','ER'],axis=1)
X_test2 = data.diff().truncate('1992-11-01','1994-08-01').drop(['T1H','R','ER'],axis=1)
X_train2 = (X_train2 - X_train2.mean()) / (X_train2.max() - X_train2.min())
X_test2 = (X_test2 - X_test2.mean()) / (X_test2.max() - X_test2.min())

X_train3 = data.diff().truncate('1979-11-01','1996-06-01').drop(['T1H','R','ER'],axis=1)
X_test3 = data.diff().truncate('1996-07-01','1998-04-01').drop(['T1H','R','ER'],axis=1)
X_train3 = (X_train3 - X_train3.mean()) / (X_train3.max() - X_train3.min())
X_test3 = (X_test3 - X_test3.mean()) / (X_test3.max() - X_test3.min())

X_train4 = data.diff().truncate('1981-09-01','1998-04-01').drop(['T1H','R','ER'],axis=1)
X_test4 = data.diff().truncate('1998-05-01','1999-12-01').drop(['T1H','R','ER'],axis=1)
X_train4 = (X_train4 - X_train4.mean()) / (X_train4.max() - X_train4.min())
X_test4 = (X_test4 - X_test4.mean()) / (X_test4.max() - X_test4.min())


# In[456]:


#Build and Evaluate Neural Net Performances
accuracy = []
model = MLPClassifier()
tscv = TimeSeriesSplit(n_splits=5)

model.fit(X_train1[features1.index],Y_train1)
accuracy.append(model.score(X_test1[features1.index],Y_test1))
print(model.score(X_test1[features1.index],Y_test1))

model.fit(X_train2[features2.index],Y_train2)
accuracy.append(model.score(X_test2[features2.index],Y_test2))
print(model.score(X_test2[features2.index],Y_test2))

model.fit(X_train3[features3.index],Y_train3)
accuracy.append(model.score(X_test3[features3.index],Y_test3))
print(model.score(X_test3[features3.index],Y_test3))

model.fit(X_train4[features4.index],Y_train4)
accuracy.append(model.score(X_test4[features4.index],Y_test4))
print(model.score(X_test4[features4.index],Y_test4))

# Compute and display results
mean_acc = sum(accuracy)/len(accuracy)
plt.plot(['Sliding Window 1','Sliding Window 2','Sliding Window 3','Sliding Window 4'],accuracy,'.-')
plt.title('Sliding Window Neural Net Accuracy')
plt.hlines(mean_acc, 0, 3,linestyles='dashed')
plt.legend(['Neural Net Accuracies',"Mean Accuracy {}".format(round(mean_acc,2))])


# In[452]:


# Non-sliding data accuracy
# Set up features and labels
X = data.diff().iloc[1:].drop(['T1H','R','ER'],axis=1).dropna()
y = np.sign(data['ER'])[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Feature Importance
tree_model = DecisionTreeClassifier(criterion='entropy')
tree_model.fit(X_train,y_train)
features = pd.DataFrame(dict(zip(X_train.columns,tree_model.feature_importances_)),index=[0]).T
features = features.loc[(features > 0.001).any(axis=1),:].sort_values(0,ascending=False)
features.plot(title='Full Data Feature Importance',kind='bar',legend=False)
plt.xticks(rotation=45)
plt.show()

# 5-Fold Cross Validation and Test/Train split
model = MLPClassifier()
tscv = TimeSeriesSplit(n_splits=5)
accuracy_full = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit and Evaluate Neural Net Performance
    model.fit(X_train[features.index],y_train)
    print(model.score(X_test[features.index],y_test))
    accuracy_full.append(model.score(X_test[features.index],y_test))


# In[454]:


# Compute and display results
mean_acc_full = sum(accuracy_full)/len(accuracy_full)
plt.plot(['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5'],accuracy_full,'.-')
plt.title('Full Data 5-Fold CV Neural Net Results')
plt.hlines(mean_acc_full, 0, 4,linestyles='dashed')
plt.legend(['Fold Accuracies',"Mean Accuracy {}".format(round(mean_acc_full,2))])


# In[457]:


from graphviz import Source
from sklearn import tree
Source( tree.export_graphviz(tree_model,
                             out_file=None,
                             feature_names = X_train.columns,
                             class_names = ['1','-1'],
                             rounded = True,
                             proportion = False,
                             precision = 3,
                             filled = True))

