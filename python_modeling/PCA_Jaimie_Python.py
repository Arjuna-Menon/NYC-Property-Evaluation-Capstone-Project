#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:42:35 2019

@author: jaimiecapps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df=pd.read_csv('documents/Github/Capstone/sample/sample_0.011.csv')
df.head()
df.isnull().sum()

# drop  block and lot
df.drop(['block', 'lot'], axis=1, inplace=True)

categorical_vars = ['cd', 'schooldist', 'council', 'zipcode', 'firecomp',
       'policeprct', 'healtharea', 'sanitboro', 'sanitsub', 'zonedist1',
       'spdist1', 'ltdheight', 'landuse',  'ext', 'proxcode', 'irrlotcode', 'lottype',
       'borocode','edesignum', 'sanitdistrict', 'healthcenterdistrict', 'pfirm15_flag']

df_dummies = pd.get_dummies(df[categorical_vars], drop_first=False) #keep all dummies to evaluate importance, for the prediction should say drop_first=True
df.drop(categorical_vars, axis=1, inplace=True)
df = pd.concat([df, df_dummies], axis=1)

X = df[df.columns]
X.drop('assesstot', axis=1, inplace=True)
predictors = X.columns
X = X.values
Y = df['assesstot'].values


X = np.array(df)
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)  # [0.9036461  0.06491982]
print(pca.singular_values_) # 41669675.46870463 11168878.84708291]


from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


#x=data.iloc[:,2:3].values
#y=data.iloc[:,3:4].values

# split data into test or training set
X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42)

# simple linear regression to training set
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# predict result
y_pred=regressor.predict(X_test)

# see relationship between training data values
plt.scatter(X_train,y_train,c='red')
plt.show()

# See the relationship between the predicted 
# values using scattered graph
plt.plot(X_test,y_pred)   
plt.scatter(X_test,y_test)
plt.xlabel(' ')
plt.ylabel(' ')

#combined rmse value
rss=((y_test-y_pred)**2).sum()
mse=np.mean((y_test-y_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((y_test-y_pred)**2)))
# 215109.8736885