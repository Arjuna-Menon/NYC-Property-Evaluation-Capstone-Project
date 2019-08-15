# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:47:50 2019

@author: grzechu
"""
#####################################################
########### REGRESSIONS (Lasso, Ridge) ##############
######################################################


############################# DATA READING ################################

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv("sample/sample_0.011.csv")
# drop also block and lot
data=data.drop(['lot','block'], axis=1)
#Scpecify what columns are factors
to_factors = ["cd","schooldist","council","zipcode","policeprct","firecomp",
               "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
               "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
               "healthcenterdistrict", "pfirm15_flag"]

####################### MODELING with LABEL ENCODING #########################
## Make label Encoding
le = LabelEncoder()

#Converting in the loop
for i in to_factors: 
    data[i] = le.fit_transform(data[i].astype(str))
    print(i) 

#Iterate thru dataset and convert columns from "to_factors" into 
for i in to_factors: 
    data[i] = data[i].astype('category')
    print(i)    

## Target variables is Assessland
df1 = data.drop(['assesstot'], axis=1)

## Create X and Y
Xs = data.drop(['assessland'], axis=1)
y = data['assessland'].values.reshape(-1,1)


'''
######### Multiple linear regression - least squares fitting #######

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print(mean_MSE)


############# Ridge regression #################
alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(Xs, y)
ridge_regressor.score
ridge_regressor.best_params_

ridge_regressor.best_score_



############ Lasso ###################
lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(Xs, y)

lasso_regressor.best_params_

lasso_regressor.best_score_

#Compare
print(mean_MSE)
print(ridge_regressor.best_score_) ## the smallers MSE with alpha = 20
print(lasso_regressor.best_score_)
'''

################# IMPLEMENTATION ###################

## Create X and Y
Xs = df1.drop(['assessland'], axis=1)
y = df1['assessland'].values.reshape(-1,1)

## Implement Ridge Regression
X_train, X_test , y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=1)

ridge2 = Ridge(alpha = 20, normalize = True)
ridge2.fit(X_train, y_train)
ridge2.coef_             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)
           # Use this model to predict the test data
coefs_ridge = pd.DataFrame(ridge2.coef_.T, index =[Xs.columns]) # Print coefficients
print(sqrt(mean_squared_error(y_test, pred2)))         # Calculate the test MSE
#ridge2.score(X_train,y_train)

## Lasso
from sklearn import linear_model
clf = linear_model.Lasso(alpha=20)
clf.fit(X_train, y_train)     
coefs_lasso = pd.DataFrame(clf.coef_.T, index =[Xs.columns])
pred = clf.predict(X_test)
print(sqrt(mean_squared_error(y_test, pred)))  
#clf.score(X_train,y_train)




########################### DUMMIE/ ONE-HOT ENCODING #######################
## Convert all to dummies, AND DELETE factors which means we do k-1 variables
df_dummies = pd.get_dummies(df1[to_factors], drop_first=True)
#Drop old factors from the dataset (oryginal one, those not one-hot encoded)
df2=df1.drop(to_factors, axis=1)
#Concat numeric variables wiht converted factors
df2 = pd.concat([df2, df_dummies], axis=1)

################# IMPLEMENTATION ###################

## Create X and Y
Xs = df2.drop(['assessland'], axis=1)
y = df2['assessland'].values.reshape(-1,1)

## Implement Ridge Regression
X_train, X_test , y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=1)

ridge2 = Ridge(alpha = 20, normalize = True)
ridge2.fit(X_train, y_train)
ridge2.coef_             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)
           # Use this model to predict the test data
coefs_ridge = pd.DataFrame(ridge2.coef_.T, index =[Xs.columns]) # Print coefficients
print(sqrt(mean_squared_error(y_test, pred2)))          # Calculate the test MSE
#ridge2.score(X_train,y_train)

## Lasso - make many many zeros
from sklearn import linear_model
clf = linear_model.Lasso(alpha=20)
clf.fit(X_train, y_train)     
coefs_lasso = pd.DataFrame(clf.coef_.T, index =[Xs.columns])
pred = clf.predict(X_test)
print(sqrt(mean_squared_error(y_test, pred)))




#### SUMMARY OF RMSE:
# Label Encoding
# Ridge: 75714.61355475917
# Lasso: 65330.62868953153 (no zeros)

# One Hot Encoding
# Ridge: 73701.44441019912
# Lasso:  61989.31464644037 (many zeros)