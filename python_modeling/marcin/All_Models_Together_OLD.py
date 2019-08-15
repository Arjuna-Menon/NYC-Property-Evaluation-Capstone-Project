# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:29:44 2019

@author: grzechu
"""

############################# IMPORTS ################################

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

#Ridge Regression
from sklearn.linear_model import RidgeCV
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler


############################# DATA READING ################################

#Read data
data = pd.read_csv("pluto4.csv")

######################## TARGET IS: ASSESSLAND #############################
data = data.drop(['assesstot'], axis=1) # drop assesstot
#Drop all building info
data=data.drop(['bldgarea', 'numfloors', 'unitsres', 'unitstotal','bldgfront', 'bldgdepth', 'ext',
'proxcode', 'yearbuilt', 'yearalter','income'],axis=1)

# drop block and lot, we don't use it
data=data.drop(['lot','block'], axis=1)
#Scpecify what columns are factors
to_factors = ["cd","schooldist","council","zipcode","policeprct",
              "firecomp","healtharea","sanitboro","sanitsub","zonedist1",
              "spdist1","ltdheight","landuse","irrlotcode","lottype",
              "borocode","edesignum","sanitdistrict","healthcenterdistrict", 
              "pfirm15_flag"]
               #"ext","proxcode",

#Converte to factors
for i in to_factors: 
    data[i] = data[i].astype('category')
    print(i)    
    
########################### SCALE ONLY NUMERIC AND COMBINED WITH FACTOR DUMMIES ##################################   
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Read data
data = pd.read_csv("pluto4.csv")

# drop block and lot, we don't use it
data=data.drop(['lot','block'], axis=1)

#Scpecify what columns are factors
to_factors = ["cd","schooldist","council","zipcode","policeprct",
              "firecomp","healtharea","sanitboro","sanitsub","zonedist1",
              "spdist1","ltdheight","landuse","irrlotcode","lottype",
              "borocode","edesignum","sanitdistrict","healthcenterdistrict", 
              "pfirm15_flag","ext","proxcode"]

#Converte to factors
for i in to_factors: 
    data[i] = data[i].astype('category')
    print(i)    

# SCALE NUMERIC
not_scale=to_factors
not_scale.append('assessland') # for some reason it adds assessland to to_factors
not_scale.append('assesstot') #adds assesstot also

numeric_df=data[data.columns.difference(not_scale)]

scaler = StandardScaler()  
numeric_df_std = scaler.fit_transform(numeric_df)

numeric_df = pd.DataFrame(numeric_df_std, columns = numeric_df.columns)

to_factors.remove('assessland')
to_factors.remove('assesstot')
# DUMMIE/ ONE-HOT ENCODING 
## Convert all to dummies, AND DELETE factors which means we do k-1 variables
df_dummies = pd.get_dummies(data[to_factors], drop_first=True)
#Drop old factors from the dataset (oryginal one, those not one-hot encoded)
#data=data.drop(to_factors, axis=1)
#Concat numeric variables wiht converted factors
data = pd.concat([numeric_df, df_dummies], axis=1)


########################### SCALE THE DATA ##################################
X = data.drop(['assessland'], axis=1)
y = data['assessland']#.values.reshape(-1,1)

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
# Put column names again for X
X = pd.DataFrame(X_std, columns = X.columns)

from sklearn.preprocessing import minmax_scale
X[to_factors] = minmax_scale(df[['x','z']])

#############################################################################
############################### MODELING ######################################
#############################################################################


################################### RIDGE REGRESSION #########################
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])


