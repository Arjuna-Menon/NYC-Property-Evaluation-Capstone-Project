# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:26:19 2019

@author: grzechu
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.model_selection import train_test_split



data = pd.read_csv("sample/sample_0.011.csv")
data = data.drop(['assesstot', "block", "lot"], axis=1)

data.isnull().sum()

#Scpecify what columns are factors
to_factors = ["cd","schooldist","council","zipcode","policeprct","firecomp",
               "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
               "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
               "healthcenterdistrict", "pfirm15_flag"]

#Iterate thru dataset and convert columns from "to_factors" into 
for i in to_factors: 
    data[i] = data[i].astype('category')
    print(i) 


# Dummies
df_dummies = pd.get_dummies(data[to_factors], drop_first=True)
data.drop(to_factors, axis=1, inplace=True)
data = pd.concat([data, df_dummies], axis=1)


X = data.drop(['assessland'], axis=1)
y = data['assessland']

#train/test split data
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)


# Choose the best parameters for tree using CV:

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# So this is your setting we search among
pprint(random_grid)


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(X_train, y_train)

#View the best parameters:
print("the output:", rf_random.best_params_)
