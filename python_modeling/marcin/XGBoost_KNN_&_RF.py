# -*- coding: utf-8 -*-
## Featre selection with Random Forest

from numpy import loadtxt
import pandas as pd
from xgboost import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot
# load data
data = pd.read_csv("sample/sample_0.011.csv")
data = data.drop(['assesstot', "block", "lot"], axis=1)

data.isnull().sum()

#Scpecify what columns are factors
to_factors = ["cd","schooldist","council","zipcode","policeprct","firecomp",
               "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
               "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
               "healthcenterdistrict", "pfirm15_flag"]

## Make label Encoding
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()

#Converting in the loop
#for i in to_factors: 
#    data[i] = le.fit_transform(data[i].astype(str))
#    print(i) 
#data['firecomp'].dtypes



#Iterate thru dataset and convert columns from "to_factors" into 
for i in to_factors: 
    data[i] = data[i].astype('category')
    print(i) 


# Dummies
df_dummies = pd.get_dummies(data[to_factors], drop_first=True)
data.drop(to_factors, axis=1, inplace=True)
data = pd.concat([data, df_dummies], axis=1)


# split data into X and y
X = data.drop('assessland', axis=1)
y = data['assessland']
# fit model no training data
model = XGBRegressor()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()

# Create data frame
results=pd.DataFrame()
results['columns']=X.columns
results['importances'] = model.feature_importances_
results.sort_values(by='importances',ascending=False,inplace=True)

import_not_0=results[results['importances']>0]
min(import_not_0['importances'])

# It gives you selected data set with only those variables
# which importance is above given threshold stated above
from sklearn.feature_selection import SelectFromModel
selection = SelectFromModel(model, threshold = min(import_not_0['importances']), prefit=True)
selected_dataset = selection.transform(X)

## So since now we have a list with variables which importance is above 0.0
list_variables=import_not_0['columns'].values.tolist()
# append assessland as a variable
list_variables.append('assessland')




#############################################################################
############################# MODELING #####################################
############################################################################

####### Build the model based on variables choosen above only #######

######################### 1. KNN #############################################
##############################################################################

#based on: https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/

#Split data
from sklearn.model_selection import train_test_split
train , test = train_test_split(data[list_variables], test_size = 0.3)

x_train = train.drop('assessland', axis=1)
y_train = train['assessland']

x_test = test.drop('assessland', axis = 1)
y_test = test['assessland']

# Preprocessing – Scaling the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)


### Build the model on train set and predict and calculate RMSE on test set

# Let us have a look at the error rate for different k values
#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
    
#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()

# ANSWER: the smallest RMSE value for k=  4 is: 62166.72061996037




######################## 2. Random Forest ###################################
#############################################################################
# Based on this: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

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


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(X_train, y_train)

#View the best parameters:
print("the output:", rf_random.best_params_)

### SOLUTION, BEST PARAMETERS
'''the output: {'n_estimators': 1000, 
             'min_samples_split': 5, 
             'min_samples_leaf': 2, 
             'max_features': 'sqrt', 
             'max_depth': 100, 
             'bootstrap': False}'''




######## MY CODE
reg = RandomForestRegressor(
            n_estimators=100, 
            max_depth=50, 
            bootstrap=True, 
            random_state=123
            )
reg.fit(x_train, y_train)
preds_train = reg.predict(x_train)
preds_test = reg.predict(x_test)
    
# Calcualte evaluation metrics for TRAIN
sqrt(mean_squared_error(y_train, preds_train))

# Calcualte evaluation metrics for TEST
sqrt(mean_squared_error(y_test, preds_test))

# ANSWER: Smallers RMSE for the random forest: 55961.89767176862

print('Parameters currently in use:\n')
pprint(reg.get_params())
