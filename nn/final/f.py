#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 08:01:14 2019

@author: mikaelapisanileal
"""

import nn_functions
import pandas as pd
x_train, x_test, y_train, y_test, predictors = nn_functions.get_data('../../python_modeling/arjuna/plutoPCA_fullland.csv','assessland')
input_nodes = len(predictors)
epochs = 30
hidden_nodes = [int((input_nodes+1)*(2/3)), int(len(x_train)/(30*2)), int(len(x_train)/(30*4)), int(len(x_train)/(30*6)), int(len(x_train)/(30*8))]
optimizers = ['rmsprop', 'adam']
h = []
o = []
df_models = pd.DataFrame()
for i in hidden_nodes:
    for j in optimizers:
        h.append(i)
        o.append(j)
        
df_models['hidden_nodes'] = h
df_models['optimizer']  = o
df_models
mse_list = []
rmse_list = []
r2_list = []
error_list = []
for index, row in df_models.iterrows():
    try:
        print('Model:', index, 'hidden_nodes:', row['hidden_nodes'], 'optimizer:', row['optimizer'])
        y_train_pred, y_test_pred, mse,rmse,r2,error = nn_functions.run_model(input_nodes, row['hidden_nodes'], x_train, x_test, y_train, y_test, row['optimizer'], epochs)
    except Exception as ex:
        print(ex)
        mse = float('nan')
        rmse = float('nan')
        r2 = float('nan')
        error = float('nan')
        print('Error while computing model: ', index)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)
    error_list.append(error)
        
df_models['mse'] = mse_list
df_models['rmse'] = rmse_list
df_models['r2'] = r2_list
df_models['error'] = error_list
df_models.drop(16, axis=0, inplace=True)
min_r2 = df_models['r2'].idxmax(axis=0, skipna=True)
print('Min r2:', min_r2)
print(df_models.loc[min_r2])
max_rmse = df_models['rmse'].idxmin(axis=0, skipna=True)
print('Min rmse:', max_rmse)
print(df_models.loc[max_rmse])
df_models