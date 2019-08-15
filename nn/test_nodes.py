#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:36:39 2019

@author: mikaelapisanileal
"""

import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))    


def get_model(input_nodes, hidden_nodes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_nodes, tf.keras.activations.linear)) #input
    for nodes in hidden_nodes:
        model.add(tf.keras.layers.Dense(nodes, tf.keras.activations.relu)) #hidden
    model.add(tf.keras.layers.Dense(1, tf.keras.activations.linear)) #output
    return model

def get_fit(x_train, y_train, x_test, y_test, model):  
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    history_train = model.fit(x_train, y_train, epochs=100, verbose=0)
    history_test = model.fit(x_test, y_test, epochs=100, verbose=0)
        
    #plot loss
    plt.plot(history_train.history['mean_squared_error'])
    plt.plot(history_test.history['mean_squared_error'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_nn_assessland.png')
    plt.show()
    
    #plot y predict vs actual y for test set
    y_pred = model.predict(x_test)
    fig, ax = plt.subplots()
    ax.plot(y_test, color = 'blue')
    ax.plot(y_pred, color = 'red')
    ax.legend(['Real', 'Predicted'])
    fig.savefig('real_vs_pred_nn_assessland.png')
    plt.show()

    print("mean_squared_error=%f" %(np.mean(history_train.history['mean_squared_error'])))
    print("mean_absolute_percentage_error=%f" %(np.mean(history_train.history['mean_absolute_percentage_error'])))
    print('RMSE:',round(rmse(y_test, y_pred),3))
    return round(rmse(y_test, y_pred),3)

    
def check_nodes(df, target_var): 
    categorical_vars = ['cd', 'schooldist', 'council', 'zipcode', 'firecomp',
           'policeprct', 'healtharea', 'sanitboro', 'sanitsub', 'zonedist1',
           'spdist1', 'ltdheight', 'landuse',  'ext', 'proxcode', 'irrlotcode', 'lottype',
           'borocode','edesignum', 'sanitdistrict', 'healthcenterdistrict', 'pfirm15_flag']
    
    df_dummies = pd.get_dummies(df[categorical_vars], drop_first=False) 
    df.drop(categorical_vars, axis=1, inplace=True)
    df = pd.concat([df, df_dummies], axis=1)
    
    X = df[df.columns]
    X.drop(target_var, axis=1, inplace=True)
    predictors = X.columns
    X = X.values
    Y = df[target_var].values
    
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_norm, Y.reshape(-1,1), 
                                                        test_size=0.2, random_state=0)
    
    max_nodes = int((len(predictors)+1)/2)
    input_nodes = len(predictors)
    rmse_list = []
    for layer in range(1, 4):
        for nodes in range(1, max_nodes):
            input_nodes 
            model = get_model(input_nodes, nodes)
            rmse_list.append(get_fit(x_train, y_train, x_test, y_test, model))
    return rmse_list
        

df = pd.read_csv('sample_0.011.csv')
df.drop(['block', 'lot', 'assessland'], axis=1, inplace=True)
rmse_list = check_nodes_2_layers(df, 'assesstot')
df_erros = pd.DataFrame(rmse_list, index=range(1, 204))  
