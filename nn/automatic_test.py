#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 19:41:16 2019

@author: mikaelapisanileal
"""

#https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-i-hyper-parameter-8129009f131b


from hyperas.distributions import choice, uniform
from hyperas import optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe


def get_data():
    df = pd.read_csv('sample_0.011.csv')
    df.drop(['block', 'lot', 'assesstot'], axis=1, inplace=True)
    
    categorical_vars = ['cd', 'schooldist', 'council', 'zipcode', 'firecomp',
           'policeprct', 'healtharea', 'sanitboro', 'sanitsub', 'zonedist1',
           'spdist1', 'ltdheight', 'landuse',  'ext', 'proxcode', 'irrlotcode', 'lottype',
           'borocode','edesignum', 'sanitdistrict', 'healthcenterdistrict', 'pfirm15_flag']
    
    df_dummies = pd.get_dummies(df[categorical_vars], drop_first=False) #keep all dummies to evaluate importance, for the prediction should say drop_first=True
    df.drop(categorical_vars, axis=1, inplace=True)
    df = pd.concat([df, df_dummies], axis=1)
    
    X = df[df.columns]
    X.drop('assessland', axis=1, inplace=True)
    predictors = X.columns
    X = X.values
    Y = df['assessland'].values
    
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(x_norm, Y.reshape(-1,1), 
                                                        test_size=0.2, random_state=0)
    return x_train, y_train, x_test, y_test, predictors

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def get_model(X_train, Y_train, X_val, Y_val, predictors):
    model = tf.keras.layers.Sequential()
    #The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
    input_nodes = len(predictors)
    max_hidden_nodes = int(input_nodes*2/3)
    model.add(tf.keras.layers.Dense({{choice(range(max_hidden_nodes,input_nodes,20))}}, input_shape=(input_nodes,)))
    model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
    model.add(tf.keras.layers.Dense({{choice(range(max_hidden_nodes,input_nodes,20))}}))
    model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
    
    if tf.keras.layers.conditional({{choice([2, 3])}}) == 3:
        model.add(tf.keras.layers.Dense({{choice(range(max_hidden_nodes,input_nodes,20))}}))
        model.add(tf.keras.layers.Activation({{choice(['relu', 'sigmoid'])}}))
        model.add(tf.keras.layers.Dropout({{uniform(0, 1)}}))
        
    model.add(tf.keras.layers.DenseDense(1))
    model.add(tf.keras.layers.DenseActivation('linear'))
    adam = tf.keras.layers.Densekeras.optimizers.Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    rmsprop = tf.keras.layers.Densekeras.optimizers.RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    sgd = tf.keras.optimizers.SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})
   
    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd
    
    model.compile(loss='mse', metrics=['mse', 'mae', 'mape'], optimizer=optim)
    model.fit(X_train, Y_train,batch_size={{choice([128,256,512])}},nb_epoch=40,verbose=2)
    predicted = model.evaluate(X_val, Y_val, verbose=0)
    return {'loss': rmse(Y_val, predicted), 'status': STATUS_OK, 'model': model}

#x_train, y_train, x_test, y_test, predictors = get_data()
best_run, best_model = optim.minimize(model=get_model,
                                      data=get_data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())
print(best_run)