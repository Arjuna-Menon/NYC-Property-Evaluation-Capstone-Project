#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:20:18 2019

@author: mikaelapisanileal
"""

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# load data
df = pd.read_csv('/Users/mikaelapisanileal/Desktop/Capstone/sample/sample_0.011.csv')
df.drop(['block', 'lot', 'assessland'], axis=1, inplace=True)

categorical_vars = ['cd', 'schooldist', 'council', 'zipcode', 'firecomp',
       'policeprct', 'healtharea', 'sanitboro', 'sanitsub', 'zonedist1',
       'spdist1', 'ltdheight', 'landuse',  'ext', 'proxcode', 'irrlotcode', 'lottype',
       'borocode','edesignum', 'sanitdistrict', 'healthcenterdistrict', 'pfirm15_flag']

df_dummies = pd.get_dummies(df[categorical_vars], drop_first=True) 
df.drop(categorical_vars, axis=1, inplace=True)
df = pd.concat([df, df_dummies], axis=1)

X = df[df.columns]
X.drop('assesstot', axis=1, inplace=True)
predictors = X.columns
X = X.values
Y = df['assesstot'].values

scaler = MinMaxScaler()
x_norm = scaler.fit_transform(X)
y_norm = scaler.fit_transform(Y.reshape(-1,1))
x_train, x_test, y_train, y_test = train_test_split(x_norm, Y.reshape(-1,1), test_size=0.2, random_state=0)

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Range('units',
                                          min_value=1,
                                          max_value=512,
                                          step=5),
                           activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['mse'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='/Users/mikaelapisanileal/Desktop/Capstone/nn/',
    project_name='test_nn')

tuner.search_space_summary()
tuner.search(x, y,
             epochs=5,
             validation_data=(x_test, y_test))
models = tuner.get_best_models(num_models=2)
