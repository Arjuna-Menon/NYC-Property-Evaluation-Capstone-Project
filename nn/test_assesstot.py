#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:20:04 2019

@author: mikaelapisanileal
"""

import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# load data
df = pd.read_csv('sample_0.011.csv')
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
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(len(predictors), tf.keras.activations.linear))
model.add(tf.keras.layers.Dense(int(len(predictors)/2), tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(int(len(predictors)/6), tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(1, tf.keras.activations.linear))

from keras import backend

def rmse_metric(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
 

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', rmse_metric])

history_train = model.fit(x_train, y_train, epochs=20, verbose=0, validation_data=(x_test, y_test))

#loss graph
plt.plot(history_train.history['loss'])
plt.plot(history_train.history['val_loss'])
plt.title('loss curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    

#plot y predict vs actual y for test set
y_pred = model.predict(x_test)
fig, ax = plt.subplots()
ax.plot(y_test, color = 'blue')
ax.plot(y_pred, color = 'red')
ax.legend(['Real', 'Predicted'])
fig.savefig('real_vs_pred_nn_assesstot.png')
plt.show()

print("mean_squared_error=%f" %(np.mean(history_train.history['mean_squared_error'])))
print("mean_absolute_percentage_error=%f" %(np.mean(history_train.history['mean_absolute_percentage_error'])))
print("rmse_metric=%f" %(np.mean(history_train.history['rmse_metric'])))
