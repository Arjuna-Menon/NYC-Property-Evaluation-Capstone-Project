#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:25:31 2019

@author: mikaelapisanileal
"""

    
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import keras.backend as K


def generate_report(y_actual, y_pred):
    mse = round(mean_squared_error(y_actual, y_pred),3)
    rmse = round(sqrt(mean_squared_error(y_actual, y_pred)),3)
    r2 = round(r2_score(y_actual, y_pred),3)
    error = np.mean(pd.DataFrame(y_actual) - pd.DataFrame(y_pred))[0]
    print('mse',mse)
    print('RMSE', rmse)
    print('R2', r2)
    print('error', error)
    return mse,rmse,r2,error

def generate_loss_plot(history, filename=None):
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_title('loss curve')
    ax.legend(['train', 'test'], loc='upper left')
    if (filename!=None):
        fig.savefig(filename)
    plt.show()
    


def generate_hist_plot(y_actual, y_pred, ax, title):    
    y = pd.DataFrame(y_actual)
    y['new']=y.index
    pred = pd.DataFrame(y_pred)
    pred.index=y['new'].values
    y = y.drop('new',axis=1)
    pred = pred.rename(columns={0:'predicted'})
    x =pd.DataFrame(y[0]-pred['predicted'])
    x = x.rename(columns={0:'difference'})
    p = x['difference'].values
    ax.hist(p, bins='auto', range=(-75000, 75000), color='#168BBD')
    ax.set_title(title) 
    return ax


def get_data(filename, target): 
    df = pd.read_csv(filename)
    X = df.copy()
    X.drop(target, axis=1, inplace=True)
    predictors = X.columns
    X = X.values
    Y = df[target].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test, predictors

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def fit_model(model, x_train, x_test, y_train, y_test, optimizer, epochs, model_id=None):
    model.compile(loss=root_mean_squared_error, optimizer=optimizer, metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_data=(x_test, y_test), shuffle=True)
    filename = None
    if (model_id!=None):
        filename = 'loss_' + str(model_id) + '.png'
    generate_loss_plot(history, filename)
    return model

def plot_compare(y_test, y_test_pred, ax, show_legend=False):
    ax.plot(y_test, color = '#7DB4C4')
    ax.plot(y_test_pred, color = '#E65353')
    if (show_legend):
        ax.legend(['Real', 'Predicted'])
    return ax

def predict(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print('ERROR Training')
    generate_report(y_train, y_train_pred)
    print('ERROR Test')
    mse,rmse,r2,error = generate_report(y_test, y_test_pred)
    return y_train_pred, y_test_pred, mse, rmse, r2, error
    
def run_model(input_nodes, hidden_nodes, x_train, x_test, y_train, y_test, optimizer, epochs, model_id=None):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_nodes, tf.keras.activations.linear))
    model.add(tf.keras.layers.Dense(hidden_nodes, tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1, tf.keras.activations.linear))
    model = fit_model(model, x_train, x_test, y_train, y_test, optimizer, epochs, model_id)    
    y_train_pred, y_test_pred, mse,rmse,r2,error = predict(model, x_train, y_train, x_test, y_test)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True)
    ax1 = generate_hist_plot(y_train, y_train_pred, ax1, 'Training error')
    ax2 = generate_hist_plot(y_test, y_test_pred, ax2, 'Test error')
    if (model_id!=None):
        plt.savefig('nn_hist_' + str(model_id) + '.png')
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=True)
    ax1 = plot_compare(y_train, y_train_pred, ax1)
    ax2 = plot_compare(y_test, y_test_pred, ax2, True)

    if (model_id!=None):
        plt.savefig('nn_' + str(model_id) + '.png')
    return y_train_pred, y_test_pred, mse,rmse,r2,error

