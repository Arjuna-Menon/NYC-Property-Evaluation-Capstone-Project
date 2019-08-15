#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:12:14 2019

@author: jaimiecapps
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt


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

# sigmoid function 
# A sigmoid function maps any value to a value between 0 and 1.
# We use it to convert numbers to probabilities. 
def nonlin(x,deriv=False):  # defines non-linearity
    if(deriv==True):        # deriviative creation
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input dataset
#X = np.array(df[1: ], dtype = np.float)  # Each row is a single "training example".
                    # Each column corresponds to one of our input nodes.

# output dataset           
#y = np.array(df[1: ], dtype = np.float).T    # ".T" is the transpose 

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1    # It's called "syn0" to imply "synapse zero". Since we only have 2 layers (input and output), 
syn0 = 2*np.random.random((3,1)) - 1    # we only need one matrix of weights to connect them.
                                        # Its dimension is (3,1) because we have 3 inputs and 1 output.

for iter in xrange(10000):  # This for loop "iterates" multiple times over the training code to optimize our network to the dataset.
    
    # forward propagation
    l0 = X                          # first layer, l0, is simply our data.
    l1 = nonlin(np.dot(l0,syn0))    # prediction step. Basically, 
                                    # we first let the network "try" to predict the output given the input.

    # how much did we miss?
    l1_error = y - l1               # l1_error is just a vector of positive and 
                                    # negative numbers reflecting how much the network missed.

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)   # Secret sauce
    
    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print ("Output After Training:")
print (l1)
