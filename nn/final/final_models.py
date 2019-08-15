#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:51:13 2019

@author: mikaelapisanileal
"""

import nn_functions

print('PCA assesstot - sample')
x_train, x_test, y_train, y_test, predictors = nn_functions.get_data('../../plutoPCA_tot.csv','assesstot')
epochs = 30
input_nodes = 170
hidden_nodes = 114 
optimizer = 'adam'
nn_functions.run_model(input_nodes, hidden_nodes, x_train, x_test, y_train, y_test, optimizer, epochs, 1)

print('PCA assessland - sample')
x_train, x_test, y_train, y_test, predictors = nn_functions.get_data('../../plutoPCA_land.csv','assessland')
epochs = 10
input_nodes = 170
hidden_nodes = 51 
optimizer = 'rmsprop'
nn_functions.run_model(input_nodes, hidden_nodes, x_train, x_test, y_train, y_test, optimizer, epochs, 2)

print('PCA assesstot - full')
x_train, x_test, y_train, y_test, predictors = nn_functions.get_data('../../python_modeling/arjuna/plutoPCA_fulltot.csv','assesstot')
epochs = 30
input_nodes = 170
hidden_nodes = 114 
optimizer = 'adam'
nn_functions.run_model(input_nodes, hidden_nodes, x_train, x_test, y_train, y_test, optimizer, epochs, 3)

print('PCA assessland - full')
x_train, x_test, y_train, y_test, predictors = nn_functions.get_data('../../python_modeling/arjuna/plutoPCA_fullland.csv','assessland')
epochs = 100
input_nodes = 170
hidden_nodes = 51 
optimizer = 'rmsprop'
nn_functions.run_model(input_nodes, hidden_nodes, x_train, x_test, y_train, y_test, optimizer, epochs, 4)

