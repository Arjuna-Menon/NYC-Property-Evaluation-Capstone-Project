#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:47:20 2019

@author: jaimiecapps
"""
import pandas as pd
# Train/Test Split
from sklearn.model_selection import train_test_split
# PreProcessing
from sklearn.preprocessing import StandardScaler
# Training the Model
from sklearn.neural_network import MLPClassifier
# Predictions and Evaluation
from sklearn.metrics import classification_report, confusion_matrix

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


import seaborn as sns # visualization
###############################
sns.pairplot( data=df, vars=(df))


df.shape    # (15536, 413)

X = df[df.columns]
X.drop('assessland', axis=1, inplace=True)
predictors = X.columns
X = X.values                    # Array
Y = df['assessland'].values     # Array

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y)

# PreProcessing
scaler = StandardScaler()
scaler.fit(X_train)     # StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)     # Array
X_test = scaler.transform(X_test)       # Array


# Training the Model
mlp = MLPClassifier(hidden_layer_sizes = (7,7,7), max_iter = 20) # 7 is random and 3 hidden layers is just a guess
mlp.fit(X_train, y_train)
#MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       #beta_2=0.999, early_stopping=False, epsilon=1e-08,
       #hidden_layer_sizes=(7, 7, 7), learning_rate='constant',
       #learning_rate_init=0.001, max_iter=20, momentum=0.9,
       #nesterovs_momentum=True, power_t=0.5, random_state=None,
       #shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       #verbose=False, warm_start=False)
print("Training set score: %f" % mlp.score(X_train, y_train)) # 0.020512
print("Test set score: %f" % mlp.score(X_test, y_test)) # 0.001030


# Predictions and Evaluation
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

len(mlp.coefs_)         # 4
len(mlp.coefs_[0])      # 412
print("weights between input and first hidden layer:")
mlp.coefs_[0]
print("\nweights between first hidden and second hidden layer:")
mlp.coefs_[1]
print("\nweights between second hidden and third hidden layer:")
mlp.coefs_[2]
len(mlp.intercepts_[0]) # 7


# Biases per layer
print("Bias values for first hidden layer:")
print(mlp.intercepts_[0])
print("\nBias values for second hidden layer:")
print(mlp.intercepts_[1])
print("\nBias values for third hidden layer:")
print(mlp.intercepts_[2])


fig, axes = plt.subplots(2,2)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[2].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(20.5, 20.5), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()


