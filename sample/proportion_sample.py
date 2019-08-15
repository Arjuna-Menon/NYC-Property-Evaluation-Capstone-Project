#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:09:02 2019

@author: mikaelapisanileal
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#create sample for df with the same proportion by zipcode
#target variable would be the target and the rest of the variables the predictors
#(without zipcode)
def create_sample(df, target_var, train_p, test_p):
    prop_vars = ["zipcode"]
    predictors = df.columns.drop([target_var])
    target = [target_var]
    
    #transform to categorical variables, variables choosen to take proportion
    df[prop_vars] = df[prop_vars].apply(lambda x:x.astype('category'))
    
    #calculate proportions and remove the ones that has less than 2 in one class
    a = df[['zipcode']].groupby(["zipcode"]).size()
    to_remove = a[a<2]
    a = a[a>1]
    df = df.loc[df['zipcode'].isin(a.index.values)]
    df['zipcode'] = df['zipcode'].cat.remove_categories(to_remove.index.values)
    
    #set seet
    RANDOM_SEED = 101
    #transform predictors and target to array
    x = np.array(df[predictors])
    y = np.array(df[target])
    #get values for proportion variables
    categorical_prop_values = df.select_dtypes(include=['category'])
    #choose 20% for training and 5% for test
    #reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_p, 
                                                        test_size=test_p, 
                                                        random_state = RANDOM_SEED, 
                                                        stratify=categorical_prop_values)

    return (X_train, X_test, y_train, y_test, predictors)


