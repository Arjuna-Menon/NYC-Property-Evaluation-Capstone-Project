#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:09:04 2019

@author: mikaelapisanileal
"""
from proportion_sample import create_sample
import pandas as pd

def get_model_assessland(dep_var):
    df = pd.read_csv("../pluto4.csv")
    df.drop(dep_var, axis=1, inplace=True)
    df_assessland = pd.DataFrame.copy(df)
    df_assessland.drop("assesstot", axis=1, inplace=True)
    return create_sample(df_assessland, "assessland", 0.2, 0.05)

def get_model_assesstot(dep_var):
    df = pd.read_csv("../pluto4.csv")
    df.drop(dep_var, axis=1, inplace=True)
    df_assestot = pd.DataFrame.copy(df)
    df_assestot.drop("assessland", axis=1, inplace=True)
    return create_sample(df_assestot, "assesstot", 0.2, 0.05)

def save_dataframes(X_train, X_test, y_train, y_test, predictors, target_var, model_id):
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)
    X_train.columns = predictors
    y_train.columns = ["assessland"]
    X_test.columns = predictors
    y_test.columns = ["assessland"]
    X_train.to_csv("X_train_"+ target_var+ "_"+ model_id + ".csv", index=False)
    y_train.to_csv("y_train_"+ target_var+ "_"+ model_id + ".csv", index=False)
    X_test.to_csv("X_test_"+ target_var+ "_"+ model_id +".csv", index=False)
    y_test.to_csv("y_test_"+ target_var+ "_"+ model_id + ".csv", index=False)
    

dep_var1 =["cd", "council", "firecomp", "policeprct", "healtharea",
              "sanitboro", "sanitsub", "sanitdistrict", "healthcenterdistrict", 
              "block", "xcoord", "ycoord","block", "lot", "borocode",
              "xcoord", "ycoord"]
X_train, X_test, y_train, y_test, predictors = get_model_assessland(dep_var1)
save_dataframes(X_train, X_test, y_train, y_test, predictors, "assessland", "1")
X_train, X_test, y_train, y_test, predictors = get_model_assesstot(dep_var1)
save_dataframes(X_train, X_test, y_train, y_test, predictors, "assesstot", "1")

dep_var2 = dep_var1 + ["facilfar", "schooldist", "zonedist1", "spdist1", 
           "landuse", "ext", "proxcode", "irrlotcode", "edesignum", "pfirm15_flag"]

X_train, X_test, y_train, y_test, predictors = get_model_assessland(dep_var2)
save_dataframes(X_train, X_test, y_train, y_test, predictors, "assessland", "2")
X_train, X_test, y_train, y_test, predictors = get_model_assesstot(dep_var2)
save_dataframes(X_train, X_test, y_train, y_test, predictors, "assesstot", "2")

dep_var3 = dep_var1 + ["irrlotcode", "numfloors", "unitsres", "lotfront", "bldgfront",
                           "bldgdepth","yearbuilt", "residfar", "commfar", "facilfar", 
                           "yearalter", "income"]

X_train, X_test, y_train, y_test, predictors = get_model_assessland(dep_var3)
save_dataframes(X_train, X_test, y_train, y_test, predictors, "assessland", "3")
X_train, X_test, y_train, y_test, predictors = get_model_assesstot(dep_var3)
save_dataframes(X_train, X_test, y_train, y_test, predictors, "assesstot", "3")
