#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:51:39 2019

@author: mikaelapisanileal
"""

from proportion_sample import create_sample
import pandas as pd

df1 = pd.read_csv("../Brooklyn.csv")
df2 = pd.read_csv("../Brooklyn_Merge.csv")
df = pd.concat([df1, df2["zipcode"]], axis=1)
X_train, X_test, y_train, y_test, predictors = create_sample(df, "assessland", 0.6, 0.4)

df = pd.DataFrame(X_train)
df.columns = predictors
df.drop("zipcode", axis = 1, inplace = True)
df["assessland"] = pd.DataFrame(y_train)

#Save data
df.to_csv("Brooklyn_sample.csv", index=False)
