# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:06:10 2019

@author: grzechu
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from sklearn.feature_selection import RFE

X = df1.drop('assessland',axis=1)
y = df1['assessland']

Xtrn, Xtest, Ytrn, Ytest = train_test_split(X,y,test_size=0.3, random_state=42)
models = [LinearRegression(), linear_model.Lasso(alpha=0.1), Ridge(alpha=100.0), RandomForestRegressor(n_estimators=100, max_features='sqrt'), KNeighborsRegressor(n_neighbors=6),DecisionTreeRegressor(max_depth=4), ensemble.GradientBoostingRegressor()]

TestModels = pd.DataFrame()
tmp = {}
 
for model in models:
    print(model)
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
    model.fit(Xtrn, Ytrn)
    tmp['R2_Price'] = r2_score(Ytest, model.predict(Xtest))
    print('score on training',model.score(Xtrn, Ytrn))
    print('r2 score',r2_score(Ytest, model.predict(Xtest)))
    TestModels = TestModels.append([tmp])
TestModels.set_index('Model', inplace=True)
 
fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()