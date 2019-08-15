import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv("sample/sample_0.011.csv")
data=data.drop(['firecomp'], axis=1)
# drop also block and lot
data=data.drop(['lot','block'], axis=1)


#Scpecify what columns are factors
to_factors = ["cd","schooldist","council","zipcode","policeprct",
              "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
              "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
              "healthcenterdistrict", "pfirm15_flag"]

numeric =  ["lotarea", "bldgarea","numbldgs","numfloors","unitsres","unitstotal","lotfront",
            "lotdepth","bldgfront","bldgdepth","yearbuilt",
            "residfar","commfar","facilfar","yearalter"]
  

#Iterate thru dataset and convert columns from "to_factors" into 
for i in to_factors: 
    data[i] = data[i].astype('category')
    print(i) 
## Target variables is assesstot
df1 = data.drop(['assessland'], axis=1)
## Convert all to dummies, AND DELETE factors which means we do k-1 variables
df_dummies = pd.get_dummies(df1[to_factors], drop_first=True)
#Drop old factors from the dataset (oryginal one, those not one-hot encoded)
df1.drop(to_factors, axis=1, inplace=True)
scaler = MinMaxScaler()
df1[numeric] = scaler.fit_transform(df1[numeric])
#Concat numeric variables wiht converted factors
df1 = pd.concat([df1, df_dummies], axis=1)
X = df1.drop(['assesstot'], axis=1)
y = data['assesstot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

reg = MLPRegressor(hidden_layer_sizes=(100), alpha=0.0001)
reg.fit(X_train,y_train)
pred_reg = reg.predict(X_test)
rms = sqrt(mean_squared_error(y_test, pred_reg))
rms


y_test = pd.DataFrame(y_test)
y_test['new']=y_test.index
pred_reg = pd.DataFrame(pred_reg)
pred_reg.index=y_test['new'].values
y_test = y_test.drop('new',axis=1)
pred_reg = pred_reg.rename(columns={0:'predicted'})
x =pd.DataFrame(y_test['assesstot']-pred_reg['predicted'])
x = x.rename(columns={0:'difference'})
done = pd.concat([x,y_test,pred_reg],axis=1)

max(x['difference'])
x.index[x['difference']==31122447.893091712]
data.iloc[2607,:]

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
p = x['difference'].values
type(p)

plt.hist(p, bins='auto', range=(-100000, 100000))
