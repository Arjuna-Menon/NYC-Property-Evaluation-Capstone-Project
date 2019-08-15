## I followed this tutorial:
#https://stackabuse.com/decision-trees-in-python-with-scikit-learn/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#READ the dataset
dataset = pd.read_csv('pluto3.csv')

#Check DATATYPE of df
dataset.info()
#List of variables to convert to FACTOR
to_factors = ["block","lot","cd","schooldist","council","zipcode","firecomp","policeprct",
               "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
               "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
               "healthcenterdistrict", "pfirm15_flag"]
#Iterate thru dataset and convert columns from "to_factors" into 
for i in to_factors: 
    dataset[i] = dataset[i].astype('category')
    print(i) 

#Drop X and Y coordinates
dataset = dataset.drop('xcoord', axis=1)
dataset = dataset.drop('ycoord', axis=1)



for i in to_factors: 
    conv = pd.get_dummies(dataset[i])
    # Drop column B as it is now encoded
    dataset = df.drop(i,axis = 1)
    # Join the encoded df
    dataset = dataset.join(conv)
    print(i) 


















## All below is done with LabelEncoder (with assumes ordinal data)


#Pandas doesnt like values such as "L190" even when it's categorical.
#Here we convert such values into numbers, so "L190" can be coded for example as: 1, etc.
# Loead the function LabelEncoder.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#Convert those columns
convert_for_le =['firecomp', 'sanitsub','zonedist1','irrlotcode']
#Converting in the loop
for i in convert_for_le: 
    dataset[i] = le.fit_transform(dataset[i].astype(str))
    print(i) 

### TEMPORARLY: drop income
dataset = dataset.drop('income', axis=1)
dataset = dataset.drop('assessland', axis=1)



#Divide the data
X = dataset.drop('assesstot', axis=1)
y = dataset['assesstot']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Training and making predictions
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df

#Evaluate the model:
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


regressor.feature_importances_



#### From here is random forest to choose best predictors: ####

## Import the random forest model.
from sklearn.ensemble import RandomForestClassifier 
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
regressor.score(X_test, y_test)

import pandas as pd
feature_importances = pd.DataFrame(regressor.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances['importance'].nlargest(10).plot(kind='barh')
