"""
Created on Thu Jul 25 21:46:17 2019
@author: marcin
"""

## Based on:
# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

# Imports
import pandas as pd
import numpy as np

#select_kbest_reg
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Tree
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Read the data

############# Read the data and transform 
data = pd.read_csv("sample/sample_0.011.csv")

data.isnull().sum()


# drop also block and lot
data=data.drop(['lot','block'], axis=1)


#Scpecify what columns are factors
to_factors = ["cd","schooldist","council","zipcode","policeprct","firecomp",
               "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
               "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
               "healthcenterdistrict", "pfirm15_flag"]

## Make label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Converting in the loop
for i in to_factors: 
    data[i] = le.fit_transform(data[i].astype(str))
    print(i) 
data['firecomp'].dtypes


#Iterate thru dataset and convert columns from "to_factors" into 
for i in to_factors: 
    data[i] = data[i].astype('category')
    print(i)    
data['firecomp'].dtypes    


## Target variables is Assessland
df1 = data.drop(['assesstot'], axis=1)



#### Function I ->  Feature Selection #####
#### 1. f_regression using SelectKBest
# About the F-Test etc.: https://stats.stackexchange.com/questions/204141/difference-between-selecting-features-based-on-f-regression-and-based-on-r2


## Select the best k predictors from data and return a list with predictors
def select_kbest_reg(data_frame, target, k):
    """
    Selecting K-Best features regression
    :param data_frame: A pandas dataFrame with the training data
    :param target: target variable name in DataFrame
    :param k: desired number of features from the data
    :returns feature_scores: scores for each feature in the data as 
    pandas DataFrame
    """
    feat_selector = SelectKBest(f_regression, k=k)
    _ = feat_selector.fit(data_frame.drop(target, axis=1), data_frame[target])
    
    feat_scores = pd.DataFrame()
    feat_scores["F-Score"] = feat_selector.scores_
    feat_scores["P-Value"] = feat_selector.pvalues_
    feat_scores["Support"] = feat_selector.get_support()
    feat_scores["Attribute"] = data_frame.drop(target, axis=1).columns
    
    #top_predictors=select_kbest_reg(df1, 'assessland', 10)
    top_predictors_true=feat_scores.loc[feat_scores['Support']]
    #Convert column with predictors into list
    predictors_list=top_predictors_true['Attribute'].values.tolist()
    
    return predictors_list




#### Function II -> Build the tree model

## Run random Forest with k predictors choosen by previous function
## and calculate the rmse    
### Use those predictors to build a tree
def a_tree(df1, top_predictors_list):
    # X contains only predictors choosen by the function select_kbest_reg
    X = df1[top_predictors_list]
    
    # create 'y' but first scale it.
    #mms = MinMaxScaler()
    #df1[['assessland']] = mms.fit_transform(df1[['assessland']])
    #Scale 'y' so we can compare train/test results
    y = df1['assessland']
    
    #train/test split data
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42)

    #Build a Ridge
    ridgeReg = Ridge(alpha=0.001, normalize=True)

    ridgeReg.fit(X_train,y_train)

    preds_train = ridgeReg.predict(X_train)
    preds_test = ridgeReg.predict(X_test)
    
    # Calcualte evaluation metrics for TRAIN
    rms_train = sqrt(mean_squared_error(y_train, preds_train))
    #mae_train = mean_absolute_error(y_train, preds_train)
    #mean_err_train=np.square(np.subtract(y_train, preds_train)).mean()

    # Calcualte evaluation metrics for TEST
    rms_test = sqrt(mean_squared_error(y_test, preds_test))
    #mae_test = mean_absolute_error(y_test, preds_test)
    #mean_err_test=np.square(np.subtract(y_test, preds_test)).mean()

    return rms_train, rms_test, y_train, preds_train, y_test, preds_test


######## LOOP - error depends on number of predictors ######
k=40 # check until 30 max number of pre
error_train=[]
error_test=[]
for i in range(1, k+1):
    print("number of predictors",i)

    # Call the function which gives you k best predictors
    top_predictors_list=select_kbest_reg(df1, 'assessland', k=i)
    print(top_predictors_list)

    # Call the function to create a tree and give rmse
    err=a_tree(df1, top_predictors_list)
    print('Error train:',round(err[0],2), 'Error test:',round(err[1],2))
    error_train.append(err[0])
    error_test.append(err[1])

#Check min error
min(error_test)


######## LINE PLOT - Error for each iteration
from matplotlib import pyplot as plt
## Train Set
plt.plot(error_train)
plt.title('Error in Train set')
plt.xlabel('Number of predictors')
plt.ylabel('Error')


## Test Set
plt.plot(error_test)
plt.title('Error: in Test Set')
plt.xlabel('Number of predictors')
plt.ylabel('Error')
plt.show()




######## HISTOGRAM - Error Difference
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


y_train=err[2]
preds_train=err[3]
y_test=err[4]
preds_test=err[5]

#### Error for TRAIN set
y_train = pd.DataFrame(y_train)
y_train['new']=y_train.index
pred_reg = pd.DataFrame(preds_train)
pred_reg.index=y_train['new'].values
y_train = y_train.drop('new',axis=1)
pred_reg = pred_reg.rename(columns={0:'predicted'})
x =pd.DataFrame(y_train['assessland']-pred_reg['predicted'])
x = x.rename(columns={0:'difference'})
done = pd.concat([x,y_train,pred_reg],axis=1)

p = x['difference'].values
type(p)
plt.hist(p, bins='auto', range=(-10000, 10000))


#### Error for TEST set
y_test = pd.DataFrame(y_test)
y_test['new']=y_test.index
pred_reg = pd.DataFrame(preds_test)
pred_reg.index=y_test['new'].values
y_test = y_test.drop('new',axis=1)
pred_reg = pred_reg.rename(columns={0:'predicted'})
x =pd.DataFrame(y_test['assessland']-pred_reg['predicted'])
x = x.rename(columns={0:'difference'})
done = pd.concat([x,y_test,pred_reg],axis=1)

p = x['difference'].values
type(p)
plt.hist(p, bins='auto', range=(-10000, 10000))
