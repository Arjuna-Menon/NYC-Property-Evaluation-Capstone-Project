
#Load the libraries
import pandas as pd
import numpy as np


#Read the data
data = pd.read_csv("sample/sample_0.011.csv")

numeric =  ["lotarea", "bldgarea","numbldgs","numfloors","unitsres","unitstotal","lotfront",
            "lotdepth","bldgfront","bldgdepth","yearbuilt",
            "residfar","commfar","facilfar","yearalter", 'assessland']

from scipy import stats
data=data[(np.abs(stats.zscore(data[numeric])) < 3).all(axis=1)]

data=data[numeric]
df1=data

####################################################TEST ON WITHOUT LOCATION VAR
#data = pd.read_csv("pluto3.csv")
#data=data.drop(['xcoord','ycoord','firecomp'], axis=1)


## Check Outliers
#data.boxplot(column=['assessland'])
#data.nlargest(100, ['assessland'])['assessland']
#data=data.loc[data['assessland'] <= 1159650]
#data = data.reset_index()


data.isnull().sum()

##### Feature Selection #####
#### 1. f_regression using SelectKBest
# About the F-Test etc.: https://stats.stackexchange.com/questions/204141/difference-between-selecting-features-based-on-f-regression-and-based-on-r2



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
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


## Run random Forest with k predictors choosen by previous function
## and calculate the rmse    
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

    #Build a tree
    reg = RandomForestRegressor(
            n_estimators=300, 
            max_depth=100, 
            bootstrap=True, 
            random_state=123
            )
    reg.fit(X_train, y_train)
    preds_train = reg.predict(X_train)
    preds_test = reg.predict(X_test)
    
    # Calcualte evaluation metrics for TRAIN
    rms_train = sqrt(mean_squared_error(y_train, preds_train))
    #mae_train = mean_absolute_error(y_train, preds_train)
    #mean_err_train=np.square(np.subtract(y_train, preds_train)).mean()

    
    # Calcualte evaluation metrics for TEST
    rms_test = sqrt(mean_squared_error(y_test, preds_test))
    #mae_test = mean_absolute_error(y_test, preds_test)
    #mean_err_test=np.square(np.subtract(y_test, preds_test)).mean()


    return rms_train, rms_test

# max number of predictors you want:
k=30
error_train=[]
error_test=[]
for i in range(1, k+1):
    print("number of predictors",i)

    # Call the function which gives you k best predictors
    top_predictors_list=select_kbest_reg(df1, 'assessland', k=i)
    print(top_predictors_list)

    # Call the function to create a tree and give rmse
    err=a_tree(df1, top_predictors_list)
    print('Error:',err)
    error_train.append(err[0])
    error_test.append(err[1])

#Plot two errors
from matplotlib import pyplot as plt
## Train Set
plt.plot(error_train)
plt.title('Error Train and Test (train)' )
plt.xlabel('Number of predictors')
plt.ylabel('Error')


## Test Set
plt.plot(error_test)
plt.title('Error: Train and Test')
plt.xlabel('Number of predictors')
plt.ylabel('Error')
plt.show()


#### Histogram of errors
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#for test set
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
plt.title('Error Diff in Test set')
plt.hist(p, bins='auto', range=(-50000, 50000))


#for train set
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
plt.title('Error Diff in Train set')
plt.hist(p, bins='auto', range=(-50000, 50000))
