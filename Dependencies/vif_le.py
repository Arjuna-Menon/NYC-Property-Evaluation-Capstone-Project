import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

df = pd.read_csv("Dependencies/sample.csv")

to_factors = [ "cd", "firecomp", "schooldist","council","zipcode","policeprct",
               "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
               "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
               "healthcenterdistrict", "pfirm15_flag"]

# Deleting block and lot since it provides too much details:
df = df.drop(['lot','block'], axis=1)

# Converting to factors:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Converting in the loop
for i in to_factors: 
    df[i] = le.fit_transform(df[i].astype(str))
df['firecomp'].dtypes


#Iterate thru dataset and convert columns from "to_factors" into 
for i in to_factors: 
    df[i] = df[i].astype('category')
    print(i)    
df['firecomp'].dtypes    


# For predicting assessland:
X = df
X = X.drop('assesstot', axis = 1)
X = X.drop('assessland', axis = 1)
X = add_constant(X)


# We don't consider index = 0 since it is the const VIF value
while True:
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, j) for j in range(X.shape[1])]
    vif["features"] = X.columns
    if max(vif["VIF Factor"][1:]>5):
        k = vif.index[vif['VIF Factor'] == max(vif['VIF Factor'][1:])]
        k=k.tolist()
        k=k[0]
        l.append(vif["features"][k])
        X.drop(vif["features"][k], axis = 1, inplace = True)
    else:
        break;
vif.round(1)

# The deleted variables are:
l1
