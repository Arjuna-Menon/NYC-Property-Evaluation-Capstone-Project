# PCA:

import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("pluto5_samplestd.csv")
# For predicting assessland, drop the following:
df.drop(['bldgarea', 'numfloors', 'unitsres', 'unitstotal','bldgfront', 'bldgdepth', 
                'ext.1','proxcode.1', 'proxcode.2',
                'yearbuilt', 'yearalter','income'],axis=1,inplace=True)
df1 = df.drop(["assessland", "assesstot"], axis =1)
pca = PCA(n_components = 170)
pc = pca.fit_transform(df1)
a = pca.explained_variance_ratio_
sum(a) # 93.4% of data variation explained 
for i in a:
    print('{:f}'.format(i))
pdf = pd.DataFrame(data = pc)
df2 = pd.concat([pdf,df["assessland"]], axis = 1)
df2.to_csv("plutoPCA_land.csv", index = False)

# For predicting assesstot:
df = pd.read_csv("pluto5_samplestd.csv")
df1 = df.drop(["assessland", "assesstot"], axis =1)
pca = PCA(n_components = 170)
pc = pca.fit_transform(df1)
a = pca.explained_variance_ratio_
sum(a) # 95.2% of data variation explained
for i in a:
    print('{:f}'.format(i))
pdf = pd.DataFrame(data = pc)
df2 = pd.concat([pdf,df["assesstot"]], axis = 1)
df2.to_csv("plutoPCA_tot.csv", index = False)
