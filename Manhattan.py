import pandas as pd

df= pd.read_csv("pluto6_fullstd.csv")
df['borocode.1'] = 1- (df['borocode.2'] + df['borocode.3'] + df['borocode.4'] + df['borocode.5'])
df = df.drop(df[df['borocode.1']!=1].index)
df.drop(['borocode.1'], axis = 1, inplace = True) #axis=1 for column

df.to_csv("Manhattan_full.csv", index = False)

df1 = pd.read_csv("pluto4.csv")
j = df1[df1['borocode']!=1].index
df1 = df1.drop(j)
df1.to_csv("Manhattan_Merge.csv", index = False)
