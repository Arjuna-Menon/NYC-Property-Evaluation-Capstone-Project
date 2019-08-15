import pandas as pd
import numpy as np

# Reading pluto3 csv:
data = pd.read_csv("pluto3.csv")

# Numerical variables:
numeric =  ["lotarea", "bldgarea","numbldgs","numfloors","unitsres","unitstotal","lotfront",
            "lotdepth","bldgfront","bldgdepth","yearbuilt", "assessland", "assesstot",
            "residfar","commfar","facilfar","yearalter"]

# Deleting all outliers that are three standard deviations from the mean for at least one of the numeric variables:
from scipy import stats
data = data[(np.abs(stats.zscore(data[numeric])) < 3).all(axis = 1)]

data.isnull().sum()
# Deleting the 29 rows containing NAs in xcoord and ycoord:
data = data.dropna(axis = 0, how = "any")

# Writing to csv:
data.to_csv("pluto4.csv", index = False)
