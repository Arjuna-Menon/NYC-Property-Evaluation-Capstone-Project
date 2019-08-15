# Scaling the data and converting to dummies:
source("data_type_fun.R")
source("libraries.R")
library(caret)
gc()
df <- read.csv("pluto4.csv")

# Delete block and lot:
df <- subset(df, select = -c(block,lot))

numeric <- c("lotarea", "bldgarea","numbldgs","numfloors","unitsres","unitstotal",
             "lotfront","lotdepth","bldgfront","bldgdepth","yearbuilt","residfar",
             "commfar","facilfar","xcoord","ycoord","yearalter", "income")


# Scaling numeric variables except the target variables assessland and assesstot:
for (colName in numeric){
  df[,colName] <- scale(df[,colName])
}

# Convert to factors:
df <- data_type(df)

# Convert to dummies:
dmy <- dummyVars("~.", data = df, fullRank = T)
df <- data.frame(predict(dmy, newdata = df)) # Have to convert to dummies for avoiding 
#the error of some factors' levels present in training set but not in test set.
write.csv(df, file = "pluto6_fullstd.csv", row.names=FALSE)
