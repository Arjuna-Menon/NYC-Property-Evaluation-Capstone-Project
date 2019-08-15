#Read functions
source("data_type_fun.R")
source("libraries.R")

# Read data and change types:
df <- read.csv("pluto2.csv")
df <- data_type(df) # function changing data types

# Choose columns with small level (lower than 53)
df <- subset(df ,select = -c(block, lot, zipcode, firecomp, policeprct, healtharea,sanitsub))
#Get rid of coordinates columns (x and y)
df <- subset(df, select = -c(xcoord, ycoord))

# Sample for test
sample <- sample_n(df, 100000)
sample <- na.omit(sample)

rf <- randomForest(assesstot ~ ., data = sample, ntree = 1, nodesize = 5, importance = TRUE)  
# Can not handle categorical predictors with more than 53 categories.
varImpPlot(rf, type = 1)
