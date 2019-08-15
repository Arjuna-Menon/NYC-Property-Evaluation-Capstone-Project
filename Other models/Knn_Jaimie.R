### Standardize pluto3

pluto <- read.csv("~/Desktop/plutostuff/pluto3.csv")

df <- pluto

head(df)

# Random sample
rand <- sample(1:nrow(df), 0.9 * nrow(df))

# Normalize
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }

# Run normalization on predictors
pluto_norm <- as.data.frame(lapply(df[,c(    ,    )], nor)) # Need predictors!
summary(pluto_norm)

# Extract training set
pluto_train <- pluto_norm[rand,]

# Extract testing set
pluto_test <- pluto_norm[-rand,]

# Extract 'cl' column to go into knn function (target variable)
pluto_target_category <- df[rand,5]

library(class)

# KNN
pluto_knn <- knn(pluto_train, pluto_test, cl=pluto_target_category, k=    ) # need k

# Confusion Matrix
pred <- table(pluto_knn, pluto_test_category)

# Accuracy (correct predictions/total num predictions)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

accuracy(pred)












