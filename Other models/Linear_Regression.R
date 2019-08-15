# Loading the libraries required:
library(glmnet)
library(caret)
source("libraries.R")
library(Metrics)

options("scipen"=100, "digits"=4)

set.seed(123)
df <- read.csv("pluto5_samplestd.csv")

# 1. Since we're predicting assessland, drop the following variables:
df_land <- subset(df, select = -c(bldgarea, numfloors, unitsres, unitstotal, bldgfront, 
                                  bldgdepth, ext.1, proxcode.1, proxcode.2, yearbuilt,
                                  yearalter, income))
df_land <- subset(df_land, select = -c(assesstot))
df_land[,'assessland'] <- log(df[,'assessland'])

# Getting the independent variables:
x_var <- data.matrix(subset(df_land, select = -c(assessland)))

# Getting the dependent variable:
y_var <- df_land[,'assessland']

# Linear regression:
# Split data into train and test
index <- createDataPartition(df_land$assessland, p = .80, list = FALSE)
train <- df_land[index, ]
test <- df_land[-index, ]

# Taining model
lmModel <- lm(assessland ~ . , data = train)
# Printing the model object
print(lmModel)

# RMSE for training set:
rmse(actual = exp(train$assessland), predicted = exp(lmModel$fitted.values)) #58,517.89

# R squared value for training set:
actual <- train$assessland
preds <- lmModel$fitted.values
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq # 0.69

# Histogram for errors in training set:
hist(exp(train$assessland) - exp(lmModel$fitted.values))

# Predicting assessland in test dataset
test$predassessland <- predict(lmModel, test)
# Priting top 6 rows of actual and predited price
head(test[ , c("assessland", "predassessland")])

# RMSE for test set:
rmse(actual = exp(test$assessland), predicted = exp(test$predassessland)) #37,565

# R squared value for test set:
actual <- test$assessland
preds <- test$predassessland
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq # 0.56


# 2. Predicting assesstot:
df_tot <- subset(df, select = -c(assessland))
df_tot[,"assesstot"] <- log(df_tot[,"assesstot"])

# Getting the independent variables:
x_var <- data.matrix(subset(df_tot, select = -c(assesstot)))

# Getting the dependent variable:
y_var <- df_tot[,'assesstot']

# Split data into train and test
index <- createDataPartition(df_tot$assesstot, p = .80, list = FALSE)
train <- df_tot[index, ]
test <- df_tot[-index, ]

# Taining model
lmModel <- lm(assesstot ~ . , data = train)

# Histogram for errors in test set:
hist(exp(test$assesstot) - exp(lmModel$fitted.values))

# RMSE for training set:
rmse(actual = exp(train$assesstot), predicted = exp(lmModel$fitted.values)) #507,087

# R squared value for training set:
actual <- train$assesstot
preds <- lmModel$fitted.values
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq # 0.87

# Histogram for training set:
hist(exp(train$assesstot) - exp(lmModel$fitted.values))

# Predicting assessland in test dataset
test$predassesstot <- predict(lmModel, test)
# Priting top 6 rows of actual and predited price
head(test[ , c("assesstot", "predassesstot")])

# RMSE for test set:
rmse(actual = exp(test$assesstot), predicted = exp(test$predassesstot)) #366,138

# R squared value for test set:
actual <- test$assesstot
preds <- test$predassesstot
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq # 0.83
