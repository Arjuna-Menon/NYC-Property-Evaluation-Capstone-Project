# Loading the libraries required:
library(glmnet)
library(dummies)
source("data_type_fun.R")
source("libraries.R")

df <- read.csv("sample/sample_0.011.csv")

# Delete block and lot:
df <- subset(df, select = -c(block,lot))

# Categorical variables and converting to dummies:
col_var <- c("cd","schooldist","council","zipcode","firecomp","policeprct",
             "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
             "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
             "healthcenterdistrict", "pfirm15_flag")

df <- data_type(df)

df <- dummy.data.frame(df, sep = "_")  # 1 dummy not dropped for each category!
# Have to convert to dummies for avoiding the error of some factors' levels present
# in training set but not in test set.

# Since we're predicting assessland:
df <- subset(df, select = -c(assesstot))
str(df)

# Getting the independent variables:
x_var <- data.matrix(subset(df, select = -c(assessland)))
  
# Getting the dependent variable:
y_var <- df[,'assessland']

# Setting the range of lambda values
lambda_seq <- 10^seq(2, -2, by = -.1)
# Using glmnet function to build the ridge regression model
fit <- glmnet(x_var, y_var, alpha = 0, lambda  = lambda_seq)
# Checking the model
summary(fit)

# Using cross validation glmnet to find the best lambda value:
lambdas <- 10^seq(3, -2, by = -.1)
ridge_cv <- cv.glmnet(x_var, y_var, alpha = 0, lambda = lambda_seq)
# Best lambda value
best_lambda <- ridge_cv$lambda.min
best_lambda

best_fit <- ridge_cv$glmnet.fit
head(best_fit)

# Rebuilding the model with optimal lambda value
best_ridge <- glmnet(x_var, y_var, alpha = 0, lambda = 100)

coef(best_ridge)

# Linear regression:
library(caret)
# Split data into train and test
index <- createDataPartition(df$assessland, p = .80, list = FALSE)
train <- df[index, ]
test <- df[-index, ]

# Taining model
lmModel <- lm(assessland ~ . , data = train)
# Printing the model object
print(lmModel)

library(Metrics)
rmse(actual = train$assessland, predicted = lmModel$fitted.values)

hist(lmModel$residuals)
#plot(lmModel)

# Predicting assessland in test dataset
test$predassessland <- predict(lmModel, test)
# Priting top 6 rows of actual and predited price
head(test[ , c("assessland", "predassessland")])

actual <- test$assessland
preds <- test$predassessland
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq

# Using the ridge regression results (x is the test data)
#x <- model.matrix(~.-test$assessland, data = test)
pred <- predict(best_ridge, s = best_lambda, newx = x)

# R squared formula
actual <- test$Price
preds <- test$PreditedPrice
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq