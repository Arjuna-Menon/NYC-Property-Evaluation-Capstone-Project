# Loading the libraries required:
library(glmnet)
library(Metrics)
source("data_type_fun.R")
source("libraries.R")
dplyset.seed(123)

df <- read.csv("pluto5_samplestd.csv")

# 1. Since we're predicting assessland, drop the following variables:
df_land <- subset(df, select = -c(bldgarea, numfloors, unitsres, unitstotal, 
                                  bldgfront, bldgdepth, ext.1, proxcode.1, 
                                  proxcode.2, yearbuilt, yearalter, income))
df_land <- subset(df_land, select = -c(assesstot))

x<- subset(df_land, select = -c(assessland))
y<- df_land[,'assessland']
data<-cbind(x,y)
model<-model.matrix(y~., data=data)
ridgedata= model[,-1]
train<- sample(1:dim(ridgedata)[1], round(0.8*dim(ridgedata)[1]))
test<- setdiff(1:dim(ridgedata)[1],train)
x_train <- data[train, ]
y_train <- data$y[train]
x_test <- data[test, ]
y_test <- data$y[test]
k=5 # 5 folds in cross-validation
grid =10^ seq (5,-2, length =30)
fit <- cv.glmnet(model,y,alpha=0,k=k,lambda = grid)
lambda_min<-fit$lambda.min #32903
newX <- model.matrix(~.-y,data=x_test)
fit_test<-predict(fit, newx=newX,s=lambda_min)

# R squared value:
actual <- y_test
preds <- fit_test
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq # 0.632

# RMSE:
rmse(actual = y_test, predicted = fit_test) # 34,418

# 2. Predicting assesstot:
df_tot <- subset(df, select = -c(assessland))

x<- subset(df_tot, select = -c(assesstot))
y<- df_tot[,'assesstot']
data<-cbind(x,y)
model<-model.matrix(y~., data=data)
ridgedata= model[,-1]
train<- sample(1:dim(ridgedata)[1], round(0.8*dim(ridgedata)[1]))
test<- setdiff(1:dim(ridgedata)[1],train)
x_train <- data[train, ]
y_train <- data$y[train]
x_test <- data[test, ]
y_test <- data$y[test]
k=5 # 5 folds in cross-validation
grid =10^ seq (5,-2, length =30)
fit <- cv.glmnet(model,y,alpha=0,k=k,lambda = grid)
lambda_min<-fit$lambda.min #10826
newX <- model.matrix(~.-y,data=x_test)
fit_test<-predict(fit, newx=newX,s=lambda_min)

# R squared value:
actual <- y_test
preds <- fit_test
rss <- sum((preds - actual) ^ 2)
tss <- sum((actual - mean(actual)) ^ 2)
rsq <- 1 - rss/tss
rsq # 0.77

# RMSE:
rmse(actual = y_test, predicted = fit_test) # 189,973