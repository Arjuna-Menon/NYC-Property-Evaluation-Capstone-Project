#http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/153-penalized-regression-essentials-ridge-lasso-elastic-net/
source("data_type_fun.R")
source("libraries.R")
library(tidyverse)
library(caret)
#install.packages("glmnet", repos = "http://cran.us.r-project.org")
library(glmnet)
library(parallel)
library(doMC)
no_cores <- max(1, detectCores() - 2)

df <- read.csv("pluto3.csv")
df <- data_type(df)

execute_lasso_assesstot <- function(df) {
  # Split the data into training and test set
  set.seed(123)
  training.samples <- df$assesstot %>% createDataPartition(p = 0.8, list = FALSE)
  train.data  <- df[training.samples, ]
  test.data <- df[-training.samples, ]
  
  # Predictor variables: transform categorical to dummpy 
  x <- model.matrix(assesstot~., train.data)[,-1]
  
  # Outcome variable
  y <- train.data$assesstot
  
  # Find the best lambda using cross-validation (lambda is to penalize)
  set.seed(1234)
  registerDoMC(cores=no_cores)
  cv <- cv.glmnet(x, y, alpha = 1, parallel = TRUE) # lasso regression alpha=1
  # Display the best lambda value
  print(paste("min lambda", cv$lambda.min))
  
  print("Fitting final model on the training data")
  # Fit the final model on the training data
  model <- glmnet(x, y, alpha = 1, lambda = cv$lambda.min)
  print("Predictions on the test data")
  x.test <- model.matrix(assesstot ~., test.data)[,-1]
  predictions_coef <- model %>% predict(newdata=x.test, type="coefficient") 
  print(predictions_coef) #the coef=0 means that the variables are not significant
  predictions <- model %>% predict(x.test) %>% as.vector()
  error = mean((predictions - test.data$assesstot)^2)
  
  # Model performance metrics
  RMSE = RMSE(predictions, test.data$assesstot)
  R2 = R2(predictions, test.data$assesstot)
  
  print(paste("error=", error))
  print(paste("RMSE=", RMSE))
  print(paste("R2=", R2))
  
  return (list(error,RMSE, R2))
}


execute_lasso_assessland <- function(df) {
  # Split the data into training and test set
  set.seed(123)
  training.samples <- df$assessland %>% createDataPartition(p = 0.8, list = FALSE)
  train.data  <- df[training.samples, ]
  test.data <- df[-training.samples, ]
  
  # Predictor variables: transform categorical to dummpy 
  x <- model.matrix(assessland~., train.data)[,-1]
  
  # Outcome variable
  y <- train.data$assessland
  
  # Find the best lambda using cross-validation (lambda is to penalize)
  set.seed(1234)
  registerDoMC(cores=no_cores)
  cv <- cv.glmnet(x, y, alpha = 1, parallel = TRUE) # lasso regression alpha=1
  # Display the best lambda value
  print(paste("min lambda", cv$lambda.min))
  
  print("Fitting final model on the training data")
  # Fit the final model on the training data
  model <- glmnet(x, y, alpha = 1, lambda = cv$lambda.min)
  print("Predictions on the test data")
  x.test <- model.matrix(assessland ~., test.data)[,-1]
  predictions_coef <- model %>% predict(newdata=x.test, type="coefficient") 
  print(predictions_coef) #the coef=0 means that the variables are not significant
  predictions <- model %>% predict(x.test) %>% as.vector()
  error = mean((predictions - test.data$assessland)^2)
  
  # Model performance metrics
  RMSE = RMSE(predictions, test.data$assessland)
  R2 = R2(predictions, test.data$assessland)
  
  print(paste("error=", error))
  print(paste("RMSE=", RMSE))
  print(paste("R2=", R2))
  
  return (list(error,RMSE, R2))
}


dep_var1 = c("cd", "council", "firecomp", "policeprct", "healtharea",
             "sanitboro", "sanitsub", "sanitdistrict", "healthcenterdistrict", 
             "block", "xcoord", "ycoord","block", "lot", "borocode",
             "xcoord", "ycoord", "zipcode")
dep_var2 = c("facilfar", "schooldist", "zonedist1", "spdist1", 
             "landuse", "ext", "proxcode", "irrlotcode", "edesignum", "pfirm15_flag")
dep_var3 <- c("irrlotcode", "numfloors", "unitsres", "lotfront", "bldgfront", "bldgdepth",
              "yearbuilt", "residfar", "commfar", "facilfar", "yearalter", "income")

df1 <- df[, !(colnames(df) %in% c(dep_var1, "assessland"))]
errors_1 = unlist(execute_lasso_assesstot(df1))

df2 <- df1[,!(colnames(df1) %in% dep_var2)]
errors_2 = unlist(execute_lasso_assesstot(df2))

df3 <- df1[,!(colnames(df1) %in% dep_var3)]
errors_3 = unlist(execute_lasso_assesstot(df3))

df4 <- df[, !(colnames(df) %in% c(dep_var1, "assesstot"))]
errors_4 = unlist(execute_lasso_assessland(df4))

df5 <- df4[, !(colnames(df4) %in% dep_var2)]
errors_5 = unlist(execute_lasso_assessland(df5))

df6 <- df4[, !(colnames(df4) %in% dep_var3)]
errors_6 = unlist(execute_lasso_assessland(df6))


#lower is better
plot(c(errors_1[2], errors_2[2], errors_3[2]), main = "Compare RMSE", 
     xlab = "Model", ylab="Assesstot RMSE", xlim=range(1:3))

plot(c(errors_4[2], errors_5[2], errors_6[2]), main = "Compare RMSE", 
     xlab = "Model", ylab="AssessLand RMSE", xlim=range(1:3))


#higher is better
plot(c(errors_1[3], errors_2[3], errors_3[3]), main = "Compare RMSE", 
     xlab = "Model", ylab="Assesstot R-squared", xlim=range(1:3))

plot(c(errors_4[3], errors_5[3], errors_6[3]), main = "Compare RMSE", 
     xlab = "Model", ylab="AssessLand R-squared", xlim=range(1:3))

