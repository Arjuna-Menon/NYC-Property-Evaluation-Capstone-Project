## Install dependencies
. /home/mikleal/conda/etc/profile.d/conda.sh
conda activate R
R
install.packages("readr")
install.packages("readxl")
install.packages("ggplot2")
install.packages("triangle")
install.packages("randomForest")


install.packages("tidyverse", depdendencies=TRUE)
install.packages("caret", depdendencies=TRUE)
install.packages("glmnet", repos = "http://cran.us.r-project.org")
install.packages("parallel", depdendencies=TRUE)
install.packages("doMC", depdendencies=TRUE)

## Check dependencies
library(dplyr)
library(readr)
library(readxl)
library(ggplot2)
library(triangle)
library(randomForest)

library(tidyverse)
library(caret)
library(glmnet)
library(parallel)
library(doMC)

glmnet, by default, standardizes the predictor variables before fitting the model.
https://think-lab.github.io/d/205/#5

