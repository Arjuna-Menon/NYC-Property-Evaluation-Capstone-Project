source("libraries.R")
df <- read.csv("pluto3.csv")
col_var <- c("block","lot","cd","schooldist","council","zipcode","firecomp","policeprct",
             "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
             "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
             "healthcenterdistrict", "pfirm15_flag")
#keep numeric vars
df <- select(df, -c(col_var, "xcoord", "ycoord"))

library(car)
layout(matrix(c(1:length(df)), nrow = 2, ncol = 3, byrow = TRUE))
for (i in colnames(df)[1:5]) {
  with(df, Boxplot(df[,i],main=i, cex.main=2))
}
layout(matrix(c(1:length(df)), nrow = 2, ncol = 3, byrow = TRUE))
for (i in colnames(df)[6:10]) {
  with(df, Boxplot(df[,i],main=i, cex.main=2))
}
layout(matrix(c(1:length(df)), nrow = 2, ncol = 3, byrow = TRUE))
for (i in colnames(df)[11:15]) {
  with(df, Boxplot(df[,i],main=i, cex.main=2))
}
layout(matrix(c(1:length(df)), nrow = 2, ncol = 3, byrow = TRUE))
for (i in colnames(df)[16:20]) {
  with(df, Boxplot(df[,i],main=i, cex.main=2))
}


