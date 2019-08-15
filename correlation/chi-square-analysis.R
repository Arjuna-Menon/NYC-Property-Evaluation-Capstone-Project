# Chi-squared test for independence between categorical variables:
library("dplyr")
library("gdata")
source("libraries.R")

df <- read.csv("correlation/chisquare_0.011.csv")
row.names(df) <- colnames(df)

# p values:
df1 <- df
for (i in colnames(df1)) {
  df1[,i] <- gsub(".*,(.*)", "\\1", df1[,i])
  df1[,i] <- as.numeric(df1[,i])
}

# Chi-squared values:
df2 <- df
for (i in colnames(df2)) {
  df2[,i] <- gsub("(.*),.*", "\\1", df2[,i])
  df2[,i] <- as.numeric(df2[,i])
}

# Those variables for which the independence hypothesis is not rejeceted:
a <- colnames(df1)
j <- 0
print("These pairs of variables might be independent:")
for (i in 1:dim(df1)[1]){
  for (j in 1:dim(df1)[2]){
    if((j >= i) & (df1[a[i],a[j]] > 0.05)){
      print(c(a[i],a[j]))
    }
  }
}
# No independence.

# Upper triangle values of chi-squared values saved in a data frame:
# d <- as.data.frame(upperTriangle(df2, diag = FALSE, byrow = TRUE))
# colnames(d) <- "value"

# Choosing a threshold chi-squared value:
# boxplot(d$value)
# abline(h=quantile(d$value,0.75),col="red",lty=2)
# m <- as.numeric(quantile(d$value, 0.75))
# table(d$value>m)

# Displaying those points that are greater than this threshold in skyblue color:
df3 <- df1
df3[df3>0.05] <- "blue" #independent
df3[df3!="blue"] <- "pink"

# White for the upper triangle so that only lower triangle is displayed:
# df3[upper.tri(df3, diag = "TRUE")] <- "white" # Removing the dependencies of a variable with itself by coloring white for the diagonal.
df3 <- as.table(t(as.matrix(df3)))

# Balloon plot:
library("gplots")
gplots::balloonplot(df3, main ="Independence Test", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt=90, dotcolor = df3,
            hide.duplicates=TRUE, text.size=0.7,dotsize=3)

# Delete cd, council, zipcode, firecomp,
# policeprct, healtharea, sanitboro, sanitsub, borocode, sanitdistrict,
# healthcenterdistrict since they depend highly on schooldist.
df <- read.csv("pluto3.csv")
# Categorical variables:
col_var <- c("block","lot","cd","schooldist","council","zipcode","firecomp","policeprct",
             "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
             "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
             "healthcenterdistrict", "pfirm15_flag")
# Keep categorical variables:
df <- select(df, col_var)
# Deleting the extremely dependent variables:
dep_var1 <- c("cd", "council", "zipcode", "firecomp", "policeprct", "healtharea",
                    "sanitboro", "sanitsub", "sanitdistrict", "healthcenterdistrict")
# borocode was also supposed to be deleted since it is dependent, but it is kept for sampling.

df <- select(df, -dep_var1, -c("block", "lot"))
df <- lapply(df, as.factor)
str(df)
