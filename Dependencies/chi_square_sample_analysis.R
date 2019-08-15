# Chi-squared test for independence between categorical variables:
library("dplyr")
library("gdata")
source("libraries.R")

df <- read.csv("Dependencies/chisquare_0.011.csv")
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

# Displaying those points that are greater than p-value of 0.05 in blue color:
df3 <- df1
df3[df3>0.05] <- "blue" #independent
df3[df3!="blue"] <- "pink"

df3 <- as.table(t(as.matrix(df3)))

# Balloon plot:
library("gplots")
gplots::balloonplot(df3, main ="Independence Test", xlab ="", ylab="",
            label = FALSE, show.margins = FALSE, colsrt=90, dotcolor = df3,
            hide.duplicates=TRUE, text.size=0.7,dotsize=3)