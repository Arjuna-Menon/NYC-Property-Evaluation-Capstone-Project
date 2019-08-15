library("dplyr")
df <- read.csv("Dependencies/anova_sample.csv")

rownames(df) <- df$X
df <- df[,-1]
rownames(df)
# Delete assess total and assesset land from data:
df <- df[!rownames(df) %in% c("assessland", "assesstot"), ]

a <- colnames(df)

# p values:
df1 <- df
for (i in a) {
  df1[,i] <- gsub(".*,(.*)", "\\1", df1[,i])
  df1[,i] <- as.numeric(df1[,i])
}

# F-statistic values:
df2 <- df
for (i in a) {
  df2[,i] <- gsub("(.*),.*", "\\1", df2[,i])
  df2[,i] <- as.numeric(df2[,i])
}

# Those variables for which the independence hypothesis is not rejeceted:
b <- rownames(df1)
print("These pairs of variables might be independent:")
for (i in b){
  for (j in a){
    if((df1[i,j] > 0.05)){
      print(c(i,j))
    }
  }
}

d <- c()
for (i in b){
  for (j in a){
    d <- c(d,df2[i,j])
  }
}
d <- as.data.frame(d)
colnames(d) <- "value"

# Displaying those points that have p-value greater than 0.05 in blue color:
df3 <- df1
df3[df3>0.05] <- "blue" #independent
df3[df3!="blue"] <- "pink"

# Balloon plot:
df3 <- as.table(as.matrix(df3))
library("gplots")
gplots::balloonplot(df3, main ="Independence Test", xlab ="", ylab="",
                    label = FALSE, show.margins = FALSE, colsrt=90, dotcolor = df3,
                    hide.duplicates=TRUE, text.size=0.7,dotsize=3)
