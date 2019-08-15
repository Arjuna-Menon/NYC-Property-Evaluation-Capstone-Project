library("dplyr")
df <- read.csv("correlation/anova_sample.csv")

df <- select(df, -c("lot"))
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
# The variable lotarea is indepedent with respect to spdist1, ltdheight, and edesignum.

d <- c()
for (i in b){
  for (j in a){
    d <- c(d,df2[i,j])
  }
}
d <- as.data.frame(d)
colnames(d) <- "value"

# Choosing a threshold chi-squared value based on the above graphs:
#boxplot(d$value)
#abline(h=quantile(d$value,0.75),col="red",lty=2)
#m <- as.numeric(quantile(d$value, 0.75))
#table(d$value>m)

# Displaying those points that are greater than this threshold in skyblue color:
df3 <- df1
df3[df3>0.05] <- "blue" #independent
df3[df3!="blue"] <- "pink"

# Balloon plot:
df3 <- as.table(as.matrix(df3))
library("gplots")
gplots::balloonplot(df3, main ="Independence Test", xlab ="", ylab="",
                    label = FALSE, show.margins = FALSE, colsrt=90, dotcolor = df3,
                    hide.duplicates=TRUE, text.size=0.7,dotsize=3)
# Remember that lotarea is indepedent with respect to spdist1, ltdheight, and edesignum.
rownames(df2)
# Two possibilities:
# 1. Keeping commfar numerical variable along with categorical variables ltdheight, lottype,
# and numerical variables lotarea, bldgarea, numbldgs, numfloors, unitsres, 
# unitstotal, lotfront, lotdepth, bldgfront, bldgdepth, yearbuilt, residfar,  
# yearalter, and income since the other variables depend on the numerical or categorical variables.
# Hence, delete the following:
dep_var2 <- c("facilfar", "schooldist", "zonedist1", "spdist1", "landuse", "ext", "proxcode",
              "irrlotcode", "edesignum", "pfirm15_flag")


# 2. Keeping lanudse categorical variable along with the numerical variables lotarea, bldgarea,
# numbldgs, unitstotal, lotdepth, and categorical variables schooldist, zonedist1, spdist1,
# ltdheight1, ext, proxcode, lottype, edesignum, and pfirm15_flag. Hence, delete the following:
dep_var3 <- c("irrlotcode", "numfloors", "unitsres", "lotfront", "bldgfront", "bldgdepth",
              "yearbuilt", "residfar", "commfar", "facilfar", "yearalter", "income")