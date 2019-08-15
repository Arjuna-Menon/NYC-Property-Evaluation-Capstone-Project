source("libraries.R")
df <- read.csv("pluto3.csv")
col_var <- c("block","lot","cd","schooldist","council","zipcode","firecomp","policeprct",
             "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
             "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
             "healthcenterdistrict", "pfirm15_flag")
#keep numeric vars
df <- select(df, -c(col_var, "xcoord", "ycoord"))

source("http://www.sthda.com/upload/rquery_cormat.r")
rquery.cormat(df)


df1 <- select(df,-c("assessland"))
df1 <- as.data.frame(scale(df1))
m1 <- lm(assesstot ~., data=df1)
summary(m1)
corr <- cor(df1)
corr[,"assesstot"]
# highest coef for lotarea, bldgarea, numfloors and unitsres
df2 <- select(df,c("lotarea", "lotfront", "lotdepth","income"))
df2 <- as.data.frame(scale(df2))
m2 <- lm(assessland ~., data=df2)
summary(m2)
corr <- cor(df2)
corr[,"assessland"]
# highest coef for lotarea, bldgarea, 

