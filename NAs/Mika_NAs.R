source("NAs/utils.R")
colnames(df)
########
#borough
summary(df$borough)
table(is.na(df$borough))
#delete levels that are large
df$borough <- as.factor(df$borough)
l <- levels(df$borough)
#check amount of rows that have rare borough 
dim(df[df$borough==l[1],])
dim(df[df$borough==l[2],])
dim(df[df$borough==l[3],])
dim(df[df$borough==l[4],])
dim(df[df$borough==l[5],])
dim(df[df$borough==l[6],])
#remove rows within first 6 levels for borough(considered as rare)
dim(df)
df <- df[!(df$borough==l[1]), ]
df <- df[!(df$borough==l[2]), ]
df <- df[!(df$borough==l[3]), ]
df <- df[!(df$borough==l[4]), ]
df <- df[!(df$borough==l[5]), ]
df <- df[!(df$borough==l[6]), ]
dim(df)
df <- droplevels(df)

########
#block
table(is.na(df$block))
########
#lot
table(is.na(df$lot))
########
#cd
table(is.na(df$cd))
df$cd[is.na(df$cd)] <- as.factor(fill_NAs_by_borough(df,"cd"))
df$cd <- as.factor(substring(as.character(df$cd),2))

########
#schooldist
table(is.na(df$schooldist))
df$schooldist[is.na(df$schooldist)] <- as.factor(fill_NAs_by_borough(df,"schooldist"))
########
#council
table(is.na(df$council))
df$council[is.na(df$council)] <- as.factor(fill_NAs_by_borough(df,"council"))
########
#zipcode : TODO
table(is.na(df$zipcode))
########
#firecomp
table(is.na(df$firecomp))
########
#policeprct
table(is.na(df$policeprct))
df$policeprct[is.na(df$policeprct)] <- as.factor(fill_NAs_by_borough(df,"policeprct"))
########
#healtharea
table(is.na(df$healtharea))
df$healtharea[is.na(df$healtharea)] <- as.factor(fill_NAs_by_borough(df,"healtharea"))
########
#healtharea
table(is.na(df$sanitboro))
df$sanitboro[is.na(df$sanitboro)] <- as.factor(fill_NAs_by_borough(df,"sanitboro"))
########
#sanitsub
table(is.na(df$sanitsub))
df$sanitsub <- as.factor(df$sanitsub)
#zonedist1
table(is.na(df$zonedist1))
#delete columns
df <- df[, !(colnames(df) %in% c('address','zonedist2','zonedist3','zonedist4', 'overlay1', 'overlay2', 'spdist2', 'spdist3', 'splitzone', 'bldgclass', 'easements', 'ownertype', 'ownername', 'ct2010', 'cb2010'))]
#spdist1
table(df$spdist1=="")
df$spdist1 <- as.factor(ifelse(df$spdist1=="", 0, 1))
#ltdheight
table(df$ltdheight=="")
df$ltdheight <- as.factor(ifelse(df$ltdheight=="", 0, 1))
#landuse
summary(as.factor(df$landuse))
df$landuse[is.na(df$landuse)] <- "0" #add Unkown category 
df$landuse <- as.factor(df$landuse)

summary(as.factor(df$easements))
summary(as.factor(df$ownertype))
table(is.na(df$lotarea))

colnames(df)[19]

