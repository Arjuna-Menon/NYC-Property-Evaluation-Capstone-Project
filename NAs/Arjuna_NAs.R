#df=read.csv("pluto_18v2_1.csv")
t(t(colnames(df)))

sum(df$lotarea==0, na.rm = TRUE)
summary(df$lotarea)
# Changing zeroes to NAs:
df$lotarea[df$lotarea==0] <- NA
# Replace NAs with median according to borough:
df$lotarea[is.na(df$lotarea)] <- fill_NAs_median(df, "lotarea")
summary(df$lotarea)

sum(df$bldgarea==0, na.rm = TRUE)
summary(df$bldgarea)
# Removing NAs by changing NAs to 5s and then filtering these out:
df$bldgarea[is.na(df$bldgarea)] <- 5
df <- filter(df, bldgarea !=5)
# Bldgarea=0 considered for predicting assessed land value and not for assessed total value.

sum(df$comarea==0, na.rm = TRUE)
summary(df$comarea)
# Removing this column since it has 49282 NAs and 700296 zeroes:
df <- subset(df, select = -c(comarea))

sum(df$resarea==0, na.rm = TRUE)
summary(df$resarea)
# Changing zeroes to NAs:
df$resarea[df$resarea==0] <- NA
# Replace NAs with median according to borough:
df$resarea[is.na(df$resarea)] <- fill_NAs_median(df, "resarea")
summary(df$resarea)

sum(df$officearea==0, na.rm = TRUE)
summary(df$officearea)
# Removing this column since it has 49282 NAs and 783721 zeroes:
df <- subset(df, select = -c(officearea))

sum(df$retailarea==0, na.rm = TRUE)
summary(df$retailarea)
# Removing this column since it has 49282 NAs and 745806 zeroes:
df <- subset(df, select = -c(retailarea))

sum(df$garagearea==0, na.rm = TRUE)
summary(df$garagearea)
# Removing this column since it has 49282 NAs and 799879 zeroes:
df <- subset(df, select = -c(garagearea))

sum(df$strgearea==0, na.rm = TRUE)
summary(df$strgearea)
# Removing this column since it has 49282 NAs and 801983 zeroes:
df <- subset(df, select = -c(strgearea))

sum(df$factryarea==0,na.rm = TRUE)
summary(df$factryarea)
# Removing this column since it has 49282 NAs and 802591 zeroes:
df <- subset(df, select = -c(factryarea))

sum(df$otherarea==0,na.rm = TRUE)
summary(df$otherarea)
# Removing this coulumn since it has 49282 NAs and 790871 zeroes:
df <- subset(df, select = -c(otherarea))

# Removing areasource since it provides details about the source from which building
# area measurements is obtained, and this is unecessary for our prediction:
df <- subset(df, select = -c(areasource))

summary(df$numbldgs)
# Replacing NAs with medians for each corresponding borough:
df$numbldgs[is.na(df$numbldgs)] <- fill_NAs_median(df, "numbldgs")
summary(df$numbldgs)

summary(df$numfloors)
# Replacing NAs with medians for each corresponding borough:
df$numfloors[is.na(df$numfloors)] <- fill_NAs_median(df, "numfloors")
summary(df$numfloors)

summary(df$unitsres)
# Replacing NAs with medians for each corresponding borough:
df$unitsres[is.na(df$unitsres)] <- fill_NAs_median(df, "unitsres")
summary(df$unitsres)

summary(df$unitstotal)
# Replacing NAs with medians for each corresponding borough:
df$unitstotal[is.na(df$unitstotal)] <- fill_NAs_median(df, "unitstotal")
summary(df$unitstotal)

summary(df$lotfront)
# Replacing NAs with medians:
df$lotfront[is.na(df$lotfront)] <- fill_NAs_median(df, "lotfront")
summary(df$lotfront)

summary(df$lotdepth)
# Replacing NAs with median for each corresponding borough:
df$lotdepth[is.na(df$lotdepth)] <- fill_NAs_median(df, "lotdepth")
summary(df$lotdepth)

summary(df$bldgfront)
# Replacing NAs with medians for each corresponding borough:
df$bldgfront[is.na(df$bldgfront)] <- fill_NAs_median(df, "bldgfront")
summary(df$bldgfront)

summary(df$bldgdepth)
# Replacing NAs with medians for each corresponding borough:
df$bldgdepth[is.na(df$bldgdepth)] <- fill_NAs_median(df, "bldgdepth")
summary(df$bldgdepth)

summary(df$ext)
# Convert to 2 levels: 1 for extension for E, EG, and G, and 0 for no extension (blank):
levels(df$ext)[levels(df$ext)!=""] <- 1
levels(df$ext)[levels(df$ext)==""] <- 0
summary(df$ext)

summary(df$proxcode)
class(df$proxcode)
# Convert 3s to 2s to have only 1 level for attached instead of a sepearte level for attached and semi-attached.
# I.E. Semi-attached comes under attached. Level 2: Attached
df$proxcode[df$proxcode==3] <- 2
# Converting NAs to zeroes since 0 is not available:
df$proxcode[is.na(df$proxcode)] <- 0
# Converting to factor since 0 to 3 has different meanings (PDF):
df$proxcode <- as.factor(df$proxcode)
summary(df$proxcode)

summary(df["irrlotcode"])
levels(df$irrlotcode)
# Only 475 blanks and since the vast majority is N (723544 N and only 134963 Y), blanks replaced with N:
df$irrlotcode[df$irrlotcode==""] <- "N"
df$irrlotcode <- droplevels(df$irrlotcode)
summary(df$irrlotcode)

class(df$lottype)
summary(df$lottype)
# 475 NAs converted to 0s since 0 is unknown. After this, convert to factor since 0 to 9 have different meanings (given in PDF):
df$lottype[is.na(df$lottype)] <- 0
df$lottype <- as.factor(df$lottype)
summary(df$lottype)

summary(df$bsmtcode)
# Deleted since it is too detailed: 
df <- subset(df, select = -c(bsmtcode))
