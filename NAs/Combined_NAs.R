df=read.csv("pluto_18v2_1.csv")
source("NAs/utils.R")
colnames(df)
str(df)
#######################################################################################################################################
# borough
class(df$borough)
summary(df$borough)
table(is.na(df$borough))
# Delete levels that are extremely large (errors):
df$borough <- as.factor(df$borough)
l <- levels(df$borough)
# Check amount of rows that have these rare (erronous values) boroughs:
dim(df[df$borough==l[1],])
dim(df[df$borough==l[2],])
dim(df[df$borough==l[3],])
dim(df[df$borough==l[4],])
dim(df[df$borough==l[5],])
dim(df[df$borough==l[6],])
# Remove rows within first 6 levels for borough (considered as rare):
dim(df)
df <- df[!(df$borough==l[1]), ]
df <- df[!(df$borough==l[2]), ]
df <- df[!(df$borough==l[3]), ]
df <- df[!(df$borough==l[4]), ]
df <- df[!(df$borough==l[5]), ]
df <- df[!(df$borough==l[6]), ]
dim(df)
df$borough <- droplevels(df$borough)
levels(df$borough) # 5 levels
#######################################################################################################################################

# zipcode
class(df$zipcode)
table(is.na(df$zipcode))
# Rows containing zipcode as NA (20535 NAs) transfered into Python and repaired there.
# Saving these rows in the CSV file Zips_code in the folder zipcode_fill:
# subset(df, is.na(zipcode)) %>% write.csv(., "zipcode_fill/Zips_code.csv")

# Deleting rows with NAs in zipcode from the data fram df:
df <- filter(df, !is.na(zipcode))

# Read the repaired CSV file which was created through the Python program:
fixed_zips <- read.csv('zipcode_fill/zips_fixed.csv')

# Row bind the old data frame (df) with the corrected zipcodes obtained through Python:
df <- rbind(df, fixed_zips)
dim(df)
summary(df$zipcode)

# Converting to factor:
df$zipcode <- as.factor(df$zipcode)
levels(df$zipcode) # 216 levels
#######################################################################################################################################

# borocode
class(df$borocode)
summary(df$borocode) # No NAs
# Convert to factor:
df$borocode <- as.factor(df$borocode)
levels(df$borocode) # 5 levels
#######################################################################################################################################

# block
class(df$block)
table(is.na(df$block)) # No NAs
# Converting to factor:
df$block <- as.factor(df$block)
levels(df$block) # 13968 levels
#######################################################################################################################################

# lot
class(df$lot)
table(is.na(df$lot)) # No NAs
# Converting to factor:
df$lot <- as.factor(df$lot)
levels(df$lot) # 2521 levels
#######################################################################################################################################

# cd
class(df$cd)
table(is.na(df$cd))
# Replace 11 NAs with random cd values of the corresponding zipcodes:
df$cd[is.na(df$cd)] <- fill_NAs_by_zipcode(df,"cd")
# Converting to factor:
df$cd <- as.factor(df$cd)
levels(df$cd) # 79 levels
# Removing rows that have cd values of 8, 32, and, 78 which are erronoeous:
df <- filter(df, !(cd == 8 | cd == 32 | cd == 78))
df$cd <- droplevels(df$cd)
levels(df$cd) # 76 levels
#######################################################################################################################################

# ct2010 and cb2010 are deleted since we only use zipcode to match to census data.
# schooldist
class(df$schooldist)
table(is.na(df$schooldist))
# Replace 78 NAs with random schooldist values of the corresponding zipcodes and converting to factor:
df$schooldist[is.na(df$schooldist)] <- fill_NAs_by_zipcode(df,"schooldist")
df$schooldist <- as.factor(df$schooldist)
levels(df$schooldist) # 32 levels
#######################################################################################################################################

# council
class(df$council)
table(is.na(df$council))
# Replace 11 NAs with random council values of the corresponding zipcodes and converting to factor:
df$council[is.na(df$council)] <- (fill_NAs_by_zipcode(df,"council"))
df$council <- as.factor(df$council)
levels(df$council) # 51 levels
#######################################################################################################################################

# firecomp
class(df$firecomp)
table(is.na(df$firecomp)) # No NAs
table(df$firecomp=="")
# 102 blanks replaced with NAs:
df$firecomp[df$firecomp==""] <- NA
# Replace 102 NAs with random firecomp values of the corresponding zipcode:
df$firecomp[is.na(df$firecomp)] <- (fill_NAs_by_zipcode(df,"firecomp"))
df$firecomp <- droplevels(df$firecomp)
levels(df$firecomp) # 348 levels
#######################################################################################################################################

# policeprct
class(df$policeprct)
table(is.na(df$policeprct))
# Replace 83 NAs with random policeprct values of the corresponding zipcodes and converting to factor:
df$policeprct[is.na(df$policeprct)] <- fill_NAs_by_zipcode(df,"policeprct")
df$policeprct <- as.factor(df$policeprct)
levels(df$policeprct) # 82 levels
#######################################################################################################################################

# healtharea
class(df$healtharea)
table(is.na(df$healtharea))
# Replace 82 NAs with random healtharea values of the corresponding zipcodes and converting to factor:
df$healtharea[is.na(df$healtharea)] <- fill_NAs_by_zipcode(df,"healtharea")
df$healtharea <- as.factor(df$healtharea)
levels(df$healtharea) # 228 levels
#######################################################################################################################################

# sanitboro
class(df$sanitboro)
table(is.na(df$sanitboro))
# Replace 259 NAs with random sanitboro values of the corresponding zipcodes:
df$sanitboro[is.na(df$sanitboro)] <- fill_NAs_by_zipcode(df,"sanitboro")
# Replace the 1 NA with a random sanitboro value of the corresponding borough:
df$sanitboro[is.na(df$sanitboro)] <- fill_NAs_by_borough(df,"sanitboro")
# Converting to factor:
df$sanitboro <- as.factor(df$sanitboro)
levels(df$sanitboro) # 5 levels
#######################################################################################################################################

# sanitsub
class(df$sanitsub)
table(is.na(df$sanitsub)) # No NAs
table(df$sanitsub == " ")
# Convert 404 blanks to NAs:
df$sanitsub[df$sanitsub==" "] <- NA
# Replace NAs with random sanitsub values of the corresponding zipcodes:
df$sanitsub[is.na(df$sanitsub)] <- fill_NAs_by_zipcode(df,"sanitsub")
# Replace the 25 NAs with random sanitsub values of the corresponding boroughs:
df$sanitsub[is.na(df$sanitsub)] <- fill_NAs_by_borough(df,"sanitsub")
# The new levels:
df$sanitsub <- droplevels(df$sanitsub)
levels(df$sanitsub) # 62 levels
#######################################################################################################################################

# address is deleted since it is textual and not required for the predictions.
# zonedist1
class(df$zonedist1)
table(is.na(df$zonedist1)) # No NAs
table(df$zonedist1=="")
# Replacing 967 blanks with NAs:
df$zonedist1[df$zonedist1==""] <- NA
# Replacing NAs with random zonedist1 values for the corresponding zipcodes:
df$zonedist1[is.na(df$zonedist1)] <- fill_NAs_by_zipcode(df,"zonedist1")
df$zonedist1 <- droplevels(df$zonedist1)
# Converting to R for residential, C for commercial, M for manufacturing, MR for mixed-manufacturing and residential, 
# BPC for Battery Park City:
df$zonedist1 <- as.factor(ifelse(grepl("/",as.character(df$zonedist1)),"MR",
                                       ifelse(grepl("M",as.character(df$zonedist1)),"M",
                                              ifelse(grepl("C",as.character(df$zonedist1)),"C",
                                                     ifelse(grepl("R",as.character(df$zonedist1)),"R",
                                                            "BPA")))))
levels(df$zonedist1) # 5 levels
#######################################################################################################################################

# zonedist2, zonedist3, zonedist4, overlay1, and overlay2 are deleted since they provide too 
# much unnecessary detail and contain a vast majority of NAs or blanks.
# spdist1
class(df$spdist1)
table(df$spdist1=="")
# Replacing 756231 blanks with 0s and non-blank values with 1 and converting to factor:
df$spdist1 <- as.factor(ifelse(df$spdist1=="", 0, 1))
levels(df$spdist1)
#######################################################################################################################################

# spdist2 and spdist3 are deleted since they have a vast majority of blanks.
# ltdheight
class(df$ltdheight)
table(df$ltdheight=="")
# Replacing 854317 blanks with 0s and non-blank values with 1 and converting to factor:
df$ltdheight <- as.factor(ifelse(df$ltdheight=="", 0, 1))
levels(df$ltdheight)
#######################################################################################################################################

# splitzone deleted since assumed buildings are only in 1 zone, and actually very few buildings are in multiple zones.
# bldgclass deleted since this information is present in the landuse variable.
# landuse
class(df$landuse)
summary(as.factor(df$landuse))
# Replace 2359 NAs with 0s for unknown category:
df$landuse[is.na(df$landuse)] <- "0"
# Convert to factor:
df$landuse <- as.factor(df$landuse)
levels(df$landuse) # 12 levels
#######################################################################################################################################

# easements:
class(df$easements)
summary(as.factor(df$easements))
# Deleted since 853597 zeroes.
#######################################################################################################################################

# ownertype:
class(df$ownertype)
summary(df$ownertype)
# Deleted since 823448 blanks.
#######################################################################################################################################

# Delete the above chosen columns to be deleted:
df <- df[, !(colnames(df) %in% c('address','zonedist2','zonedist3','zonedist4', 'overlay1', 'overlay2', 'spdist2', 'spdist3', 'splitzone', 'bldgclass', 'easements', 'ownertype', 'ownername', 'ct2010', 'cb2010'))]
#######################################################################################################################################

# lotarea
class(df$lotarea)
table(df$lotarea==0)
summary(df$lotarea)
# Changing 3153 zeroes to NAs: (Also we have 144 NAs)
df$lotarea[df$lotarea==0] <- NA
# Replace NAs with median according to zipcodes:
df$lotarea[is.na(df$lotarea)] <- fill_NAs_median(df, "lotarea")
# Replace the 1 NAs with a random lotarea value of the corresponding borough:
df$lotarea[is.na(df$lotarea)] <- fill_NAs_median_borough(df,"lotarea")
df <- filter(df, !is.na(lotarea))
#######################################################################################################################################

# bldgarea
class(df$bldgarea)
table(df$bldgarea==0) # Bldgarea=0 considered for predicting assessed land value and not for assessed total value
summary(df$bldgarea)
# Removing NAs by changing NAs to 5s and then filtering these out:
df$bldgarea[is.na(df$bldgarea)] <- 5
df <- filter(df, bldgarea !=5)
#######################################################################################################################################

# comarea
class(df$comarea)
table(df$comarea==0)
summary(df$comarea)
# Removing this column since it has a vast majority of 0s and NAs (700290 0s and 47715 NAs):
df <- subset(df, select = -c(comarea))
#######################################################################################################################################

# resarea
class(df$resarea)
table(df$resarea==0)
summary(df$resarea)
# Since a total of nearly 100,000 rows contain 0 or NA values and as these cannot be replaced with the medians for 
# each zipcode without biasing the prediction model, this variable is removed:
df <- subset(df, select = -c(resarea))
#######################################################################################################################################

# officearea
class(df$officearea)
table(df$officearea==0)
summary(df$officearea)
# Removing this column since it has a vast majority of 0s and NAs (783666 0s and 47715 NAs):
df <- subset(df, select = -c(officearea))
#######################################################################################################################################

# retailarea
class(df$retailarea)
table(df$retailarea==0)
summary(df$retailarea)
# Removing this column since it has a vast majority of 0s and NAs (745749 0s and 47715 NAs):
df <- subset(df, select = -c(retailarea))
#######################################################################################################################################

# garagearea
class(df$garagearea)
table(df$garagearea==0)
summary(df$garagearea)
# Removing this column since it has a vast majority of 0s and NAs (799824 0s and 47715 NAs):
df <- subset(df, select = -c(garagearea))
#######################################################################################################################################

# strgearea
class(df$strgearea)
table(df$strgearea==0)
summary(df$strgearea)
# Removing this column since it has a vast majority of 0s and NAs (801926 0s and 47715 NAs):
df <- subset(df, select = -c(strgearea))
#######################################################################################################################################

# factryarea
class(df$factryarea)
table(df$factryarea==0)
summary(df$factryarea)
# Removing this column since it has a vast majority of 0s and NAs (802537 0s and 47715 NAs):
df <- subset(df, select = -c(factryarea))
#######################################################################################################################################

# otherarea
table(df$otherarea==0)
summary(df$otherarea)
# Removing this coulumn since it has a vast majority of 0s and NAs (790856 0s and 47715 NAs):
df <- subset(df, select = -c(otherarea))
#######################################################################################################################################

# areasource
# Removing areasource since it provides details about the source from which building area measurements is obtained,
# and this is unecessary for our prediction:
df <- subset(df, select = -c(areasource))
#######################################################################################################################################

# numbldgs
class(df$numbldgs)
summary(df$numbldgs)
# Replacing 144 NAs with medians for each corresponding zipcode:
df$numbldgs[is.na(df$numbldgs)] <- fill_NAs_median(df, "numbldgs")
#######################################################################################################################################

# numfloors
class(df$numfloors)
summary(df$numfloors)
# Replacing 144 NAs with medians for each corresponding zipcode:
df$numfloors[is.na(df$numfloors)] <- fill_NAs_median(df, "numfloors")
#######################################################################################################################################

# unitsres
class(df$unitsres)
summary(df$unitsres)
# Replacing NAs with medians for each corresponding zipcode:
df$unitsres[is.na(df$unitsres)] <- fill_NAs_median(df, "unitsres")
df$unitsres <- unlist(df$unitsres)
#######################################################################################################################################

# unitstotal
class(df$unitstotal)
summary(df$unitstotal) # No NAs
#######################################################################################################################################

# lotfront
class(df$lotfront)
summary(df$lotfront)
# Replacing 144 NAs with medians for each corresponding zipcode:
df$lotfront[is.na(df$lotfront)] <- fill_NAs_median(df, "lotfront")
#######################################################################################################################################

# lotdepth
class(df$lotdepth)
summary(df$lotdepth)
# Replacing 1928 NAs with median for each corresponding zipcode:
df$lotdepth[is.na(df$lotdepth)] <- fill_NAs_median(df, "lotdepth")
#######################################################################################################################################

# bldgfront
class(df$bldgfront)
summary(df$bldgfront)
# Replacing 144 NAs with medians for each corresponding zipcode:
df$bldgfront[is.na(df$bldgfront)] <- fill_NAs_median(df, "bldgfront")
#######################################################################################################################################

# bldgdepth
class(df$bldgdepth)
summary(df$bldgdepth)
# Replacing 144 NAs with medians for each corresponding zipcode:
df$bldgdepth[is.na(df$bldgdepth)] <- fill_NAs_median(df, "bldgdepth")
#######################################################################################################################################

# ext
class(df$ext)
summary(df$ext)
# Convert to 2 levels: 1 for extension for E, EG, and G, and 0 for no extension (blank):
levels(df$ext)[levels(df$ext)!=""] <- 1
levels(df$ext)[levels(df$ext)==""] <- 0
levels(df$ext)
#######################################################################################################################################

# proxcode
class(df$proxcode)
summary(df$proxcode)
# Convert 3s to 2s to have only 1 indicator for attached instead of sepearte indicators for attached and semi-attached.
# (I.E. Semi-attached comes under attached, level 1 for detached and level 2 for attached.)
df$proxcode[df$proxcode==3] <- 2
# Converting NAs to zeroes since 0 is not available:
df$proxcode[is.na(df$proxcode)] <- 0
# Convert to factor:
df$proxcode <- as.factor(df$proxcode)
levels(df$proxcode)
#######################################################################################################################################

# irrlotcode
class(df$irrlotcode)
summary(df["irrlotcode"])
levels(df$irrlotcode)
# Only few blanks and since the vast majority is N, blanks replaced with N:
df$irrlotcode[df$irrlotcode==""] <- "N"
df$irrlotcode <- droplevels(df$irrlotcode)
levels(df$irrlotcode)
#######################################################################################################################################

# lottype
class(df$lottype)
summary(df$lottype)
# NAs converted to 0s since 0 is unknown.
df$lottype[is.na(df$lottype)] <- 0
# Convert to factor:
df$lottype <- as.factor(df$lottype)
levels(df$lottype)
#######################################################################################################################################

# bsmtcode
# Deleted since it provides unnecessary details:
df <- subset(df, select = -c(bsmtcode))
#######################################################################################################################################

# assessland
class(df$assessland)
summary(df$assessland)
table(df$assessland == 0)
boxplot(df$assessland)
hist(df$assessland)
# Replace NAs by zero:
df$assessland[is.na(df$assessland)] <- 0
# Delete all values equal to zero:
df <- filter(df, assessland !=0)
#######################################################################################################################################

# assesstot
class(df$assesstot)
summary(df$assesstot)
str(df$assesstot)
boxplot(df$assesstot)
hist(df$assesstot)
#######################################################################################################################################

# exemptland
table(is.na(df$exemptland))
# Deleted since tax exemption related to factors other than just the location:
df <- subset(df, select = -c(exemptland))
#######################################################################################################################################

# exempttot
table(is.na(df$exempttot))
# Deleted since tax exemption related to factors other than the location alone (such as the business present there):
df <- subset(df, select = -c(exempttot))
#######################################################################################################################################

# yearbuilt
class(df$yearbuilt)
# Plot the data (we have some zeros):
library(ggplot2)
ggplot(df, aes(x=yearbuilt)) + geom_histogram() +
  labs(title="Year built",x="Year", y = "Count")

table(is.na(df$yearbuilt)) # No NAs
table(df$yearbuilt==0) # 39614 0s

#Mode function:
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Replace zeroes with mode when the building is built:
df$yearbuilt <- ifelse(df$yearbuilt == 0 & df$bldgarea != 0, getmode(df$yearbuilt), df$yearbuilt)

levels(as.factor(df$yearbuilt))
# Delete the row containing yearbuilt as 2040:
df <- filter(df, yearbuilt !=2040)
levels(as.factor(df$yearbuilt)) # 219 levels
#######################################################################################################################################

# yearalter1 & yearalter2
# If yearalter2 is empty than take yearalter1, otherwise keep 0, which means that the bulding was not 
# modified at all:
table(df$yearalter1==0)
table(df$yearalter2==0)
class(df$yearalter1)
df$yearalter3 <- ifelse(df$yearalter2 == 0 , df$yearalter1, df$yearalter2)
# Change the name of the column:
names(df)[names(df) == "yearalter3"] <- "yearalter"
# Delete previous columns yearalter1 & yearalter2:
df <- subset(df, select = -c(yearalter2))
df <- subset(df, select = -c(yearalter1))
#######################################################################################################################################

# histdist
table(df$histdist == "")
# Deleted since 823561 blanks:
df <- subset(df, select = -c(histdist))
#######################################################################################################################################

# landmark
table(df$landmark == "")
# Delted since 852457 blanks:
df <- subset(df, select = -c(landmark))
#######################################################################################################################################

# builtfar deleted since we have the values for computing this ratio in other variables
df <- subset(df, select = -c(builtfar))
#######################################################################################################################################

# residfar
class(df$residfar)
summary(df$residfar) # No NAs
#######################################################################################################################################

# commfar
class(df$commfar)
summary(df$commfar) # No NAs
#######################################################################################################################################

# facilfar
class(df$facilfar)
summary(df$facilfar) # No NAs
#######################################################################################################################################

# bbl deleted since it is a combination of borough, block, and lot information:
df <- subset(df, select = -c(bbl))
#######################################################################################################################################

# condono deleted since it is unnecessary for predictions and contains mostly zeroes:
table(is.na(df$condono))
table(df$condono==0)
df <- subset(df, select = -c(condono))
#######################################################################################################################################

# tract2010 deleted since census data merged with zip code:
df <- subset(df, select = -c(tract2010))
#######################################################################################################################################

# xcoord and ycoord kept for visualization but not predictions.
table(is.na(df$xcoord))
table(is.na(df$ycoord))
# zonemap deleted since it is not required for visualization:
df <- subset(df, select = -c(zmcode))
#######################################################################################################################################

# zmcode deleted since assumed that buildings are in only 1 zone:
df <- subset(df, select = -c(zonemap))
#######################################################################################################################################

# sanborn deleted since this information is present in tax block and lot:
df <- subset(df, select = -c(sanborn))
#######################################################################################################################################

# taxmap deleted since it refers to a volume number which is unnecessay information:
df <- subset(df, select = -c(taxmap))
#######################################################################################################################################

# edesignum
# E codes mean that there is some hazardous material affecting this property. These are converted to 1 here.
class(df$edesignum)
levels(df$edesignum) # 105 levels.
table(is.na(df$edesignum))
df$edesignum <- as.character(df$edesignum) # Convert to character
df$edesignum[df$edesignum == ""] <- "0" # If empty, then replace with 0
df$edesignum[df$edesignum != "0"] <- "1" # If not zero, then replace with 1
df$edesignum <- as.factor(df$edesignum) # Save as factor
levels(df$edesignum)
#######################################################################################################################################

# appbbl deleted since it contains information already present in borough, block, and lot:
df <- subset(df, select = -c(appbbl))
#######################################################################################################################################

# appdate
str(df$appdate)
# Deleted since it has 5102 levels and this is extra information not needed for the predictions:
df <- subset(df, select = -c(appdate))
#######################################################################################################################################

# mappluto_f
summary(df$mappluto_f)
# Deleted since it contains 853863 NAs:
df <- subset(df, select = -c(mappluto_f))
#######################################################################################################################################

# plutomapid deleted since zipcode used instead for merging with census data:
table(is.na(df$plutomapid))
df <- subset(df, select = -c(plutomapid))
#######################################################################################################################################

# version
summary(df$version)
# Deleted since it has only 1 level, which is version 18v2.1:
df <- subset(df, select = -c(version))
#######################################################################################################################################

# sanitdistrict
class(df$sanitdistrict)
table(is.na(df$sanitdistrict))
# Replace 221 NAs with random sanitdistrict values of the corresponding zipcodes:
df$sanitdistrict[is.na(df$sanitdistrict)] <- fill_NAs_by_zipcode(df,"sanitdistrict")
# Replace the 1 NA with a random sanitdistrict value of the corresponding borough:
df$sanitdistrict[is.na(df$sanitdistrict)] <- fill_NAs_by_borough(df,"sanitdistrict")
# Converting to factor:
df$sanitdistrict <- as.factor(df$sanitdistrict)
levels(df$sanitdistrict) # 27 levels
#######################################################################################################################################

# healthcenterdistrict
class(df$healthcenterdistrict)
table(is.na(df$healthcenterdistrict))
# Replace 74 NAs with random healthcenterdistrict values of the corresponding zipcodes and converting to factor:
df$healthcenterdistrict[is.na(df$healthcenterdistrict)] <- fill_NAs_by_zipcode(df,"healthcenterdistrict")
df$healthcenterdistrict <- as.factor(df$healthcenterdistrict)
levels(df$healthcenterdistrict) # 37 levels
#######################################################################################################################################

# firm07_flag deleted since more recent 2015 data present:
df <- subset(df, select = -c(firm07_flag))
#######################################################################################################################################

# pfirm15_flag indicates if a tax lot is vulnerable to flooding.
class(df$pfirm15_flag)
table(is.na(df$pfirm15_flag))
# 789461 NAs and these are replaced with 0s:
df$pfirm15_flag[is.na(df$pfirm15_flag)] <- 0
# Convert to factor:
df$pfirm15_flag <- as.factor(df$pfirm15_flag)
levels(df$pfirm15_flag)
#######################################################################################################################################

# rpaddate
summary(df$rpaddate)
# Deleted since only 1 date present here:
df <- subset(df, select = -c(rpaddate))
#######################################################################################################################################

# dcasdate
summary(df$dcasdate)
# Deleted since only 1 date present here:
df <- subset(df, select = -c(dcasdate))
#######################################################################################################################################

# zoningdate
summary(df$zoningdate)
# Deleted since only 1 date present here:
df <- subset(df, select = -c(zoningdate))
#######################################################################################################################################

# landmkdate
summary(df$landmkdate)
# Deleted since only 1 date present here:
df <- subset(df, select = -c(landmkdate))
#######################################################################################################################################

# basempdate
summary(df$basempdate)
# Deleted since only 1 date present here:
df <- subset(df, select = -c(basempdate))
#######################################################################################################################################

# masdate
summary(df$masdate)
# Deleted since all data is NA:
df <- subset(df, select = -c(masdate))
#######################################################################################################################################

# polidate
summary(df$polidate)
# Deleted since all data is NA:
df <- subset(df, select = -c(polidate))
#######################################################################################################################################

# edesigdate
table(df$edesigdate)
# Deleted since only 1 date present here:
df <- subset(df, select = -c(edesigdate))
#######################################################################################################################################

# Deleting borough since we have borocode. It was used at the beginning to remove erronous values:
df <- subset(df, select = -c(borough))
#######################################################################################################################################

# From Tableau:
table(df$zipcode==11430) # Outliers as seen from Tableau. JFK airport that has too high of an 
# assessed land and assessed total
table(df$zipcode == 12345) # Outlier as seen from Tableau.
# Deleting these outliers:
df <- filter(df, !(zipcode == 11430 | zipcode == 12345))
#######################################################################################################################################

# Writing the new partially cleaned CSV file:
write.csv(df, file = "pluto2.csv", row.names=FALSE)
#######################################################################################################################################
