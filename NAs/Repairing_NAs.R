#df <- read.csv("pluto_18v2_1.csv")
colnames(df)

# Column 'edesigdate'
# Delete the column. It has only 2 levels, check below. Delete it cuz almost all rows have the same date.
table(df$edesigdate)
df <- subset(df, select = -c(edesigdate))

# Column 'polidate'
# Delete the column. All data is NA.
summary(df$polidate)
df <- subset(df, select = -c(polidate))

# Column 'masdate'
# Delete the column. All data is NA.
summary(df$masdate)
df <- subset(df, select = -c(masdate))

# Column 'basempdate' the same explanation as # Column 'edesigdate' (column at the begining).
summary(df$basempdate)
str(df$basempdate)
df <- subset(df, select = -c(basempdate))

# Column 'landmkdate' the same explanation as # Column 'edesigdate' (column at the begining).
summary(df$landmkdate)
df <- subset(df, select = -c(landmkdate))
 
# Column 'zoningdate' the same explanation as # Column 'edesigdate' (column at the begining).
summary(df$zoningdate)
df <- subset(df, select = -c(zoningdate))

# Column 'dcasdate' the same explanation as # Column 'edesigdate' (column at the begining).
summary(df$dcasdate)
df <- subset(df, select = -c(dcasdate))

# Column 'rpaddate' the same explanation as # Column 'edesigdate' (column at the begining).
summary(df$rpaddate)
df <- subset(df, select = -c(rpaddate))

# Column 'pfirm15_flag' this column indicated if a tax lot is vulnerable to be flooded.  KEEP THIS COLUMN
# We transform NA into 0. 92% of data is 1.
summary(df$pfirm15_flag)
str(df$pfirm15_flag)
793294/858982*100 # 92%
858982-793294
# replace NA into 0
df$pfirm15_flag[is.na(df$pfirm15_flag)] <- 0
# Change type as factor
df$pfirm15_flag <- as.factor(df$pfirm15_flag)

# Column 'firm07_flag' # Decided to delete cuz we have the same data for the year 2015.
# We keep only 2015 because it is more accurate.
str(df$firm07_flag)
summary(df$firm07_flag)
df <- subset(df, select = -c(firm07_flag))

# Column 'healthcenterdistrict' KEEP THIS COLUMN
## To repare this, you have to execute Mika's code first! She repaired problem with borough lebels.
str(df$healthcenterdistrict)
summary(df$healthcenterdistrict)

#Check how many NAs are there
table(is.na(df$healthcenterdistrict))
#We find an NA value, later we wanna figure out from which district this NA comes from. If for example
#it is from BX we wanna sample healthcenterdistrict only from BX area (it is in order to be the most accurate).
health_per_borough <- unique(df[c("borough", "healthcenterdistrict")]) #keep cds for each borough
health_per_borough <- health_per_borough[!(is.na(health_per_borough$healthcenterdistrict)), ] #remove cd NAs
get_sample<-function(borough) { #get a sample from cds where the borough is received 
  return (sample(health_per_borough[health_per_borough$borough==borough,]$healthcenterdistrict,1)) 
}
df$healthcenterdistrict[is.na(df$healthcenterdistrict)] <- lapply(df$borough[is.na(df$healthcenterdistrict)], FUN=get_sample)

df$healthcenterdistrict <- unlist(df$healthcenterdistrict)
df$healthcenterdistrict <- as.factor(df$healthcenterdistrict)


# Column 'sanitdistrict' KEEP THIS COLUMN
str(df$sanitdistrict)
summary(df$sanitdistrict)

# Let's repate the same steps like `healthcenterdistrict`, while imputing NAs
table(is.na(df$sanitdistrict))
#We find an NA value, later we wanna figure out from which district this NA comes from. If for example
#it is from BX we wanna sample sanitdistrict only from BX area (it is in order to be the most accurate).
sanit_per_borough <- unique(df[c("borough", "sanitdistrict")]) #keep cds for each borough
sanit_per_borough <- sanit_per_borough[!(is.na(sanit_per_borough$sanitdistrict)), ] #remove cd NAs
get_sample<-function(borough) { #get a sample from cds where the borough is received 
  return (sample(sanit_per_borough[sanit_per_borough$borough==borough,]$sanitdistrict,1)) 
}
df$sanitdistrict[is.na(df$sanitdistrict)] <- lapply(df$borough[is.na(df$sanitdistrict)], FUN=get_sample)

df$sanitdistrict <- unlist(df$sanitdistrict) # unlist the result column
df$sanitdistrict <- as.factor(df$sanitdistrict) # and save as factor




#Column 'version' DROP IT. it has only 1 level, which is version: 18v2.1
str(df$version)
summary(df$version)
# Drop column
df <- subset(df, select = -c(version))



# Column 'plutomapid' 
# this column is useful for VISULIZATION not for PREDICTION.
# Deleted, restore if you wanna use it!
str(df$plutomapid)
summary(df$plutomapid)
df <- subset(df, select = -c(plutomapid))


# Column 'mappluto_f'
str(df$mappluto_f)
summary(df$mappluto_f)
df <- subset(df, select = -c(mappluto_f))

# Column 'appdate'
str(df$appdate)
summary(df$appdate)
df <- subset(df, select = -c(appdate))


# Column 'appbbl'
str(df$appbbl)
summary(df$appbbl)
df <- subset(df, select = -c(appbbl))

# Column 'edesignum'
# CONVERTE IT: E means there is some hazadrus regarding this property
# converted -> E =1; NA=0
summary(df$edesignum)
str(df$edesignum)
## Converting data E=1 and empty = 0
df$edesignum <- as.character(df$edesignum) # as character
df$edesignum[df$edesignum==""] <- "0" # if empty then 0
df$edesignum[df$edesignum != "0"] <- "1" # if not zero then 1
df$edesignum <- as.factor(df$edesignum) # save as factor


# Column 'taxmap' # DROP IT
df <- subset(df, select = -c(taxmap))


# Column 'sanborn'
df <- subset(df, select = -c(sanborn))


# Column 'zmcode'
str(df$zmcode)
summary(df$zonemap)
df <- subset(df, select = -c(zonemap))

### COLUMNS ABOUT VISUALIZATION X,Y, ZMCODE #### DELETED
# Column: xcoord
df <- subset(df, select = -c(xcoord))
# Column: ycoord
df <- subset(df, select = -c(ycoord))
# COlumn: zmcode
df <- subset(df, select = -c(zmcode))



# Column: tract2010 DELETED
df <- subset(df, select = -c(tract2010))


# Column: condono DELETED
df <- subset(df, select = -c(condono))

# Column: bbl
df <- subset(df, select = -c(bbl))


# Column: borocode (KEEP IT, no NAs)
summary(df$borocode)

# Column: facilfar (DELETED)
df <- subset(df, select = -c(facilfar))

#Column: commfar
df <- subset(df, select = -c(commfar))

# Column: residfar
df <- subset(df, select = -c(residfar))

# Column: builtfar
df <- subset(df, select = -c(builtfar))
828658/858976 # this % is not specified. Too many.
# Column: landmark
df <- subset(df, select = -c(landmark))

# Column: histdist
summary(df$histdist)
df <- subset(df, select = -c(histdist))


# Column: yearalter2 & yearalter1
# If yearalter2 is empty than take Alter 1, otherwise keep 0, which means that bulding was
# not modified at all.
df$yearalter3 <- ifelse(df$yearalter2 == 0 , df$yearalter1, df$yearalter2) # If else...
# Change the name of the column
names(df)[names(df) == "yearalter3"] <- "last_modif"
# Delete previous columns yearalter1 & yearalter2
df <- subset(df, select = -c(yearalter2))
df <- subset(df, select = -c(yearalter1))


#Column: yearbuilt
# Plot the data (we have some zeros):
library(ggplot2)
ggplot(df, aes(x=yearbuilt)) + geom_histogram() +
  labs(title="Year built",x="Year", y = "Count")

table(is.na(df$yearbuilt)) # No NAs
#Mode function:
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Replace zeroes with mode when the building is built:
df$yearbuilt <- ifelse(df$yearbuilt == 0 & df$bldgarea != 0, getmode(df$yearbuilt), df$yearbuilt)

# Check it: 
# ggplot(df, aes(x=yearbuilt)) + geom_histogram() +
#   labs(title="Year built",x="Year", y = "Count")


# Column: exempttot
str(df$exempttot)
summary(df$exempttot)

# Column: exempttot
df <- subset(df, select = -c(exempttot))

#Column: exemptland
df <- subset(df, select = -c(exemptland))


# Column: assesstot (TARGET VARIABLE !!!!!)
#Check statistics and outliers
summary(df$assesstot)
str(df$assesstot)
boxplot(df$assesstot)
hist(df$assesstot)

# Replace NAs by zero
df$assesstot[is.na(df$assesstot)] <- 0
# Delete all values equal to zero
df <- filter(df, assesstot !=0)


# Column: assessland (TARGET VARIABLE !!!!!)
summary(df$assessland)
boxplot(df$assessland)
hist(df$assessland)


<<<<<<< HEAD

=======
## delete borough
# cUZ WE HAVE BOROCOD
df <- subset(df, select = -c(borough))


summary(df$resarea)
str(df$resarea)
>>>>>>> 60dff98ac8a95cfd8c3725230f8852973567492e
