# This is the main file.
# Only agreed and final code goes here.
# If you wanna try sth, create file with name.

# Download the data
url <- "https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyc_pluto_18v2_1_csv.zip"
filename <- "pluto_data.zip"
download.file(url, filename)
files <- unzip(filename)
df <- read.csv("pluto_18v2_1.csv")
file.remove(c(filename, "PLUTODD18v2.1.pdf", "PlutoReadme18v2.1.pdf"))
colnames(df)

## Removing variables (not necessary probably...)
# df <- read.csv("pluto_18v2_1.csv")
# t(t(names(df)))
# 
# varlist <- df(c("block", "lot", "cd", "schooldist", "council", "zipcode", "firecomp", "policeprct",
#                 "zonedist1", "spdist1", "ltdheight", "landuse", "ownertype", "lotarea", "numbldgs",
#                 "numfloors", "unitsres", "unitstotal", "ext", "irrlotcode", "lottype",
#                 "assessland", "assesstot", "yearbuilt", "yearalter1", "yearalter2", "borocode", "xcoord",
#                 "ycoord", "zonemap", "zmcode", "edesignum", "sanitdistrict", "healthcenterdistrict"))