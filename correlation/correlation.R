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
