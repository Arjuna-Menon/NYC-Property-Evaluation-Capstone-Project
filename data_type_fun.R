# The function get a dataframe and return the same data frame but with specified types of column

#df <- read.csv("pluto2.csv")
data_type <- function(df) {
  
  #Those columns are saved as factors:
  factors <- c("block","lot","cd","schooldist","council","zipcode","firecomp","policeprct",
               "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
               "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
               "healthcenterdistrict", "pfirm15_flag")
  
  #Those columns should be numeric
  # numeric <- c("lotarea", "bldgarea","numbldgs","numfloors","unitsres","unitstotal","lotfront",
  #              "lotdepth","bldgfront","bldgdepth","assessland","assesstot","yearbuilt",           
  #              "residfar","commfar","facilfar","xcoord","ycoord","yearalter", "income")
  
  for (i in colnames(df)) {
    if (i %in% factors) {
      df[,i] <- as.factor(df[,i])
    } else {
      df[,i] <- as.numeric(df[,i])
    }

  }
  
  return (df)
  
}
