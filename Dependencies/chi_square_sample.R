# For independence test between categorical variables:

source("libraries.R")
source("data_type_fun.R")

test_sample <- read.csv("Dependencies/sample.csv")

col_var <- c("cd","schooldist","council","zipcode","firecomp","policeprct",
             "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
             "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
             "healthcenterdistrict", "pfirm15_flag")
# block and lot not considered here

row_var <- col_var

test_sample <- data_type(test_sample)

counter <- 0
test_samp <- data.frame()

for (j in col_var) {
  for(i in row_var) {
    chi_res <- chisq.test(test_sample[,j], test_sample[,i], simulate.p.value = TRUE)
    test_samp[i,j] <- paste(round(chi_res$statistic,3), chi_res$p.value, sep=", ")
    print(paste0("done ",i," ",j))
    print(counter <- counter+1)
  }
}

write_csv(test_samp, path = "Dependencies/chisquare_0.011.csv")
