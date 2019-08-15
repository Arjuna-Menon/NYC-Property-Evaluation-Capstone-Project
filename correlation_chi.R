####Chi-Square Test
source("libraries.R")
source("data_type_fun.R")
df <- read.csv("pluto2.csv")

df <- data_type(df)

col_var <- c("cd","schooldist","council","zipcode","firecomp","policeprct",
             "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
             "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
             "healthcenterdistrict", "pfirm15_flag")

row_var <- col_var


#####  This test is on the whole dataset
counter <- 0
test2 <- data.frame()

for (j in col_var) {
  for(i in row_var) {
    chi_res <- chisq.test(df[,j], df[,i], simulate.p.value = TRUE)
    test2[i,j] <- paste(round(chi_res$statistic,3), chi_res$p.value, sep=", ")
    print(paste0("done ",i," ",j))
    print(counter <- counter+1)
  }
}

write_csv(test2, path = "correlation/factors_chi2.csv")



##### Repeate the process only for sample #####

source("libraries.R")
source("data_type_fun.R")

test2_sample <- read.csv("sample/sample_0.011.csv")

col_var <- c("cd","schooldist","council","zipcode","firecomp","policeprct",
             "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
             "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
             "healthcenterdistrict", "pfirm15_flag")

row_var <- col_var

test2_sample <- data_type(test2_sample)

counter <- 0
test2_samp <- data.frame()

for (j in col_var) {
  for(i in row_var) {
    chi_res <- chisq.test(test2_sample[,j], test2_sample[,i], simulate.p.value = TRUE)
    test2_samp[i,j] <- paste(round(chi_res$statistic,3), chi_res$p.value, sep=", ")
    print(paste0("done ",i," ",j))
    print(counter <- counter+1)
  }
}

write_csv(test2_samp, path = "correlation/chisquare_0.011.csv")
