df <- read.csv("sample/sample_0.011.csv")
categorical_vars <- c("cd","schooldist","council","zipcode","firecomp","policeprct",
                      "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
                      "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
                      "healthcenterdistrict", "pfirm15_flag") #remove block
#keep numeric vars
numeric_vars <- setdiff(colnames(df), c(categorical_vars, "xcoord", "ycoord", "block", "lot"))

#transform to factor categorical variables
df[categorical_vars] <- lapply(df[categorical_vars], factor)
result <- data.frame()
for (i in numeric_vars) {
  for (j in categorical_vars) {
    print(paste(i, j, sep=" and "))
    fit <- aov(df[,i] ~ df[,j], data=df)
    print(summary(fit))
    print(unlist(summary(fit)))
    result[i,j] <- paste(unlist(summary(fit))["F value1"], unlist(summary(fit))["Pr(>F)1"], sep=",")
  }
}
write.csv(result,"correlation/anova_sample.csv")