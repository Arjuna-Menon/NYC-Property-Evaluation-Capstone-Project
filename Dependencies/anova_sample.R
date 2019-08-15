df <- read.csv("Dependencies/sample.csv")
categorical_vars <- c("cd","schooldist","council","zipcode","firecomp","policeprct",
                      "healtharea","sanitboro","sanitsub","zonedist1","spdist1","landuse",
                      "ext","proxcode","irrlotcode","borocode","edesignum","sanitdistrict",
                      "healthcenterdistrict")
# We don't consider lot and block

# Delete ltdheight, lottype, pfirm15_flag categorical vars and unitstotal numerical variable:

# Numerical variables:
numeric_vars <- setdiff(colnames(df), c(categorical_vars, "block", "lot", "ltdheight", "lottype",
                                        "pfirm15_flag", "unitstotal"))

# Categorical variables:
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
write.csv(result,"Dependencies/anova_sample.csv")