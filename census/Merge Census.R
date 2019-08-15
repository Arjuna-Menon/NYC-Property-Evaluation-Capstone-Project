#Read Libraries
source("libraries.R")
source("NAs/utils.R")
pluto <- read.csv("pluto2.csv")
income <- read.csv("census/census_income.csv")

families <- select(income, "GEO.id2", "HC02_EST_VC15", "HC04_EST_VC15")
colnames(families) <- c("zipcode", "mean_income_fam", "mean_income_non")

data <- merge(pluto, families, by="zipcode", all.x = TRUE)

#fill NAs values for mean income taking median from zipcode, if then is still NA fill by borough
data$mean_income_fam[is.na(data$mean_income_fam)] <- fill_NAs_median(data, "mean_income_fam")
data$mean_income_non[is.na(data$mean_income_non)] <- fill_NAs_median(data, "mean_income_non")
data$mean_income_fam[is.na(data$mean_income_fam)] <- fill_NAs_median_borough(data, "mean_income_fam")
data$mean_income_non[is.na(data$mean_income_non)] <- fill_NAs_median_borough(data, "mean_income_non")

table(is.na(data$mean_income_fam))
table(is.na(data$mean_income_non))

min_income_fam <- min(data$mean_income_fam)
max_income_fam <- max(data$mean_income_fam)
min_income_non <- min(data$mean_income_non)
max_income_non <- max(data$mean_income_non)

get_income <- function(mean_income_fam, mean_income_non) {
  income_fam <- rtriangle(1,a=min_income_fam,b=max_income_fam,c = mean_income_fam)
  income_non <- rtriangle(1,a=min_income_non,b=max_income_non,c = mean_income_non)
  return (sample(c(income_fam, income_non), size=1))
}
data["income"] <- 0
for(i in 1:nrow(data)) {
  print(i)
  if (as.numeric(data[i,"landuse"])<6) {
    data[i,"income"] <- get_income(data[i, "mean_income_fam"], data[i, "mean_income_non"])
  }
}

data <- subset(data, select = -c(mean_income_fam, mean_income_non))

# Put zeroes for income when bldgarea is zero:
table(data$income==0)
table(data$bldgarea == 0)
table(data$bldgarea==0 & data$income==0)
# Hence, the remaining (39418 - 37433 = 1985) rows are replaced with income = 0:
data$income[data$bldgarea == 0] <- 0

write_csv(data, path = "pluto3.csv")
summary(data)