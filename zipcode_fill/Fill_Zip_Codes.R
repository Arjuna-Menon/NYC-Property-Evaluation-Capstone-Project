
na_zips <- subset(df, is.na(zipcode))
write.csv(na_zips, "zipcode_fill/Zips_code.csv")

