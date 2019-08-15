source("libraries.R")

get_sample<-function(b, col_per_zipcode, colname) { # Get a sample where the zipcode is equal to the received one
  list_to_sample <- col_per_zipcode[colname][col_per_zipcode["zipcode"]==b]
  if (length(list_to_sample)==0) { # If all rows have NAs for that colname for the given zipcode, return NA.
    return (NA)
  }
  return (sample(list_to_sample, size=1)) 
}

fill_NAs_by_zipcode <- function(df, colname) {
  col_per_zipcode <- unique(df[,c("zipcode", colname)]) #keep unique values for each zipcode
  col_per_zipcode <- col_per_zipcode[!(is.na(col_per_zipcode[colname])), ] #remove NAs
  return (sapply(df["zipcode"][is.na(df[colname])], FUN=get_sample, col_per_zipcode, colname))
}

get_median<-function(b, col_per_zipcode, colname){
  index <- which(col_per_zipcode["zipcode"]==b)
  if (length(index)==0) { # If all rows have NAs for that colname for the given zipcode, return NA.
    return (NA)
  }
  return (unlist(col_per_zipcode[index, colname]))
}
fill_NAs_median<-function(df, colname) {
  col_per_zipcode <- df[,c("zipcode", colname)]
  col_per_zipcode <- col_per_zipcode[!(is.na(col_per_zipcode[colname])), ]
  col_per_zipcode <- col_per_zipcode[!(is.na(col_per_zipcode["zipcode"])), ]
  col_per_zipcode <- col_per_zipcode %>% group_by_at("zipcode") %>% summarise_at(vars(colname), median)
  return (sapply(df["zipcode"][is.na(df[colname])], FUN=get_median, col_per_zipcode, colname))
}

get_sample_borough<-function(b, col_per_borough, colname) { # Get a sample where the borocode is equal to the received one
  list_to_sample <- col_per_borough[colname][col_per_borough["borocode"]==b]
  if (length(list_to_sample)==0) { # If all rows have NAs for that colname for the given borocode, return NA.
    return (NA)
  }
  return (sample(list_to_sample, size=1)) 
}

fill_NAs_by_borough <- function(df, colname) {
  col_per_borough <- unique(df[,c("borocode", colname)]) #keep unique values for each borocode
  col_per_borough <- col_per_borough[!(is.na(col_per_borough[colname])), ] #remove NAs
  return (sapply(df["borocode"][is.na(df[colname])], FUN=get_sample_borough, col_per_borough, colname))
}

get_median_borough<-function(b, col_per_borough, colname){
  index <- which(col_per_borough["borocode"]==b)
  if (length(index)==0) { # If all rows have NAs for that colname for the given borocode, return NA.
    return (NA)
  }
  return (unlist(col_per_borough[index, colname]))
}

fill_NAs_median_borough<-function(df, colname) {
  col_per_borough <- df[,c("borocode", colname)]
  col_per_borough <- col_per_borough[!(is.na(col_per_borough[colname])), ]
  col_per_borough <- col_per_borough[!(is.na(col_per_borough["borocode"])), ]
  col_per_borough <- col_per_borough %>% group_by_at("borocode") %>% summarise_at(vars(colname), median)
  return (sapply(df["borocode"][is.na(df[colname])], FUN=get_median_borough, col_per_borough, colname))
}

