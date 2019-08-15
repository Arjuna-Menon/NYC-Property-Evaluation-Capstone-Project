## Clustering with numerical data only

## Read stuff
source("libraries.R")
source("data_type_fun.R")
df <- read.csv("pluto4.csv")
df <- data_type(df)



## Choose only numerical values
#Specify what are factors
col_var <- c("block","lot","cd","schooldist","council",
             #"zipcode", We need zipcode for visualization.
             "firecomp","policeprct",
             "healtharea","sanitboro","sanitsub","zonedist1","spdist1","ltdheight","landuse",
             "ext","proxcode","irrlotcode","lottype","borocode","edesignum","sanitdistrict",
             "healthcenterdistrict", "pfirm15_flag","xcoord", "ycoord")

#Dont choose factors
df_num <- select(df, -c(col_var))

#Aggregate by unique_block
df_agg = aggregate(df_num,
                   by = list(df_num$zipcode),
                   FUN = mean)

#Delete the original zipcode column
df_agg <- subset(df_agg, select=-zipcode)
#Change the name of grouping variable into zipcode
colnames(df_agg)[which(names(df_agg) == "Group.1")] <- "zipcode"


########### CLUSTERING TECHNIQUES: ########### 

######### Model Based:######### 
set.seed(123)
clus_mc <- Mclust(df_agg[, !names(df_agg) %in% c('zipcode')]) # don't use zipcode
summary(clus_mc)

#Cbind results with df
mc_results <- cbind(clus_mc$classification, df_agg)
#rename column
colnames(mc_results)[which(names(mc_results) == "clus_mc$classification")] <- "clust_mc"
#select only class + zipcode
agg_clust <- select(mc_results, zipcode, clust_mc)

write.csv(agg_clust,  file = "Clustering/Clusters_MC.csv")


######### K-Means ######### 

#Scale data
clus_kmeans <- scale(df_agg[, !names(df_agg) %in% c('zipcode')])

#Function to check # of clusters in a data
plot.wgss = function(mydata, maxc) {
  wss = numeric(maxc)
  for (i in 1:maxc)
    wss[i] = kmeans(mydata, centers=i, nstart = 10)$tot.withinss
  plot(1:maxc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares", main="Scree Plot")
}

## scree plot
scree_plot2 <- plot.wgss(clus_kmeans, 10)

## K-means version I (3 groups)
km1 <- kmeans(clus_kmeans, centers=3, nstart = 20)
table(km1$cluster)

kmeans_1 <- cbind(km1$cluster, df_agg)
kmeans_1 <- select(kmeans_1, zipcode, `km1$cluster`)

write.csv(kmeans_1,  file = "Clustering/kmeans_3groups.csv")


## K-means version II (4 groups)
set.seed(1212)
km2 <- kmeans(clus_kmeans, centers=4, nstart = 20)
table(km2$cluster)

kmeans_2 <- cbind(km2$cluster, df_agg)
kmeans_2 <- select(kmeans_1, zipcode, `km2$cluster`)

write.csv(kmeans_2,  file = "Clustering/kmeans_4groups.csv")
