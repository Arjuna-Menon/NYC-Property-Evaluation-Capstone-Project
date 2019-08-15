################# Functions descriptions #################
##########################################################
# Here we are looking for intersection between point (incidents) and poligons 
# (area, which is zip-codes)
# Note: the process of looking for intersection is based on shapefile. 
# For this reason a shapefile with zipcodes where donwladed from:
# https://data.austintexas.gov/Locations-and-Maps/Zipcodes/ghsj-v65t 
# What function does is: 
# a) transform a csv file (number 5 accidents) which contains Longitude and Latitude 
#    of every single accident in Austin into poligon object (so the intersection
#    later can be found easier).
# b) read a poligon (shapefile) with zip-codes from the file.
# c) Intersect two poligons
# d) as a result we assign to every accident a zip-code, where the accident
#    occured. Rest of the work is done in the Function 4. Where accidents are
#    grouped.


#################################### Imports #################################
# PART I: Converte csv file into shapefile
from geopandas import GeoDataFrame
from shapely.geometry import Point
    
# PART II: Find intersection between shapefiles
import geopandas as gpd
import os
import pandas as pd


############################ Function #######################################       
def intersect(df, zipcodes):

    

    #Convert points (csv file) into shapefile (just changin format)
    geometry = [Point(xy) for xy in zip(df.xcoord, df.ycoord)]
    #df = df.drop(['Longitude', 'Latitude'], axis=1)
    crs = {'init': 'epsg:2263'}
    gdf = GeoDataFrame(df, crs=crs, geometry=geometry)
    
    
    # PART II
    #Find intersection between points (accidents) and poligons (zipcodes)
    gdfLeft = gpd.read_file(os.path.join(zipcodes))
    gdfRight = gdf
    #Join the data
    gdfJoined = gpd.sjoin(gdfLeft, gdfRight, how="inner", op='intersects')
    
    
    # PART III
    ## CHECK (uncommend if needed)
    # In order to check if process went correct. Below code was used. 
    # Random sample of 2000 points were chosen, save into csv file and 
    # plotted in QGIS software and compared with zipcodes.
    '''test_random=gdfJoined.sample(2000) 
    test_random.to_csv('random.csv',sep=',')'''
    
    
   # logging.info("Intersection succes!")
    
    return gdfJoined


# Read data
zipcodes='shape/zip_poligons.shp' # do not change
df=pd.read_csv('Zips_code.csv') # this file might be changed. Depend to what points you need to find zipcodes

# Run the function
zips_fixed=intersect(df,zipcodes)
#Drop columns
zips_fixed = zips_fixed.drop(['geometry', 'zipcode', 'index_right','Unnamed: 0'], axis=1)
# Rename ZIPCODE to zipcode (in order to be correct with all data set we gonna join)
zips_fixed.rename(columns={'ZIPCODE': 'zipcode'}, inplace=True)

# Save the new fixed df to csv. file
zips_fixed.to_csv('zips_fixed.csv',index=False)

