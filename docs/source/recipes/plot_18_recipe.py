"""
Plot to compare two linked datasets : Snow against Elevation
============================================
In this recipe, we will plot a dependant variable (in this example snow cover) 
against an independent variable (elevation). This will be on a contour plot 
separately and together, then finding the coefficient of correlation and making an 
elevation line plot.
"""

# %%
# 1. Import cf-python, cf-plot and other required packages:

import cfplot as cfp
import cf
import sys
import scipy.stats.mstats as mstats
import matplotlib.pyplot as plt

#%%
# 2. Read the data in:
# We are investigating the influence of the land height on the snow cover, 
#so snow cover is the dependent variable. You can use different variable
# names for easier understanding.
# We are selecting the first field in the data with [0]
PATH="~/summerstudents/final-recipes/new-required-datasets"
orog = cf.read(f"{PATH}/1km_elevation.nc")[0]
snow = cf.read(f"{PATH}/snowcover")[0]

snow_day = snow[0]  # first day, Jan 1st of this year (2024)

#%%
# 3. Choose the regions:
# Ensuring both datasets cover it, and also that its not too large as then 
# regridding could cause the program to crash on small computers (dont 
# regrid the whole map), as well as being mainly land.
# Tool for finding rectangular  area for lat and lon points
# https://www.findlatitudeandlongitude.com/l/Yorkshire%2C+England/5009414/
region_in_mid_uk = [(-3.0, -1.0), (52.0, 55.0)]  # lon then lat
use_orog = orog.subspace(
    longitude=cf.wi(*region_in_mid_uk[0]),
    latitude=cf.wi(*region_in_mid_uk[1])
)

#%%
# 4. Subspace snow to same bounding box as orography data (orog)
use_snow = snow_day.subspace(
    longitude=cf.wi(*region_in_mid_uk[0]),
    latitude=cf.wi(*region_in_mid_uk[1])
)

#%%
# 5. Normalise the data
# This normalises to between 0 and 1 so multiply by 100 to get a percentage
use_snow_normal = ((use_snow - use_snow.minimum())/ (use_snow.range()))*100 

#%%
# 6. Reassign the units as they are removed by cf-python after calculation
# Only do this if you are certain the units are convertible to %
use_snow_normal.override_units("percentage", inplace=True) 

#%%
# 7. Plot of Snowcover contour
# First it outputs the newly formatted data, you can change the file names
# the plots will save as.
# colour_scale.txt is a colour scale specifically made and saved to show 
# the data but you could make your own or use existing ones from
# https://ncas-cms.github.io/cf-plot/build/colour_scales.html
# You will also need to adjust the labels and axis for your region
cfp.gopen(file="snow-1.png")
###cfp.cscale("~/cfplot_data/colour_scale.txt")
cfp.mapset(resolution="10m")
cfp.con(use_snow_normal, 
        lines=False, 
        title = "Snow Cover Contour Plot", 
        xticklabels = ("3W", "2W", "1W"), 
        yticklabels =("52N", "53N", "54N", "55N"), 
        xticks = (-3, -2, -1), 
        yticks= (52, 53, 54, 55))
cfp.gclose()

#%%
# 8. Plot of 1km Resolution Orography Contour
# Here the ocean doesnt get coloured out because when it is regridded it gets 
# masked by the lack of data for the dependant value (snow) over the seas
cfp.gopen(file="orog-1.png")
cfp.cscale("wiki_2_0_reduced")
cfp.mapset(resolution="10m")
cfp.con(use_orog, 
        lines=False, 
        title = "1km resolution Orography Contour Plot", 
        xticklabels = ("3W", "2W", "1W"), 
        yticklabels =("52N", "53N", "54N", "55N"), 
        xticks = (-3, -2, -1), 
        yticks= (52, 53, 54, 55))
cfp.gclose()

#%%
# 9. Plot of Vertical Orography Lineplot of 1km Resolution Elevation
lonz = use_orog.construct("longitude").data[0]
elevation_orog = use_orog.subspace(longitude = lonz)
xticks_elevation_orog = [52, 52.5, 53, 53.5, 54, 54.5, 55]

cfp.gopen(figsize=(12, 6), file ="orography_vertical.png")

cfp.lineplot(
    x=elevation_orog.coord('latitude').array,
    y=elevation_orog.array.squeeze(),
    color="black",
    title="Orography Profile over part of UK",
    ylabel="Elevation (m)",
    xlabel="Latitude",
    xticks=xticks_elevation_orog,
)

cfp.plotvars.plot.fill_between(
    elevation_orog.coord('latitude').array,
    0,
    elevation_orog.array.squeeze(),
    color='black',
    alpha=0.7
)
cfp.gclose()

#%%
# 10. Regrid the data to have a comparable array.
# Here the orography file is regridded to the snow file 
# as snow is higher resolution (if this doesnt work switch them).
reg = use_orog.regrids(use_snow_normal, method="linear")


# This plots the regridded file data
cfp.gopen(file="regridded-1.png")
cfp.con(reg, lines=False)
cfp.gclose()

#%%
# 11. Squeeze snow data to remove the size 1 axes
snow_data = snow_data.squeeze()

#%%
#  12. Final statistical calculations
coefficient = mstats.pearsonr(reg_data.array, snow_data.array)
print(f"The Pearson correlation coefficient is {coefficient}")
