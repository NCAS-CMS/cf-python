"""
Plotting Contour Subplots with different Projections
============================================
In this recipe, we will plot different projections for the same data to illustrate
visually the ones available in order to more visually decide which is suitable.
"""

# %%
# 1. Import cf-python and cf-plot:

import cfplot as cfp

import cf

#%%
# 2. Read the field in:
# Here I've used sample data ggap.nc (and later pressure=850), 
# but you could use tas_A1.nc (with time=15)

f=cf.read("~/cfplot_data/ggap.nc")[0] 

#%%
# 3. Create the file with subplots:
# If you are changing the number of subplots ensure the number of rows * number 
# of columns = the number of subplots/projections
# Here we are doing 6 projections so 2x3 is fine

cfp.gopen(rows=2, columns=3, bottom=0.2, file="projections.png")

#%%
# 4. List the projection types being used:
# Here we are using Cylindrical/Default, North Pole Stereographic, 
# South Pole Stereographic, Mollweide, Cropped Lambert Conformal and Robinson
# However you could also use other such as "rotated", "ortho" or 
# "merc", "ukcp", "osgb", or "EuroPP"
# https://ncas-cms.github.io/cf-plot/build/user_guide.html#appendixc

projtypes = ["cyl", "npstere", "spstere", "moll", "lcc", "robin"] 

#%%
# 5. We then use a for loop to cycle through all the different projection types:
# Only gpos has 1 added because it can only take 1 as its first value, otherwise there are 
# errors. There are if statements for some projections (lcc, OSGB and EuroPP) as they have
# specific requirements for their contour.
# However, OSGB and EuroPP will require very different data anyway.

for i, proj in enumerate(projtypes):
    cfp.gpos(i+1) 
    if projtypes[i] == "lcc":
        cfp.mapset(proj='lcc', lonmin=-50, lonmax=50, latmin=20, latmax=85)
    if (projtypes[i]== "OSGB") or (projtypes[i] =="EuroPP"):
        cfp.mapset(proj=projtypes[i], resolution='50m')
    else:
        cfp.mapset(proj=projtypes[i])
    if i ==len(projtypes)-1:
         cfp.con(f.subspace(pressure=850), lines=False, title = projtypes[i], colorbar_position=[0.1, 0.1, 0.8, 0.02], colorbar_orientation='horizontal') #to see the marking lines need to be True, but this can be hard to read so if not needed use False
    else:
        cfp.con(f.subspace(pressure=850), lines = False, title = projtypes[i], colorbar = False)
cfp.gclose()