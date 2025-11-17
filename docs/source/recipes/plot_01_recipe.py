"""
Calculating global mean temperature timeseries
==============================================

In this recipe we will calculate and plot monthly and annual global mean temperature timeseries.
"""

# %%
# 1. Import cf-python and cf-plot:

import cfplot as cfp

import cf

# %%
# 2. Read the field constructs:

f = cf.read("~/recipes/cru_ts4.06.1901.2021.tmp.dat.nc")
print(f)

# %%
# 3. Select near surface temperature by index and look at its contents:

temp = f[1]
print(temp)

# %%
# 4. Select latitude and longitude dimensions by identities, with two different techniques:

lon = temp.coordinate("long_name=longitude")
lat = temp.coordinate("Y")

# %%
# 5. Print the description of near surface temperature using the dump method to show properties of all constructs:

temp.dump()

# %%
# 6. Latitude and longitude dimension coordinate cell bounds are absent, which are created and set:

a = lat.create_bounds()
lat.set_bounds(a)
lat.dump()

# %%

b = lon.create_bounds()
lon.set_bounds(b)
lon.dump()

# %%

print(b.array)

# %%
# 7. Time dimension coordinate cell bounds are similarly created and set for cell sizes of one calendar month:

time = temp.coordinate("long_name=time")
c = time.create_bounds(cellsize=cf.M())
time.set_bounds(c)
time.dump()

# %%
# 8. Calculate and plot the area weighted mean surface temperature for each time:

global_avg = temp.collapse("area: mean", weights=True)
cfp.lineplot(global_avg, color="red", title="Global mean surface temperature")

# %%
# 9. Calculate and plot the annual global mean surface temperature:

annual_global_avg = global_avg.collapse("T: mean", group=cf.Y())
cfp.lineplot(
    annual_global_avg,
    color="red",
    title="Annual global mean surface temperature",
)
