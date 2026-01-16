"""
Calculating and plotting the global average temperature anomalies
=================================================================

In this recipe we will calculate and plot the global average temperature anomalies.
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
# 5. Print the description of near surface temperature to show properties of all constructs:

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
# 8. Calculate the area weighted mean surface temperature for each time using the collapse method:

global_avg = temp.collapse("area: mean", weights=True)

# %%
# 9. Calculate the annual global mean surface temperature:

annual_global_avg = global_avg.collapse("T: mean", group=cf.Y())

# %%
# 10. The temperature values are averaged for the climatological period of 1961-1990 by defining a subspace within these years using `cf.wi` query instance over subspace and doing a statistical collapse with the collapse method:

annual_global_avg_61_90 = annual_global_avg.subspace(
    T=cf.year(cf.wi(1961, 1990))
)
print(annual_global_avg_61_90)

# %%

temp_clim = annual_global_avg_61_90.collapse("T: mean")
print(temp_clim)

# %%
# 11. The temperature anomaly is then calculated by subtracting these climatological temperature values from the annual global average temperatures and plotted:

temp_anomaly = annual_global_avg - temp_clim
cfp.lineplot(
    temp_anomaly,
    color="red",
    title="Global Average Temperature Anomaly (1901-2021)",
    ylabel="1961-1990 climatology difference ",
    yunits="degree Celcius",
)
