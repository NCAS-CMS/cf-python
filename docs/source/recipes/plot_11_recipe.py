"""
Plotting the Warming Stripes
============================

In this recipe, we will plot the `Warming Stripes (Climate Stripes)
<https://en.wikipedia.org/wiki/Warming_stripes>`_ created by
Professor Ed Hawkins at NCAS, University of Reading. Here we will use the
ensemble mean of the 
`HadCRUT.5.0.1.0 analysis gridded data
<https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/download.html>`_ for
the same.

"""

# %%
# 1. Import cf-python and matplotlib.pyplot:

import matplotlib.pyplot as plt

import cf

# %%
# 2. Read the field constructs:
temperature_data = cf.read(
    "~/recipes/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc"
)[0]
print(temperature_data)

# %%
# 3. Calculate the annual mean temperature anomalies. The ``'weights=True'``
# argument is used to take the varying lengths of months into account which
# ensures that the calculated mean is more accurate:
annual_temperature = temperature_data.collapse(
    "T: mean", weights=True, group=cf.Y()
)

# %%
# 4. Select the data from 1850 to 2022:
period = annual_temperature.subspace(T=cf.year(cf.wi(1850, 2022)))

# %%
# 5. Calculate the global average temperature for each year:
global_temperature = period.collapse("X: Y: mean")

# %%
# 6. Get the global average temperature and squeeze it to remove the size 1 axis:
global_avg_temp = global_temperature.array.squeeze()

# %%
# 7. Create a normalization function that maps the interval from the minimum to 
# the maximum temperature to the interval [0, 1] for colouring:
norm_global = plt.Normalize(global_avg_temp.min(), global_avg_temp.max())

# %%
# 8. Set the colormap instance:
cmap = plt.get_cmap("RdBu_r")

# %%
# 9. Create the figure and the axes for the global plot. Loop over the selected 
# years, plot a colored vertical stripe for each and remove the axes:
fig_global, ax_global = plt.subplots(figsize=(10, 2))

for i in range(global_avg_temp.shape[0]):
    ax_global.axvspan(
        xmin=i-0.5,
        xmax=i+0.5,
        color=cmap(norm_global(global_avg_temp[i]))
    )

ax_global.axis("off")

plt.show()

# %%
# 10. For the regional warming stripes, steps 5 to 9 are repeated for the 
# specific region. Here, we define the bounding box for UK by subspacing over 
# a domain spanning 49.9 to 59.4 degrees north and -10.5 to 1.8 degrees east:
uk_temperature = period.subspace(X=cf.wi(-10.5, 1.8), Y=cf.wi(49.9, 59.4))
uk_avg_temperature = uk_temperature.collapse("X: Y: mean")
uk_avg_temp = uk_avg_temperature.array.squeeze()
norm_uk = plt.Normalize(uk_avg_temp.min(), uk_avg_temp.max())

# %%

fig_uk, ax_uk = plt.subplots(figsize=(10, 2))

for i in range(uk_avg_temp.shape[0]):
    ax_uk.axvspan(xmin=i-0.5, xmax=i+0.5, color=cmap(norm_uk(uk_avg_temp[i])))

ax_uk.axis("off")

plt.show()
