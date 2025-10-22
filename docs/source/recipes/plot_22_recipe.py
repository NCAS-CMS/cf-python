"""
Plotting a wind rose as a scatter plot
======================================

Given a file containing northerly and easterly wind components, we can
calculate the magnitude and bearing of the resultant wind at each point
in the region and plot them using a scatter plot on a polar grid to
create a wind rose representing wind vectors in the given area.
"""
# %%
# 1. Import cf-python, Dask.array, NumPy, and Matplotlib:

import cf
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt

# %%
# 2. Read the field constructs and load the wind speed component fields:

f = cf.read('~/recipes/data1.nc')
print(f)

U = f[2].squeeze() # Easterly wind speed component
V = f[3].squeeze() # Northerly wind speed component


# %%
# 3. Set a bounding region for the data and discard readings outside of it:

tl = (41, 72) # (long, lat) of top left of bounding box.
br = (65, 46) # (long, lat) of bottom right of bounding box.

U_region = U.subspace(X=cf.wi(tl[0], br[0]), Y=cf.wi(br[1], tl[1]))
V_region = V.subspace(X=cf.wi(tl[0], br[0]), Y=cf.wi(br[1], tl[1]))

# %%
# 4. Select measurements for a specific pressure using the subspace method,
# then use squeeze to remove the size 1 axis:

U_sub = U_region.subspace(pressure=500.0)
V_sub = V_region.subspace(pressure=500.0)

U_sub.squeeze(inplace=True)
V_sub.squeeze(inplace=True)

# %%
# 5. Calculate the magnitude of each resultant vector using Dask's hypot
# function:

magnitudes = da.hypot(U_sub.data, V_sub.data)

# %%
# 6. Calculate the angle of the resultant vector (relative to an Easterly ray)
# using Dask's arctan2 function, then convert to a clockwise bearing:

azimuths = da.arctan2(V_sub.data, U_sub.data)

bearings = ((np.pi/2) - azimuths) % (np.pi*2)

# %%
# 7. Flatten the two dimensions of each array for plotting with Matplotlib:

bearings_flattened = da.ravel(bearings)

magnitudes_flattened = da.ravel(magnitudes)

# %%
# 8. Draw the scatter plot using Matplotlib:

plt.figure(figsize=(5, 6))
# sphinx_gallery_start_ignore
fig = plt.gcf()
# sphinx_gallery_end_ignore
ax = plt.subplot(polar=True)
ax.set_theta_zero_location("N") # Place 0 degrees at the top.
ax.set_theta_direction(-1) # Arrange bearings clockwise around the plot.
# sphinx_gallery_start_ignore
plt.close()
# sphinx_gallery_end_ignore

# %%
# 9. Draw a scatter plot on the polar plot, using the wind direction bearing as
# the angle and the magnitude of the resultant wind speed as the distance
# from the pole.
ax.scatter(bearings_flattened.compute(), magnitudes_flattened.compute(), s=1.2)
# sphinx_gallery_start_ignore
plt.close()
# sphinx_gallery_end_ignore

# %%
# 10. Label the axes and add a title.

# sphinx_gallery_start_ignore
plt.figure(fig)
# sphinx_gallery_end_ignore
plt.title(f'Wind Rose Scatter Plot\nLat: {br[1]}°-{tl[1]}°, Long: {tl[0]}°-{br[0]}°')

ax.set_xlabel("Bearing [°]")

ax.set_ylabel("Speed [m/s]", rotation=45, labelpad=30, size=8)

ax.yaxis.set_label_coords(0.45, 0.45)

ax.yaxis.set_tick_params(which='both', labelrotation=45, labelsize=8)

ax.set_rlabel_position(45)
# sphinx_gallery_start_ignore
plt.close()
# sphinx_gallery_end_ignore

# %%
# 11. Display the plot.

# sphinx_gallery_start_ignore
plt.figure(fig)
# sphinx_gallery_end_ignore
plt.show()
