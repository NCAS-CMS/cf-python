"""
Combining cf and Matplotlib plots in one figure
===============================================

Presently, cf-plot has very few exposed interfaces to its Matplotlib and
Cartopy backend. This makes it difficult to combine plots from the three
in one figure, but not impossible.

A combined cf and Matplotlib plot can be achieved by amending the figure
stored at ``cfp.plotvars.master_plot``, and then redrawing it with the
new subplots.
"""

# %%
# 1. Import cf-python, cf-plot, Matplotlib, NumPy, and Dask.array:

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_end_ignore

import matplotlib.pyplot as plt
import cfplot as cfp
import cf

import numpy as np
import dask.array as da

# %%
# 2. Read example data field constructs, and set region for our plots:

f = cf.read(f"~/recipes/data1.nc")

u = f.select_by_identity("eastward_wind")[0]
v = f.select_by_identity("northward_wind")[0]
t = f.select_by_identity("air_temperature")[0]

# Subspace to get values for a specified pressure, here 500 mbar
u = u.subspace(pressure=500)
v = v.subspace(pressure=500)
t = t.subspace(pressure=500)

lonmin, lonmax, latmin, latmax = 10, 120, -30, 30

# %% [markdown]
#
# Outlining the figure with cf-plot
# ---------------------------------
#

# %%
# 1. Set desired dimensions for our final figure:

rows, cols = 2, 2

# %%
# 2. Create a figure of set dimensions with ``cfp.gopen()``, then set the
# position of the cf plot:


cfp.gopen(rows, cols)

pos = 2  # Second position in the figure

cfp.gpos(pos)
# sphinx_gallery_start_ignore
plt.close()
# sphinx_gallery_end_ignore

# %%
# 3. Create a simple vector plot:

cfp.mapset(lonmin=lonmin, lonmax=lonmax, latmin=latmin, latmax=latmax)
cfp.vect(u=u, v=v, key_length=10, scale=120, stride=4)

# %% [markdown]
#
# Creating our Matplotlib plots
# -----------------------------
#

# %%
# 1. Access the newly-created figure:

fig = cfp.plotvars.master_plot

# %%
# 2. Reduce fields down to our test data for a wind rose scatter plot:

# Limit to specific geographic region
u_region = u.subspace(X=cf.wi(lonmin, lonmax), Y=cf.wi(latmin, latmax))
v_region = v.subspace(X=cf.wi(lonmin, lonmax), Y=cf.wi(latmin, latmax))
t_region = t.subspace(X=cf.wi(lonmin, lonmax), Y=cf.wi(latmin, latmax))

# Remove size 1 axes
u_squeeze = u_region.squeeze()
v_squeeze = v_region.squeeze()
t_squeeze = t_region.squeeze()

# Flatten to one dimension for plot
u_f = da.ravel(u_squeeze.data)
v_f = da.ravel(v_squeeze.data)
t_f = da.ravel(t_squeeze.data)

# %%
# 3. Perform calculations to create appropriate plot data:

mag_f = da.hypot(u_f, v_f)  # Wind speed magnitude

azimuths_f = da.arctan2(v_f, u_f)
rad_f = ((np.pi / 2) - azimuths_f) % (np.pi * 2)  # Wind speed bearing

# Normalise temperature data into a range appropriate for setting point sizes (1-10pt).
temp_scaled = 1 + (t_f - t_f.min()) / (t_f.max() - t_f.min()) * (10 - 1)

# %%
# 4. Add Matplotlib subplot to our existing cf figure:

pos = 1  # First position in the figure

ax = fig.add_subplot(rows, cols, pos, polar=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.scatter(
    rad_f.compute(),
    mag_f.compute(),
    s=temp_scaled.compute(),
    c=temp_scaled.compute(),
    alpha=0.5,
)

ax.set_xlabel("Bearing [°]")
ax.set_ylabel("Speed [m/s]", rotation=45, labelpad=30, size=8)
ax.yaxis.set_label_coords(0.45, 0.45)
ax.yaxis.set_tick_params(which="both", labelrotation=45, labelsize=8)
ax.set_rlabel_position(45)

# %%
# 5. Create and add a third plot, for example:

x = np.linspace(0, 10, 100)
y = np.sin(x)

pos = 3  # Third position in the figure

ax1 = fig.add_subplot(rows, cols, pos)

ax1.plot(x, y, label="sin(x)")
ax1.legend()

# %% [markdown]
#
# Drawing the new figure
# ----------------------
#

# %%
# 1. Draw final figure:

fig = plt.figure(fig)
fig.tight_layout()
fig.show()

# %% [markdown]
#
# Summary
# -------
#
# In summary, to use other plotting libraries with cf-plot, you must first
# create your figure with cf-plot with placeholders for your other plots,
# then add subplots by accessing the ``cfp.plotvars.master_plot`` object,
# and finally redraw the figure containing the new plots.
