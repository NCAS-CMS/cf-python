"""
Plotting a joint histogram
==========================

In this recipe, we will be creating a joint histogram of PM2.5 mass 
concentration and 2-metre temperature.
"""

# %%
# 1. Import cf-python, numpy and matplotlib.pyplot:

import matplotlib.pyplot as plt
import numpy as np

import cf

# %%
# 2. Read the field constructs using read function:
f = cf.read("~/recipes/levtype_sfc.nc")
print(f)

# %%
# 3. Select the PM2.5 mass concentration and 2-metre temperature fields by
# index and print the description to show properties of all constructs:
pm25_field = f[2]
pm25_field.dump()

# %%

temp_field = f[3]
temp_field.dump()

# %%
# 4. Convert the units to degree celsius for the temperature field:
temp_field.units = "degC"
temp_field.get_property("units")

# %%
# 5. Digitize the PM2.5 mass concentration and 2-metre temperature fields.
# This step counts the number of values in each of the 10 equally sized bins
# spanning the range of the values. The ``'return_bins=True'`` argument makes 
# sure that the calculated bins are also returned:
pm25_indices, pm25_bins = pm25_field.digitize(10, return_bins=True)
temp_indices, temp_bins = temp_field.digitize(10, return_bins=True)

# %%
# 6. Create a joint histogram of the digitized fields:
joint_histogram = cf.histogram(pm25_indices, temp_indices)

# %%
# 7. Get histogram data from the ``joint_histogram``:
histogram_data = joint_histogram.array

# %%
# 8. Calculate bin centers for PM2.5 and temperature bins:
pm25_bin_centers = [(a + b) / 2 for a, b in pm25_bins]
temp_bin_centers = [(a + b) / 2 for a, b in temp_bins]


# %%
# 9. Create grids for PM2.5 and temperature bins using `numpy.meshgrid
# <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_:
temp_grid, pm25_grid = np.meshgrid(temp_bin_centers, pm25_bin_centers)

# %%
# 10. Plot the joint histogram using `matplotlib.pyplot.pcolormesh
# <https://matplotlib.org/stable/api/_as_gen/
# matplotlib.pyplot.pcolormesh.html>`_. Use `cf.Field.unique
# <https://ncas-cms.github.io/cf-python/method/cf.Field.unique.html>`_ to get
# the unique data array values and show the bin boundaries as ticks on the
# plot. ``'style='plain'`` in `matplotlib.axes.Axes.ticklabel_format
# <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html>`_ 
# disables the scientific notation on the y-axis:
plt.pcolormesh(temp_grid, pm25_grid, histogram_data, cmap="viridis")
plt.xlabel("2-metre Temperature (degC)")
plt.ylabel("PM2.5 Mass Concentration (kg m**-3)")
plt.xticks(temp_bins.unique().array, rotation=45)
plt.yticks(pm25_bins.unique().array)
plt.colorbar(label="Frequency")
plt.title("Joint Histogram of PM2.5 and 2-metre Temperature", y=1.05)
plt.ticklabel_format(style="plain")
plt.show()
