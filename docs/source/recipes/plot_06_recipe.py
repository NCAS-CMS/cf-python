"""
Converting from rotated latitude-longitude to regular latitude-longitude
========================================================================

In this recipe, we will be regridding from a rotated latitude-longitude source domain to a regular latitude-longitude destination domain.
"""

# %%
# 1. Import cf-python, cf-plot and numpy:

import cfplot as cfp

import cf

# %%
# 2. Read the field constructs using read function:

f = cf.read("~/recipes/au952a.pd20510414.pp")
print(f)

# %%
# 3. Select the field by index and print its description to show properties of all constructs:

gust = f[0]
gust.dump()

# %%
# 4. Access the time coordinate of the gust field and retrieve the datetime values of the time coordinate:

print(gust.coordinate("time").datetime_array)

# %%
# 5. Create a new instance of the `cf.dt` class with a specified year, month, day, hour, minute, second and microsecond. Then store the result in the variable ``test``:
test = cf.dt(2051, 4, 14, 1, 30, 0, 0)
print(test)

# %%
# 6. Plot the wind gust by creating a subspace for the specified variable ``test`` using `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_. Here `cfplot.mapset <http://ajheaps.github.io/cf-plot/mapset.html>`_ is used to set the mapping parameters like setting the map resolution to 50m:
cfp.mapset(resolution="50m")
cfp.con(gust.subspace(T=test), lines=False)

# %%
# 7. To see the rotated pole data on the native grid, the above steps are repeated and projection is set to rotated in `cfplot.mapset <http://ajheaps.github.io/cf-plot/mapset.html>`_:
cfp.mapset(resolution="50m", proj="rotated")
cfp.con(gust.subspace(T=test), lines=False)

# %%
# 8. Create dimension coordinates for the destination grid with the latitude and
# longitude values for Europe. `cf.Domain.create_regular
# <https://ncas-cms.github.io/cf-python/docs/method/cf.Domain.create_regular.html>`_
# method is used to
# create a regular grid with longitudes and latitudes. Spherical regridding is
# then performed on the gust variable by passing the target domain as argument.
# The method also takes an argument ``'linear'`` which specifies the type of
# regridding method to use. The description of the ``regridded_data`` is finally
# printed to show properties of all its constructs:

target_domain = cf.Domain.create_regular((-25, 45, 10), (32, 72, 10))
regridded_data = gust.regrids(target_domain, "linear")
regridded_data.dump()

# %%
# 9. Step 6 is similarly repeated for the ``regridded_data`` to plot the wind gust on a regular latitude-longitude domain:
cfp.mapset(resolution="50m")
cfp.con(regridded_data.subspace(T=test), lines=False)
