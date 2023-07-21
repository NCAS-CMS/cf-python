"""
Using mask to plot Aerosol Optical Depth
========================================

In this recipe, we will make use of a
`masked array
<https://ncas-cms.github.io/cf-python/constant.html#cf.cf.masked>`_
to plot the `high-quality` retrieval of Aerosol Optical Depth (AOD) from all other
retrievals.

"""

# %%
# 1. Import cf-python, cf-plot and matplotlib.pyplot:

import cfplot as cfp
import matplotlib.pyplot as plt

import cf

# %%
# 2. Read the field constructs:
fl = cf.read(
    "~/recipes/JRR-AOD_v3r0_npp_s202012310752331_e202012310753573_c202100000000000.nc"
)
print(fl)

# %%
# 3. Select AOD from the field list by identity and look at the contents:
aod = fl.select_field("long_name=AOT at 0.55 micron for both ocean and land")
print(aod)

# %%
# 4. Select AOD retrieval quality by index and look at the quality flags:
quality = fl[13]
print(quality)

# %%
# 5. Select latitude and longitude dimensions by identities, with two different
# techniques:
lon = aod.coordinate("long_name=Longitude")
lat = aod.coordinate("Y")

# %%
# 6. Plot the AOD for all the retrievals using
# `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_. Here the argument
# ``'ptype'`` specifies the type of plot to use (latituide-longitude here) and
# the argument ``'lines=False'`` does not draw contour lines:
cfp.con(f=aod.array, x=lon.array, y=lat.array, ptype=1, lines=False)

# %%
# 7. Create a mask for AOD based on the quality of the retrieval. The
# ``'__ne__'`` method is an implementation of the ``!=`` operator. It is used to
# create a mask where all the `high-quality` AOD points (with the flag 0) are
# marked as ``False``, and all the other data points (medium quality, low
# quality, or no retrieval) are marked as ``True``:
mask = quality.array.__ne__(0)

# %%
# 8. Apply the mask to the AOD dataset. The ``'where'`` function takes the
# mask as an input and replaces all the values in the AOD dataset that
# correspond to ``True`` in the mask with a masked value using `cf.masked
# <https://ncas-cms.github.io/cf-python/constant.html#cf.cf.masked>`_.
# In this case, all AOD values that are not of `high-quality` (since they were
# marked as ``True`` in the mask) are masked. This means that the ``high``
# variable contains only the AOD data that was retrieved with `high-quality`:
high = aod.where(mask, cf.masked)

# %%
# 9. Now plot both the AOD from `high-quality` retrieval and all other retrievals 
# using `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_. Here:
#
# - `cfplot.gopen <http://ajheaps.github.io/cf-plot/gopen.html>`_ is used to 
#   define the parts of the plot area, specifying that the figure should have 
#   1 row and 2 columns, which is closed by 
#   `cfplot.gclose <http://ajheaps.github.io/cf-plot/gclose.html>`_;
# - `plt.suptitle <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html>`_
#   is used to add a title for the whole figure;
# - the subplots for plotting are selected using 
#   `cfplot.gpos <https://ajheaps.github.io/cf-plot/gpos.html>`_ after which 
#   `cfplot.mapset <https://ajheaps.github.io/cf-plot/mapset.html>`_ is used to 
#   set the map limits and resolution for the subplots;  
# - and as cf-plot stores the plot in a plot object with the name 
#   ``cfp.plotvars.plot``, country borders are added using normal 
#   `Cartopy operations <https://scitools.org.uk/cartopy/docs/latest/reference/index.html>`_ 
#   on the ``cfp.plotvars.mymap`` object:
import cartopy.feature as cfeature

cfp.gopen(rows=1, columns=2, bottom=0.2)
plt.suptitle("AOD for both ocean and land", fontsize=20)
cfp.gpos(1)
cfp.mapset(resolution="50m", lonmin=68, lonmax=98, latmin=7, latmax=36)
cfp.con(
    f=aod.array,
    x=lon.array,
    y=lat.array,
    ptype=1,
    lines=False,
    title="All retrievals",
    colorbar=None,
)
cfp.plotvars.mymap.add_feature(cfeature.BORDERS)
cfp.gpos(2)
cfp.mapset(resolution="50m", lonmin=68, lonmax=98, latmin=7, latmax=36)
cfp.con(
    f=high.array,
    x=lon.array,
    y=lat.array,
    ptype=1,
    lines=False,
    title="High quality retrieval",
    colorbar_position=[0.1, 0.20, 0.8, 0.02],
    colorbar_orientation="horizontal",
    colorbar_title="AOD at 0.55 $\mu$",
)
cfp.plotvars.mymap.add_feature(cfeature.BORDERS)
cfp.gclose()
