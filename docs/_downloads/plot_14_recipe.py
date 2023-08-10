"""
Overlay Geopotential height contours over Temperature anomalies
===============================================================

In this recipe, we will overlay Geopotential height contours over Temperature
anomalies to help analyse meteorological conditions during July 2018,
specifically focusing on the significant concurrent extreme events that occurred
during the 2018 boreal spring/summer season in the Northern Hemisphere.

"""

# %%
# 1. Import cf-python and cf-plot:
import cfplot as cfp

import cf

# %%
# 2. Read and select the 200 hpa geopotential by index and look at its contents:
gp = cf.read("~/recipes/ERA5_monthly_averaged_z200.nc")[0]
print(gp)

# %%
# 3. Convert the geopotential data to geopotential height by dividing it by the
# acceleration due to gravity (approximated as 9.81 :math:`m \cdot {s}^{-2}`):
gph = gp / 9.81

# %%
# 4. Subset the geopotential height to extract data specifically for July 2018,
# a significant month due to heat extremes and heavy rainfall:
gph_july = gph.subspace(T=cf.month(7) & cf.year(2018)).squeeze()

# %%
# 5. Plot contour lines of this geopotential height for July 2018. Here:
#
# - `cfplot.gopen <http://ajheaps.github.io/cf-plot/gopen.html>`_ is used to
#   define the parts of the plot area, which is closed by
#   `cfplot.gclose <http://ajheaps.github.io/cf-plot/gclose.html>`_;
# - `cfplot.mapset <https://ajheaps.github.io/cf-plot/mapset.html>`_ is used to
#   set the map projection to North Polar Stereographic;
# - `cfplot.setvars <http://ajheaps.github.io/cf-plot/setvars.html>`_ is used to
#   set various attributes of the plot, like setting the thickness of the lines
#   that represent continents;
# - `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_ plots the contour
#   lines representing the 200 hpa geopotential height values without filling
#   between the contour lines (``fill=False``) and no colour bar
#   (``colorbar=False``);
# - `cfplot.levs <https://ajheaps.github.io/cf-plot/levs.html>`_ is used to
#   specify two contour levels, 12000 and 12300 m, corresponding to the
#   approximate polar-front jet and subtropical jet respectively;
# - `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_ is again used to
#   plot the contour lines for polar-front jet and subtropical jet with a
#   thicker line width;
# - `cfp.plotvars.mymap.stock_img() <https://scitools.org.uk/cartopy/docs/v0.15/matplotlib/geoaxes.html#cartopy.mpl.geoaxes.GeoAxes.stock_img>`_
#   then finally visualises the Earth's surface in cf-plot's
#   ``cfp.plotvars.mymap`` plot object:
cfp.gopen()
cfp.mapset(proj="npstere")
cfp.setvars(continent_thickness=0.5)

cfp.con(
    f=gph_july,
    fill=False,
    lines=True,
    line_labels=False,
    colors="black",
    linewidths=1,
    colorbar=False,
)

cfp.levs(manual=[12000, 12300])
cfp.con(
    f=gph_july,
    fill=False,
    lines=True,
    colors="black",
    linewidths=3.0,
    colorbar=False,
)

cfp.plotvars.mymap.stock_img()
cfp.gclose()

# %%
# 6. Read and select the 2-metre temperature by index and look at its contents:
t2m = cf.read("~/recipes/ERA5_monthly_averaged_t2m.nc")[0]
print(t2m)

# %%
# 7. Set the units from Kelvin to degrees Celsius:
t2m.Units = cf.Units("degreesC")

# %%
# 8. Extract a subset for July across the years for ``t2m``:
t2m_july = t2m.subspace(T=cf.month(7))

# %%
# 9. The 2-meter temperature climatology is then calculated for the month of
# July over the period from 1981 to 2010, which provides a baseline against
# which anomalies in later years are compared:
t2m_july_climatology = t2m_july.subspace(
    T=cf.year(cf.wi(1981, 2010))
).collapse("T: mean")

# %%
# 10. Calculate the temperature anomaly for the month of July in the year 2018
# relative to the climatological baseline (``t2m_july_climatology``). This
# indicates how much the temperatures for that month in that year deviated from
# the long-term average for July across the 1981-2010 period:
t2m_july_anomaly_2018 = (
    t2m_july.subspace(T=cf.year(2018)).squeeze() - t2m_july_climatology
)

# %%
# 11.
# The July 2018 season experienced extreme heat in many parts of the Northern
# Hemisphere. This period's extreme events were related to unusual
# meteorological conditions, particularly abnormalities in the jet stream. To
# provide an insight into the atmospheric conditions, the temperature anomalies
# and the geopotential height contours are plotted using cf-plot. Here:
#
# - `cfplot.gopen <http://ajheaps.github.io/cf-plot/gopen.html>`_ is used to
#   define the parts of the plot area, which is closed by
#   `cfplot.gclose <http://ajheaps.github.io/cf-plot/gclose.html>`_;
# - `cfplot.mapset <https://ajheaps.github.io/cf-plot/mapset.html>`_ is used to
#   set the map projection to Robinson;
# - `cfplot.setvars <http://ajheaps.github.io/cf-plot/setvars.html>`_ is used to
#   set various attributes of the plot, like setting the thickness of the lines
#   that represent continents and master title properties;
# - `cfplot.levs <https://ajheaps.github.io/cf-plot/levs.html>`_ is used to
#   specify the contour levels for temperature anomalies, starting from -2 to 2
#   with an interval of 0.5;
# - `cfplot.cscale <http://ajheaps.github.io/cf-plot/cscale.html>`_ is used to
#   choose one of the colour maps amongst many available;
# - `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_ plots contour fill
#   of temperature anomalies without contour lines (``lines=False``);
# - `cfplot.levs() <https://ajheaps.github.io/cf-plot/levs.html>`_ is used to
#   reset contour levels to default after which the steps to plot the contour
#   lines representing the 200 hpa geopotential height values, the approximate
#   polar-front jet and subtropical jet from Step 5 are repeated:
cfp.gopen()
cfp.mapset(proj="robin")
cfp.setvars(
    continent_thickness=0.5,
    master_title="July 2018",
    master_title_fontsize=22,
    master_title_location=[0.53, 0.83],
)

cfp.levs(min=-2, max=2, step=0.5)
cfp.cscale("temp_19lev")
cfp.con(
    f=t2m_july_anomaly_2018,
    lines=False,
    colorbar_title="Temperature anomaly relative to 1981-2010 ($\degree C$)",
    colorbar_fontsize=13,
    colorbar_thick=0.04,
)

cfp.levs()
cfp.con(
    f=gph_july,
    fill=False,
    lines=True,
    line_labels=False,
    colors="black",
    linewidths=1,
    colorbar=False,
)

cfp.levs(manual=[12000, 12300])
cfp.con(
    f=gph_july,
    fill=False,
    lines=True,
    colors="black",
    linewidths=3.0,
    colorbar=False,
)

cfp.gclose()
