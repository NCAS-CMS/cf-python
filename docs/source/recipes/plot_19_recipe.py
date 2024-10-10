"""
Recipe 1: Calculating and Plotting Seasonal Mean Pressure at Mean Sea Level

Objective: Calculate and plot the seasonal mean pressure at mean sea level
from ensemble simulation data for 1941.
"""

import cfplot as cfp
import matplotlib.pyplot as plt

import cf

# 1. Load the dataset
f = cf.read("~/recipes_break/ERA5_monthly_averaged_SST.nc")
sst = f[0]  # Select the SST variable

# Collapse data by area mean (average over spatial dimensions)
am_max = sst.collapse("area: maximum")  # equivalent to "X Y: mean"
am_min = sst.collapse("area: minimum")

# Reduce all timeseries down to just 1980+ since there are some data
# quality issues before 1970
am_max = am_max.subspace(T=cf.ge(cf.dt("1980-01-01")))
am_min = am_min.subspace(T=cf.ge(cf.dt("1980-01-01")))

# TODO COMMENT
colours_seasons_map = {
    "red": (cf.mam(), "Mean across MAM: March, April and May"),
    "blue": (cf.jja(), "Mean across JJA: June, July and August"),
    "green": (cf.son(), "Mean across SON: September, October and November"),
    "purple": (cf.djf(), "Mean across DJF: December, January and February"),
}

cfp.gopen(
    rows=2, columns=1, bottom=0.1, top=0.85, file="global_avg_sst_plot.png"
)

# Put maxima subplot at top since these values are higher, given
# increasing x axis
cfp.gpos(1)
plt.suptitle(
    "Global Average Sea Surface Temperature monthly minima\nand maxima "
    "including seasonal means of these extrema",
    fontsize=18,
)
# Set limits manually only to allow space so the legend doesn't overlap the
# data, which isn't possible purely from positioning it anywhere
cfp.gset(xmin="1980-01-01", xmax="2022-12-01", ymin=304, ymax=312)
for colour, season_query in colours_seasons_map.items():
    query_on_season, season_description = season_query
    am_sub = am_max.collapse("T: mean", group=query_on_season)
    cfp.lineplot(
        am_sub,
        color=colour,
        markeredgecolor=colour,
        marker="o",
        label=season_description,
        title="Maxima per month or season",
    )
cfp.lineplot(
    am_max,
    color="grey",
    xlabel="",
    label="All months",
)

# Minima subplot below the maxima one
cfp.gpos(2)
cfp.gset(xmin="1980-01-01", xmax="2022-12-01", ymin=269, ymax=272)
for colour, season_query in colours_seasons_map.items():
    query_on_season, season_description = season_query
    am_sub = am_min.collapse("T: mean", group=query_on_season)
    cfp.lineplot(
        am_sub,
        color=colour,
        markeredgecolor=colour,
        marker="o",
        xlabel="",
        title="Minima per month or season",
    )
cfp.lineplot(
    am_min,
    color="grey",
)

cfp.gclose()
