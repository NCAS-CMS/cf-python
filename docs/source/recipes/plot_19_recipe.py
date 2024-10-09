"""
Recipe 1: Calculating and Plotting Seasonal Mean Pressure at Mean Sea Level

Objective: Calculate and plot the seasonal mean pressure at mean sea level
from ensemble simulation data for 1941.
"""

import cfplot as cfp
import cf
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

# 1. Load the dataset
f = cf.read("~/recipes_break/ERA5_monthly_averaged_SST.nc")
sst = f[0]  # Select the SST variable

# Collapse data by area mean (average over spatial dimensions)
am_max = sst.collapse("area: maximum")  # equivalent to "X Y: mean"
am_min = sst.collapse("area: minimum")  # equivalent to "X Y: mean"
print("AM SEASONAL IS", am_min, am_max)

# Reduce all timeseries down to just 1980+ since there are some data
# quality issues before 1970
am_max = am_max.subspace(T=cf.ge(cf.dt("1980-01-01")))
am_min = am_min.subspace(T=cf.ge(cf.dt("1980-01-01")))
print("FINAL FIELDS ARE", am_max, am_min)

am_sub_1 = am_min.collapse("T: mean", group=cf.mam())
am_sub_2 = am_min.collapse("T: mean", group=cf.jja())
am_sub_3 = am_min.collapse("T: mean", group=cf.son())
am_sub_4 = am_min.collapse("T: mean", group=cf.djf())

am_sub_5 = am_max.collapse("T: mean", group=cf.mam())
am_sub_6 = am_max.collapse("T: mean", group=cf.jja())
am_sub_7 = am_max.collapse("T: mean", group=cf.son())
am_sub_8 = am_max.collapse("T: mean", group=cf.djf())


cfp.gopen(rows=2, columns=1, bottom=0.2, file="global_avg_sst_plot.png")

# Put maxima subplot at top since these values are higher, given
# increasing x axis
xticks = list(range(1980, 2024))
xlabels = [None for i in xticks]

cfp.gpos(1)
cfp.lineplot(
    am_max,
    color="grey",
    xlabel="",
    #xticks=xticks,
    #xticklabels=xlabels,
)
cfp.lineplot(
    am_sub_5,
    color="red",
    markeredgecolor="red",
    marker="o",
    xlabel="",
    #xticks=xticks,
    #xticklabels=xlabels,
)
cfp.lineplot(
    am_sub_6,
    color="green",
    markeredgecolor="green",
    marker="o",
    xlabel="",
    #xticks=xticks,
    #xticklabels=xlabels,
)
cfp.lineplot(
    am_sub_7,
    color="blue",
    markeredgecolor="blue",
    marker="o",
    xlabel="",
    #xticks=xticks,
    #xticklabels=xlabels,
)
cfp.lineplot(
    am_sub_8,
    color="purple",
    markeredgecolor="purple",
    marker="o",
    xlabel="",
    #xticks=xticks,
    #xticklabels=xlabels,
)

# Minima subplot below the maxima one
cfp.gpos(2)
cfp.lineplot(
    am_min,
    color="grey",
)
#cfp.lineplot(
#    am,
#    color="blue",
#    title="Global Average Sea Surface Temperature",
#    ylabel="Temperature (K)",
#    xlabel="Time"
#)
cfp.lineplot(
    am_sub_1,
    color="red",
    markeredgecolor="red",
    marker="o"
)
cfp.lineplot(
    am_sub_2,
    color="green",
    markeredgecolor="green",
    marker="o"
)
cfp.lineplot(
    am_sub_3,
    color="blue",
    markeredgecolor="blue",
    marker="o"
)
cfp.lineplot(
    am_sub_4,
    color="purple",
    markeredgecolor="purple",
    marker="o"
)

cfp.gclose()
