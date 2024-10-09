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
#am.squeeze(inplace=True)cf.seasons()
print("AM SEASONAL IS", am_min, am_max)
# REDUCE TO TEST
am_min = am_min[-100:]  # final 100 points
am_max = am_max[-100:]  # final 100 points

# Check available coordinates (already found 'dimensioncoordinate0' as the
# time coordinate)
###print("Available coordinates:", am.coordinates())

###am_dim_key, am_data = am.coordinate("dimensioncoordinate0", item=True)
am_sub_1 = am_min.collapse("T: mean", group=cf.mam())
am_sub_2 = am_min.collapse("T: mean", group=cf.jja())
am_sub_3 = am_min.collapse("T: mean", group=cf.son())
am_sub_4 = am_min.collapse("T: mean", group=cf.djf())
am_sub_5 = am_max.collapse("T: mean", group=cf.mam())
am_sub_6 = am_max.collapse("T: mean", group=cf.jja())
am_sub_7 = am_max.collapse("T: mean", group=cf.son())
am_sub_8 = am_max.collapse("T: mean", group=cf.djf())



"""
am_sub_1 = am.subspace(**{am_dim_key: cf.mam()})
am_sub_2 = am.subspace(**{am_dim_key: cf.month(3)})
am_sub_3 = am.subspace(**{am_dim_key: cf.month(4)})
am_sub_4 = am.subspace(**{am_dim_key: cf.month(5)})
am_sub_2 = am_sub_2 - am_sub_1
am_sub_3 = am_sub_3 - am_sub_1
am_sub_4 = am_sub_4 - am_sub_1
"""


cfp.gopen(file="global_avg_sst_plot.png")
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
)
cfp.lineplot(
    am_sub_2,
    color="green",
)
cfp.lineplot(
    am_sub_3,
    color="blue",
)
cfp.lineplot(
    am_sub_4,
    color="purple",
)
cfp.lineplot(
    am_sub_5,
    color="red",
)
cfp.lineplot(
    am_sub_6,
    color="green",
)
cfp.lineplot(
    am_sub_7,
    color="blue",
)
cfp.lineplot(
    am_sub_8,
    color="purple",
)
cfp.gclose()
