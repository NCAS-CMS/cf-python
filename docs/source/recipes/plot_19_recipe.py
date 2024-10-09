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
am = sst.collapse("area: mean")  # equivalent to "X Y: mean"
am.squeeze(inplace=True)

# Check available coordinates (already found 'dimensioncoordinate0' as the
# time coordinate)
print("Available coordinates:", am.coordinates())

am_dim_key, am_data = am.coordinate("dimensioncoordinate0", item=True)
am_sub = am.subspace(**{am_dim_key: cf.mam()})

cfp.gopen(file="global_avg_sst_plot.png")
cfp.lineplot(
    am,
    color="blue",
    title="Global Average Sea Surface Temperature",
    ylabel="Temperature (K)",
    xlabel="Time"
)
cfp.lineplot(
    am_sub,
    color="red",
)
cfp.gclose()
