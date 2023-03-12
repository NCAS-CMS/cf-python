"""
Comparing two datasets with different resolutions using regridding
==================================================================

In this recipe, we will regrid two different datasets with different resolutions. An example use case could be one where the observational dataset with a higher resolution needs to be regridded to that of the model dataset so that they can be compared with each other.
"""

# %%
# 1. Import cf-python:

import cf

# %%
# 2. Read the field constructs:

obs = cf.read("~/recipes/cru_ts4.06.1901.2021.tmp.dat.nc", chunks=None)
print(obs)

# %%

model = cf.read(
    "~/recipes/tas_Amon_HadGEM3-GC3-1_hist-1p0_r3i1p1f2_gn_185001-201412.nc"
)
print(model)

# %%
# 3. Select observation and model temperature fields by identity and index respectively, and look at their contents:

obs_temp = obs.select_field("long_name=near-surface temperature")
print(obs_temp)

# %%

model_temp = model[0]
print(model_temp)

# %%
# 4. Regrid observational data to that of the model data and create a new low resolution observational data using bilinear interpolation:
obs_temp_regrid = obs_temp.regrids(model_temp, method="linear")
print(obs_temp_regrid)
