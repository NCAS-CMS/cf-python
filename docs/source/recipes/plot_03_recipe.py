"""
Plotting global mean temperatures spatially
===========================================

In this recipe, we will plot the global mean temperature spatially.
"""

# %%
# 1. Import cf-python and cf-plot:

import cfplot as cfp

import cf

# %%
# 2. Read the field constructs:

f = cf.read("~/recipes/cru_ts4.06.1901.2021.tmp.dat.nc")
print(f)

# %%
# 3. Select near surface temperature by index and look at its contents:

temp = f[1]
print(temp)

# %%
# 4. Average the monthly mean surface temperature values by the time axis using the collapse method:

global_avg = temp.collapse("mean", axes="long_name=time")

# %%
# 5. Plot the global mean surface temperatures:

cfp.con(global_avg, lines=False, title="Global mean surface temperature")
