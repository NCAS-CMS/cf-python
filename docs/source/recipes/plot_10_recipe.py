"""
Calculating and plotting the relative vorticity
===============================================

Vorticity, the microscopic measure of rotation in a fluid, is a vector field defined as the curl of velocity `(James R. Holton, Gregory J. Hakim, An Introduction to Dynamic Meteorology, 2013, Elsevier : Academic Press p95-125) <https://www.sciencedirect.com/science/article/pii/B9780123848666000040>`_. In this recipe, we will be calculating and plotting the relative vorticity from the wind components.
"""

# %%
# 1. Import cf-python and cf-plot:

import cf
import cfplot as cfp

# %%
# 2. Read the field constructs:
f = cf.read("~/recipes/ERA5_monthly_averaged_pressure_levels.nc")
print(f)

# %%
# 3. Select wind components and look at their contents:
u = f[1]
print(u)

# %%

v = f[2]
print(v)

# %%
# 4. Create a date-time object for the required time period:
jan_2023 = cf.year(2023) & cf.month(1)

# %%
# 5. The wind components are subspaced for January 2023 while selecting one of the `experiment version` dimensions:
u2 = u.subspace(T=jan_2023)[:,0,:,:]
v2 = v.subspace(T=jan_2023)[:,0,:,:]

# %%
# 6. The relative vorticity is calculated using `cf.curl_xy 
# <https://ncas-cms.github.io/cf-python/function/cf.curl_xy.html>`_ and 
# plotted using `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_. 
# Setting ``radius="earth"`` in the the `cf.curl_xy 
# <https://ncas-cms.github.io/cf-python/function/cf.curl_xy.html>`_ function 
# takes into account the Earth's spherical geometry when calculating the 
# spatial derivatives in the horizontal directions, leading to a more accurate 
# representation of the relative vorticity: 

rv = cf.curl_xy(u2, v2, radius="earth")
cfp.con(rv, lines=False, title="Relative Vorticity")

# %%
# 7. The discontinuities in the calculated vorticity field along the periodic 
# boundary conditions in the longitudinal direction (wrap-around of data at 0° 
# longitude and 180° longitude) is a result of the X axis not being cyclic. 
# This can be corrected by setting ``'x_wrap=True'`` while calculating the 
# relative vorticity. Setting ``rv.units = "s-1"``, ensures that the units of 
# the relative vorticity field are consistent with the calculation and the 
# physical interpretation of the quantity:

u2.iscyclic('X')

# %%

rv = cf.curl_xy(u2, v2, x_wrap=True, radius="earth")
rv.units = "s-1"
print(rv)

# %%

cfp.con(rv, lines=False, title="Relative Vorticity")
