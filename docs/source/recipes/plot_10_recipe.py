"""
Calculating and plotting the relative vorticity
===============================================

Vorticity, the microscopic measure of rotation in a fluid, is a vector field
defined as the curl of velocity
`(James R. Holton, Gregory J. Hakim, An Introduction to Dynamic Meteorology,
2013, Elsevier : Academic Press p95-125)
<https://www.sciencedirect.com/science/article/pii/B9780123848666000040>`_.
In this recipe, we will be calculating and plotting the relative vorticity
from the wind components.
"""

# %%
# 1. Import cf-python and cf-plot:

import cfplot as cfp

import cf

# %%
# 2. Read the field constructs:
f = cf.read("~/recipes/ERA5_monthly_averaged_pressure_levels.nc")
print(f)

# %%
# 3. Select wind components and look at their contents:
u = f.select_field("eastward_wind")
print(u)

# %%

v = f.select_field("northward_wind")
print(v)

# %%
# 4. Create a date-time object for the required time period:
jan_2023 = cf.year(2023) & cf.month(1)

# %%
# 5. The relative vorticity is calculated using `cf.curl_xy
# <https://ncas-cms.github.io/cf-python/function/cf.curl_xy.html>`_ and
# plotted using `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_.
# The ``with cf.relaxed_identities(True)`` context manager statement prevents
# the curl opereration broadcasting across the two ``expver`` dimensions because
# it can't be certain that they are the same as they lack the standardised
# metadata. Setting
# ``cf.relaxed_identities(True)`` allows the ``long_name`` to be treated
# as standardised metadata. Since the horizontal coordinates are latitude and
# longitude, the
# `cf.curl_xy <https://ncas-cms.github.io/cf-python/function/cf.curl_xy.html>`_
# function automatically accounts for the Earth's spherical geometry when
# calculating the spatial derivatives in the horizontal directions, and for this
# it requires the Earth's radius. In this case the radius is not stored in the
# wind fields, so must be provided by setting ``radius="earth"`` keyword
# parameter. While plotting, the relative vorticity is subspaced for January
# 2023 and one of the `experiment versions` using the dictionary unpacking
# operator (``**``) as there is an equal to sign in the identifier
# (``"long_name=expver"``):

with cf.relaxed_identities(True):
    rv = cf.curl_xy(u, v, radius="earth")

cfp.con(
    rv.subspace(T=jan_2023, **{"long_name=expver": 1}),
    lines=False,
    title="Relative Vorticity",
)

# %%
# 6. Although the X axis is cyclic, it is not recognised as such, owing to the
# fact that the longitude coordinate bounds are missing. This results in
# discontinuities in the calculated vorticity field on the plot at the
# wrap-around location of 0 degrees east. The cyclicity could either be set on
# the field itself or just in the curl command  by setting ``'x_wrap=True'``
# while calculating the relative vorticity. Setting ``rv.units = "s-1"``,
# ensures that the units of the relative vorticity field are consistent with
# the calculation and the physical interpretation of the quantity:

print(v.coordinate("X").has_bounds())

# %%

with cf.relaxed_identities(True):
    rv = cf.curl_xy(u, v, x_wrap=True, radius="earth")

rv.units = "s-1"
print(rv)

# %%

cfp.con(
    rv.subspace(T=jan_2023, **{"long_name=expver": 1}),
    lines=False,
    title="Relative Vorticity",
)
