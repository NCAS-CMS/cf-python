"""
Plotting wind vectors overlaid on precipitation data
====================================================

In this recipe we will plot wind vectors, derived from northward and eastward wind components, over precipitation data.
"""

# %%
# 1. Import cf-python and cf-plot:

import cfplot as cfp

import cf

# %%
# 2. Read the field constructs:

f1 = cf.read("~/recipes/northward.nc")
print(f1)

# %%

f2 = cf.read("~/recipes/eastward.nc")
print(f2)

# %%

f3 = cf.read("~/recipes/monthly_precipitation.nc")
print(f3)

# %%
# 3. Select wind vectors and precipitation data by index and look at their contents:
v = f1[0]
print(v)

# %%

u = f2[0]
print(u)

# %%

pre = f3[0]
print(pre)

# %%
# 4. Plot the wind vectors on top of precipitation data for June 1995 by creating a subspace with a date-time object and using `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_. Here `cfplot.gopen <http://ajheaps.github.io/cf-plot/gopen.html>`_ is used to define the parts of the plot area, which is closed by `cfplot.gclose <http://ajheaps.github.io/cf-plot/gclose.html>`_; `cfplot.cscale <http://ajheaps.github.io/cf-plot/cscale.html>`_ is used to choose one of the colour maps amongst many available; `cfplot.levs <http://ajheaps.github.io/cf-plot/levs.html>`_ is used to set the contour levels for precipitation data; and `cfplot.vect <http://ajheaps.github.io/cf-plot/vect.html>`_ is used to plot the wind vectors for June 1995:
june_95 = cf.year(1995) & cf.month(6)
cfp.gopen()
cfp.cscale("precip4_11lev")
cfp.levs(step=100)
cfp.con(
    pre.subspace(T=june_95),
    lines=False,
    title="June 1995 monthly global precipitation",
)
cfp.vect(
    u=u.subspace(T=june_95),
    v=v.subspace(T=june_95),
    key_length=10,
    scale=35,
    stride=5,
)
cfp.gclose()
