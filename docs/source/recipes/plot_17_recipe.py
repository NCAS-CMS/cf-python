"""
Plotting contour subplots with different colour maps/scales
===========================================================

In this recipe, we will plot data with different colour maps to illustrate
the importance of choosing the correct one for a plot. This is to ensure
the use of perceptually uniform scales and avoid unintended bias.

"""

# %%
# 1. Import cf-python and cf-plot:

import cfplot as cfp

import cf

# %%
# 2. Read the field in:
# Here I've used sample data ggap.nc (and later pressure=850), but you
# could use tas_A1.nc (with time=15)
PATH="~/git-repos/cf-plot/cfplot/test/cfplot_data"
f = cf.read(f"{PATH}/ggap.nc")[0]

# %%
# 3. Choose a set of predefined colour scales to view (based on NCAR)
# Choose a set of predefined colour scales to view (based on NCAR)
# You could also choose your own from
# https://ncas-cms.github.io/cf-plot/build/colour_scales.html
# Simply change the name in quotes and ensure the
# number of rows * number of columns = number of colour scales

# %%
# a. Perceptually uniform colour scales, with no zero value
colour_scale = ["viridis", "magma", "inferno", "plasma", "parula", "gray"]
cfp.gopen(rows=2, columns=3, bottom=0.2)

# %%
# b. NCAR Command Language - Enhanced to help with colour blindness
colour_scale = [
    "StepSeq25",
    "posneg_2",
    "posneg_1",
    "BlueDarkOrange18",
    "BlueDarkRed18",
    "GreenMagenta16",
    "BlueGreen14",
    "BrownBlue12",
    "Cat12",
]
cfp.gopen(rows=3, columns=3, bottom=0.1)

# %%
# c. Orography/bathymetry colour scales
# These are used to show the shape/contour of landmasses, bear in mind the
# example data we use is with pressure so doesnt accurately represent this.
# You could instead use cfp.cscale('wiki_2_0', ncols=16, below=2, above=14)
# or any other orography colour scale in a similar way.
colour_scale = [
    "os250kmetres",
    "wiki_1_0_2",
    "wiki_1_0_3",
    "wiki_2_0",
    "wiki_2_0_reduced",
    "arctic",
]
cfp.gopen(rows=2, columns=3, bottom=0.2, file="ColourPlot.png")

# %%
# 4. We then use a for loop to cycle through all the different colour maps:
# Only gpos has 1 added because it can only take 1 as its first value,
# otherwise there are errors.
for i, colour_scale in enumerate(colour_scale):
    cfp.gpos(i + 1)
    cfp.mapset(proj="cyl")
    cfp.cscale(colour_scale)
    if i == len(colour_scale) + 1:
        cfp.con(
            f.subspace(pressure=850),
            lines=False,
            title=colour_scale,
            colorbar_position=[0.1, 0.1, 0.8, 0.02],
            colorbar_orientation="horizontal",
        )
    else:
        cfp.con(f.subspace(pressure=850), title=colour_scale, lines=False)
cfp.gclose(view=True)
