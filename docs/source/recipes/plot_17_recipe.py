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
import matplotlib.pyplot as plt

import cf

# %%
# 2. Read the field in:
# Here I've used sample data ggap.nc (and later pressure=850), but you
# could use tas_A1.nc (with time=15)
PATH = "~/git-repos/cf-plot/cfplot/test/cfplot_data"
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
colour_scale_pu = [
    "viridis",
    "magma",
    "inferno",
]  # "plasma", "parula", "gray"]


# %%
# b. NCAR Command Language - Enhanced to help with colour blindness
colour_scale_ncl = [
    "StepSeq25",
    "posneg_2",
    # "posneg_1",
    # "BlueDarkOrange18",
    # "BlueDarkRed18",
    "GreenMagenta16",
    # "BlueGreen14",
    # "BrownBlue12",
    # "Cat12",
]


# %%
# c. Orography/bathymetry colour scales
# These are used to show the shape/contour of landmasses, bear in mind the
# example data we use is with pressure so doesnt accurately represent this.
# You could instead use cfp.cscale('wiki_2_0', ncols=16, below=2, above=14)
# or any other orography colour scale in a similar way.
colour_scale_ob = [
    "os250kmetres",
    "wiki_1_0_2",
    # "wiki_1_0_3",
    # "wiki_2_0",
    # "wiki_2_0_reduced",
    "arctic",
]


# We plot each category of colourmap as columns, but given the gpos
# function positions subplots from left to right, row by row from the top,
# we need to interleave the values in a list. We can use zip to do this.
#
colour_scales_columns = [
    val
    for category in zip(colour_scale_pu, colour_scale_ncl, colour_scale_ob)
    for val in category
]

zip(colour_scale_pu, colour_scale_ncl, colour_scale_ob)
# %%
# 4. We then use a for loop to cycle through all the different colour maps:
# Only gpos has 1 added because it can only take 1 as its first value,
# otherwise there are errors.
cfp.gopen(rows=3, columns=3, bottom=0.1, top=0.85, file="ColourPlot.png")
plt.suptitle(
    (
        "Air temperature (K) at 850 mbar pressure shown in different "
        "colourmap categories"
    ),
    fontsize=18,
)
for i, colour_scale in enumerate(colour_scales_columns):
    cfp.gpos(i + 1)
    cfp.cscale(colour_scale)
    if i == len(colour_scale) + 1:
        # For the final plot, don't plot the colourbar across all subplots
        # as is the default
        cfp.con(
            f.subspace(pressure=850),
            lines=False,
            axes=False,
            colorbar_drawedges=False,
            colorbar_title=f"Shown in '{colour_scale}'",
            colorbar_fraction=0.03,
            colorbar_fontsize=11,
        )
    elif i < 3:
        if i == 0:
            set_title = "Perceptually uniform\ncolour maps"
        elif i == 1:
            set_title = (
                "NCL colour maps enhanced to \nhelp with colour blindness"
            )
        elif i == 2:
            set_title = "Orography/bathymetry\ncolour maps"

        cfp.con(
            f.subspace(pressure=850),
            lines=False,
            axes=False,
            title=set_title,
            colorbar_drawedges=False,
            colorbar_title=f"Shown in '{colour_scale}'",
            colorbar_fraction=0.03,
            colorbar_fontsize=11,
        )

    else:
        cfp.con(
            f.subspace(pressure=850),
            lines=False,
            axes=False,
            colorbar_drawedges=False,
            colorbar_title=f"Shown in '{colour_scale}'",
            colorbar_fontsize=11,
            colorbar_fraction=0.03,
        )

cfp.gclose(view=True)
