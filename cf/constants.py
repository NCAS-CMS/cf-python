import logging
import sys
from enum import Enum, auto
from tempfile import gettempdir

import numpy as np
from dask import config
from dask.utils import parse_bytes
from psutil import virtual_memory

from .units import Units

# --------------------------------------------------------------------
# Find the total amount of memory, in bytes
# --------------------------------------------------------------------
_TOTAL_MEMORY = float(virtual_memory().total)

_CHUNKSIZE = "128 MiB"
config.set({"array.chunk-size": _CHUNKSIZE})


"""A dictionary of useful constants.

Whilst the dictionary may be modified directly, it is safer to
retrieve and set the values with a function where one is
provided. This is due to interdependencies between some values.

Note ATOL and RTOL are constants that in essence belong in this dict,
but since they can be read and manipulated directly from cfdm, it is
safest to work with ``cfdm.constants.CONSTANTS['ATOL']`` (and 'RTOL'
equivalent) instead of storing separately and synchronising them here
in cf.

:Keys:

    TOTAL_MEMORY: `float`
      Find the total amount of physical memory (in bytes).

    CHUNKSIZE: `int`
      The chunk size (in bytes) for data storage and processing.

    TEMPDIR: `str`
      The location to store temporary files. By default it is the
      default directory used by the :mod:`tempfile` module.

    REGRID_LOGGING: `bool`
      Whether or not to enable `esmpy` logging. If it is logging is
      performed after every call to `esmpy`. By default logging is
      disabled.

    LOG_LEVEL: `str`
      The minimal level of seriousness for which log messages are
      shown. See `cf.log_level`.

"""
CONSTANTS = {
    "ATOL": sys.float_info.epsilon,
    "RTOL": sys.float_info.epsilon,
    "TEMPDIR": gettempdir(),
    "TOTAL_MEMORY": _TOTAL_MEMORY,
    "REGRID_LOGGING": False,
    "RELAXED_IDENTITIES": False,
    "LOG_LEVEL": logging.getLevelName(logging.getLogger().level),
    "BOUNDS_COMBINATION_MODE": "AND",
    "CHUNKSIZE": parse_bytes(_CHUNKSIZE),
}

masked = np.ma.masked

repr_prefix = "CF "
repr_suffix = ""

_stash2standard_name = {}

# ---------------------------------------------------------------------
# Coordinate reference constants TODO: turn these into functions
# ---------------------------------------------------------------------
cr_canonical_units = {
    "earth_radius": Units("m"),
    "grid_north_pole_latitude": Units("degrees_north"),
    "grid_north_pole_longitude": Units("degrees_east"),
    "inverse_flattening": Units("1"),
    "latitude_of_projection_origin": Units("degrees_north"),
    "longitude_of_central_meridian": Units("degrees_east"),
    "longitude_of_prime_meridian": Units("degrees_east"),
    "longitude_of_projection_origin": Units("degrees_east"),
    "north_pole_grid_longitude": Units("degrees"),
    "perspective_point_height": Units("m"),
    "scale_factor_at_central_meridian": Units("1"),
    "scale_factor_at_projection_origin": Units("1"),
    "semi_major_axis": Units("m"),
    "semi_minor_axis": Units("m"),
    "standard_parallel": Units("degrees_north"),
    "straight_vertical_longitude_from_pole": Units("degrees_north"),
}

cr_coordinates = {
    "grid_mapping_name:albers_conical_equal_area": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:azimuthal_equidistant": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:geostationary": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:lambert_azimuthal_equal_area": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:lambert_conformal_conic": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:lambert_cylindrical_equal_area": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:latitude_longitude": ("latitude", "longitude"),
    "grid_mapping_name:mercator": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:orthographic": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:polar_stereographic": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:rotated_latitude_longitude": (
        "grid_latitude",
        "grid_longitude",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:sinusoidal": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:stereographic": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "grid_mapping_name:transverse_mercator": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "standard_name:vertical_perspective": (
        "projection_x_coordinate",
        "projection_y_coordinate",
        "latitude",
        "longitude",
    ),
    "standard_name:atmosphere_ln_pressure_coordinate": (
        "atmosphere_ln_pressure_coordinate",
    ),
    "standard_name:atmosphere_sigma_coordinate": (
        "atmosphere_sigma_coordinate",
    ),
    "standard_name:atmosphere_hybrid_sigma_pressure_coordinate": (
        "atmosphere_hybrid_sigma_pressure_coordinate",
    ),
    "standard_name:atmosphere_hybrid_height_coordinate": (
        "atmosphere_hybrid_height_coordinate",
    ),
    "standard_name:atmosphere_sleve_coordinate": (
        "atmosphere_sleve_coordinate",
    ),
    "standard_name:ocean_sigma_coordinate": ("ocean_sigma_coordinate",),
    "standard_name:ocean_s_coordinate": ("ocean_s_coordinate",),
    "standard_name:ocean_sigma_z_coordinate": ("ocean_sigma_z_coordinate",),
    "standard_name:ocean_double_sigma_coordinate": (
        "ocean_double_sigma_coordinate",
    ),
}
# Coordinate conversion terms and their default values.
# Column  1 : Coordinate conversion term
# Columns 2+: Default values
# See appendices D and F in the CF conventions for details.
cr_default_values = {
    "a": 0.0,
    "b": 0.0,
    "b1": 0.0,
    "b2": 0.0,
    "depth": 0.0,
    "depth_c": 0.0,
    "eta": 0.0,
    "href": 0.0,
    "k_c": 0.0,
    "lev": 0.0,
    "longitude_of_prime_meridian": 0.0,
    "north_pole_grid_longitude": 0.0,
    "nsigma": 0.0,
    "orog": 0.0,
    "p0": 0.0,
    "ps": 0.0,
    "ptop": 0.0,
    "s": 0.0,
    "sigma": 0.0,
    "z1": 0.0,
    "z2": 0.0,
    "zlev": 0.0,
    "zsurf1": 0.0,
    "zsurf2": 0.0,
    "ztop": 0.0,
}


# --------------------------------------------------------------------
# Define the sets of standard names described by Table D.1 of the CF
# conventions.
#
# The element order of these tuple values is important. The standard
# names at each index location form a consistent set, as described
# Appendix D: Parametric Vertical Coordinates of the CF conventions.
# --------------------------------------------------------------------
formula_terms_D1 = {
    "computed_standard_name": (
        "altitude",
        "height_above_geopotential_datum",
        "height_above_reference_ellipsoid",
        "height_above_mean_sea_level",
    ),
    "eta": (
        "sea_surface_height_above_geoid",
        "sea_surface_height_above_geopotential_datum",
        "sea_surface_height_above_reference_ellipsoid",
        "sea_surface_height_above_mean_sea_level",
    ),
    "depth": (
        "sea_floor_depth_below_geoid",
        "sea_floor_depth_below_geopotential_datum",
        "sea_floor_depth_below_reference_ellipsoid",
        "sea_floor_depth_below_mean_sea_level",
    ),
    "zlev": (
        "altitude",
        "height_above_geopotential_datum",
        "height_above_reference_ellipsoid",
        "height_above_mean_sea_level",
    ),
}

# --------------------------------------------------------------------
# Define the standard names that are allowed for each formula term, as
# described in Appendix D: Parametric Vertical Coordinates of the CF
# conventions.
#
# A domain ancillary or coordinate construct may have any of the
# specified names. A value of None means that no standard name has
# been defined for that term.
# --------------------------------------------------------------------
formula_terms_standard_names = {
    "atmosphere_ln_pressure_coordinate": {
        "p0": ("reference_air_pressure_for_atmosphere_vertical_coordinate",),
        "lev": ("atmosphere_ln_pressure_coordinate",),
    },
    "atmosphere_sigma_coordinate": {
        "sigma": ("atmosphere_sigma_coordinate",),
        "ptop": ("air_pressure_at_top_of_atmosphere_model",),
        "ps": ("surface_air_pressure",),
    },
    "atmosphere_hybrid_sigma_pressure_coordinate": {
        "p0": ("reference_air_pressure_for_atmosphere_vertical_coordinate",),
        "ps": ("surface_air_pressure",),
        "a": (None,),
        "ap": (None,),
        "b": (None,),
    },
    "atmosphere_hybrid_height_coordinate": {
        "a": ("atmosphere_hybrid_height_coordinate",),
        "b": (None,),
        "orog": (
            "surface_altitude",
            "surface_height_above_geopotential_datum",
        ),
    },
    "atmosphere_sleve_coordinate": {
        "ztop": (
            "altitude_at_top_of_atmosphere_model",
            "height_above_geopotential_datum_at_top_of_atmosphere_model",
        ),
        "a": (None,),
        "b1": (None,),
        "b2": (None,),
        "zsurf1": (None,),
        "zsurf2": (None,),
    },
    "ocean_sigma_coordinate": {
        "eta": formula_terms_D1["eta"],
        "depth": formula_terms_D1["depth"],
        "sigma": ("ocean_sigma_coordinate",),
    },
    "ocean_s_coordinate": {
        "eta": formula_terms_D1["eta"],
        "depth": formula_terms_D1["depth"],
        "a": (None,),
        "b": (None,),
        "depth_c": (None,),
        "C": (None,),
        "s": ("ocean_s_coordinate",),
    },
    "ocean_s_coordinate_g1": {
        "eta": formula_terms_D1["eta"],
        "depth": formula_terms_D1["depth"],
        "depth_c": (None,),
        "C": (None,),
        "s": ("ocean_s_coordinate_g1",),
    },
    "ocean_s_coordinate_g2": {
        "eta": formula_terms_D1["eta"],
        "depth": formula_terms_D1["depth"],
        "depth_c": (None,),
        "C": (None,),
        "s": ("ocean_s_coordinate_g2",),
    },
    "ocean_sigma_z_coordinate": {
        "eta": formula_terms_D1["eta"],
        "depth": formula_terms_D1["depth"],
        "zlev": formula_terms_D1["zlev"],
        "nsigma": (None,),
        "depth_c": (None,),
        "sigma": ("ocean_sigma_z_coordinate",),
    },
    "ocean_double_sigma_coordinate": {
        "depth": formula_terms_D1["depth"],
        "a": (None,),
        "href": (None,),
        "k_c": (None,),
        "z1": (None,),
        "z2": (None,),
        "sigma": ("ocean_double_sigma_coordinate",),
    },
}

# --------------------------------------------------------------------
# Set the maximum number of dimensions allowed for each formula term,
# as described in Appendix D: Parametric Vertical Coordinates of the
# CF conventions.
#
# A given domain ancillary construct may have this number of
# dimensions or fewer.
# --------------------------------------------------------------------
formula_terms_max_dimensions = {
    "atmosphere_ln_pressure_coordinate": {"p0": 0, "lev": 1},  # (k)
    "atmosphere_sigma_coordinate": {
        "sigma": 1,  # (k)
        "ptop": 0,
        "ps": 3,  # (n,j,i)
    },
    "atmosphere_hybrid_sigma_pressure_coordinate": {
        "p0": 0,
        "ps": 3,  # (n,j,i)
        "a": 1,  # (k)
        "ap": 1,  # (k)
        "b": 1,  # (k)
    },
    "atmosphere_hybrid_height_coordinate": {
        "a": 1,  # (k)
        "b": 1,  # (k)
        "orog": 3,  # (n,j,i)
    },
    "atmosphere_sleve_coordinate": {
        "ztop": 0,
        "a": 1,  # (k)
        "b1": 1,  # (k)
        "b2": 1,  # (k)
        "zsurf1": 3,  # (n,j,i)
        "zsurf2": 3,  # (n,j,i)
    },
    "ocean_sigma_coordinate": {
        "eta": 3,  # (n,j,i)
        "depth": 2,  # (j,i)
        "sigma": 1,  # (k)
    },
    "ocean_s_coordinate": {
        "eta": 3,  # (n,j,i)
        "depth": 2,  # (j,i)
        "a": 0,
        "b": 0,
        "depth_c": 0,
        "C": 1,  # (k)
        "s": 1,  # (k)
    },
    "ocean_s_coordinate_g1": {
        "eta": 3,  # (n,j,i)
        "depth": 2,  # (j,i)
        "depth_c": 0,
        "C": 1,  # (k)
        "s": 1,  # (k)
    },
    "ocean_s_coordinate_g2": {
        "eta": 3,  # (n,j,i)
        "depth": 2,  # (j,i)
        "depth_c": 0,
        "C": 1,  # (k)
        "s": 1,  # (k)
    },
    "ocean_sigma_z_coordinate": {
        "eta": 3,  # (n,j,i)
        "depth": 2,  # (j,i)
        "zlev": 1,  # (k)
        "nsigma": 0,
        "depth_c": 0,
        "sigma": 1,  # (k)
    },
    "ocean_double_sigma_coordinate": {
        "depth": 2,  # (j,i)
        "a": 0,
        "href": 0,
        "k_c": 0,
        "z1": 0,
        "z2": 0,
        "sigma": 1,  # (k)
    },
}

# --------------------------------------------------------------------
# Define the computed standard name of the computed vertical
# coordinate values, as described in Appendix D: Parametric Vertical
# Coordinates of the CF conventions.
#
# A string value means that there can only be one computed standard
# name.
#
# A dictionary value means that the computed standard name depends on
# the standard name of the given term. For example, the computed
# standard name for 'atmosphere_sleve_coordinate' depends on the
# standard name of the 'ztop' term.
# --------------------------------------------------------------------
_D1_depth_mapping = {
    "sea_floor_depth_below_geoid": "altitude",
    "sea_floor_depth_below_geopotential_datum": "height_above_geopotential_ datum",
    "sea_floor_depth_below_reference_ellipsoid": "height_above_reference_ ellipsoid",
    "sea_floor_depth_below_mean_sea_level": "height_above_mean_sea_ level",
}

formula_terms_computed_standard_names = {
    "atmosphere_ln_pressure_coordinate": "air_pressure",
    "atmosphere_sigma_coordinate": "air_pressure",
    "atmosphere_hybrid_sigma_pressure_coordinate": "air_pressure",
    "atmosphere_hybrid_height_coordinate": {
        "orog": {
            "surface_altitude": "altitude",
            "surface_height_above_geopotential_datum": "height_above_geopotential_datum",
        }
    },
    "atmosphere_sleve_coordinate": {
        "ztop": {
            "altitude_at_top_of_atmosphere_model": "altitude",
            "height_above_geopotential_datum_at_top_of_atmosphere_model": "height_above_geopotential_datum",
        }
    },
    "ocean_sigma_coordinate": {"depth": _D1_depth_mapping},
    "ocean_s_coordinate": {"depth": _D1_depth_mapping},
    "ocean_s_coordinate_g1": {"depth": _D1_depth_mapping},
    "ocean_s_coordinate_g2": {"depth": _D1_depth_mapping},
    "ocean_sigma_z_coordinate": {"depth": _D1_depth_mapping},
    "ocean_double_sigma_coordinate": {"depth": _D1_depth_mapping},
}

# --------------------------------------------------------------------
# Define the canonical units of formula terms, as described in
# Appendix D: Parametric Vertical Coordinates of the CF conventions.
# --------------------------------------------------------------------
formula_terms_units = {
    "atmosphere_ln_pressure_coordinate": {"p0": "Pa", "lev": ""},
    "atmosphere_sigma_coordinate": {"sigma": "", "ptop": "Pa", "ps": "Pa"},
    "atmosphere_hybrid_sigma_pressure_coordinate": {
        "p0": "Pa",
        "ps": "Pa",
        "ap": "Pa",
        "a": "",
        "b": "",
    },
    "atmosphere_hybrid_height_coordinate": {"a": "m", "b": "", "orog": "m"},
    "atmosphere_sleve_coordinate": {
        "ztop": "m",
        "a": "",
        "b1": "",
        "b2": "",
        "zsurf1": "m",
        "zsurf2": "m",
    },
    "ocean_sigma_coordinate": {"eta": "m", "depth": "m", "sigma": ""},
    "ocean_s_coordinate": {
        "eta": "m",
        "depth": "m",
        "a": "",
        "b": "",
        "depth_c": "m",
        "C": "",
        "s": "",
    },
    "ocean_s_coordinate_g1": {
        "eta": "m",
        "depth": "m",
        "depth_c": "m",
        "C": "",
        "s": "",
    },
    "ocean_s_coordinate_g2": {
        "eta": "m",
        "depth": "m",
        "depth_c": "m",
        "C": "",
        "s": "",
    },
    "ocean_sigma_z_coordinate": {
        "eta": "m",
        "depth": "m",
        "zlev": "m",
        "nsigma": "",
        "depth_c": "m",
        "sigma": "",
    },
    "ocean_double_sigma_coordinate": {
        "depth": "m",
        "href": "m",
        "k_c": "",
        "a": "m",
        "z1": "m",
        "z2": "m",
        "sigma": "",
    },
}


# --------------------------------------------------------------------
# Logging level setup
# --------------------------------------------------------------------
# For explicitness, define here rather than importing identical Enum
# from cfdm
class ValidLogLevels(Enum):
    DISABLE = 0
    WARNING = 1
    INFO = 2
    DETAIL = 3
    DEBUG = -1


# --------------------------------------------------------------------
# Controlled vocabulary for bounds combination options
# --------------------------------------------------------------------
class OperandBoundsCombination(Enum):
    AND = auto()
    OR = auto()
    XOR = auto()
    NONE = auto()
