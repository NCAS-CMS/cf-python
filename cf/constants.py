from .units import Units

import logging
import sys

from psutil   import virtual_memory
from tempfile import gettempdir

from numpy.ma import masked as numpy_ma_masked
from numpy.ma import nomask as numpy_ma_nomask

from . import mpi_on
from . import mpi_size
if mpi_on:
    from . import mpi_comm
# --- End: if


# platform = sys.platform
# if platform == 'darwin':
#     from psutil import virtual_memory

# --------------------------------------------------------------------
# Find the total amount of memory, in bytes
# --------------------------------------------------------------------
_TOTAL_MEMORY = float(virtual_memory().total)
# if platform == 'darwin':
#     # MacOS
#    _MemTotal = float(virtual_memory().total)
# else:
#     # Linux
#     _meminfo_file = open('/proc/meminfo', 'r', 1)
#     for line in _meminfo_file:
#         field_size = line.split()
#         if field_size[0] == 'MemTotal:':
#             _MemTotal = float(field_size[1]) * 1024
#             break
#     # --- End: for
#     _meminfo_file.close()
# # --- End: if


"""
A dictionary of useful constants.

Whilst the dictionary may be modified directly, it is safer to
retrieve and set the values with a function where one is
provided. This is due to interdependencies between some values.

:Keys:

    ATOL : float
      The value of absolute tolerance for testing numerically
      tolerant equality.

    TOTAL_MEMORY : float
      Find the total amount of physical memory (in bytes).

    CHUNKSIZE : float
      The chunk size (in bytes) for data storage and
      processing.

    FM_THRESHOLD : float
      The minimum amount of memory (in bytes) to be kept free
      for temporary work space. This should always be
      MINNCFM*CHUNKSIZE.

    MINNCFM : int
      The number of chunk sizes to be kept free for temporary
      work space.

    OF_FRACTION : float
      The fraction of the maximum number of concurrently open
      files which may be used for files containing data
      arrays.

    RTOL : float
      The value of relative tolerance for testing numerically
      tolerant equality.

    TEMPDIR : str
      The location to store temporary files. By default it is
      the default directory used by the :mod:`tempfile` module.

    REGRID_LOGGING : bool
      Whether or not to enable ESMPy logging. If it is logging
      is performed after every call to ESMPy. By default logging
      is disabled.

    FREE_MEMORY_FACTOR : int
      Factor to divide the free memory by. If MPI is on this is
      equal to the number of PEs. Otherwise it is equal to 1 and
      is ignored in any case.

    COLLAPSE_PARALLEL_MODE : int
      The mode to use when parallelising collapse. By default
      this is 0 to try and automatically determine which mode to
      use.

    LOG_LEVEL : str
      The minimal level of seriousness for which log messages are shown.
      See functions.LOG_LEVEL().
"""
CONSTANTS = {
    'RTOL': sys.float_info.epsilon,
    'ATOL': sys.float_info.epsilon,
    'TEMPDIR': gettempdir(),
    'OF_FRACTION': 0.5,
    'TOTAL_MEMORY': _TOTAL_MEMORY,
    'FREE_MEMORY_FACTOR': 0.1,
    'WORKSPACE_FACTOR_1': 2.0,
    'WORKSPACE_FACTOR_2': 8.0,
    'REGRID_LOGGING': False,
    'COLLAPSE_PARALLEL_MODE': 0,
    'RELAXED_IDENTITIES': False,
    'IGNORE_IDENTITIES': False,
    'LOG_LEVEL': logging.getLevelName(logging.getLogger().level),
}

CONSTANTS['FM_THRESHOLD'] = (
    CONSTANTS['FREE_MEMORY_FACTOR'] * CONSTANTS['TOTAL_MEMORY']
)

if mpi_on:
    CONSTANTS['MIN_TOTAL_MEMORY'] = min(
        mpi_comm.allgather(CONSTANTS['TOTAL_MEMORY']))
else:
    CONSTANTS['MIN_TOTAL_MEMORY'] = CONSTANTS['TOTAL_MEMORY']
# --- End: if

CONSTANTS['CHUNKSIZE'] = (
    (CONSTANTS['FREE_MEMORY_FACTOR'] * CONSTANTS['MIN_TOTAL_MEMORY']) /
    (mpi_size *
     CONSTANTS['WORKSPACE_FACTOR_1'] + CONSTANTS['WORKSPACE_FACTOR_2'])
)

masked = numpy_ma_masked
# nomask = numpy_ma_nomask

repr_prefix = 'CF '
repr_suffix = ''

_file_to_fh = {}

_stash2standard_name = {}

# ---------------------------------------------------------------------
# Coordinate reference constants TODO: turn these into functions
# ---------------------------------------------------------------------
cr_canonical_units = {
    'earth_radius': Units('m'),
    'false_easting': Units('m'),
    'projection_x_coordinate': Units('m'),
    'false_northing': Units('m'),
    'projection_y_coordinate': Units('m'),
    'grid_north_pole_latitude': Units('degrees_north'),
    'grid_north_pole_longitude': Units('degrees_east'),
    'inverse_flattening': Units('1'),
    'latitude_of_projection_origin': Units('degrees_north'),
    'longitude_of_central_meridian': Units('degrees_east'),
    'longitude_of_prime_meridian': Units('degrees_east'),
    'longitude_of_projection_origin': Units('degrees_east'),
    'north_pole_grid_longitude': Units('degrees'),
    'perspective_point_height': Units('m'),
    'scale_factor_at_central_meridian': Units('1'),
    'scale_factor_at_projection_origin': Units('1'),
    'semi_major_axis': Units('m'),
    'semi_minor_axis': Units('m'),
    'standard_parallel': Units('degrees_north'),
    'straight_vertical_longitude_from_pole': Units('degrees_north'),
}

cr_coordinates = {
    'grid_mapping_name:albers_conical_equal_area': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:azimuthal_equidistant': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:geostationary': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:lambert_azimuthal_equal_area': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:lambert_conformal_conic': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:lambert_cylindrical_equal_area': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:latitude_longitude': (
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:mercator': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:orthographic': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:polar_stereographic': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:rotated_latitude_longitude': (
        'grid_latitude',
        'grid_longitude',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:sinusoidal': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:stereographic': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'grid_mapping_name:transverse_mercator': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'standard_name:vertical_perspective': (
        'projection_x_coordinate',
        'projection_y_coordinate',
        'latitude',
        'longitude'
    ),
    'standard_name:atmosphere_ln_pressure_coordinate': (
        'atmosphere_ln_pressure_coordinate',
    ),
    'standard_name:atmosphere_sigma_coordinate': (
        'atmosphere_sigma_coordinate',
    ),
    'standard_name:atmosphere_hybrid_sigma_pressure_coordinate': (
        'atmosphere_hybrid_sigma_pressure_coordinate',
    ),
    'standard_name:atmosphere_hybrid_height_coordinate': (
        'atmosphere_hybrid_height_coordinate',
    ),
    'standard_name:atmosphere_sleve_coordinate': (
        'atmosphere_sleve_coordinate',
    ),
    'standard_name:ocean_sigma_coordinate': (
        'ocean_sigma_coordinate',
    ),
    'standard_name:ocean_s_coordinate': (
        'ocean_s_coordinate',
    ),
    'standard_name:ocean_sigma_z_coordinate': (
        'ocean_sigma_z_coordinate',
    ),
    'standard_name:ocean_double_sigma_coordinate': (
        'ocean_double_sigma_coordinate',
    ),
}
# Coordinate conversion terms and their default values.
# Column  1 : Coordinate conversion term
# Columns 2+: Default values
# See appendices D and F in the CF conventions for details.
cr_default_values = {
    'a': 0.0,
    'b': 0.0,
    'b1': 0.0,
    'b2': 0.0,
    'depth': 0.0,
    'depth_c': 0.0,
    'eta': 0.0,
    'href': 0.0,
    'k_c': 0.0,
    'lev': 0.0,
    'longitude_of_prime_meridian': 0.0,
    'north_pole_grid_longitude': 0.0,
    'nsigma': 0.0,
    'orog': 0.0,
    'p0': 0.0,
    'ps': 0.0,
    'ptop': 0.0,
    's': 0.0,
    'sigma': 0.0,
    'z1': 0.0,
    'z2': 0.0,
    'zlev': 0.0,
    'zsurf1': 0.0,
    'zsurf2': 0.0,
    'ztop': 0.0,
}


# --------------------------------------------------------------------
# Logging level setup
# --------------------------------------------------------------------
valid_log_levels = [  # order (highest to lowest severity) must be preserved
    'DISABLE',
    'WARNING',
    'INFO',
    'DETAIL',
    'DEBUG',
]
# Map string level identifiers to ints from 0 to len(valid_log_levels):
numeric_log_level_map = dict(enumerate(valid_log_levels))
# We treat 'DEBUG' as a special case so assign to '-1' rather than highest int:
numeric_log_level_map[-1] = numeric_log_level_map.pop(
    len(valid_log_levels) - 1)
# Result for print(numeric_log_level_map) is:
# {0: 'DISABLE', 1: 'WARNING', 2: 'INFO', 3: 'DETAIL', -1: 'DEBUG'}
