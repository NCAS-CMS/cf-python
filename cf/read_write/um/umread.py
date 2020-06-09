import logging
import os
import netCDF4
import csv
import re
import textwrap

from datetime import datetime

from numpy import any          as numpy_any
from numpy import arange       as numpy_arange
from numpy import arccos       as numpy_arccos
from numpy import arcsin       as numpy_arcsin
from numpy import array        as numpy_array
from numpy import clip         as numpy_clip
from numpy import column_stack as numpy_column_stack
from numpy import cos          as numpy_cos
from numpy import deg2rad      as numpy_deg2rad
from numpy import dtype        as numpy_dtype
from numpy import empty        as numpy_empty
from numpy import isnan        as numpy_isnan
from numpy import mean         as numpy_mean
from numpy import nan          as numpy_nan
from numpy import pi           as numpy_pi
from numpy import rad2deg      as numpy_rad2deg
from numpy import resize       as numpy_resize
from numpy import result_type  as numpy_result_type
from numpy import sin          as numpy_sin
from numpy import sum          as numpy_sum
from numpy import transpose    as numpy_transpose
from numpy import where        as numpy_where

from netCDF4 import date2num as netCDF4_date2num

import cftime
import cfdm

from ...                   import __version__, __Conventions__, __file__
from ...decorators         import (_manage_log_level_via_verbosity,
                                   _manage_log_level_via_verbose_attr)
from ...functions          import (RTOL, ATOL, equals,
                                   open_files_threshold_exceeded,
                                   close_one_file, abspath,
                                   load_stash2standard_name)
from ...units              import Units

from ...data.data import Data, Partition, PartitionMatrix

from ...data              import UMArray
from ...data.functions    import _open_um_file, _close_um_file
from ...umread_lib.umfile import UMFileException


logger = logging.getLogger(__name__)

_cached_runid = {}
_cached_latlon = {}
_cached_time = {}
_cached_ctime = {}
_cached_size_1_height_coordinate = {}
_cached_z_coordinate = {}
_cached_date2num = {}
_cached_model_level_number_coordinate = {}

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
_pi_over_180 = numpy_pi/180.0

# PP missing data indicator
_pp_rmdi = -1.0e+30

# No no-missing-data value of BMDI (as described in UMDP F3 v805)
_BMDI_no_missing_data_value = -1.0e+30

# Reference surface pressure in Pascals
_pstar = 1.0e5

# --------------------------------------------------------------------
# Characters used in decoding LBEXP into a runid
# --------------------------------------------------------------------
_characters = (
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
)

_n_characters = len(_characters)

# # --------------------------------------------------------------------
# # Number matching regular expression
# # --------------------------------------------------------------------
# _number_regex = '([-+]?\d*\.?\d+(e[-+]?\d+)?)'

_Units = {
    None: Units(),
    '': Units(''),
    '1': Units('1'),
    'Pa': Units('Pa'),
    'm': Units('m'),
    'hPa': Units('hPa'),
    'K': Units('K'),
    'degrees': Units('degrees'),
    'degrees_east': Units('degrees_east'),
    'degrees_north': Units('degrees_north'),
    'days': Units('days'),
    'gregorian 1752-09-13': Units('days since 1752-09-13', 'gregorian'),
    '365_day 1752-09-13': Units('days since 1752-09-13', '365_day'),
    '360_day 0-1-1': Units('days since 0-1-1', '360_day'),
}

# --------------------------------------------------------------------
# Names of PP integer and real header items
# --------------------------------------------------------------------
_header_names = (
    'LBYR', 'LBMON', 'LBDAT', 'LBHR', 'LBMIN', 'LBDAY',
    'LBYRD', 'LBMOND', 'LBDATD', 'LBHRD', 'LBMIND',
    'LBDAYD', 'LBTIM', 'LBFT', 'LBLREC', 'LBCODE', 'LBHEM',
    'LBROW', 'LBNPT', 'LBEXT', 'LBPACK', 'LBREL', 'LBFC',
    'LBCFC', 'LBPROC', 'LBVC', 'LBRVC', 'LBEXP', 'LBEGIN',
    'LBNREC', 'LBPROJ', 'LBTYP', 'LBLEV', 'LBRSVD1',
    'LBRSVD2', 'LBRSVD3', 'LBRSVD4', 'LBSRCE', 'LBUSER1',
    'LBUSER2', 'LBUSER3', 'LBUSER4', 'LBUSER5', 'LBUSER6',
    'LBUSER7',
    'BRSVD1', 'BRSVD2', 'BRSVD3', 'BRSVD4',
    'BDATUM', 'BACC', 'BLEV', 'BRLEV', 'BHLEV', 'BHRLEV',
    'BPLAT', 'BPLON', 'BGOR',
    'BZY', 'BDY', 'BZX', 'BDX', 'BMDI', 'BMKS'
)

# --------------------------------------------------------------------
# Positions of PP header items in their arrays
# --------------------------------------------------------------------
(lbyr, lbmon, lbdat, lbhr, lbmin, lbday,
 lbyrd, lbmond, lbdatd, lbhrd, lbmind,
 lbdayd, lbtim, lbft, lblrec, lbcode, lbhem,
 lbrow, lbnpt, lbext, lbpack, lbrel, lbfc,
 lbcfc, lbproc, lbvc, lbrvc, lbexp, lbegin,
 lbnrec, lbproj, lbtyp, lblev, lbrsvd1,
 lbrsvd2, lbrsvd3, lbrsvd4, lbsrce, lbuser1,
 lbuser2, lbuser3, lbuser4, lbuser5, lbuser6,
 lbuser7,
 ) = list(range(45))

(brsvd1, brsvd2, brsvd3, brsvd4,
 bdatum, bacc, blev, brlev, bhlev, bhrlev,
 bplat, bplon, bgor,
 bzy, bdy, bzx, bdx, bmdi, bmks,
 ) = list(range(19))

# --------------------------------------------------------------------
# Map PP axis codes to CF standard names (The full list of field code
# keys may be found at
# http://cms.ncas.ac.uk/html_umdocs/wave/@header.)
# --------------------------------------------------------------------
_coord_standard_name = {
    0: None,                  # Sigma (or eta, for hybrid coordinate data).
    1: 'air_pressure',        # Pressure (mb).
    2: 'height',              # Height above sea level (km)
    # Eta (U.M. hybrid coordinates) only:
    3: 'atmosphere_hybrid_sigma_pressure_coordinate',
    4: 'depth',               # Depth below sea level (m)
    5: 'model_level_number',  # Model level.
    6: 'air_potential_temperature',    # Theta
    7: 'atmosphere_sigma_coordinate',  # Sigma only.
    8: None,                  # Sigma-theta
    10: 'latitude',           # Latitude (degrees N).
    11: 'longitude',          # Longitude (degrees E).
    # Site number (set of parallel rows or columns e.g.Time series):
    13: 'region',
    14: 'atmosphere_hybrid_height_coordinate',
    15: 'height',
    20: 'time',          # Time (days) (Gregorian calendar (not 360 day year))
    21: 'time',               # Time (months)
    22: 'time',               # Time (years)
    23: 'time',               # Time (model days with 360 day model calendar)
    40: None,                 # pseudolevel
    99: None,                 # Other
    -10: 'grid_latitude',     # Rotated latitude (degrees).
    -11: 'grid_longitude',    # Rotated longitude (degrees).
    -20: 'radiation_wavelength',
}

# --------------------------------------------------------------------
# Map PP axis codes to CF long names
# --------------------------------------------------------------------
_coord_long_name = {}

# --------------------------------------------------------------------
# Map PP axis codes to UDUNITS strings
# --------------------------------------------------------------------
# _coord_units = {
_axiscode_to_units = {
    0: '1',               # Sigma (or eta, for hybrid coordinate data)
    1: 'hPa',             # air_pressure
    2: 'm',               # altitude
    3: '1',               # atmosphere_hybrid_sigma_pressure_coordinate
    4: 'm',               # depth
    5: '1',               # model_level_number
    6: 'K',               # air_potential_temperature
    7: '1',               # atmosphere_sigma_coordinate
    10: 'degrees_north',  # latitude
    11: 'degrees_east',   # longitude
    13: '',               # region
    14: '1',              # atmosphere_hybrid_height_coordinate
    15: 'm',              # height
    20: 'days',           # time (gregorian)
    23: 'days',           # time (360_day)
    40: '1',              # pseudolevel
    -10: 'degrees',  # rotated latitude  (not an official axis code)
    -11: 'degrees',  # rotated longitude (not an official axis code)
}

# --------------------------------------------------------------------
# Map PP axis codes to Units objects
# --------------------------------------------------------------------
_axiscode_to_Units = {
    0: _Units['1'],               # Sigma (or eta, for hybrid coordinate data)
    1: _Units['hPa'],             # air_pressure
    2: _Units['m'],               # altitude
    3: _Units['1'],          # atmosphere_hybrid_sigma_pressure_coordinate
    4: _Units['m'],               # depth
    5: _Units['1'],               # model_level_number
    6: _Units['K'],               # air_potential_temperature
    7: _Units['1'],               # atmosphere_sigma_coordinate
    10: _Units['degrees_north'],  # latitude
    11: _Units['degrees_east'],   # longitude
    13: _Units[''],               # region
    14: _Units['1'],              # atmosphere_hybrid_height_coordinate
    15: _Units['m'],              # height
    20: _Units['days'],           # time (gregorian)
    23: _Units['days'],           # time (360_day)
    40: _Units['1'],              # pseudolevel
    -10: _Units['degrees'],  # rotated latitude  (not an official axis code)
    -11: _Units['degrees'],  # rotated longitude (not an official axis code)
}

# --------------------------------------------------------------------
# Map PP axis codes to CF axis attributes
# --------------------------------------------------------------------
_coord_axis = {
    1: 'Z',    # air_pressure
    2: 'Z',    # altitude
    3: 'Z',    # atmosphere_hybrid_sigma_pressure_coordinate
    4: 'Z',    # depth
    5: 'Z',    # model_level_number
    6: 'Z',    # air_potential_temperature
    7: 'Z',    # atmosphere_sigma_coordinate
    10: 'Y',   # latitude
    11: 'X',   # longitude
    13: None,  # region
    14: 'Z',   # atmosphere_hybrid_height_coordinate
    15: 'Z',   # height
    20: 'T',   # time (gregorian)
    23: 'T',   # time (360_day)
    40: None,  # pseudolevel
    -10: 'Y',  # rotated latitude  (not an official axis code)
    -11: 'X',  # rotated longitude (not an official axis code)
}

# --------------------------------------------------------------------
# Map PP axis codes to CF positive attributes
# --------------------------------------------------------------------
_coord_positive = {
    1: 'down',  # air_pressure
    2: 'up',    # altitude
    3: 'down',  # atmosphere_hybrid_sigma_pressure_coordinate
    4: 'down',  # depth
    5: None,    # model_level_number
    6: 'up',    # air_potential_temperature
    7: 'down',  # atmosphere_sigma_coordinate
    10: None,   # latitude
    11: None,   # longitude
    13: None,   # region
    14: 'up',   # atmosphere_hybrid_height_coordinate
    15: 'up',   # height
    20: None,   # time (gregorian)
    23: None,   # time (360_day)
    40: None,   # pseudolevel
    -10: None,  # rotated latitude  (not an official axis code)
    -11: None,  # rotated longitude (not an official axis code)
}

# --------------------------------------------------------------------
# Map LBVC codes to PP axis codes. The full list of field code keys
# may be found at http://cms.ncas.ac.uk/html_umdocs/wave/@fcodes
# --------------------------------------------------------------------
_lbvc_to_axiscode = {
    1: 2,       # altitude (Height)
    2: 4,       # depth (Depth)
    3: None,    # (Geopotential (= g*height))
    4: None,    # (ICAO height)
    6: 4,       # model_level_number  # Changed from 5 !!!
    7: None,    # (Exner pressure)
    8: 1,       # air_pressure  (Pressure)
    9: 3,       # atmosphere_hybrid_sigma_pressure_coordinate (Hybrid pressure)
    # dch check:
    10: 7,      # atmosphere_sigma_coordinate (Sigma (= p/surface p))
    16: None,   # (Temperature T)
    19: 6,      # air_potential_temperature (Potential temperature)
    27: None,   # (Atmospheric) density
    28: None,   # (d(p*)/dt .  p* = surface pressure)
    44: None,   # (Time in seconds)
    65: 14,     # atmosphere_hybrid_height_coordinate (Hybrid height)
    129: None,  # Surface
    176: 10,    # latitude    (Latitude)
    177: 11,    # longitude   (Longitude)
}

# --------------------------------------------------------------------
# Map model identifier codes to model names. The model identifier code
# is the last four digits of LBSRCE.
# --------------------------------------------------------------------
_lbsrce_model_codes = {1111: 'UM'}

# --------------------------------------------------------------------
# Names of PP extra data codes
# --------------------------------------------------------------------
_extra_data_name = {
    1: 'x',
    2: 'y',
    3: 'y_domain_lower_bound',
    4: 'x_domain_lower_bound',
    5: 'y_domain_upper_bound',
    6: 'x_domain_upper_bound',
    7: 'z_domain_lower_bound',
    8: 'x_domain_upper_bound',
    9: 'title',
    10: 'domain_title',
    11: 'x_lower_bound',
    12: 'x_upper_bound',
    13: 'y_lower_bound',
    14: 'y_upper_bound',
}

# --------------------------------------------------------------------
# LBCODE values for unrotated latitude longitude grids
# --------------------------------------------------------------------
_true_latitude_longitude_lbcodes = set((1, 2))

# --------------------------------------------------------------------
# LBCODE values for rotated latitude longitude grids
# --------------------------------------------------------------------
_rotated_latitude_longitude_lbcodes = set((101, 102, 111))

# _axis = {'t'   : 'dim0',
#          'z'   : 'dim1',
#          'y'   : 'dim2',
#          'x'   : 'dim3',
#          'r'   : 'dim4',
#          'p'   : 'dim5',
#          'area': None,
#      }

_axis = {'area': None}


class UMField:
    '''TODO

    '''

    def __init__(self, var, fmt, byte_ordering, word_size, um_version,
                 set_standard_name, height_at_top_of_model, verbose=None,
                 implementation=None, **kwargs):
        '''**Initialization**

    :Parameters:

        var: `umfile.Var`

        byte_ordering: `str`
            ``'little_endian'` or ``'big_endian'``.

        word_size: `int`
            Word size in bytes (4 or 8).

        fmt: `str`
            ``'PP'` or ``'FF'``

        um_version: number

        set_standard_name: `bool`
            If True then set the standard_name CF property.

        height_at_top_of_model: `float`

        verbose: `int` or `None`, optional
            If an integer from ``0`` to ``3``, corresponding to increasing
            verbosity (else ``-1`` as a special case of maximal and extreme
            verbosity), set for the duration of the method call (only) as
            the minimum severity level cut-off of displayed log messages,
            regardless of the global configured `cf.LOG_LEVEL`.

            Else, if `None` (the default value), log messages will be
            filtered out, or otherwise, according to the value of the
            `cf.LOG_LEVEL` setting.

            Overall, the higher a non-negative integer that is set (up to
            a maximum of ``3``) the more description that is printed
            about the read process.

        kwargs: *optional*
            Keyword arguments providing extra CF properties for each
            return field constuct.

        '''
        self._bool = False

        self.implementation = implementation

        self.verbose = verbose

        self.fmt = fmt
        self.height_at_top_of_model = height_at_top_of_model
        self.byte_ordering = byte_ordering
        self.word_size = word_size

        self.atol = ATOL()

        self.field = self.implementation.initialise_Field()

        cf_properties = {}
        attributes = {}

        self.fields = []

        filename = abspath(var.file.path)
        self.filename = filename

        groups = var.group_records_by_extra_data()

        n_groups = len(groups)

        if n_groups == 1:
            # There is one group of records
            groups_nz = [var.nz]
            groups_nt = [var.nt]
        elif n_groups > 1:
            # There are multiple groups of records, distinguished by
            # different extra data.
            groups_nz = []
            groups_nt = []
            groups2 = []
            for group in groups:
                group_size = len(group)
                if group_size == 1:
                    # There is only one record in this group
                    split_group = False
                    nz = 1
                elif group_size > 1:
                    # There are multiple records in this group
                    # Find the lengths of runs of identical times
                    times = [(self.header_vtime(rec), self.header_dtime(rec))
                             for rec in group]
                    lengths = [len(tuple(g)) for k, g in
                               itertools.groupby(times)]
                    if len(set(lengths)) == 1:
                        # Each run of identical times has the same
                        # length, so it is possible that this group
                        # forms a variable of nz x nt records.
                        split_group = False
                        nz = lengths.pop()
                        z0 = [self.z for rec in group[:nz]]
                        for i in range(nz, group_size, nz):
                            z1 = [self.header_z(rec) for rec in group[i:i+nz]]
                            if z1 != z0:
                                split_group = True
                                break
                    else:
                        # Different runs of identical times have
                        # different lengths, so it is not possible for
                        # this group to form a variable of nz x nt
                        # records.
                        split_group = True
                        nz = 1
                # --- End: if

                if split_group:
                    # This group doesn't form a complete nz x nt
                    # matrix, so split it up into 1 x 1 groups.
                    groups2.extend([[rec] for rec in group])
                    groups_nz.extend([1] * group_size)
                    groups_nt.extend([1] * group_size)
                else:
                    # This group forms a complete nz x nt matrix, so
                    # it may be considered as a variable in its own
                    # right and doesn't need to be split up.
                    groups2.append(group)
                    groups_nz.append(nz)
                    groups_nt.append(group_size/nz)
            # --- End: for

            groups = groups2
        # --- End: if

        rec0 = groups[0][0]

        int_hdr = rec0.int_hdr
        self.int_hdr_dtype = int_hdr.dtype

        int_hdr = int_hdr.tolist()
        real_hdr = rec0.real_hdr.tolist()
        self.int_hdr = int_hdr
        self.real_hdr = real_hdr

        # ------------------------------------------------------------
        # Set some metadata quantities which are guaranteed to be the
        # same for all records in a variable
        # ------------------------------------------------------------
        LBNPT = int_hdr[lbnpt]
        LBROW = int_hdr[lbrow]
        LBTIM = int_hdr[lbtim]
        LBCODE = int_hdr[lbcode]
        LBPROC = int_hdr[lbproc]
        LBVC = int_hdr[lbvc]
        LBUSER5 = int_hdr[lbuser5]
        BPLAT = real_hdr[bplat]
        BPLON = real_hdr[bplon]
        BDX = real_hdr[bdx]
        BDY = real_hdr[bdy]

        self.lbnpt = LBNPT
        self.lbrow = LBROW
        self.lbtim = LBTIM
        self.lbproc = LBPROC
        self.lbvc = LBVC
        self.bplat = BPLAT
        self.bplon = BPLON
        self.bdx = BDX
        self.bdy = BDY

        # ------------------------------------------------------------
        # Set some derived metadata quantities which are (as good as)
        # guaranteed to be the same for all records in a variable
        # ------------------------------------------------------------
        self.lbtim_ia, ib = divmod(LBTIM, 100)
        self.lbtim_ib, ic = divmod(ib, 10)

        if ic == 1:
            calendar = 'gregorian'
        elif ic == 4:
            calendar = '365_day'
        else:
            calendar = '360_day'

        self.calendar = calendar
        self.reference_time_Units()

        header_um_version, source = divmod(int_hdr[lbsrce], 10000)

        if header_um_version > 0 and int(um_version) == um_version:
            model_um_version = header_um_version
            self.um_version = header_um_version
        else:
            model_um_version = None
            self.um_version = um_version

        # Set source
        source = _lbsrce_model_codes.setdefault(source, None)
        if source is not None and model_um_version is not None:
            source += ' vn{0}'.format(model_um_version)
        if source:
            cf_properties['source'] = source

        # ------------------------------------------------------------
        # Set the T, Z, Y and X axis codes. These are guaranteed to be
        # the same for all records in a variable.
        # ------------------------------------------------------------
        if LBCODE == 1 or LBCODE == 2:
            # 1 = Unrotated regular lat/long grid
            # 2 = Regular lat/lon grid boxes (grid points are box
            #     centres)
            ix = 11
            iy = 10
        elif LBCODE == 101 or LBCODE == 102:
            # 101 = Rotated regular lat/long grid
            # 102 = Rotated regular lat/lon grid boxes (grid points
            #       are box centres)
            ix = -11  # rotated longitude (not an official axis code)
            iy = -10  # rotated latitude  (not an official axis code)
        elif LBCODE >= 10000:
            # Cross section
            ix, iy = divmod(divmod(LBCODE, 10000)[1], 100)
        else:
            ix = None
            iy = None

        iz = _lbvc_to_axiscode.setdefault(LBVC, None)

        # Set it from the calendar type
        if iy in (20, 23) or ix in (20, 23):
            # Time is dealt with by x or y
            it = None
        elif calendar == 'gregorian':
            it = 20
        else:
            it = 23

        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.it = it

        self.cf_info = {}

        # Set a identifying name based on the submodel and STASHcode
        # (or field code).
        stash = int_hdr[lbuser4]
        submodel = int_hdr[lbuser7]
        self.stash = stash

        # The STASH code has been set in the PP header, so try to find
        # its standard_name from the conversion table
        stash_records = stash2standard_name.get((submodel, stash), None)

        um_Units = None
        um_condition = None

        long_name = None
        standard_name = None

        if stash_records:
            um_version = self.um_version
            for (long_name,
                 units,
                 valid_from,
                 valid_to,
                 standard_name,
                 cf_info,
                 um_condition) in stash_records:

                # Check that conditions are met
                if not self.test_um_version(valid_from, valid_to, um_version):
                    continue

                if um_condition:
                    if not self.test_um_condition(um_condition,
                                                  LBCODE, BPLAT, BPLON):
                        continue

                # Still here? Then we have our standard_name, etc.
#                if standard_name:
#                    if set_standard_name:
#                        cf_properties['standard_name'] = standard_name
#                    else:
#                        attributes['_standard_name'] = standard_name
                if standard_name and set_standard_name:
                    cf_properties['standard_name'] = standard_name

                cf_properties['long_name'] = long_name.rstrip()

                um_Units = _Units.get(units, None)
                if um_Units is None:
                    um_Units = Units(units)
                    _Units[units] = um_Units

                self.um_Units = um_Units
                self.cf_info = cf_info

                break
        # --- End: if

        if stash:
            section, item = divmod(stash, 1000)
            um_stash_source = 'm%02ds%02di%03d' % (submodel, section, item)
            cf_properties['um_stash_source'] = um_stash_source
            identity = 'UM_{0}_vn{1}'.format(um_stash_source,
                                             self.um_version)
        else:
            identity = 'UM_{0}_fc{1}_vn{2}'.format(submodel,
                                                   int_hdr[lbfc],
                                                   self.um_version)

        if um_Units is None:
            self.um_Units = _Units[None]

        if um_condition:
            identity += '_{0}'.format(um_condition)

        if long_name is None:
            cf_properties['long_name'] = identity

        for recs, nz, nt in zip(groups, groups_nz, groups_nt):
            self.recs = recs
            self.nz = nz
            self.nt = nt
            self.z_recs = recs[:nz]
            self.t_recs = recs[::nz]

            LBUSER5 = recs[0].int_hdr.item(lbuser5,)

#            self.cell_method_axis_name = {'area': 'area'}

            self.down_axes = set()
            self.z_axis = 'z'

            # --------------------------------------------------------
            # Get the extra data for this group
            # --------------------------------------------------------
            extra = recs[0].get_extra_data()
            self.extra = extra

            # --------------------------------------------------------
            # Set some derived metadata quantities
            # --------------------------------------------------------
            logger.detail(self.__dict__)  # pragma: no cover
            self.printfdr()      # pragma: no cover

            # --------------------------------------------------------
            # Create the 'T' dimension coordinate
            # --------------------------------------------------------
            axiscode = it
            if axiscode is not None:
                c = self.time_coordinate(axiscode)

            # --------------------------------------------------------
            # Create the 'Z' dimension coordinate
            # --------------------------------------------------------
            axiscode = iz
            if axiscode is not None:
                # Get 'Z' coordinate from LBVC
                if axiscode == 3:
                    c = self.atmosphere_hybrid_sigma_pressure_coordinate(
                        axiscode)
                elif axiscode == 2 and 'height' in self.cf_info:
                    # Create the height coordinate from the information
                    # given in the STASH to standard_name conversion table
                    height, units = self.cf_info['height']
                    c = self.size_1_height_coordinate(axiscode, height, units)
                elif axiscode == 14:
                    c = self.atmosphere_hybrid_height_coordinate(axiscode)
                else:
                    c = self.z_coordinate(axiscode)

                # Create a model_level_number auxiliary coordinate
                LBLEV = int_hdr[lblev]
                if LBVC in (2, 9, 65) or LBLEV in (7777, 8888):  # CHECK!
                    self.LBLEV = LBLEV
                    c = self.model_level_number_coordinate(aux=bool(c))
            # --- End: if

            # --------------------------------------------------------
            # Create the 'Y' dimension coordinate
            # --------------------------------------------------------
            axiscode = iy
            yc = None
            if axiscode is not None:
                if axiscode in (20, 23):
                    # 'Y' axis is time-since-reference-date
                    if extra.get('y', None) is not None:
                        c = self.time_coordinate_from_extra_data(
                            axiscode, 'y')
                    else:
                        LBUSER3 = int_hdr[lbuser3]
                        if LBUSER3 == LBROW:
                            self.lbuser3 = LBUSER3
                            c = self.time_coordinate_from_um_timeseries(
                                axiscode, 'y')
                else:
                    ykey, yc = self.xy_coordinate(axiscode, 'y')
            # --- End: if

            # --------------------------------------------------------
            # Create the 'X' dimension coordinate
            # --------------------------------------------------------
            axiscode = ix
            xc = None
            if axiscode is not None:
                if axiscode in (20, 23):
                    # X axis is time since reference date
                    if extra.get('x', None) is not None:
                        c = self.time_coordinate_from_extra_data(axiscode, 'x')
                    else:
                        LBUSER3 = int_hdr[lbuser3]
                        if LBUSER3 == LBNPT:
                            self.lbuser3 = LBUSER3
                            c = self.time_coordinate_from_um_timeseries(
                                axiscode, 'x')
                else:
                    xkey, xc = self.xy_coordinate(axiscode, 'x')
            # --- End: if

            # -10: rotated latitude  (not an official axis code)
            # -11: rotated longitude (not an official axis code)

            if (iy, ix) == (-10, -11) or (iy, ix) == (-11, -10):
                # ----------------------------------------------------
                # Create a ROTATED_LATITUDE_LONGITUDE coordinate
                # reference
                # ----------------------------------------------------
                ref = self.implementation.initialise_CoordinateReference()

                cc = self.implementation.initialise_CoordinateConversion(
                    parameters={
                        'grid_mapping_name': 'rotated_latitude_longitude',
                        'grid_north_pole_latitude': BPLAT,
                        'grid_north_pole_longitude': BPLON
                    })

                self.implementation.set_coordinate_conversion(ref, cc)

                self.implementation.set_coordinate_reference(
                    self.field, ref, copy=False)

                # ----------------------------------------------------
                # Create UNROTATED, 2-D LATITUDE and LONGITUDE
                # auxiliary coordinates
                # ----------------------------------------------------
                self.latitude_longitude_2d_aux_coordinates(
                    yc, xc)  # , rotated_pole)

            # --------------------------------------------------------
            # Create a RADIATION WAVELENGTH dimension coordinate
            # --------------------------------------------------------
            try:
                rwl, rwl_units = self.cf_info['below']
            except (KeyError, TypeError):
                pass
            else:
                c = self.radiation_wavelength_coordinate(rwl, rwl_units)

                # Set LBUSER5 to zero so that it is not confused for a
                # pseudolevel
                LBUSER5 = 0

            # --------------------------------------------------------
            # Create a PSEUDOLEVEL dimension coordinate. This must be
            # done *after* the possible creation of a radiation
            # wavelength dimension coordinate.
            # --------------------------------------------------------
            if LBUSER5 != 0:
                self.pseudolevel_coordinate(LBUSER5)

            attributes['int_hdr'] = int_hdr[:]
            attributes['real_hdr'] = real_hdr[:]
            attributes['file'] = filename
            attributes['id'] = identity

            cf_properties['Conventions'] = __Conventions__
            cf_properties['runid'] = self.decode_lbexp()
            cf_properties['lbproc'] = str(LBPROC)
            cf_properties['lbtim'] = str(LBTIM)
            cf_properties['stash_code'] = str(stash)
            cf_properties['submodel'] = str(submodel)

            # --------------------------------------------------------
            # Set the data and extra data
            # --------------------------------------------------------
            data = self.create_data()

            # --------------------------------------------------------
            # Insert data into the field
            # --------------------------------------------------------
            field = self.field

            self.implementation.set_data(field, self.data,
                                         axes=self.data_axes,
                                         copy=False)

            # --------------------------------------------------------
            # Insert attributes and CF properties into the field
            # --------------------------------------------------------
            fill_value = data.fill_value
            if fill_value is not None:
                cf_properties['_FillValue'] = data.fill_value

            # Add kwargs to the CF properties
            cf_properties.update(kwargs)

            self.implementation.set_properties(
                field, cf_properties, copy=False)

            field.id = identity

            if standard_name and not set_standard_name:
                field._custom['standard_name'] = standard_name

            self.implementation.nc_set_variable(field, identity)

            # --------------------------------------------------------
            # Create and insert cell methods
            # --------------------------------------------------------
            cell_methods = self.create_cell_methods()
            for cm in cell_methods:
                self.implementation.set_cell_method(field, cm)

            # Check for decreasing axes that aren't decreasing
            down_axes = self.down_axes
            logger.info(
                'down_axes = {}'.format(down_axes))  # pragma: no cover

            if down_axes:
                field.flip(down_axes, inplace=True)

            # Force cyclic X axis for paritcular values of LBHEM
            if int_hdr[lbhem] in (0, 1, 2, 4):
                field.cyclic('X', period=360)

            self.fields.append(field)
        # --- End: for

        self._bool = True

    def __bool__(self):
        '''x.__bool__() <==> bool(x)

        '''
        return self._bool

    def __repr__(self):
        '''x.__repr__() <==> repr(x)

        '''
        return self.fdr()

    def __str__(self):
        '''x.__str__() <==> str(x)

        '''
        out = [self.fdr()]

        attrs = ('endian',
                 'reftime', 'vtime', 'dtime',
                 'um_version', 'source',
                 'it', 'iz', 'ix', 'iy',
                 'site_time_cross_section', 'timeseries',
                 'file')

        for attr in attrs:
            out.append('{0}={1}'.format(attr, getattr(self, attr, None)))

        out.append('')

        return '\n'.join(out)

    def atmosphere_hybrid_height_coordinate(self, axiscode):
        '''TODO

    **From appendix A of UMDP F3**

    From UM Version 5.2, the method of defining the model levels in PP
    headers was revised. At vn5.0 and 5.1, eta values were used in the
    PP headers to specify the levels of model data, which was of
    limited use when plotting data on model levels. From 5.2, the PP
    headers were redefined to give information on the height of the
    level. Given a 2D orography field, the height field for a given
    level can then be derived. The height coordinates for PP-output
    are defined as:

      Z(i,j,k)=Zsea(k)+C(k)*orography(i,j)

    where Zsea(k) and C(k) are height based hybrid coefficients.

      Zsea(k) = eta_value(k)*Height_at_top_of_model

      C(k)=[1-eta_value(k)/eta_value(first_constant_rho_level)]**2 for
      levels less than or equal to first_constant_rho_level

      C(k)=0.0 for levels greater than first_constant_rho_level

    where eta_value(k) is the eta_value for theta or rho level k. The
    eta_value is a terrain-following height coordinate; full details
    are given in UMDP15, Appendix B.

    The PP headers store Zsea and C as follows :-

      * 46 = bulev = brsvd1  = Zsea of upper layer boundary
      * 47 = bhulev = brsvd2 = C of upper layer boundary
      * 52 = blev            = Zsea of level
      * 53 = brlev           = Zsea of lower layer boundary
      * 54 = bhlev           = C of level
      * 55 = bhrlev          = C of lower layer boundary

    :Parameters:

        axiscode: `int`

    :Returns:

        `DimensionCoordinate` or `None`

        '''
        field = self.field

        # "a" domain ancillary
        array = numpy_array(
            [rec.real_hdr[blev] for rec in self.z_recs],  # Zsea
            dtype=float)
        bounds0 = numpy_array(
            [rec.real_hdr[brlev] for rec in self.z_recs],  # Zsea lower
            dtype=float)
        bounds1 = numpy_array(
            [rec.real_hdr[brsvd1] for rec in self.z_recs],  # Zsea upper
            dtype=float)
        bounds = numpy_column_stack((bounds0, bounds1))

        # Insert new Z axis
        da = self.implementation.initialise_DomainAxis(size=array.size)
        axis_key = self.implementation.set_domain_axis(self.field, da)
        _axis['z'] = axis_key

        ac = self.implementation.initialise_DomainAncillary()
        ac = self.coord_data(ac, array, bounds, units=_Units['m'])
        ac.id = 'UM_atmosphere_hybrid_height_coordinate_a'
        self.implementation.set_properties(
            ac, {'long_name': 'height based hybrid coeffient a'})
        key_a = self.implementation.set_domain_ancillary(
            field, ac, axes=[_axis['z']], copy=False)

        # atmosphere_hybrid_height_coordinate dimension coordinate
        TOA_height = bounds1.max()
        if TOA_height <= 0:
            TOA_height = self.height_at_top_of_model

        if not TOA_height:
            dc = None
        else:
            array = array / TOA_height
            dc = self.implementation.initialise_DimensionCoordinate()
            dc = self.coord_data(dc, array, bounds, units=_Units[''])
            self.implementation.set_properties(
                dc, {'standard_name': 'atmosphere_hybrid_height_coordinate'})
            dc = self.coord_axis(dc, axiscode)
            dc = self.coord_positive(dc, axiscode, _axis['z'])
            self.implementation.set_dimension_coordinate(
                field, dc, axes=[_axis['z']], copy=False)

        # "b" domain ancillary
        array = numpy_array([rec.real_hdr[bhlev] for rec in self.z_recs],
                            dtype=float)
        bounds0 = numpy_array([rec.real_hdr[bhrlev] for rec in self.z_recs],
                              dtype=float)
        bounds1 = numpy_array([rec.real_hdr[brsvd2] for rec in self.z_recs],
                              dtype=float)
        bounds = numpy_column_stack((bounds0, bounds1))

        ac = self.implementation.initialise_DomainAncillary()
        ac = self.coord_data(ac, array, bounds, units=_Units['1'])
        ac.id = 'UM_atmosphere_hybrid_height_coordinate_b'
        self.implementation.set_properties(
            ac, {'long_name': 'height based hybrid coeffient b'})
        key_b = self.implementation.set_domain_ancillary(
            field, ac, axes=[_axis['z']], copy=False)

        if bool(dc):
            # atmosphere_hybrid_height_coordinate coordinate reference
            ref = self.implementation.initialise_CoordinateReference()
            cc = self.implementation.initialise_CoordinateConversion(
                parameters={
                    'standard_name': 'atmosphere_hybrid_height_coordinate'
                },
                domain_ancillaries={
                    'a': key_a,
                    'b': key_b,
                    'orog': None
                })
            self.implementation.set_coordinate_conversion(ref, cc)
            # TODO set coordinates?
            self.implementation.set_coordinate_reference(
                field, ref, copy=False)

        return dc

    def depth_coordinate(self, axiscode):
        '''TODO

    :Parameters:

        axiscode: `int`

    :Returns:

        `DimensionCoordinate` or `None`

        '''
        dc = self.model_level_number_coordinate(aux=False)

        field = self.field

        array = numpy_array([rec.real_hdr[blev] for rec in self.z_recs],
                            dtype=float)
        bounds0 = numpy_array([rec.real_hdr[brlev] for rec in self.z_recs],
                              dtype=float)
        bounds1 = numpy_array([rec.real_hdr[brsvd1] for rec in self.z_recs],
                              dtype=float)
        bounds = numpy_column_stack((bounds0, bounds1))

        # Create Z domain axis construct
        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisZ = self.implementation.set_domain_axis(self.field, da)
        _axis['z'] = axisZ

        # ac = AuxiliaryCoordinate()
        ac = self.implementation.initialise_AuxiliaryCoordinate()
        ac = self.coord_data(ac, array, bounds, units=_Units['m'])
        ac.id = 'UM_atmosphere_hybrid_height_coordinate_ak'
        ac.long_name = 'atmosphere_hybrid_height_coordinate_ak'
#        field.insert_aux(ac, axes=[zdim], copy=False)
        self.implementation.set_auxiliary_coordinate(
            self.field, ac, axes=[_axis['z']], copy=False)

        array = numpy_array([rec.real_hdr[bhlev] for rec in self.z_recs],
                            dtype=float)
        bounds0 = numpy_array([rec.real_hdr[bhrlev] for rec in self.z_recs],
                              dtype=float)
        bounds1 = numpy_array([rec.real_hdr[brsvd2] for rec in self.z_recs],
                              dtype=float)
        bounds = numpy_column_stack((bounds0, bounds1))

        # ac = AuxiliaryCoordinate()
        ac = self.implementation.initialise_AuxiliaryCoordinate()
        ac = self.coord_data(ac, array, bounds, units=_Units['1'])
        ac.id = 'UM_atmosphere_hybrid_height_coordinate_bk'
        ac.long_name = 'atmosphere_hybrid_height_coordinate_bk'
        self.implementation.set_auxiliary_coordinate(
            self.field, ac, axes=[_axis['z']], copy=False)

        return dc

    def atmosphere_hybrid_sigma_pressure_coordinate(self, axiscode):
        '''atmosphere_hybrid_sigma_pressure_coordinate when not an array axis

    46 BULEV Upper layer boundary or BRSVD(1)

    47 BHULEV Upper layer boundary or BRSVD(2)

        For hybrid levels:
        - BULEV is B-value at half-level above.
        - BHULEV is A-value at half-level above.

        For hybrid height levels (vn5.2-, Smooth heights)
        - BULEV is Zsea of upper layer boundary
            * If rho level: Zsea for theta level above
        * If theta level: Zsea for rho level above
        - BHLEV is C of upper layer boundary
            * If rho level: C for theta level above
            * If theta level: C for rho level above

    :Parameters:

        axiscode: `int`

    :Returns:

        `DimensionCoordinate`

        '''
        array = []
        bounds = []
        ak_array = []
        ak_bounds = []
        bk_array = []
        bk_bounds = []

        for rec in self.z_recs:
            BLEV, BRLEV, BHLEV, BHRLEV, BULEV, BHULEV = self.header_bz(rec)

            array.append(BLEV + BHLEV/_pstar)
            bounds.append([BRLEV + BHRLEV/_pstar, BULEV + BHULEV/_pstar])

            ak_array.append(BHLEV)
            ak_bounds.append((BHRLEV, BHULEV))

            bk_array.append(BLEV)
            bk_bounds.append((BRLEV, BULEV))

        array = numpy_array(array, dtype=float)
        bounds = numpy_array(bounds, dtype=float)
        ak_array = numpy_array(ak_array, dtype=float)
        ak_bounds = numpy_array(ak_bounds, dtype=float)
        bk_array = numpy_array(bk_array, dtype=float)
        bk_bounds = numpy_array(bk_bounds, dtype=float)

        # Insert new Z axis
        da = self.implementation.initialise_DomainAxis(size=array.size)
        axis_key = self.implementation.set_domain_axis(self.field, da)
        _axis['z'] = axis_key

        field = self.field

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(
            dc, array, bounds,
            units=_axiscode_to_Units.setdefault(axiscode, None))
        dc = self.coord_positive(dc, axiscode, _axis['z'])
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        self.implementation.set_dimension_coordinate(self.field, dc,
                                                     axes=[_axis['z']],
                                                     copy=False)

        ac = self.implementation.initialise_AuxiliaryCoordinate()
        ac = self.coord_data(ac, ak_array, ak_bounds, units=_Units['Pa'])
        ac.id = 'UM_atmosphere_hybrid_sigma_pressure_coordinate_ak'
        ac.long_name = 'atmosphere_hybrid_sigma_pressure_coordinate_ak'

        self.implementation.set_auxiliary_coordinate(self.field, ac,
                                                     axes=[_axis['z']],
                                                     copy=False)

        ac = self.implementation.initialise_AuxiliaryCoordinate()
        ac = self.coord_data(ac, bk_array, bk_bounds, units=_Units['1'])

        self.implementation.set_auxiliary_coordinate(self.field, ac,
                                                     axes=[_axis['z']],
                                                     copy=False)

        ac.id = 'UM_atmosphere_hybrid_sigma_pressure_coordinate_bk'
        ac.long_name = 'atmosphere_hybrid_sigma_pressure_coordinate_bk'

        return dc

    def create_cell_methods(self):
        '''Create the cell methods

    :Returns:

        `list`

        '''
        cell_methods = []

        LBPROC = self.lbproc
        LBTIM_IB = self.lbtim_ib
        tmean_proc = 0
        if LBTIM_IB in (2, 3) and LBPROC in (128, 192, 2176, 4224, 8320):
            tmean_proc = 128
            LBPROC -= 128

        # ------------------------------------------------------------
        # Area cell methods
        # ------------------------------------------------------------
        # -10: rotated latitude  (not an official axis code)
        # -11: rotated longitude (not an official axis code)
        if self.ix in (10, 11, 12, -10, -11) and self.iy in (
                10, 11, 12, -10, -11):
            cf_info = self.cf_info

            if 'where' in cf_info:
                cell_methods.append('area: mean')

                cell_methods.append(cf_info['where'])
                if 'over' in cf_info:
                    cell_methods.append(cf_info['over'])
            # --- End: if

            if LBPROC == 64:
                cell_methods.append('x: mean')

            # dch : do special zonal mean as as in pp_cfwrite

        # ------------------------------------------------------------
        # Vertical cell methods
        # ------------------------------------------------------------
        if LBPROC == 2048:
            cell_methods.append('z: mean')

        # ------------------------------------------------------------
        # Time cell methods
        # ------------------------------------------------------------
        if 't' in _axis:
            axis = 't'
        else:
            axis = 'time'

        if LBTIM_IB == 0 or LBTIM_IB == 1:
            if axis == 't':
                cell_methods.append(axis+': point')
        elif LBPROC == 4096:
            cell_methods.append(axis+': minimum')
        elif LBPROC == 8192:
            cell_methods.append(axis+': maximum')
        if tmean_proc == 128:
            if LBTIM_IB == 2:
                cell_methods.append(axis+': mean')
            elif LBTIM_IB == 3:
                cell_methods.append(axis+': mean within years')
                cell_methods.append(axis+': mean over years')
        # --- End: if

        if not cell_methods:
            return []

        cell_methods = self.implementation.initialise_CellMethod().create(
            ' '.join(cell_methods))

        for cm in cell_methods:
            cm.change_axes(_axis, inplace=True)

        return cell_methods

    def coord_axis(self, c, axiscode):
        '''TODO

        '''
        axis = _coord_axis.setdefault(axiscode, None)
        if axis is not None:
            c.axis = axis

        return c

    def coord_data(self, c, array=None, bounds=None, units=None,
                   fill_value=None, climatology=False):
        '''Set the data array of a coordinate construct.

    :Parameters:

        c: Coordinate construct

        data: array-like, optional
            The data array.

        bounds: array-like, optional
            The Cell bounds for the data array.

        units: `Units`, optional
            The units of the data array.

        fill_value: optional

        climatology: `bool`, optional
            Whether or not the coordinate construct is a time
            climatology. By default it is not.

    :Returns:

        Coordinate construct

        '''
        if array is not None:
            array = Data(array, units=units, fill_value=fill_value)
            self.implementation.set_data(c, array, copy=False)

        if bounds is not None:
            bounds_data = Data(bounds, units=units, fill_value=fill_value)
            bounds = self.implementation.initialise_Bounds()
            self.implementation.set_data(bounds, bounds_data, copy=False)
            self.implementation.set_bounds(c, bounds, copy=False)

        return c

    def coord_names(self, coord, axiscode):
        '''TODO

    :Parameters:

        coord: Coordinate construct

        axiscode: `int`

    :Returns:

        Coordinate construct

        '''
        standard_name = _coord_standard_name.setdefault(axiscode, None)

        if standard_name is not None:
            coord.set_property('standard_name', standard_name)
            coord.ncvar = standard_name
        else:
            long_name = _coord_long_name.setdefault(axiscode, None)
            if long_name is not None:
                coord.long_name = long_name

        return coord

    def coord_positive(self, c, axiscode, domain_axis_key):
        '''TODO

    :Parameters:

        c: Coordinate construct

        axiscode: `int`

        domain_axis_key: `str`

    :Returns:

        Coordinate construct

        '''
        positive = _coord_positive.setdefault(axiscode, None)
        if positive is not None:
            c.positive = positive
            if positive == 'down' and axiscode != 4:
                self.down_axes.add(domain_axis_key)
        # --- End: if

        return c

    def ctime(self, rec):
        '''TODO

        '''
        reftime = self.refUnits
        LBVTIME = tuple(self.header_vtime(rec))
        LBDTIME = tuple(self.header_dtime(rec))

        key = (LBVTIME, LBDTIME, self.refunits, self.calendar)
        ctime = _cached_ctime.get(key, None)
        if ctime is None:
            LBDTIME = list(LBDTIME)
            LBDTIME[0] = LBVTIME[0]

            ctime = cftime.datetime(*LBDTIME)

            if ctime < cftime.datetime(*LBVTIME):
                LBDTIME[0] += 1
                ctime = cftime.datetime(*LBDTIME)

            ctime = Data(ctime, reftime).array.item()
            _cached_ctime[key] = ctime

        return ctime

    def header_vtime(self, rec):
        '''Return the list [LBYR, LBMON, LBDAT, LBHR, LBMIN] for the given
    record.

    :Parameters:

        rec:

    :Returns:

        `list`

    **Examples:**

    >>> u.header_vtime(rec)
    [1991, 1, 1, 0, 0]

        '''
        return rec.int_hdr[lbyr:lbmin+1]

    def header_dtime(self, rec):
        '''Return the list [LBYRD, LBMOND, LBDATD, LBHRD, LBMIND] for the
    given record.

    :Parameters:

        rec:

    :Returns:

        `list`

    **Examples:**

    >>> u.header_dtime(rec)
    [1991, 2, 1, 0, 0]

        '''
        return rec.int_hdr[lbyrd:lbmind+1]

    def header_bz(self, rec):
        '''Return the list [BLEV, BRLEV, BHLEV, BHRLEV, BULEV, BHULEV] for the
    given record.

    :Parameters:

        rec:

    :Returns:

        `list`

    **Examples:**

    >>> u.header_bz(rec)

        '''
        real_hdr = rec.real_hdr
        return (
            real_hdr[blev:bhrlev+1].tolist() +  # BLEV, BRLEV, BHLEV, BHRLEV
            real_hdr[brsvd1:brsvd2+1].tolist()  # BULEV, BHULEV
        )

    def header_lz(self, rec):
        '''Return the list [LBLEV, LBUSER5] for the given record.

    :Parameters:

        rec:

    :Returns:

        `list`

    **Examples:**

    >>> u.header_lz(rec)

        '''
        int_hdr = rec.int_hdr
        return [int_hdr.item(lblev,), int_hdr.item(lbuser5,)]

    def header_z(self, rec):
        '''Return the list [LBLEV, LBUSER5, BLEV, BRLEV, BHLEV, BHRLEV, BULEV,
    BHULEV] for the given record.

    These header items are used by the compare_levels function in
    compare.c

    :Parameters:

        rec:

    :Returns:

        `list`

    **Examples:**

    >>> u.header_z(rec)

        '''
        return self.header_lz + self.header_bz

    @_manage_log_level_via_verbose_attr
    def create_data(self):
        '''Sets the data and data axes.

    :Returns:

        `Data`

        '''
        logger.info('Creating data:')  # pragma: no cover

        LBROW = self.lbrow
        LBNPT = self.lbnpt

        yx_shape = (LBROW, LBNPT)
        yx_size = LBROW * LBNPT

        nz = self.nz
        nt = self.nt
        recs = self.recs

        units = self.um_Units

        data_type_in_file = self.data_type_in_file

        filename = self.filename

        data_axes = [_axis['y'], _axis['x']]

        if len(recs) == 1:
            # --------------------------------------------------------
            # 0-d partition matrix
            # --------------------------------------------------------
            rec = recs[0]

            fill_value = rec.real_hdr.item(bmdi,)
            if fill_value == _BMDI_no_missing_data_value:
                fill_value = None

            data = Data(UMArray(filename=filename,
                                ndim=2,
                                shape=yx_shape,
                                size=yx_size,
                                dtype=data_type_in_file(rec),
                                header_offset=rec.hdr_offset,
                                data_offset=rec.data_offset,
                                disk_length=rec.disk_length,
                                fmt=self.fmt,
                                word_size=self.word_size,
                                byte_ordering=self.byte_ordering),
                        units=units,
                        fill_value=fill_value)

            logger.info(
                '    location = {}'.format(
                    yx_shape)
            )  # pragma: no cover
        else:
            # --------------------------------------------------------
            # 1-d or 2-d partition matrix
            # --------------------------------------------------------
            file_data_types = set()
            word_sizes = set()

            # Find the partition matrix shape
            pmshape = [n for n in (nt, nz) if n > 1]
            pmndim = len(pmshape)

            partitions = []
            empty_list = []
            partitions_append = partitions.append

            zero_to_LBROW = (0, LBROW)
            zero_to_LBNPT = (0, LBNPT)

            if pmndim == 1:
                # ----------------------------------------------------
                # 1-d partition matrix
                # ----------------------------------------------------
                data_ndim = 3
                if nz > 1:
                    pmaxes = [_axis[self.z_axis]]
                    data_shape = (nz, LBROW, LBNPT)
                    data_size = nz * yx_size
                else:
                    pmaxes = [_axis['t']]
                    data_shape = (nt, LBROW, LBNPT)
                    data_size = nt * yx_size

                partition_shape = [1, LBROW, LBNPT]

                for i, rec in enumerate(recs):
                    # Find the data type of the array in the file
                    file_data_type = data_type_in_file(rec)
                    file_data_types.add(file_data_type)

                    subarray = UMArray(filename=filename,
                                       ndim=2,
                                       shape=yx_shape,
                                       size=yx_size,
                                       dtype=file_data_type,
                                       header_offset=rec.hdr_offset,
                                       data_offset=rec.data_offset,
                                       disk_length=rec.disk_length,
                                       fmt=self.fmt,
                                       word_size=self.word_size,
                                       byte_ordering=self.byte_ordering)

                    location = [(i, i+1), zero_to_LBROW, zero_to_LBNPT]

                    partitions_append(Partition(
                        subarray=subarray,
                        location=location,
                        shape=partition_shape,
                        axes=data_axes,
                        flip=empty_list,
                        part=empty_list,
                        Units=units
                    ))

                    logger.info(
                        '    header_offset = {}, location = {}, '
                        'subarray[...].max() = {}'.format(
                            rec.hdr_offset, location, subarray[...].max())
                    )  # pragma: no cover
                # --- End: for

                # Populate the 1-d partition matrix
                matrix = numpy_array(partitions, dtype=object)
            else:
                # ----------------------------------------------------
                # 2-d partition matrix
                # ----------------------------------------------------
                pmaxes = [_axis['t'], _axis[self.z_axis]]
                data_shape = (nt, nz, LBROW, LBNPT)
                data_size = nt * nz * yx_size
                data_ndim = 4

                partition_shape = [1, 1, LBROW, LBNPT]

                for i, rec in enumerate(recs):
                    # Find T and Z axis indices
                    t, z = divmod(i, nz)

                    # Find the data type of the array in the file
                    file_data_type = data_type_in_file(rec)
                    file_data_types.add(file_data_type)

                    subarray = UMArray(filename=filename,
                                       ndim=2,
                                       shape=yx_shape,
                                       size=yx_size,
                                       dtype=file_data_type,
                                       header_offset=rec.hdr_offset,
                                       data_offset=rec.data_offset,
                                       disk_length=rec.disk_length,
                                       fmt=self.fmt,
                                       word_size=self.word_size,
                                       byte_ordering=self.byte_ordering)

                    location = [(t, t+1), (z, z+1), zero_to_LBROW,
                                zero_to_LBNPT]

                    partitions_append(Partition(
                        subarray=subarray,
                        location=location,
                        shape=partition_shape,
                        axes=data_axes,
                        flip=empty_list,
                        part=empty_list,
                        Units=units))

                    logger.info(
                        '    location = {}, subarray[...].max() = {}'.format(
                            location, subarray[...].max())
                    )  # pragma: no cover
                # --- End: for

                # Populate the 2-d partition matrix
                matrix = numpy_array(partitions, dtype=object)
                matrix.resize(pmshape)
            # --- End: if

            data_axes = pmaxes + data_axes

            # Set the data array
            fill_value = recs[0].real_hdr.item(bmdi,)
            if fill_value == _BMDI_no_missing_data_value:
                fill_value = None

            data = Data(units=units, fill_value=fill_value)

            data._axes = data_axes
            data._shape = data_shape
            data._ndim = data_ndim
            data._size = data_size
            data.partitions = PartitionMatrix(matrix, pmaxes)
            data.dtype = numpy_result_type(*file_data_types)
        # --- End: if

        self.data = data
        self.data_axes = data_axes

        return data

    def decode_lbexp(self):
        '''Decode the integer value of LBEXP in the PP header into a runid.

    If this value has already been decoded, then it will be returned
    from the cache, otherwise the value will be decoded and then added
    to the cache.

    :Returns:

        `str`
           A string derived from LBEXP. If LBEXP is a negative integer
           then that number is returned as a string.

    **Examples:**

    >>> self.decode_lbexp()
    'aaa5u'
    >>> self.decode_lbexp()
    '-34'

        '''
        LBEXP = self.int_hdr[lbexp]

        runid = _cached_runid.get(LBEXP, None)
        if runid is not None:
            # Return a cached decoding of this LBEXP
            return runid

        if LBEXP < 0:
            runid = str(LBEXP)
        else:
            # Convert LBEXP to a binary string, filled out to 30 bits with
            # zeros
            bits = bin(LBEXP)
            bits = bits.lstrip('0b').zfill(30)

            # Step through 6 bits at a time, converting each 6 bit chunk into
            # a decimal integer, which is used as an index to the characters
            # lookup list.
            runid = []
            for i in range(0, 30, 6):
                index = int(bits[i:i+6], 2)
                if index < _n_characters:
                    runid.append(_characters[index])
            # --- End: for

            runid = ''.join(runid)

        # Enter this runid into the cache
        _cached_runid[LBEXP] = runid

        # Return the runid
        return runid

    def dtime(self, rec):
        '''Return the elapsed time since the data time of the given record.

    :Parameters:

        rec:

    :Returns:

        `float`

    **Examples:**

    >>> u.dtime(rec)
    31.5

        '''
        reftime = self.refUnits
        units = self.refunits
        calendar = self.calendar

        LBDTIME = tuple(self.header_dtime(rec))

        key = (LBDTIME, units, calendar)
        time = _cached_date2num.get(key, None)
        if time is None:
            # It is important to use the same time_units as vtime
            try:
                if self.calendar == 'gregorian':
                    time = netCDF4_date2num(
                        datetime(*LBDTIME), units, calendar)
                else:
                    time = netCDF4_date2num(
                        cftime.datetime(*LBDTIME), units, calendar)

                _cached_date2num[key] = time
            except ValueError:
                time = numpy_nan  # ppp
        # --- End: if

        return time

    def fdr(self):
        '''Return a the contents of PP field headers as strings.

    This is a bit like printfdr in the UKMO IDL PP library.

    :Returns:

        `list`

        '''
        out2 = []
        for i, rec in enumerate(self.recs):
            out = ['Field {0}:'.format(i)]

            x = ['{0}::{1}'.format(name, value)
                 for name, value in zip(_header_names,
                                        self.int_hdr + self.real_hdr)]

            x = textwrap.fill(' '.join(x), width=79)
            out.append(x.replace('::', ': '))

            if self.extra:
                out.append('EXTRA DATA:')
                for key in sorted(self.extra):
                    out.append('{0}: {1}'.format(key, str(self.extra[key])))
            # --- End: if

            out.append('file: '+self.filename)
            out.append('format, byte order, word size: {}, {}, {}'.format(
                self.fmt, self.byte_ordering, self.word_size))

            out.append('')

            out2.append('\n'.join(out))

        return out2

    def latitude_longitude_2d_aux_coordinates(self, yc, xc):
        '''TODO

    :Parameters:

        yc: `DimensionCoordinate`

        xc: `DimensionCoordinate`

    :Returns:

        `None`

        '''
        BDX = self.bdx
        BDY = self.bdy
        LBNPT = self.lbnpt
        LBROW = self.lbrow
        BPLAT = self.bplat
        BPLON = self.bplon

        # Create the unrotated latitude and longitude arrays if we
        # couldn't find them in the cache
        cache_key = (LBNPT, LBROW, BDX, BDY, BPLAT, BPLON)
        lat, lon = _cached_latlon.get(cache_key, (None, None))

        if lat is None:
            lat, lon = self.unrotated_latlon(yc.varray, xc.varray,
                                             BPLAT, BPLON)

            atol = self.atol
            if abs(BDX) >= atol and abs(BDY) >= atol:
                _cached_latlon[cache_key] = (lat, lon)
        # --- End: if

        if xc.has_bounds() and yc.has_bounds():  # TODO push to implementation
            cache_key = ('bounds',) + cache_key
            lat_bounds, lon_bounds = _cached_latlon.get(
                cache_key, (None, None))
            if lat_bounds is None:
                xb = numpy_empty(xc.size+1)
                xb[:-1] = xc.bounds.subspace[:, 0].squeeze(1).array
                xb[-1] = xc.bounds.datum(-1, 1)

                yb = numpy_empty(yc.size+1)
                yb[:-1] = yc.bounds.subspace[:, 0].squeeze(1).array
                yb[-1] = yc.bounds.datum(-1, 1)

                temp_lat_bounds, temp_lon_bounds = self.unrotated_latlon(
                    yb, xb, BPLAT, BPLON)

                lat_bounds = numpy_empty(lat.shape + (4,))
                lon_bounds = numpy_empty(lon.shape + (4,))

                lat_bounds[..., 0] = temp_lat_bounds[0:-1, 0:-1]
                lon_bounds[..., 0] = temp_lon_bounds[0:-1, 0:-1]

                lat_bounds[..., 1] = temp_lat_bounds[1:, 0:-1]
                lon_bounds[..., 1] = temp_lon_bounds[1:, 0:-1]

                lat_bounds[..., 2] = temp_lat_bounds[1:, 1:]
                lon_bounds[..., 2] = temp_lon_bounds[1:, 1:]

                lat_bounds[..., 3] = temp_lat_bounds[0:-1, 1:]
                lon_bounds[..., 3] = temp_lon_bounds[0:-1, 1:]

                atol = self.atol
                if abs(BDX) >= atol and abs(BDY) >= atol:
                    _cached_latlon[cache_key] = (lat_bounds, lon_bounds)
        else:
            lat_bounds = None
            lon_bounds = None

        axes = [_axis['y'], _axis['x']]

        for axiscode, array, bounds in zip(
                (10, 11), (lat, lon), (lat_bounds, lon_bounds)):
            # ac = AuxiliaryCoordinate()
            ac = self.implementation.initialise_AuxiliaryCoordinate()
            ac = self.coord_data(
                ac, array, bounds=bounds,
                units=_axiscode_to_Units.setdefault(axiscode, None)
            )
            ac = self.coord_names(ac, axiscode)

            key = self.implementation.set_auxiliary_coordinate(self.field, ac,
                                                               axes=axes,
                                                               copy=False)

    def model_level_number_coordinate(self, aux=False):
        '''model_level_number dimension or auxiliary coordinate

    :Parameters:

        aux: `bool`

    :Returns:

        out : `AuxiliaryCoordinate` or `DimensionCoordinate` or `None`

    '''
        array = tuple([rec.int_hdr.item(lblev,) for rec in self.z_recs])

        key = array
        c = _cached_model_level_number_coordinate.get(key, None)

        if c is not None:
            if aux:
                self.field.insert_aux(c, axes=[_axis['z']], copy=True)
                self.implementation.set_auxiliary_coordinate(
                    self.field, c, axes=[_axis['z']], copy=True)
            else:
                self.implementation.set_dimension_coordinate(
                    self.field, c, axes=[_axis['z']], copy=True)
        else:
            array = numpy_array(array, dtype=self.int_hdr_dtype)

            if array.min() < 0:
                return

            array = numpy_where(array == 9999, 0, array)

            axiscode = 5

            if aux:
                ac = self.implementation.initialise_AuxiliaryCoordinate()
                ac = self.coord_data(ac, array, units=Units('1'))
                ac = self.coord_names(ac, axiscode)
                self.implementation.set_auxiliary_coordinate(
                    self.field, ac, axes=[_axis['z']], copy=False)

            else:
                dc = self.implementation.initialise_DimensionCoordinate()
                dc = self.coord_data(dc, array, units=Units('1'))
                dc = self.coord_names(dc, axiscode)
                dc = self.coord_axis(dc, axiscode)
                self.implementation.set_dimension_coordinate(
                    self.field, dc, axes=[_axis['z']], copy=False)

            _cached_model_level_number_coordinate[key] = c

        return c

    def data_type_in_file(self, rec):
        '''Return the data type of the data array.

    :Parameters:

        rec: `umfile.Rec`

    :Returns:

        `numpy.dtype`

        '''
        # Find the data type
        if rec.int_hdr.item(lbuser2,) == 3:
            # Boolean
            return numpy_dtype(bool)
        else:
            # Int or float
            return rec.get_type_and_num_words()[0]
#            rec_file = rec.file
#            # data_type = rec_file.c_interface.get_type_and_length(
#            data_type = rec_file.c_interface.get_type_and_num_words(
#                rec.int_hdr)[0]
#            if data_type == 'int':
#                # Integer
#                data_type = 'int%d' % (rec_file.word_size * 8)
#            else:
#                # Float
#                data_type = 'float%d' % (rec_file.word_size * 8)
#        # --- End: if
#
#        return numpy_dtype(data_type)

    def printfdr(self):
        '''Print out the contents of PP field headers.

    This is a bit like printfdr in the UKMO IDL PP library.

    **Examples:**

    >>> u.printfdr()

        '''
        for header in self.fdr():
            logger.info(header)

    def pseudolevel_coordinate(self, LBUSER5):
        '''TODO

        '''
        if self.nz == 1:
            array = numpy_array((LBUSER5,), dtype=self.int_hdr_dtype)
        else:
            # 'Z' aggregation has been done along the pseudolevel axis
            array = numpy_array([rec.int_hdr.item(lbuser5,)
                                 for rec in self.z_recs],
                                dtype=self.int_hdr_dtype)
            self.z_axis = 'p'

        axiscode = 40

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(
            dc, array, units=_axiscode_to_Units.setdefault(axiscode, None))
        self.implementation.set_properties(dc, {'long_name': 'pseudolevel'})
        dc.id = 'UM_pseudolevel'

        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisP = self.implementation.set_domain_axis(self.field, da)
        _axis['p'] = axisP

        self.implementation.set_dimension_coordinate(
            self.field, dc, axes=[_axis['p']], copy=False)

        return dc

    def radiation_wavelength_coordinate(self, rwl, rwl_units):
        '''TODO

        '''
        array = numpy_array((rwl,), dtype=float)
        bounds = numpy_array(((0.0, rwl)), dtype=float)

        units = _Units.get(rwl_units, None)
        if units is None:
            units = Units(rwl_units)
            _Units[rwl_units] = units

        axiscode = -20
        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(dc, array, bounds, units=units)
        dc = self.coords_names(dc, axiscode)

        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisR = self.implementation.set_domain_axis(self.field, da)
        _axis['r'] = axisR

        self.implementation.set_dimension_coordinate(
            self.field, dc, axes=[_axis['r']], copy=False)

        return dc

    def reference_time_Units(self):
        '''TODO

        '''
        LBYR = self.int_hdr[lbyr]
        time_units = 'days since {}-1-1'.format(LBYR)
        calendar = self.calendar

        key = time_units+' calendar='+calendar
        units = _Units.get(key, None)
        if units is None:
            units = Units(time_units, calendar)
            _Units[key] = units

        self.refUnits = units
        self.refunits = time_units

        return units

    def size_1_height_coordinate(self, axiscode, height, units):
        '''TODO

        '''
        # Create the height coordinate from the information given in the
        # STASH to standard_name conversion table

        key = (axiscode, height, units)
        dc = _cached_size_1_height_coordinate.get(key, None)

        da = self.implementation.initialise_DomainAxis(size=1)
        axisZ = self.implementation.set_domain_axis(self.field, da)
        _axis['z'] = axisZ

        if dc is not None:
            copy = True
        else:
            height_units = _Units.get(units, None)
            if height_units is None:
                height_units = Units(units)
                _Units[units] = height_units

            array = numpy_array((height,), dtype=float)

            dc = self.implementation.initialise_DimensionCoordinate()
            dc = self.coord_data(dc, array, units=height_units)
            dc = self.coord_positive(dc, axiscode, _axis['z'])
            dc = self.coord_axis(dc, axiscode)
            dc = self.coord_names(dc, axiscode)

            _cached_size_1_height_coordinate[key] = dc
            copy = False

        self.implementation.set_dimension_coordinate(self.field, dc,
                                                     axes=[_axis['z']],
                                                     copy=copy)
        return dc

    def test_um_condition(self, um_condition, LBCODE, BPLAT, BPLON):
        '''Return `True` if a field satisfies the condition specified for a
    STASH code to standard name conversion.

    :Parameters:

        um_condition: `str`

        LBCODE: `int`

        BPLAT: `float`

        BPLON: `float`

    :Returns:

        `bool`
            `True` if a field satisfies the condition specified,
            `False` otherwise.

    **Examples:**

    >>> ok = u.test_um_condition('true_latitude_longitude', ...)

        '''
        if um_condition == 'true_latitude_longitude':
            if LBCODE in _true_latitude_longitude_lbcodes:
                return True

            # Check pole location in case of incorrect LBCODE
            atol = self.atol
            if (abs(BPLAT-90.0) <= atol + RTOL()*90.0 and abs(BPLON) <= atol):
                return True

        elif um_condition == 'rotated_latitude_longitude':
            if LBCODE in _rotated_latitude_longitude_lbcodes:
                return True

            # Check pole location in case of incorrect LBCODE
            atol = self.atol
            if not (abs(BPLAT-90.0) <= atol + RTOL()*90.0 and
                    abs(BPLON) <= atol):
                return True

        else:
            raise ValueError(
                "Unknown UM condition in STASH code conversion table: "
                "{!r}".format(um_condition)
            )

        # Still here? Then the condition has not been satisfied.
        return

    def test_um_version(self, valid_from, valid_to, um_version):
        '''Return `True` if the UM version applicable to this field is within
    the given range.

    If possible, the UM version is derived from the PP header and
    stored in the metadata object. Otherwise it is taken from the
    *um_version* parameter.

    :Parameters:

        valid_from: `int`, `float` or `None`

        valid_to: `int`, `float` or `None`

        um_version: `int` or `float`

    :Returns:

        `bool`
            `True` if the UM version applicable to this field
            construct is within the range, `False` otherwise.

    **Examples:**

    >>> ok = u.test_um_version(401, 505, 1001)
    >>> ok = u.test_um_version(401, None, 606.3)
    >>> ok = u.test_um_version(None, 405, 401)

        '''
        if valid_to is None:
            if valid_from is None:
                return True

            if valid_from <= um_version:
                return True
        elif valid_from is None:
            if um_version <= valid_to:
                return True
        elif valid_from <= um_version <= valid_to:
            return True

        return False

    def time_coordinate(self, axiscode):
        '''Return the T dimension coordinate

    :Parameters:

        axiscode: `int`

    :Returns:

        `DimensionCoordinate`

        '''
        recs = self.t_recs

        vtimes = numpy_array([self.vtime(rec) for rec in recs], dtype=float)
        dtimes = numpy_array([self.dtime(rec) for rec in recs], dtype=float)

        if numpy_isnan(vtimes.sum()) or numpy_isnan(dtimes.sum()):
            return  # ppp

        IB = self.lbtim_ib

        if IB <= 1 or vtimes.item(0,) >= dtimes.item(0,):
            array = vtimes
            bounds = None
            climatology = False
        elif IB == 3:
            # The field is a time mean from T1 to T2 for each year
            # from LBYR to LBYRD
            ctimes = numpy_array([self.ctime(rec) for rec in recs])
            array = 0.5*(vtimes + ctimes)
            bounds = numpy_column_stack((vtimes, dtimes))
            climatology = True
        else:
            array = 0.5*(vtimes + dtimes)
            bounds = numpy_column_stack((vtimes, dtimes))
            climatology = False

        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisT = self.implementation.set_domain_axis(self.field, da)
        _axis['t'] = axisT

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(dc, array, bounds,
                             units=self.refUnits,
                             climatology=climatology)
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        self.implementation.set_dimension_coordinate(self.field, dc,
                                                     axes=[_axis['t']],
                                                     copy=False)
        return dc

    def time_coordinate_from_extra_data(self, axiscode, axis):
        '''TODO

        '''
        extra = self.extra
        array = extra[axis]
        bounds = extra.get(axis+'_bounds', None)

        calendar = self.calendar
        if calendar == '360_day':
            units = _Units['360_day 0-1-1']
        elif calendar == 'gregorian':
            units = _Units['gregorian 1752-09-13']
        elif calendar == '365_day':
            units = _Units['365_day 1752-09-13']
        else:
            units = None

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(dc, array, bounds, units=units)
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)
        self.implementation.set_dimension_coordinate(self.field, dc,
                                                     axes=[_axis[axis]],
                                                     copy=False)

        return dc

    def time_coordinate_from_um_timeseries(self, axiscode, axis):
        '''TODO

        '''
        # This PP/FF field is a timeseries. The validity time is
        # taken to be the time for the first sample, the data time
        # for the last sample, with the others evenly between.
        rec = self.recs[0]
        vtime = self.vtime(rec)
        dtime = self.dtime(rec)

        size = self.lbuser3 - 1.0
        delta = (dtime - vtime)/size

        array = numpy_arange(vtime, vtime+delta*size, size, dtype=float)

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(dc, array, units=units)
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)
        self.implementation.set_dimension_coordinate(self.field, dc,
                                                     axes=[_axis[axis]],
                                                     copy=False)
        return dc

    def vtime(self, rec):
        '''Return the elapsed time since the validity time of the given
    record.

    :Parameters:

        rec:

    :Returns:

        `float`

    **Examples:**

    >>> u.vtime(rec)
    31.5

        '''
        reftime = self.refUnits
        units = self.refunits
        calendar = self.calendar

        LBVTIME = tuple(self.header_vtime(rec))

        key = (LBVTIME, units, calendar)

        time = _cached_date2num.get(key, None)
        if time is None:
            # It is important to use the same time_units as dtime
            try:
                if self.calendar == 'gregorian':
                    time = netCDF4_date2num(
                        datetime(*LBVTIME), units, calendar)
                else:
                    time = netCDF4_date2num(cftime.datetime(*LBVTIME),
                                            units, calendar)

                _cached_date2num[key] = time
            except ValueError:
                time = numpy_nan  # ppp
        # --- End: if

        return time

    def dddd(self):
        '''TODO

        '''
        for axis_code, extra_type in zip((11, 10), ('x', 'y')):
            coord_type = extra_type + '_domain_bounds'

            if coord_type in p.extra:
                p.extra[coord_type]
                # Create, from extra data, an auxiliary coordinate
                # with 1) data and bounds, if the upper and lower
                # bounds have no missing values; or 2) data but no
                # bounds, if the upper bound has missing values
                # but the lower bound does not.

                # Should be the axis which has axis_code 13
                file_position = ppfile.tell()
                bounds = p.extra[coord_type][...]

                # Reset the file pointer after reading the extra
                # data into a numpy array
                ppfile.seek(file_position, os.SEEK_SET)
                data = None
                # dch also test in bmdi?:
                if numpy_any(bounds[..., 1] == _pp_rmdi):
                    # dch also test in bmdi?:
                    if not numpy_any(bounds[..., 0] == _pp_rmdi):
                        data = bounds[..., 0]
                    bounds = None
                else:
                    data = numpy_mean(bounds, axis=1)

                if (data, bounds) != (None, None):
                    aux = 'aux%(auxN)d' % locals()
                    auxN += 1  # Increment auxiliary number

                    coord = _create_Coordinate(
                        domain, aux, axis_code, p=p, array=data, aux=True,
                        bounds_array=bounds, pubattr={'axis': None},
                        # DCH xdim? should be the axis which has axis_code 13:
                        dimensions=[xdim])
            else:
                coord_type = '{0}_domain_lower_bound'.format(extra_type)
                if coord_type in p.extra:
                    # Create, from extra data, an auxiliary
                    # coordinate with data but no bounds, if the
                    # data noes not contain any missing values
                    file_position = ppfile.tell()
                    data = p.extra[coord_type][...]
                    # Reset the file pointer after reading the
                    # extra data into a numpy array
                    ppfile.seek(file_position, os.SEEK_SET)
                    if not numpy_any(data == _pp_rmdi):  # dch + test in bmdi
                        aux = 'aux%(auxN)d' % locals()
                        auxN += 1  # Increment auxiliary number
                        coord = _create_Coordinate(
                            domain, aux, axis_code, p=p, aux=True,
                            array=numpy_array(data), pubattr={'axis': None},
                            dimensions=[xdim]
                        )  # DCH xdim?
            # --- End: if
        # --- End: for

    def unrotated_latlon(self, rotated_lat, rotated_lon, pole_lat, pole_lon):
        '''Create 2-d arrays of unrotated latitudes and longitudes.

    :Parameters:

        rotated_lat: `numpy.ndarray`

        rotated_lon: `numpy.ndarray`

        pole_lat: `float`

        pole_lon: `float`

    :Returns:

        lat, lon: `numpy.ndarray`, `numpy.ndarray`

        '''
        # Make sure rotated_lon and pole_lon is in [0, 360)
        pole_lon = pole_lon % 360.0

        # Convert everything to radians
        pole_lon *= _pi_over_180
        pole_lat *= _pi_over_180

        cos_pole_lat = numpy_cos(pole_lat)
        sin_pole_lat = numpy_sin(pole_lat)

        # Create appropriate copies of the input rotated arrays
        rot_lon = rotated_lon.copy()
        rot_lat = rotated_lat.view()

        # Make sure rotated longitudes are between -180 and 180
        rot_lon %= 360.0
        rot_lon = numpy_where(rot_lon < 180.0, rot_lon, rot_lon-360)

        # Create 2-d arrays of rotated latitudes and longitudes in radians
        nlat = rot_lat.size
        nlon = rot_lon.size
        rot_lon = numpy_resize(numpy_deg2rad(rot_lon), (nlat, nlon))
        rot_lat = numpy_resize(numpy_deg2rad(rot_lat), (nlon, nlat))
        rot_lat = numpy_transpose(rot_lat, axes=(1, 0))

        # Find unrotated latitudes
        CPART = numpy_cos(rot_lon) * numpy_cos(rot_lat)
        sin_rot_lat = numpy_sin(rot_lat)
        x = cos_pole_lat * CPART + sin_pole_lat * sin_rot_lat
        x = numpy_clip(x, -1.0, 1.0)
        unrotated_lat = numpy_arcsin(x)

        # Find unrotated longitudes
        x = -cos_pole_lat*sin_rot_lat + sin_pole_lat*CPART
        x /= numpy_cos(unrotated_lat)
        # dch /0 or overflow here? surely lat could be ~+-pi/2? if so,
        # does x ~ cos(lat)?
        x = numpy_clip(x, -1.0, 1.0)
        unrotated_lon = -numpy_arccos(x)

        unrotated_lon = numpy_where(rot_lon > 0.0,
                                    -unrotated_lon, unrotated_lon)
        if pole_lon >= self.atol:
            SOCK = pole_lon - numpy_pi
        else:
            SOCK = 0
        unrotated_lon += SOCK

        # Convert unrotated latitudes and longitudes to degrees
        unrotated_lat = numpy_rad2deg(unrotated_lat)
        unrotated_lon = numpy_rad2deg(unrotated_lon)

        # Return unrotated latitudes and longitudes
        return (unrotated_lat, unrotated_lon)

    def xy_coordinate(self, axiscode, axis):
        '''Create an X or Y dimension coordinate from header entries or extra
    data.

    :Parameters:

        axiscode: `int`

        axis: `str`
            'x' or 'y'

    :Returns:

        `str, `DimensionCoordinate`

        '''
        if axis == 'y':
            delta = self.bdy
            origin = self.real_hdr[bzy]
            size = self.lbrow

            da = self.implementation.initialise_DomainAxis(size=size)
            axis_key = self.implementation.set_domain_axis(self.field, da)
            _axis['y'] = axis_key
        else:
            delta = self.bdx
            origin = self.real_hdr[bzx]
            size = self.lbnpt

            da = self.implementation.initialise_DomainAxis(size=size)
            axis_key = self.implementation.set_domain_axis(self.field, da)
            _axis['x'] = axis_key

        if abs(delta) > self.atol:
            # Create regular coordinates from header items
            if axiscode == 11 or axiscode == -11:
                origin -= divmod(origin + delta*size, 360.0)[0] * 360
                while origin + delta*size > 360.0:
                    origin -= 360.0
                while origin + delta*size < -360.0:
                    origin += 360.0
            # --- End: if

            array = numpy_arange(origin+delta, origin+delta*(size+0.5), delta,
                                 dtype=float)

            # Create the coordinate bounds
            if axiscode in (13, 31, 40, 99):
                # The following axiscodes do not have bounds:
                # 13 = Site number (set of parallel rows or columns
                #      e.g.Time series)
                # 31 = Logarithm to base 10 of pressure in mb
                # 40 = Pseudolevel
                # 99 = Other
                bounds = None
            else:
                delta_by_2 = 0.5 * delta
                bounds = numpy_empty((size, 2), dtype=float)
                bounds[:, 0] = array - delta_by_2
                bounds[:, 1] = array + delta_by_2
        else:
            # Create coordinate from extra data
            array = self.extra.get(axis, None)
            bounds = self.extra.get(axis+'_bounds', None)

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(
            dc, array, bounds,
            units=_axiscode_to_Units.setdefault(axiscode, None)
        )
        dc = self.coord_positive(dc, axiscode, axis_key)  # _axis[axis])
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        key = self.implementation.set_dimension_coordinate(self.field, dc,
                                                           axes=[axis_key],
                                                           copy=False)

        return key, dc

    @_manage_log_level_via_verbose_attr
    def z_coordinate(self, axiscode):
        '''Create a Z dimension coordinate from BLEV

    :Parameters:

        axiscode: `int`

    :Returns:

        `DimensionCoordinate`

        '''
        logger.info(
            'Creating Z coordinates and bounds from BLEV, BRLEV and '
            'BRSVD1:'
        )  # pragma: no cover

        z_recs = self.z_recs
        array = tuple([rec.real_hdr.item(blev,) for rec in z_recs])
        bounds0 = tuple(
            [rec.real_hdr[brlev] for rec in z_recs])  # lower level boundary
        bounds1 = tuple([rec.real_hdr[brsvd1] for rec in z_recs])  # bulev
        if _coord_positive.get(axiscode, None) == 'down':
            bounds0, bounds1 = bounds1, bounds0

            #        key = (axiscode, array, bounds0, bounds1)
#        dc = _cached_z_coordinate.get(key, None)

#        if dc is not None:
#            copy = True
#        else:
        copy = False
        array = numpy_array(array, dtype=float)
        bounds0 = numpy_array(bounds0, dtype=float)
        bounds1 = numpy_array(bounds1, dtype=float)
        bounds = numpy_column_stack((bounds0, bounds1))

        if (bounds0 == bounds1).all():
            bounds = None
        else:
            bounds = numpy_column_stack((bounds0, bounds1))

        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisZ = self.implementation.set_domain_axis(self.field, da)
        _axis['z'] = axisZ

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(
            dc, array, bounds=bounds,
            units=_axiscode_to_Units.setdefault(axiscode, None)
        )
        dc = self.coord_positive(dc, axiscode, _axis['z'])
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        self.implementation.set_dimension_coordinate(
            self.field, dc, axes=[_axis['z']], copy=copy)

        logger.info('    ' + dc.dump(display=False))  # pragma: no cover

        return dc

    @_manage_log_level_via_verbose_attr
    def z_reference_coordinate(self, axiscode):
        '''TODO

        '''
        logger.info(
            'Creating Z reference coordinates from BRLEV'
        )  # pragma: no cover

        array = numpy_array([rec.real_hdr.item(brlev,) for rec in self.z_recs],
                            dtype=float)

        LBVC = self.lbvc

        key = (axiscode, LBVC, array)
        dc = _cached_z_reference_coordinate.get(key, None)

        if dc is not None:
            copy = True
        else:
            if not 128 <= LBVC <= 139:
                bounds = []
                for rec in self.z_recs:
                    BRLEV = rec.real_hdr.item(brlev,)
                    BRSVD1 = rec.real_hdr.item(brsvd1,)

                    if abs(BRSVD1-BRLEV) >= ATOL:
                        bounds = None
                        break

                    bounds.append((BRLEV, BRSVD1))
            else:
                bounds = None

            if bounds:
                bounds = numpy_array((bounds,), dtype=float)

            dc = self.implementation.initialise_DimensionCoordinate()
            dc = self.coord_data(
                dc, array, bounds,
                units=_axiscode_to_Units.setdefault(axiscode, None)
            )
            dc = self.coord_axis(dc, axiscode)
            dc = self.coord_names(dc, axiscode)

            if not dc.get('positive', True):  # ppp
                dc.flip(i=True)

            _cached_z_reference_coordinate[key] = dc
            copy = False
        # --- End: if

        self.implementation.set_dimension_coordinate(self.field, dc,
                                                     axes=[_axis['z']],
                                                     copy=copy)

        return dc

# --- End: class

# _stash2standard_name = {}
#
# def load_stash2standard_name(table=None, delimiter='!', merge=True):
#     '''Load a STASH to standard name conversion table.
#
# :Parameters:
#
#     table: `str`, optional
#         Use the conversion table at this file location. By default the
#         table will be looked for at
#         ``os.path.join(os.path.dirname(cf.__file__),'etc/STASH_to_CF.txt')``
#
#     delimiter: `str`, optional
#         The delimiter of the table columns. By default, ``!`` is taken
#         as the delimiter.
#
#     merge: `bool`, optional
#         If *table* is None then *merge* is taken as False, regardless
#         of its given value.
#
# :Returns:
#
#     `None`
#
# *Examples:*
#
# >>> load_stash2standard_name()
# >>> load_stash2standard_name('my_table.txt')
# >>> load_stash2standard_name('my_table2.txt', ',')
# >>> load_stash2standard_name('my_table3.txt', merge=True)
# >>> load_stash2standard_name('my_table4.txt', merge=False)
#
#     '''
#     # 0  Model
#     # 1  STASH code
#     # 2  STASH name
#     # 3  units
#     # 4  valid from UM vn
#     # 5  valid to   UM vn
#     # 6  standard_name
#     # 7  CF extra info
#     # 8  PP extra info
#
#     if table is None:
#         # Use default conversion table
#         merge = False
#         package_path = os.path.dirname(__file__)
#         table = os.path.join(package_path, 'etc/STASH_to_CF.txt')
#
#     lines = csv.reader(open(table, 'r'),
#                        delimiter=delimiter, skipinitialspace=True)
#
#     raw_list = []
#     [raw_list.append(line) for line in lines]
#
#     # Get rid of comments
#     for line in raw_list[:]:
#         if line[0].startswith('#'):
#             raw_list.pop(0)
#             continue
#         break
#
#     # Convert to a dictionary which is keyed by (submodel, STASHcode)
#     # tuples
#
#     (model, stash, name,
#      units,
#      valid_from, valid_to,
#      standard_name, cf, pp) = list(range(9))
#
#     stash2sn = {}
#     for x in raw_list:
#         key = (int(x[model]), int(x[stash]))
#
#         if not x[units]:
#             x[units] = None
#
#         try:
#             cf_info = {}
#             if x[cf]:
#                 for d in x[7].split():
#                     if d.startswith('height='):
#                         cf_info['height'] = re.split(_number_regex, d,
#                                                      re.IGNORECASE)[1:4:2]
#                         if cf_info['height'] == '':
#                             cf_info['height'][1] = '1'
#
#                     if d.startswith('below_'):
#                         cf_info['below'] = re.split(_number_regex, d,
#                                                      re.IGNORECASE)[1:4:2]
#                        if cf_info['below'] == '':
#                             cf_info['below'][1] = '1'
#
#                     if d.startswith('where_'):
#                         cf_info['where'] = d.replace('where_', 'where ', 1)
#                     if d.startswith('over_'):
#                         cf_info['over'] = d.replace('over_', 'over ', 1)
#
#             x[cf] = cf_info
#         except IndexError:
#             pass
#
#         try:
#             x[valid_from] = float(x[valid_from])
#         except ValueError:
#             x[valid_from] = None
#
#         try:
#             x[valid_to] = float(x[valid_to])
#         except ValueError:
#             x[valid_to] = None
#
#         x[pp] = x[pp].rstrip()
#
#         line = (x[name:],)
#
#         if key in stash2sn:
#             stash2sn[key] += line
#         else:
#             stash2sn[key] = line
#     # --- End: for
#
#     if not merge:
#         _stash2standard_name.clear()
#
#     _stash2standard_name.update(stash2sn)


# ---------------------------------------------------------------------
# Create the STASH code to standard_name conversion dictionary
# ---------------------------------------------------------------------
stash2standard_name = load_stash2standard_name()


class UMRead(cfdm.read_write.IORead):
    '''TODO

    '''
    @_manage_log_level_via_verbosity
    def read(self, filename, um_version=405,
             aggregate=True, endian=None, word_size=None,
             set_standard_name=True, height_at_top_of_model=None,
             fmt=None, chunk=True, verbose=None):
        '''Read fields from a PP file or UM fields file.

    The file may be big or little endian, 32 or 64 bit

    :Parameters:

        filename: `file` or `str`
            A string giving the file name, or an open file object,
            from which to read fields.

        um_version: number, optional
            The Unified Model (UM) version to be used when decoding
            the PP header. Valid versions are, for example, ``402``
            (v4.2), ``606.3`` (v6.6.3) and ``1001`` (v10.1). The
            default version is ``405`` (v4.5). The version is ignored
            if it can be inferred from the PP headers, which will
            generally be the case for files created at versions 5.3
            and later. Note that the PP header can not encode tertiary
            version elements (such as the ``3`` in ``606.3``), so it
            may be necessary to provide a UM version in such cases.

        verbose: `int` or `None`, optional
            If an integer from ``0`` to ``3``, corresponding to increasing
            verbosity (else ``-1`` as a special case of maximal and extreme
            verbosity), set for the duration of the method call (only) as
            the minimum severity level cut-off of displayed log messages,
            regardless of the global configured `cf.LOG_LEVEL`.

            Else, if `None` (the default value), log messages will be
            filtered out, or otherwise, according to the value of the
            `cf.LOG_LEVEL` setting.

            Overall, the higher a non-negative integer that is set (up to
            a maximum of ``3``) the more description that is printed about
            the read process.

        set_standard_name: `bool`, optional

    :Returns:

        `list`
            The fields in the file.

    **Examples:**

    >>> f = read('file.pp')
    >>> f = read('*/file[0-9].pp', um_version=708)

        '''
        if endian:
            byte_ordering = endian+'_endian'
        else:
            byte_ordering = None

        self.read_vars = {
            'filename': filename,
            'byte_ordering': byte_ordering,
            'word_size': word_size,
            'fmt': fmt
        }

        history = 'Converted from UM/PP by cf-python v{}'.format(__version__)

        if endian:
            byte_ordering = endian+'_endian'
        else:
            byte_ordering = None

        f = self.file_open(filename)

        um = [UMField(var, f.format, f.byte_ordering, f.word_size,
                      um_version, set_standard_name, history=history,
                      height_at_top_of_model=height_at_top_of_model,
                      verbose=verbose,
                      implementation=self.implementation)
              for var in f.vars]

        return [field for x in um for field in x.fields if field]

    def is_um_file(self, filename):
        '''Return True if a file is a PP file or UM fields file.

    Note that the file type is determined by inspecting the file's
    contents and any file suffix is not not considered.

    :Parameters:

        filename: `str`
            The file.

    :Returns:

        `bool`

    **Examples:**

    >>> r.is_um_file('myfile.pp')
    True
    >>> r.is_um_file('myfile.nc')
    False
    >>> r.is_um_file('myfile.pdf')
    False
    >>> r.is_um_file('myfile.txt')
    False

        '''
        try:
            f = _open_um_file(filename)
        except Exception as error:
            return False

        try:
            f.close_fd()
        except:
            pass

        return True

    def file_close(self):
        '''Close the file that has been read.

    :Returns:

        `None`

        '''
        _close_um_file(self.read_vars['filename'])

    def file_open(self, filename):
        '''Open the file for reading.

    :Paramters:

        filename: `str`
            The file to be read.

    :Returns:

        '''
        g = self.read_vars

        return _open_um_file(filename,
                             byte_ordering=g['byte_ordering'],
                             word_size=g['word_size'],
                             fmt=g['fmt'])


# --- End: class

'''
Problems:

Z and P coordinates
/home/david/data/pp/aaaao/aaaaoa.pmh8dec.03328.pp

/net/jasmin/chestnut/data-24/david/testpp/026000000000c.fc0607.000128.0000.00.04.0260.0020.1491.12.01.00.00.pp
skipping variable stash code=0, 0, 0 because: grid code not supported
umfile: error condition detected in routine list_copy_to_ptr_array
umfile: error condition detected in routine process_vars
umfile: error condition detected in routine file_parse
OK 2015-04-01

/net/jasmin/chestnut/data-24/david/testpp/026000000000c.fc0619.000128.0000.00.04.0260.0020.1491.12.01.00.00.pp
skipping variable stash code=0, 0, 0 because: grid code not supported
umfile: error condition detected in routine list_copy_to_ptr_array
umfile: error condition detected in routine process_vars
umfile: error condition detected in routine file_parse
OK 2015-04-01

/net/jasmin/chestnut/data-24/david/testpp/lbcode_10423.pp
skipping variable stash code=0, 0, 0 because: grid code not supported
umfile: error condition detected in routine list_copy_to_ptr_array
umfile: error condition detected in routine process_vars
umfile: error condition detected in routine file_parse
OK 2015-04-01

/net/jasmin/chestnut/data-24/david/testpp/lbcode_11323.pp
skipping variable stash code=0, 0, 0 because: grid code not supported
umfile: error condition detected in routine list_copy_to_ptr_array
umfile: error condition detected in routine process_vars
umfile: error condition detected in routine file_parse
OK 2015-04-01

EXTRA_DATA:
/net/jasmin/chestnut/data-24/david/testpp/ajnjgo.pmm1feb.pp

SLOW: (Not any more! 2015-04-01)
/net/jasmin/chestnut/data-24/david/testpp/xgdria.pdk949a.pp
/net/jasmin/chestnut/data-24/david/testpp/xhbmaa.pm27sep.pp

RUN LENGTH ENCODED dump (not fields file)
/home/david/data/um/xhlska.dak69h0
Field 115 (stash code 9)

dch@eslogin008:/nerc/n02/n02/dch> ff2pp xgvwko.piw96b0 xgvwko.piw96b0.pp

file xgvwko.piw96b0 is a byte swapped 64 bit ieee um file

'''
