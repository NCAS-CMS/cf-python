import itertools
import logging
import textwrap
from datetime import datetime
from uuid import uuid4

import cfdm
import cftime
import dask.array as da
import numpy as np
from cfdm import Constructs, is_log_level_info
from dask.array.core import getter, normalize_chunks
from dask.base import tokenize
from netCDF4 import date2num as netCDF4_date2num

from ... import __Conventions__, __version__
from ...constants import _stash2standard_name
from ...data import Data
from ...data.array import UMArray
from ...decorators import (
    _manage_log_level_via_verbose_attr,
    _manage_log_level_via_verbosity,
)
from ...functions import abspath
from ...functions import atol as cf_atol
from ...functions import load_stash2standard_name
from ...functions import rtol as cf_rtol
from ...umread_lib.umfile import File
from ...units import Units

# import numpy as np


logger = logging.getLogger(__name__)

_cached_runid = {}
_cached_latlon = {}
_cached_ctime = {}
_cached_size_1_height_coordinate = {}
_cached_date2num = {}
_cached_model_level_number_coordinate = {}
_cached_regular_array = {}
_cached_regular_bounds = {}
_cached_data = {}

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
_pi_over_180 = np.pi / 180.0

# PP missing data indicator
_pp_rmdi = -1.0e30

# No no-missing-data value of BMDI (as described in UMDP F3 v805)
_BMDI_no_missing_data_value = -1.0e30

# Reference surface pressure in Pascals
_pstar = 1.0e5

# --------------------------------------------------------------------
# Characters used in decoding LBEXP into a runid
# --------------------------------------------------------------------
_characters = (
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)

_n_characters = len(_characters)

# # --------------------------------------------------------------------
# # Number matching regular expression
# # --------------------------------------------------------------------
# _number_regex = '([-+]?\d*\.?\d+(e[-+]?\d+)?)'

_Units = {
    None: Units(),
    "": Units(""),
    "1": Units("1"),
    "Pa": Units("Pa"),
    "m": Units("m"),
    "hPa": Units("hPa"),
    "K": Units("K"),
    "degrees": Units("degrees"),
    "degrees_east": Units("degrees_east"),
    "degrees_north": Units("degrees_north"),
    "days": Units("days"),
    "gregorian 1752-09-13": Units("days since 1752-09-13", "gregorian"),
    "365_day 1752-09-13": Units("days since 1752-09-13", "365_day"),
    "360_day 0-1-1": Units("days since 0-1-1", "360_day"),
}

# --------------------------------------------------------------------
# Names of PP integer and real header items
# --------------------------------------------------------------------
_header_names = (
    "LBYR",
    "LBMON",
    "LBDAT",
    "LBHR",
    "LBMIN",
    "LBDAY",
    "LBYRD",
    "LBMOND",
    "LBDATD",
    "LBHRD",
    "LBMIND",
    "LBDAYD",
    "LBTIM",
    "LBFT",
    "LBLREC",
    "LBCODE",
    "LBHEM",
    "LBROW",
    "LBNPT",
    "LBEXT",
    "LBPACK",
    "LBREL",
    "LBFC",
    "LBCFC",
    "LBPROC",
    "LBVC",
    "LBRVC",
    "LBEXP",
    "LBEGIN",
    "LBNREC",
    "LBPROJ",
    "LBTYP",
    "LBLEV",
    "LBRSVD1",
    "LBRSVD2",
    "LBRSVD3",
    "LBRSVD4",
    "LBSRCE",
    "LBUSER1",
    "LBUSER2",
    "LBUSER3",
    "LBUSER4",
    "LBUSER5",
    "LBUSER6",
    "LBUSER7",
    "BRSVD1",
    "BRSVD2",
    "BRSVD3",
    "BRSVD4",
    "BDATUM",
    "BACC",
    "BLEV",
    "BRLEV",
    "BHLEV",
    "BHRLEV",
    "BPLAT",
    "BPLON",
    "BGOR",
    "BZY",
    "BDY",
    "BZX",
    "BDX",
    "BMDI",
    "BMKS",
)

# --------------------------------------------------------------------
# Positions of PP header items in their arrays
# --------------------------------------------------------------------
(
    lbyr,
    lbmon,
    lbdat,
    lbhr,
    lbmin,
    lbday,
    lbyrd,
    lbmond,
    lbdatd,
    lbhrd,
    lbmind,
    lbdayd,
    lbtim,
    lbft,
    lblrec,
    lbcode,
    lbhem,
    lbrow,
    lbnpt,
    lbext,
    lbpack,
    lbrel,
    lbfc,
    lbcfc,
    lbproc,
    lbvc,
    lbrvc,
    lbexp,
    lbegin,
    lbnrec,
    lbproj,
    lbtyp,
    lblev,
    lbrsvd1,
    lbrsvd2,
    lbrsvd3,
    lbrsvd4,
    lbsrce,
    lbuser1,
    lbuser2,
    lbuser3,
    lbuser4,
    lbuser5,
    lbuser6,
    lbuser7,
) = tuple(range(45))

(
    brsvd1,
    brsvd2,
    brsvd3,
    brsvd4,
    bdatum,
    bacc,
    blev,
    brlev,
    bhlev,
    bhrlev,
    bplat,
    bplon,
    bgor,
    bzy,
    bdy,
    bzx,
    bdx,
    bmdi,
    bmks,
) = tuple(range(19))

# --------------------------------------------------------------------
# Map PP axis codes to CF standard names (The full list of field code
# keys may be found at
# http://cms.ncas.ac.uk/html_umdocs/wave/@header.)
# --------------------------------------------------------------------
_coord_standard_name = {
    0: None,  # Sigma (or eta, for hybrid coordinate data).
    1: "air_pressure",  # Pressure (mb).
    2: "height",  # Height above sea level (km)
    # Eta (U.M. hybrid coordinates) only:
    3: "atmosphere_hybrid_sigma_pressure_coordinate",
    4: "depth",  # Depth below sea level (m)
    5: "model_level_number",  # Model level.
    6: "air_potential_temperature",  # Theta
    7: "atmosphere_sigma_coordinate",  # Sigma only.
    8: None,  # Sigma-theta
    10: "latitude",  # Latitude (degrees N).
    11: "longitude",  # Longitude (degrees E).
    # Site number (set of parallel rows or columns e.g.Time series):
    13: None,  # "region",
    14: "atmosphere_hybrid_height_coordinate",
    15: "height",
    20: "time",  # Time (days) (Gregorian calendar (not 360 day year))
    21: "time",  # Time (months)
    22: "time",  # Time (years)
    23: "time",  # Time (model days with 360 day model calendar)
    40: None,  # pseudolevel
    99: None,  # Other
    -10: "grid_latitude",  # Rotated latitude (degrees).
    -11: "grid_longitude",  # Rotated longitude (degrees).
    -20: "radiation_wavelength",
}

# --------------------------------------------------------------------
# Map PP axis codes to CF long names
# --------------------------------------------------------------------
_coord_long_name = {13: "site"}

# --------------------------------------------------------------------
# Map PP axis codes to UDUNITS strings
# --------------------------------------------------------------------
# _coord_units = {
_axiscode_to_units = {
    0: "1",  # Sigma (or eta, for hybrid coordinate data)
    1: "hPa",  # air_pressure
    2: "m",  # altitude
    3: "1",  # atmosphere_hybrid_sigma_pressure_coordinate
    4: "m",  # depth
    5: "1",  # model_level_number
    6: "K",  # air_potential_temperature
    7: "1",  # atmosphere_sigma_coordinate
    10: "degrees_north",  # latitude
    11: "degrees_east",  # longitude
    13: "",  # region
    14: "1",  # atmosphere_hybrid_height_coordinate
    15: "m",  # height
    20: "days",  # time (gregorian)
    23: "days",  # time (360_day)
    40: "1",  # pseudolevel
    -10: "degrees",  # rotated latitude  (not an official axis code)
    -11: "degrees",  # rotated longitude (not an official axis code)
}

# --------------------------------------------------------------------
# Map PP axis codes to Units objects
# --------------------------------------------------------------------
_axiscode_to_Units = {
    0: _Units["1"],  # Sigma (or eta, for hybrid coordinate data)
    1: _Units["hPa"],  # air_pressure
    2: _Units["m"],  # altitude
    3: _Units["1"],  # atmosphere_hybrid_sigma_pressure_coordinate
    4: _Units["m"],  # depth
    5: _Units["1"],  # model_level_number
    6: _Units["K"],  # air_potential_temperature
    7: _Units["1"],  # atmosphere_sigma_coordinate
    10: _Units["degrees_north"],  # latitude
    11: _Units["degrees_east"],  # longitude
    13: _Units[""],  # region
    14: _Units["1"],  # atmosphere_hybrid_height_coordinate
    15: _Units["m"],  # height
    20: _Units["days"],  # time (gregorian)
    23: _Units["days"],  # time (360_day)
    40: _Units["1"],  # pseudolevel
    -10: _Units["degrees"],  # rotated latitude  (not an official axis code)
    -11: _Units["degrees"],  # rotated longitude (not an official axis code)
}

# --------------------------------------------------------------------
# Map PP axis codes to CF axis attributes
# --------------------------------------------------------------------
_coord_axis = {
    1: "Z",  # air_pressure
    2: "Z",  # altitude
    3: "Z",  # atmosphere_hybrid_sigma_pressure_coordinate
    4: "Z",  # depth
    5: "Z",  # model_level_number
    6: "Z",  # air_potential_temperature
    7: "Z",  # atmosphere_sigma_coordinate
    10: "Y",  # latitude
    11: "X",  # longitude
    13: None,  # region
    14: "Z",  # atmosphere_hybrid_height_coordinate
    15: "Z",  # height
    20: "T",  # time (gregorian)
    23: "T",  # time (360_day)
    40: None,  # pseudolevel
    -10: "Y",  # rotated latitude  (not an official axis code)
    -11: "X",  # rotated longitude (not an official axis code)
}

# --------------------------------------------------------------------
# Map PP axis codes to CF positive attributes
# --------------------------------------------------------------------
_coord_positive = {
    1: "down",  # air_pressure
    2: "up",  # altitude
    3: "down",  # atmosphere_hybrid_sigma_pressure_coordinate
    4: "down",  # depth
    5: None,  # model_level_number
    6: "up",  # air_potential_temperature
    7: "down",  # atmosphere_sigma_coordinate
    10: None,  # latitude
    11: None,  # longitude
    13: None,  # region
    14: "up",  # atmosphere_hybrid_height_coordinate
    15: "up",  # height
    20: None,  # time (gregorian)
    23: None,  # time (360_day)
    40: None,  # pseudolevel
    -10: None,  # rotated latitude  (not an official axis code)
    -11: None,  # rotated longitude (not an official axis code)
}

# --------------------------------------------------------------------
# Map LBVC codes to PP axis codes. The full list of field code keys
# may be found at http://cms.ncas.ac.uk/html_umdocs/wave/@fcodes
# --------------------------------------------------------------------
_lbvc_to_axiscode = {
    1: 2,  # altitude (Height)
    2: 4,  # depth (Depth)
    3: None,  # (Geopotential (= g*height))
    4: None,  # (ICAO height)
    6: 4,  # model_level_number  # Changed from 5 !!!
    7: None,  # (Exner pressure)
    8: 1,  # air_pressure  (Pressure)
    9: 3,  # atmosphere_hybrid_sigma_pressure_coordinate (Hybrid pressure)
    # dch check:
    10: 7,  # atmosphere_sigma_coordinate (Sigma (= p/surface p))
    16: None,  # (Temperature T)
    19: 6,  # air_potential_temperature (Potential temperature)
    27: None,  # (Atmospheric) density
    28: None,  # (d(p*)/dt .  p* = surface pressure)
    44: None,  # (Time in seconds)
    65: 14,  # atmosphere_hybrid_height_coordinate (Hybrid height)
    129: None,  # Surface
    176: 10,  # latitude    (Latitude)
    177: 11,  # longitude   (Longitude)
}

# --------------------------------------------------------------------
# Map model identifier codes to model names. The model identifier code
# is the last four digits of LBSRCE.
# --------------------------------------------------------------------
_lbsrce_model_codes = {1111: "UM"}

# --------------------------------------------------------------------
# Names of PP extra data codes
# --------------------------------------------------------------------
_extra_data_name = {
    1: "x",
    2: "y",
    3: "y_domain_lower_bound",
    4: "x_domain_lower_bound",
    5: "y_domain_upper_bound",
    6: "x_domain_upper_bound",
    7: "z_domain_lower_bound",
    8: "x_domain_upper_bound",
    9: "title",
    10: "domain_title",
    11: "x_lower_bound",
    12: "x_upper_bound",
    13: "y_lower_bound",
    14: "y_upper_bound",
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

_axis = {"area": "area"}

_autocyclic_false = {"no-op": True, "X": False, "cyclic": False}


class UMField:
    """Represents Fields derived from a UM fields file."""

    def __init__(
        self,
        var,
        fmt,
        byte_ordering,
        word_size,
        um_version,
        set_standard_name,
        height_at_top_of_model,
        verbose=None,
        implementation=None,
        select=None,
        info=False,
        **kwargs,
    ):
        """**Initialisation**

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

            verbose: `int` or `str` or `None`, optional
                If an integer from ``-1`` to ``3``, or an equivalent string
                equal ignoring case to one of:

                * ``'DISABLE'`` (``0``)
                * ``'WARNING'`` (``1``)
                * ``'INFO'`` (``2``)
                * ``'DETAIL'`` (``3``)
                * ``'DEBUG'`` (``-1``)

                set for the duration of the method call only as the minimum
                cut-off for the verboseness level of displayed output (log)
                messages, regardless of the globally-configured `cf.log_level`.
                Note that increasing numerical value corresponds to increasing
                verbosity, with the exception of ``-1`` as a special case of
                maximal and extreme verbosity.

                Otherwise, if `None` (the default value), output messages will
                be shown according to the value of the `cf.log_level` setting.

                Overall, the higher a non-negative integer or equivalent string
                that is set (up to a maximum of ``3``/``'DETAIL'``) for
                increasing verbosity, the more description that is printed
                about the read process.

            kwargs: *optional*
                Keyword arguments providing extra CF properties for each
                return field construct.

        """
        self._bool = False

        self.info = info

        self.implementation = implementation

        self.verbose = verbose

        self.fmt = fmt
        self.height_at_top_of_model = height_at_top_of_model
        self.byte_ordering = byte_ordering
        self.word_size = word_size

        self.atol = cf_atol()

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
                    times = [
                        (self.header_vtime(rec), self.header_dtime(rec))
                        for rec in group
                    ]
                    lengths = [
                        len(tuple(g)) for k, g in itertools.groupby(times)
                    ]
                    if len(set(lengths)) == 1:
                        # Each run of identical times has the same
                        # length, so it is possible that this group
                        # forms a variable of nz x nt records.
                        split_group = False
                        nz = lengths.pop()
                        z0 = [self.z for rec in group[:nz]]
                        for i in range(nz, group_size, nz):
                            z1 = [
                                self.header_z(rec) for rec in group[i : i + nz]
                            ]
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
                    groups_nt.append(group_size / nz)

            groups = groups2

        rec0 = groups[0][0]

        int_hdr = rec0.int_hdr
        self.int_hdr_dtype = int_hdr.dtype
        self.real_hdr_dtype = rec0.real_hdr.dtype
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
        stash = int_hdr[lbuser4]
        LBUSER5 = int_hdr[lbuser5]
        submodel = int_hdr[lbuser7]
        BPLAT = real_hdr[bplat]
        BPLON = real_hdr[bplon]
        BDX = real_hdr[bdx]
        BDY = real_hdr[bdy]

        if not LBROW or not LBNPT:
            logger.warn(
                f"WARNING: Skipping STASH code {stash} with LBROW={LBROW}, "
                f"LBNPT={LBNPT}, LBPACK={int_hdr[lbpack]} "
                "(possibly runlength encoded)"
            )  # pragma: no cover
            self.field = (None,)
            return

        if stash:
            section, item = divmod(stash, 1000)
            um_stash_source = "m%02ds%02di%03d" % (submodel, section, item)
        else:
            um_stash_source = None

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
            source += f" vn{model_um_version}"

        # Only process the requested fields
        ok = True
        if select:
            values1 = (
                f"stash_code={stash}",
                f"lbproc={LBPROC}",
                f"lbtim={LBTIM}",
                f"runid={self.decode_lbexp()}",
                f"submodel={submodel}",
            )
            if um_stash_source is not None:
                values1 += (f"um_stash_source={um_stash_source}",)
            if source:
                values1 += (f"source={source}",)

            ok = False
            for value0 in select:
                for value1 in values1:
                    ok = Constructs._matching_values(
                        value0, None, value1, basic=True
                    )
                    if ok:
                        break

                if ok:
                    break

        if not ok:
            # This PP/UM field does not match the requested selection
            self.field = (None,)
            return

        # Still here?
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
            calendar = "gregorian"
        elif ic == 4:
            calendar = "365_day"
        else:
            calendar = "360_day"

        self.calendar = calendar
        self.reference_time_Units()

        if source:
            cf_properties["source"] = source

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
        elif calendar == "gregorian":
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
        #        stash = int_hdr[lbuser4]#
        self.stash = stash

        # The STASH code has been set in the PP header, so try to find
        # its standard_name from the conversion table
        stash_records = _stash2standard_name.get((submodel, stash), None)

        um_Units = None
        um_condition = None

        long_name = None
        standard_name = None

        if stash_records:
            um_version = self.um_version
            for (
                long_name,
                units,
                valid_from,
                valid_to,
                standard_name,
                cf_info,
                um_condition,
            ) in stash_records:
                # Check that conditions are met
                if not self.test_um_version(valid_from, valid_to, um_version):
                    continue

                if um_condition:
                    if not self.test_um_condition(
                        um_condition, LBCODE, BPLAT, BPLON
                    ):
                        continue

                # Still here? Then we have our standard_name, etc.
                #                if standard_name:
                #                    if set_standard_name:
                #                        cf_properties['standard_name'] = standard_name
                #                    else:
                #                        attributes['_standard_name'] = standard_name
                if standard_name and set_standard_name:
                    cf_properties["standard_name"] = standard_name

                cf_properties["long_name"] = long_name.rstrip()

                um_Units = _Units.get(units, None)
                if um_Units is None:
                    um_Units = Units(units)
                    _Units[units] = um_Units

                self.um_Units = um_Units
                self.cf_info = cf_info

                break

        if um_stash_source is not None:
            cf_properties["um_stash_source"] = um_stash_source
            identity = f"UM_{um_stash_source}_vn{self.um_version}"
        else:
            identity = f"UM_{submodel}_fc{int_hdr[lbfc]}_vn{self.um_version}"

        if um_Units is None:
            self.um_Units = _Units[None]

        if um_condition:
            identity += f"_{um_condition}"

        if long_name is None:
            cf_properties["long_name"] = identity

        for recs, nz, nt in zip(groups, groups_nz, groups_nt):
            self.recs = recs
            self.nz = nz
            self.nt = nt
            self.z_recs = recs[:nz]
            self.t_recs = recs[::nz]

            LBUSER5 = recs[0].int_hdr.item(lbuser5)

            #            self.cell_method_axis_name = {'area': 'area'}

            self.down_axes = set()
            self.z_axis = "z"

            # --------------------------------------------------------
            # Get the extra data for this group
            # --------------------------------------------------------
            extra = recs[0].get_extra_data()
            self.extra = extra

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
                        axiscode
                    )
                elif axiscode == 2 and "height" in self.cf_info:
                    # Create the height coordinate from the information
                    # given in the STASH to standard_name conversion table
                    height, units = self.cf_info["height"]
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

            # --------------------------------------------------------
            # Create the 'Y' dimension coordinate
            # --------------------------------------------------------
            axiscode = iy
            yc = None
            if axiscode is not None:
                if axiscode in (20, 23):
                    # 'Y' axis is time-since-reference-date
                    if extra.get("y", None) is not None:
                        c = self.time_coordinate_from_extra_data(axiscode, "y")
                    else:
                        LBUSER3 = int_hdr[lbuser3]
                        if LBUSER3 == LBROW:
                            self.lbuser3 = LBUSER3
                            c = self.time_coordinate_from_um_timeseries(
                                axiscode, "y"
                            )
                else:
                    ykey, yc, yaxis = self.xy_coordinate(axiscode, "y")
                    if axiscode == 13:
                        _axis["site_axis"] = yaxis
                        self.site_coordinates_from_extra_data()

            # --------------------------------------------------------
            # Create the 'X' dimension coordinate
            # --------------------------------------------------------
            axiscode = ix
            xc = None
            xkey = None
            if axiscode is not None:
                if axiscode in (20, 23):
                    # X axis is time since reference date
                    if extra.get("x", None) is not None:
                        c = self.time_coordinate_from_extra_data(axiscode, "x")
                    else:
                        LBUSER3 = int_hdr[lbuser3]
                        if LBUSER3 == LBNPT:
                            self.lbuser3 = LBUSER3
                            c = self.time_coordinate_from_um_timeseries(
                                axiscode, "x"
                            )
                else:
                    xkey, xc, xaxis = self.xy_coordinate(axiscode, "x")
                    if axiscode == 13:
                        _axis["site_axis"] = xaxis
                        self.site_coordinates_from_extra_data()

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
                        "grid_mapping_name": "rotated_latitude_longitude",
                        "grid_north_pole_latitude": BPLAT,
                        "grid_north_pole_longitude": BPLON,
                    }
                )

                self.implementation.set_coordinate_conversion(ref, cc)

                self.implementation.set_coordinate_reference(
                    self.field, ref, copy=False
                )

                # ----------------------------------------------------
                # Create UNROTATED, 2-D LATITUDE and LONGITUDE
                # auxiliary coordinates
                # ----------------------------------------------------
                aux_keys = self.latitude_longitude_2d_aux_coordinates(yc, xc)

                self.implementation.set_coordinate_reference_coordinates(
                    ref, [ykey, xkey] + aux_keys
                )

            # --------------------------------------------------------
            # Create a RADIATION WAVELENGTH dimension coordinate
            # --------------------------------------------------------
            try:
                rwl, rwl_units = self.cf_info["below"]
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

            attributes["int_hdr"] = int_hdr[:]
            attributes["real_hdr"] = real_hdr[:]
            attributes["file"] = filename
            attributes["id"] = identity

            cf_properties["Conventions"] = __Conventions__
            cf_properties["runid"] = self.decode_lbexp()
            cf_properties["lbproc"] = str(LBPROC)
            cf_properties["lbtim"] = str(LBTIM)
            cf_properties["stash_code"] = str(stash)
            cf_properties["submodel"] = str(submodel)

            # --------------------------------------------------------
            # Set the data and extra data
            # --------------------------------------------------------
            data = self.create_data()

            # --------------------------------------------------------
            # Insert data into the field
            # --------------------------------------------------------
            field = self.field

            self.implementation.set_data(
                field, self.data, axes=self.data_axes, copy=False
            )

            # --------------------------------------------------------
            # Insert attributes and CF properties into the field
            # --------------------------------------------------------
            fill_value = data.fill_value
            if fill_value is not None:
                cf_properties["_FillValue"] = data.fill_value

            # Add kwargs to the CF properties
            cf_properties.update(kwargs)

            self.implementation.set_properties(
                field, cf_properties, copy=False
            )

            field.id = identity

            if standard_name and not set_standard_name:
                field._custom["standard_name"] = standard_name

            self.implementation.nc_set_variable(field, identity)

            # --------------------------------------------------------
            # Create and insert cell methods
            # --------------------------------------------------------
            cell_methods = self.create_cell_methods()
            for cm in cell_methods:
                self.implementation.set_cell_method(field, cm)

            logger.info(f"down_axes = {self.down_axes}")  # pragma: no cover

            # Force cyclic X axis for particular values of LBHEM
            if xkey is not None and int_hdr[lbhem] in (0, 1, 2, 4):
                field.cyclic(
                    xkey,
                    iscyclic=True,
                    config={
                        "axis": xaxis,
                        "coord": xc,
                        "period": self.get_data(np.array(360.0), xc.Units),
                    },
                )

            self.fields.append(field)

        self._bool = True

    def __bool__(self):
        """x.__bool__() <==> bool(x)"""
        return self._bool

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return self.fdr()

    def __str__(self):
        """x.__str__() <==> str(x)"""
        out = [self.fdr()]

        attrs = (
            "endian",
            "reftime",
            "vtime",
            "dtime",
            "um_version",
            "source",
            "it",
            "iz",
            "ix",
            "iy",
            "site_time_cross_section",
            "timeseries",
            "file",
        )

        for attr in attrs:
            out.append(f"{attr}={getattr(self, attr, None)}")

        out.append("")

        return "\n".join(out)

    def _reorder_z_axis(self, indices, z_axis, pmaxes):
        """Reorder the Z axis `Rec` instances.

        :Parameters:

            indices: `list`
                Aggregation axis indices. See `create_data` for
                details.

            z_axis: `int`
                The identifier of the Z axis.

            pmaxes: sequence of `int`
                The aggregation axes, which include the Z axis.

        :Returns:

            `list`

        **Examples**

        >>> _reorder_z_axis([(0, <Rec A>), (1, <Rec B>)], 0, [0])
        [(0, <Rec B>), (1, <Rec A>)]

        >>> _reorder_z_axis(
        ...     [(0, 0, <Rec A>),
        ...      (0, 1, <Rec B>),
        ...      (1, 0, <Rec C>),
        ...      (1, 1, <Rec D>)],
        ...     1, [0, 1]
        ... )
        [(0, 0, <Rec B>), (0, 1, <Rec A>), (1, 0, <Rec D>), (1, 1, <Rec C>)]

        """
        indices_new = []
        zpos = pmaxes.index(z_axis)
        aaa0 = indices[0]
        indices2 = [aaa0]
        for aaa in indices[1:]:
            if aaa[zpos] > aaa0[zpos]:
                indices2.append(aaa)
            else:
                indices_new.extend(indices2[::-1])
                aaa0 = aaa
                indices2 = [aaa0]

        indices_new.extend(indices2[::-1])

        indices = [a[:-1] + b[-1:] for a, b in zip(indices, indices_new)]
        return indices

    def atmosphere_hybrid_height_coordinate(self, axiscode):
        """`atmosphere_hybrid_height_coordinate` when not an array axis.

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

        """
        field = self.field

        # "a" domain ancillary
        array = np.array(
            [rec.real_hdr[blev] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,  # Zsea
        )
        bounds0 = np.array(
            [rec.real_hdr[brlev] for rec in self.z_recs],  # Zsea lower
            dtype=self.real_hdr_dtype,
        )
        bounds1 = np.array(
            [rec.real_hdr[brsvd1] for rec in self.z_recs],  # Zsea upper
            dtype=self.real_hdr_dtype,
        )
        bounds = self.create_bounds_array(bounds0, bounds1)

        # Insert new Z axis
        da = self.implementation.initialise_DomainAxis(size=array.size)
        axis_key = self.implementation.set_domain_axis(
            self.field, da, copy=False
        )
        _axis["z"] = axis_key

        ac = self.implementation.initialise_DomainAncillary()
        ac = self.coord_data(ac, array, bounds, units=_Units["m"])
        ac.id = "UM_atmosphere_hybrid_height_coordinate_a"
        self.implementation.set_properties(
            ac, {"long_name": "height based hybrid coeffient a"}, copy=False
        )
        key_a = self.implementation.set_domain_ancillary(
            field, ac, axes=[_axis["z"]], copy=False
        )

        # Height at top of atmosphere
        toa_height = self.height_at_top_of_model
        if toa_height is None:
            pseudolevels = any(
                [
                    rec.int_hdr.item(
                        lbuser5,
                    )
                    for rec in self.z_recs
                ]
            )
            if pseudolevels:
                # Pseudolevels and atmosphere hybrid height
                # coordinates are both present => can't reliably infer
                # height. This is due to a current limitation in the C
                # library that means it can ony create Z-T
                # aggregations, rather than the required Z-T-P
                # aggregations.
                toa_height = -1

        if toa_height is None:
            toa_height = bounds1.max()
            if toa_height <= 0:
                toa_height = None
        elif toa_height <= 0:
            toa_height = None
        else:
            toa_height = float(toa_height)

        # atmosphere_hybrid_height_coordinate dimension coordinate
        if toa_height is None:
            dc = None
        else:
            array = array / toa_height
            bounds = bounds / toa_height
            dc = self.implementation.initialise_DimensionCoordinate()
            dc = self.coord_data(dc, array, bounds, units=_Units["1"])
            self.implementation.set_properties(
                dc,
                {"standard_name": "atmosphere_hybrid_height_coordinate"},
                copy=False,
            )
            dc = self.coord_axis(dc, axiscode)
            dc = self.coord_positive(dc, axiscode, _axis["z"])
            key_dc = self.implementation.set_dimension_coordinate(
                field,
                dc,
                axes=[_axis["z"]],
                copy=False,
                autocyclic=_autocyclic_false,
            )

        # "b" domain ancillary
        array = np.array(
            [rec.real_hdr[bhlev] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds0 = np.array(
            [rec.real_hdr[bhrlev] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds1 = np.array(
            [rec.real_hdr[brsvd2] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds = self.create_bounds_array(bounds0, bounds1)

        ac = self.implementation.initialise_DomainAncillary()
        ac = self.coord_data(ac, array, bounds, units=_Units["1"])
        ac.id = "UM_atmosphere_hybrid_height_coordinate_b"
        self.implementation.set_properties(
            ac, {"long_name": "height based hybrid coeffient b"}, copy=False
        )
        key_b = self.implementation.set_domain_ancillary(
            field, ac, axes=[_axis["z"]], copy=False
        )

        # atmosphere_hybrid_height_coordinate coordinate reference
        ref = self.implementation.initialise_CoordinateReference()
        cc = self.implementation.initialise_CoordinateConversion(
            parameters={
                "standard_name": "atmosphere_hybrid_height_coordinate"
            },
            domain_ancillaries={"a": key_a, "b": key_b, "orog": None},
        )
        self.implementation.set_coordinate_conversion(ref, cc)
        if dc is not None:
            self.implementation.set_coordinate_reference_coordinates(
                ref, (key_dc,)
            )

        self.implementation.set_coordinate_reference(field, ref, copy=False)

        return dc

    def depth_coordinate(self, axiscode):
        """`atmosphere_hybrid_height_coordinate_*k` depth coordinate.

        Only applicable when not an array axis.

        :Parameters:

            axiscode: `int`

        :Returns:

            `DimensionCoordinate` or `None`

        """
        dc = self.model_level_number_coordinate(aux=False)

        field = self.field

        array = np.array(
            [rec.real_hdr[blev] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds0 = np.array(
            [rec.real_hdr[brlev] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds1 = np.array(
            [rec.real_hdr[brsvd1] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds = self.create_bounds_array(bounds0, bounds1)

        # Create Z domain axis construct
        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisZ = self.implementation.set_domain_axis(field, da, copy=False)
        _axis["z"] = axisZ

        # ac = AuxiliaryCoordinate()
        ac = self.implementation.initialise_AuxiliaryCoordinate()
        ac = self.coord_data(ac, array, bounds, units=_Units["m"])
        ac.id = "UM_atmosphere_hybrid_height_coordinate_ak"
        ac.long_name = "atmosphere_hybrid_height_coordinate_ak"
        #        field.insert_aux(ac, axes=[zdim], copy=False)
        self.implementation.set_auxiliary_coordinate(
            field,
            ac,
            axes=[_axis["z"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )

        array = np.array(
            [rec.real_hdr[bhlev] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds0 = np.array(
            [rec.real_hdr[bhrlev] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds1 = np.array(
            [rec.real_hdr[brsvd2] for rec in self.z_recs],
            dtype=self.real_hdr_dtype,
        )
        bounds = self.create_bounds_array(bounds0, bounds1)

        # ac = AuxiliaryCoordinate()
        ac = self.implementation.initialise_AuxiliaryCoordinate()
        ac = self.coord_data(ac, array, bounds, units=_Units["1"])
        ac.id = "UM_atmosphere_hybrid_height_coordinate_bk"
        ac.long_name = "atmosphere_hybrid_height_coordinate_bk"
        self.implementation.set_auxiliary_coordinate(
            field,
            ac,
            axes=[_axis["z"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )

        return dc

    def atmosphere_hybrid_sigma_pressure_coordinate(self, axiscode):
        """`atmosphere_hybrid_sigma_pressure_coordinate`

        Only applicable when not an array axis.

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

        """
        array = []
        bounds = []
        ak_array = []
        ak_bounds = []
        bk_array = []
        bk_bounds = []

        for rec in self.z_recs:
            BLEV, BRLEV, BHLEV, BHRLEV, BULEV, BHULEV = self.header_bz(rec)

            array.append(BLEV + BHLEV / _pstar)
            bounds.append([BRLEV + BHRLEV / _pstar, BULEV + BHULEV / _pstar])

            ak_array.append(BHLEV)
            ak_bounds.append((BHRLEV, BHULEV))

            bk_array.append(BLEV)
            bk_bounds.append((BRLEV, BULEV))

        array = np.array(array, dtype=float)
        bounds = np.array(bounds, dtype=float)
        ak_array = np.array(ak_array, dtype=float)
        ak_bounds = np.array(ak_bounds, dtype=float)
        bk_array = np.array(bk_array, dtype=float)
        bk_bounds = np.array(bk_bounds, dtype=float)

        field = self.field

        # Insert new Z axis
        da = self.implementation.initialise_DomainAxis(size=array.size)
        axis_key = self.implementation.set_domain_axis(field, da, copy=False)
        _axis["z"] = axis_key

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(
            dc,
            array,
            bounds,
            units=_axiscode_to_Units.setdefault(axiscode, None),
        )
        dc = self.coord_positive(dc, axiscode, _axis["z"])
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        self.implementation.set_dimension_coordinate(
            field,
            dc,
            axes=[_axis["z"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )

        ac = self.implementation.initialise_AuxiliaryCoordinate()
        ac = self.coord_data(ac, ak_array, ak_bounds, units=_Units["Pa"])
        ac.id = "UM_atmosphere_hybrid_sigma_pressure_coordinate_ak"
        ac.long_name = "atmosphere_hybrid_sigma_pressure_coordinate_ak"

        self.implementation.set_auxiliary_coordinate(
            field,
            ac,
            axes=[_axis["z"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )

        ac = self.implementation.initialise_AuxiliaryCoordinate()
        ac = self.coord_data(ac, bk_array, bk_bounds, units=_Units["1"])

        self.implementation.set_auxiliary_coordinate(
            field,
            ac,
            axes=[_axis["z"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )

        ac.id = "UM_atmosphere_hybrid_sigma_pressure_coordinate_bk"
        ac.long_name = "atmosphere_hybrid_sigma_pressure_coordinate_bk"

        return dc

    def create_bounds_array(self, bounds0, bounds1):
        """Stack two 1-d arrays to create a bounds array.

        The returned array will have a trailing dimension of size 2.

        The leading dimension size and data type are taken from
        *bounds0*.

        :Parameters:

            bounds0: `numpy.ndarray`
                The bounds which are to occupy ``[:, 0]`` in the
                returned bounds array.

            bounds1: `numpy.ndarray`
                The bounds which are to occupy ``[:, 1]`` in the
                returned bounds array.

        :Returns:

            `numpy.ndarray`

        """
        bounds = np.empty((bounds0.size, 2), dtype=bounds0.dtype)
        bounds[:, 0] = bounds0
        bounds[:, 1] = bounds1
        return bounds

    def create_cell_methods(self):
        """Create the cell methods.

        **UMDP F3**

        LBPROC Processing code. This indicates what processing has
        been done to the basic ﬁeld. It should be 0 if no processing
        has been done, otherwise add together the relevant numbers
        from the list below:

        1 Difference from another experiment.
        2 Difference from zonal (or other spatial) mean.
        4 Difference from time mean.
        8 X-derivative (d/dx)
        16 Y-derivative (d/dy)
        32 Time derivative (d/dt)
        64 Zonal mean ﬁeld
        128 Time mean ﬁeld
        256 Product of two ﬁelds
        512 Square root of a ﬁeld
        1024 Difference between ﬁelds at levels BLEV and BRLEV
        2048 Mean over layer between levels BLEV and BRLEV
        4096 Minimum value of ﬁeld during time period
        8192 Maximum value of ﬁeld during time period
        16384 Magnitude of a vector, not speciﬁcally wind speed
        32768 Log10 of a ﬁeld
        65536 Variance of a ﬁeld
        131072 Mean over an ensemble of parallel runs

        :Returns:

            `list` of `str`
               The cell methods.

        """
        cell_methods = []

        LBPROC = self.lbproc
        LBTIM_IB = self.lbtim_ib
        tmean_proc = 0

        # ------------------------------------------------------------
        # Ensemble mean cell method
        # ------------------------------------------------------------
        if 131072 <= LBPROC < 262144:
            cell_methods.append("realization: mean")
            LBPROC -= 131072

        if LBTIM_IB in (2, 3) and LBPROC in (128, 192, 2176, 4224, 8320):
            tmean_proc = 128
            LBPROC -= 128

        # ------------------------------------------------------------
        # Area cell methods
        # ------------------------------------------------------------
        # -10: rotated latitude  (not an official axis code)
        # -11: rotated longitude (not an official axis code)
        if self.ix in (10, 11, 12, -10, -11) and self.iy in (
            10,
            11,
            12,
            -10,
            -11,
        ):
            cf_info = self.cf_info

            if "where" in cf_info:
                cell_methods.append("area: mean")

                cell_methods.append(cf_info["where"])
                if "over" in cf_info:
                    cell_methods.append(cf_info["over"])

            if LBPROC == 64:
                cell_methods.append("x: mean")

            # dch : do special zonal mean as as in pp_cfwrite

        # ------------------------------------------------------------
        # Vertical cell methods
        # ------------------------------------------------------------
        if LBPROC == 2048:
            cell_methods.append("z: mean")

        # ------------------------------------------------------------
        # Time cell methods
        # ------------------------------------------------------------
        if "t" in _axis:
            axis = "t"
        else:
            axis = "time"

        if LBTIM_IB == 0 or LBTIM_IB == 1:
            if axis == "t":
                cell_methods.append(axis + ": point")
        elif LBPROC == 4096:
            cell_methods.append(axis + ": minimum")
        elif LBPROC == 8192:
            cell_methods.append(axis + ": maximum")
        if tmean_proc == 128:
            if LBTIM_IB == 2:
                cell_methods.append(axis + ": mean")
            elif LBTIM_IB == 3:
                cell_methods.append(axis + ": mean within years")
                cell_methods.append(axis + ": mean over years")

        if not cell_methods:
            return []

        cell_methods = self.implementation.initialise_CellMethod().create(
            " ".join(cell_methods)
        )

        for cm in cell_methods:
            cm.change_axes(_axis, inplace=True)

        return cell_methods

    def coord_axis(self, c, axiscode):
        """Map axis codes to CF axis attributes for the coordinate."""
        axis = _coord_axis.setdefault(axiscode, None)
        if axis is not None:
            c.axis = axis

        return c

    def coord_data(
        self,
        c,
        array=None,
        bounds=None,
        units=None,
        fill_value=None,
        climatology=False,
    ):
        """Set the data array of a coordinate construct.

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

        """
        if array is not None:
            data = self.get_data(array, units, fill_value)
            self.implementation.set_data(c, data, copy=False)

        if bounds is not None:
            data = self.get_data(bounds, units, fill_value, bounds=True)
            bounds = self.implementation.initialise_Bounds()
            self.implementation.set_data(bounds, data, copy=False)
            self.implementation.set_bounds(c, bounds, copy=False)

        return c

    def coord_names(self, coord, axiscode):
        """Map axis codes to CF standard names for the coordinate.

        :Parameters:

            coord: Coordinate construct

            axiscode: `int`

        :Returns:

            Coordinate construct

        """
        standard_name = _coord_standard_name.setdefault(axiscode, None)

        if standard_name is not None:
            coord.set_property("standard_name", standard_name, copy=False)
            coord.ncvar = standard_name
        else:
            long_name = _coord_long_name.setdefault(axiscode, None)
            if long_name is not None:
                coord.set_property("long_name", long_name, copy=False)

        return coord

    def coord_positive(self, c, axiscode, domain_axis_key):
        """Map axis codes to CF positive attributes for the coordinate.

        :Parameters:

            c: `Coordinate`
               A 1-d coordinate construct

            axiscode: `int`

            domain_axis_key: `str`

        :Returns:

            Coordinate construct

        """
        positive = _coord_positive.setdefault(axiscode, None)
        if positive is not None:
            c.positive = positive
            if positive == "down" and axiscode != 4:
                self.down_axes.add(domain_axis_key)
                c.flip(inplace=True)

        return c

    def ctime(self, rec):
        """Return elapsed time since the clock time of the given
        record."""
        reftime = self.refUnits
        LBVTIME = tuple(self.header_vtime(rec))
        LBDTIME = tuple(self.header_dtime(rec))

        key = (LBVTIME, LBDTIME, self.refunits, self.calendar)
        ctime = _cached_ctime.get(key, None)
        if ctime is None:
            LBDTIME = list(LBDTIME)
            LBDTIME[0] = LBVTIME[0]

            ctime = cftime.datetime(*LBDTIME, calendar=self.calendar)

            if ctime < cftime.datetime(*LBVTIME, calendar=self.calendar):
                LBDTIME[0] += 1
                ctime = cftime.datetime(*LBDTIME, calendar=self.calendar)

            ctime = Data(ctime, reftime).array.item()
            _cached_ctime[key] = ctime

        return ctime

    def header_vtime(self, rec):
        """Return the list [LBYR, LBMON, LBDAT, LBHR, LBMIN] for the
        given record.

        :Parameters:

            rec:

        :Returns:

            `list`

        **Examples**

        >>> u.header_vtime(rec)
        [1991, 1, 1, 0, 0]

        """
        return rec.int_hdr[lbyr : lbmin + 1]

    def header_dtime(self, rec):
        """Return the list [LBYRD, LBMOND, LBDATD, LBHRD, LBMIND] for
        the given record.

        :Parameters:

            rec:

        :Returns:

            `list`

        **Examples**

        >>> u.header_dtime(rec)
        [1991, 2, 1, 0, 0]

        """
        return rec.int_hdr[lbyrd : lbmind + 1]

    def header_bz(self, rec):
        """Return the list [BLEV, BRLEV, BHLEV, BHRLEV, BULEV, BHULEV]
        for the given record.

        :Parameters:

            rec:

        :Returns:

            `list`

        **Examples**

        >>> u.header_bz(rec)

        """
        real_hdr = rec.real_hdr
        return (
            real_hdr[blev : bhrlev + 1].tolist()
            + real_hdr[  # BLEV, BRLEV, BHLEV, BHRLEV
                brsvd1 : brsvd2 + 1
            ].tolist()  # BULEV, BHULEV
        )

    def header_lz(self, rec):
        """Return the list [LBLEV, LBUSER5] for the given record.

        :Parameters:

            rec:

        :Returns:

            `list`

        **Examples**

        >>> u.header_lz(rec)

        """
        int_hdr = rec.int_hdr
        return [int_hdr.item(lblev), int_hdr.item(lbuser5)]

    def header_z(self, rec):
        """Return the list [LBLEV, LBUSER5, BLEV, BRLEV, BHLEV, BHRLEV,
        BULEV, BHULEV] for the given record.

        These header items are used by the compare_levels function in
        compare.c

        :Parameters:

            rec:

        :Returns:

            `list`

        **Examples**

        >>> u.header_z(rec)

        """
        return self.header_lz + self.header_bz

    @_manage_log_level_via_verbose_attr
    def create_data(self):
        """Sets the data and data axes.

        :Returns:

            `Data`

        """
        if self.info:
            logger.info("Creating data:")  # pragma: no cover

        LBROW = self.lbrow
        LBNPT = self.lbnpt

        yx_shape = (LBROW, LBNPT)

        nz = self.nz
        nt = self.nt
        recs = self.recs

        um_Units = self.um_Units
        units = getattr(um_Units, "units", None)
        calendar = getattr(um_Units, "calendar", None)

        data_type_in_file = self.data_type_in_file

        filename = self.filename

        data_axes = [_axis["y"], _axis["x"]]

        # Initialise a dask graph for the uncompressed array, and some
        # dask.array.core.getter arguments
        token = tokenize((nt, nz) + yx_shape, uuid4())
        name = (UMArray().__class__.__name__ + "-" + token,)
        dsk = {}
        full_slice = Ellipsis
        klass_name = UMArray().__class__.__name__

        fmt = self.fmt

        if len(recs) == 1:
            # --------------------------------------------------------
            # 0-d partition matrix
            # --------------------------------------------------------
            pmaxes = []
            file_data_types = set()

            rec = recs[0]

            fill_value = rec.real_hdr.item(bmdi)
            if fill_value == _BMDI_no_missing_data_value:
                fill_value = None

            data_shape = yx_shape

            subarray = UMArray(
                filename=filename,
                address=rec.hdr_offset,
                shape=yx_shape,
                dtype=data_type_in_file(rec),
                fmt=fmt,
                word_size=self.word_size,
                byte_ordering=self.byte_ordering,
                units=units,
                calendar=calendar,
            )

            key = f"{klass_name}-{tokenize(subarray)}"
            dsk[key] = subarray
            dsk[name + (0, 0)] = (getter, key, full_slice, False, False)

            dtype = data_type_in_file(rec)
            chunks = normalize_chunks((-1, -1), shape=data_shape, dtype=dtype)
        else:
            # --------------------------------------------------------
            # 1-d or 2-d partition matrix
            # --------------------------------------------------------
            file_data_types = set()

            # Find the partition matrix shape
            pmshape = [n for n in (nt, nz) if n > 1]

            if len(pmshape) == 1:
                # ----------------------------------------------------
                # 1-d partition matrix
                # ----------------------------------------------------
                z_axis = _axis.get(self.z_axis)
                if nz > 1:
                    pmaxes = [z_axis]
                    data_shape = (nz, LBROW, LBNPT)
                else:
                    pmaxes = [_axis["t"]]
                    data_shape = (nt, LBROW, LBNPT)

                word_size = self.word_size
                byte_ordering = self.byte_ordering

                indices = [(i, rec) for i, rec in enumerate(recs)]

                if nz > 1 and z_axis in self.down_axes:
                    indices = self._reorder_z_axis(indices, z_axis, pmaxes)

                for i, rec in indices:
                    # Find the data type of the array in the file
                    file_data_type = data_type_in_file(rec)
                    file_data_types.add(file_data_type)

                    shape = (1,) + yx_shape

                    subarray = UMArray(
                        filename=filename,
                        address=rec.hdr_offset,
                        shape=shape,
                        dtype=file_data_type,
                        fmt=fmt,
                        word_size=word_size,
                        byte_ordering=byte_ordering,
                        units=units,
                        calendar=calendar,
                    )

                    key = f"{klass_name}-{tokenize(subarray)}"
                    dsk[key] = subarray
                    dsk[name + (i, 0, 0)] = (
                        getter,
                        key,
                        full_slice,
                        False,
                        False,
                    )

                dtype = np.result_type(*file_data_types)
                chunks = normalize_chunks(
                    (1, -1, -1), shape=data_shape, dtype=dtype
                )
            else:
                # ----------------------------------------------------
                # 2-d partition matrix
                # ----------------------------------------------------
                z_axis = _axis[self.z_axis]
                pmaxes = [_axis["t"], z_axis]

                data_shape = (nt, nz, LBROW, LBNPT)

                word_size = self.word_size
                byte_ordering = self.byte_ordering

                indices = [
                    divmod(i, nz) + (rec,) for i, rec in enumerate(recs)
                ]
                if z_axis in self.down_axes:
                    indices = self._reorder_z_axis(indices, z_axis, pmaxes)

                for t, z, rec in indices:
                    # Find the data type of the array in the file
                    file_data_type = data_type_in_file(rec)
                    file_data_types.add(file_data_type)

                    shape = (1, 1) + yx_shape

                    subarray = UMArray(
                        filename=filename,
                        address=rec.hdr_offset,
                        shape=shape,
                        dtype=file_data_type,
                        fmt=fmt,
                        word_size=word_size,
                        byte_ordering=byte_ordering,
                        units=units,
                        calendar=calendar,
                    )

                    key = f"{klass_name}-{tokenize(subarray)}"
                    dsk[key] = subarray
                    dsk[name + (t, z, 0, 0)] = (
                        getter,
                        key,
                        full_slice,
                        False,
                        False,
                    )

                dtype = np.result_type(*file_data_types)
                chunks = normalize_chunks(
                    (1, 1, -1, -1), shape=data_shape, dtype=dtype
                )

        data_axes = pmaxes + data_axes

        # Set the data array
        fill_value = recs[0].real_hdr.item(bmdi)
        if fill_value == _BMDI_no_missing_data_value:
            fill_value = None

        # Create the dask array
        dx = da.Array(dsk, name[0], chunks=chunks, dtype=dtype)

        # Create the Data object
        data = Data(dx, units=um_Units, fill_value=fill_value)
        data._cfa_set_write(True)

        self.data = data
        self.data_axes = data_axes

        return data

    def decode_lbexp(self):
        """Decode the integer value of LBEXP in the PP header into a
        runid.

        If this value has already been decoded, then it will be returned
        from the cache, otherwise the value will be decoded and then added
        to the cache.

        :Returns:

            `str`
               A string derived from LBEXP. If LBEXP is a negative integer
               then that number is returned as a string.

        **Examples**

        >>> self.decode_lbexp()
        'aaa5u'
        >>> self.decode_lbexp()
        '-34'

        """
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
            bits = bits.lstrip("0b").zfill(30)

            # Step through 6 bits at a time, converting each 6 bit chunk into
            # a decimal integer, which is used as an index to the characters
            # lookup list.
            runid = []
            for i in range(0, 30, 6):
                index = int(bits[i : i + 6], 2)
                if index < _n_characters:
                    runid.append(_characters[index])

            runid = "".join(runid)

        # Enter this runid into the cache
        _cached_runid[LBEXP] = runid

        # Return the runid
        return runid

    def dtime(self, rec):
        """Return the elapsed time since the data time of the given
        record.

        :Parameters:

            rec:

        :Returns:

            `float`

        **Examples**

        >>> u.dtime(rec)
        31.5

        """
        units = self.refunits
        calendar = self.calendar

        LBDTIME = tuple(self.header_dtime(rec))

        key = (LBDTIME, units, calendar)
        time = _cached_date2num.get(key, None)
        if time is None:
            # It is important to use the same time_units as vtime
            try:
                if self.calendar == "gregorian":
                    time = netCDF4_date2num(
                        datetime(*LBDTIME), units, calendar
                    )
                else:
                    time = netCDF4_date2num(
                        cftime.datetime(*LBDTIME, calendar=self.calendar),
                        units,
                        calendar,
                    )

                _cached_date2num[key] = time
            except ValueError:
                time = np.nan  # ppp

        return time

    def fdr(self):
        """Return a the contents of PP field headers as strings.

        This is a bit like printfdr in the UKMO IDL PP library.

        :Returns:

            `list`

        """
        out2 = []
        for i, rec in enumerate(self.recs):
            out = [f"Field {i}:"]

            x = [
                f"{name}::{value}"
                for name, value in zip(
                    _header_names, self.int_hdr + self.real_hdr
                )
            ]

            x = textwrap.fill(" ".join(x), width=79)
            out.append(x.replace("::", ": "))

            if self.extra:
                out.append("EXTRA DATA:")
                for key in sorted(self.extra):
                    out.append(f"{key}: {str(self.extra[key])}")

            out.append("file: " + self.filename)
            out.append(
                f"fmt, byte order, word size: {self.fmt}, "
                f"{self.byte_ordering}, {self.word_size}"
            )

            out.append("")

            out2.append("\n".join(out))

        return out2

    def latitude_longitude_2d_aux_coordinates(self, yc, xc):
        """Set the latitude and longitude auxiliary coordinates.

        :Parameters:

            yc: `DimensionCoordinate`

            xc: `DimensionCoordinate`

        :Returns:

            `list`
                The keys of the auxiliary coordinates.

        """
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
            lat, lon = self.unrotated_latlon(yc.array, xc.array, BPLAT, BPLON)

            atol = self.atol
            if abs(BDX) >= atol and abs(BDY) >= atol:
                _cached_latlon[cache_key] = (lat, lon)

        if xc.has_bounds() and yc.has_bounds():  # TODO push to implementation
            cache_key = ("bounds",) + cache_key
            lat_bounds, lon_bounds = _cached_latlon.get(
                cache_key, (None, None)
            )
            if lat_bounds is None:
                xb = np.empty(xc.size + 1)
                xb[:-1] = xc.bounds.subspace[:, 0].squeeze(1).array
                xb[-1] = xc.bounds.datum(-1, 1)

                yb = np.empty(yc.size + 1)
                yb[:-1] = yc.bounds.subspace[:, 0].squeeze(1).array
                yb[-1] = yc.bounds.datum(-1, 1)

                temp_lat_bounds, temp_lon_bounds = self.unrotated_latlon(
                    yb, xb, BPLAT, BPLON
                )

                lat_bounds = np.empty(lat.shape + (4,))
                lon_bounds = np.empty(lon.shape + (4,))

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

        axes = [_axis["y"], _axis["x"]]

        keys = []
        for axiscode, array, bounds in zip(
            (10, 11), (lat, lon), (lat_bounds, lon_bounds)
        ):
            # ac = AuxiliaryCoordinate()
            ac = self.implementation.initialise_AuxiliaryCoordinate()
            ac = self.coord_data(
                ac,
                array,
                bounds=bounds,
                units=_axiscode_to_Units.setdefault(axiscode, None),
            )
            ac = self.coord_names(ac, axiscode)

            key = self.implementation.set_auxiliary_coordinate(
                self.field, ac, axes=axes, copy=False
            )
            keys.append(key)

        return keys

    def model_level_number_coordinate(self, aux=False):
        """model_level_number dimension or auxiliary coordinate.

        :Parameters:

            aux: `bool`

        :Returns:

            out : `AuxiliaryCoordinate` or `DimensionCoordinate` or `None`

        """
        array = tuple([rec.int_hdr.item(lblev) for rec in self.z_recs])

        key = array
        c = _cached_model_level_number_coordinate.get(key, None)

        if c is not None:
            if aux:
                self.field.insert_aux(c, axes=[_axis["z"]], copy=True)
                self.implementation.set_auxiliary_coordinate(
                    self.field,
                    c,
                    axes=[_axis["z"]],
                    copy=True,
                    autocyclic=_autocyclic_false,
                )
            else:
                self.implementation.set_dimension_coordinate(
                    self.field,
                    c,
                    axes=[_axis["z"]],
                    copy=True,
                    autocyclic=_autocyclic_false,
                )
        else:
            array = np.array(array, dtype=self.int_hdr_dtype)

            if array.min() < 0:
                return

            array = np.where(array == 9999, 0, array)

            axiscode = 5

            if aux:
                ac = self.implementation.initialise_AuxiliaryCoordinate()
                ac = self.coord_data(ac, array, units=Units("1"))
                ac = self.coord_names(ac, axiscode)
                self.implementation.set_auxiliary_coordinate(
                    self.field,
                    ac,
                    axes=[_axis["z"]],
                    copy=False,
                    autocyclic=_autocyclic_false,
                )

            else:
                dc = self.implementation.initialise_DimensionCoordinate()
                dc = self.coord_data(dc, array, units=Units("1"))
                dc = self.coord_names(dc, axiscode)
                dc = self.coord_axis(dc, axiscode)
                self.implementation.set_dimension_coordinate(
                    self.field,
                    dc,
                    axes=[_axis["z"]],
                    copy=False,
                    autocyclic=_autocyclic_false,
                )

            _cached_model_level_number_coordinate[key] = c

        return c

    def data_type_in_file(self, rec):
        """Return the data type of the data array.

        :Parameters:

            rec: `umfile.Rec`

        :Returns:

            `numpy.dtype`

        """
        # Find the data type
        if rec.int_hdr.item(lbuser2) == 3:
            # Boolean
            return np.dtype(bool)

        # Int or float
        return rec.get_type_and_num_words()[0]

    def printfdr(self, display=False):
        """Print out the contents of PP field headers.

        This is a bit like printfdr in the UKMO IDL PP library.

        **Examples**

        >>> u.printfdr()

        """
        if display:
            for header in self.fdr():
                print(header)
        else:
            for header in self.fdr():
                logger.info(header)

    def pseudolevel_coordinate(self, LBUSER5):
        """Create and return the pseudolevel coordinate."""
        if self.nz == 1:
            array = np.array((LBUSER5,), dtype=self.int_hdr_dtype)
        else:
            # 'Z' aggregation has been done along the pseudolevel axis
            array = np.array(
                [rec.int_hdr.item(lbuser5) for rec in self.z_recs],
                dtype=self.int_hdr_dtype,
            )
            self.z_axis = "p"

        axiscode = 40

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(
            dc, array, units=_axiscode_to_Units.setdefault(axiscode, None)
        )
        self.implementation.set_properties(
            dc, {"long_name": "pseudolevel"}, copy=False
        )
        dc.id = "UM_pseudolevel"

        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisP = self.implementation.set_domain_axis(self.field, da, copy=False)
        _axis["p"] = axisP

        self.implementation.set_dimension_coordinate(
            self.field,
            dc,
            axes=[_axis["p"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )

        return dc

    def radiation_wavelength_coordinate(self, rwl, rwl_units):
        """Creata and return the radiation wavelength coordinate."""
        array = np.array((rwl,), dtype=float)
        bounds = np.array(((0.0, rwl)), dtype=float)

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
        _axis["r"] = axisR

        self.implementation.set_dimension_coordinate(
            self.field,
            dc,
            axes=[_axis["r"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )

        return dc

    def reference_time_Units(self):
        """Return the units of the `reference_time`."""
        LBYR = self.int_hdr[lbyr]
        time_units = f"days since {LBYR}-1-1"
        calendar = self.calendar

        key = time_units + " calendar=" + calendar
        units = _Units.get(key, None)
        if units is None:
            units = Units(time_units, calendar)
            _Units[key] = units

        self.refUnits = units
        self.refunits = time_units

        return units

    def size_1_height_coordinate(self, axiscode, height, units):
        """Create and return the size-one height coordinate."""
        # Create the height coordinate from the information given in the
        # STASH to standard_name conversion table

        key = (axiscode, height, units)
        dc = _cached_size_1_height_coordinate.get(key, None)

        da = self.implementation.initialise_DomainAxis(size=1)
        axisZ = self.implementation.set_domain_axis(self.field, da, copy=False)
        _axis["z"] = axisZ

        if dc is not None:
            copy = True
        else:
            height_units = _Units.get(units, None)
            if height_units is None:
                height_units = Units(units)
                _Units[units] = height_units

            array = np.array((height,), dtype=float)

            dc = self.implementation.initialise_DimensionCoordinate()
            dc = self.coord_data(dc, array, units=height_units)
            dc = self.coord_positive(dc, axiscode, _axis["z"])
            dc = self.coord_axis(dc, axiscode)
            dc = self.coord_names(dc, axiscode)

            _cached_size_1_height_coordinate[key] = dc
            copy = False

        self.implementation.set_dimension_coordinate(
            self.field,
            dc,
            axes=[_axis["z"]],
            copy=copy,
            autocyclic=_autocyclic_false,
        )
        return dc

    def test_um_condition(self, um_condition, LBCODE, BPLAT, BPLON):
        """Return `True` if a field satisfies the condition specified
        for a STASH code to standard name conversion.

        :Parameters:

            um_condition: `str`

            LBCODE: `int`

            BPLAT: `float`

            BPLON: `float`

        :Returns:

            `bool`
                `True` if a field satisfies the condition specified,
                `False` otherwise.

        **Examples**

        >>> ok = u.test_um_condition('true_latitude_longitude', ...)

        """
        if um_condition == "true_latitude_longitude":
            if LBCODE in _true_latitude_longitude_lbcodes:
                return True

            # Check pole location in case of incorrect LBCODE
            atol = self.atol
            if (
                abs(BPLAT - 90.0) <= atol + cf_rtol() * 90.0
                and abs(BPLON) <= atol
            ):
                return True

        elif um_condition == "rotated_latitude_longitude":
            if LBCODE in _rotated_latitude_longitude_lbcodes:
                return True

            # Check pole location in case of incorrect LBCODE
            atol = self.atol
            if not (
                abs(BPLAT - 90.0) <= atol + cf_rtol() * 90.0
                and abs(BPLON) <= atol
            ):
                return True

        else:
            raise ValueError(
                "Unknown UM condition in STASH code conversion table: "
                f"{um_condition!r}"
            )

        # Still here? Then the condition has not been satisfied.
        return

    def test_um_version(self, valid_from, valid_to, um_version):
        """Return `True` if the UM version applicable to this field is
        within the given range.

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

        **Examples**

        >>> ok = u.test_um_version(401, 505, 1001)
        >>> ok = u.test_um_version(401, None, 606.3)
        >>> ok = u.test_um_version(None, 405, 401)

        """
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
        """Return the T dimension coordinate.

        :Parameters:

            axiscode: `int`

        :Returns:

            `DimensionCoordinate`

        """
        recs = self.t_recs

        vtimes = np.array([self.vtime(rec) for rec in recs], dtype=float)
        dtimes = np.array([self.dtime(rec) for rec in recs], dtype=float)

        if np.isnan(vtimes.sum()) or np.isnan(dtimes.sum()):
            return  # ppp

        IB = self.lbtim_ib

        if IB <= 1 or vtimes.item(0) >= dtimes.item(0):
            array = vtimes
            bounds = None
            climatology = False
        elif IB == 3:
            # The field is a time mean from T1 to T2 for each year
            # from LBYR to LBYRD
            ctimes = np.array([self.ctime(rec) for rec in recs])
            array = 0.5 * (vtimes + ctimes)
            bounds = self.create_bounds_array(vtimes, dtimes)

            climatology = True
        else:
            array = 0.5 * (vtimes + dtimes)
            bounds = self.create_bounds_array(vtimes, dtimes)

            climatology = False

        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisT = self.implementation.set_domain_axis(self.field, da, copy=False)
        _axis["t"] = axisT

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(
            dc, array, bounds, units=self.refUnits, climatology=climatology
        )
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        self.implementation.set_dimension_coordinate(
            self.field,
            dc,
            axes=[_axis["t"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )
        return dc

    def time_coordinate_from_extra_data(self, axiscode, axis):
        """Create the time coordinate from extra data and return it.

        :Returns:

            `DimensionCoordinate`

        """
        extra = self.extra

        array = extra[axis]
        bounds = extra.get(axis + "_bounds", None)

        calendar = self.calendar
        if calendar == "360_day":
            units = _Units["360_day 0-1-1"]
        elif calendar == "gregorian":
            units = _Units["gregorian 1752-09-13"]
        elif calendar == "365_day":
            units = _Units["365_day 1752-09-13"]
        else:
            units = None

        # Create time domain axis.
        #
        # Note that `axis` might not be "t". For instance, it could be
        # "y" if the time coordinates are coming from extra data.
        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisT = self.implementation.set_domain_axis(self.field, da, copy=False)
        _axis[axis] = axisT

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(dc, array, bounds, units=units)
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        self.implementation.set_dimension_coordinate(
            self.field,
            dc,
            axes=(axisT,),
            copy=False,
            autocyclic=_autocyclic_false,
        )

        return dc

    def time_coordinate_from_um_timeseries(self, axiscode, axis):
        """Create the time coordinate from a timeseries field."""
        # This PP/FF field is a timeseries. The validity time is
        # taken to be the time for the first sample, the data time
        # for the last sample, with the others evenly between.
        rec = self.recs[0]
        vtime = self.vtime(rec)
        dtime = self.dtime(rec)

        size = self.lbuser3 - 1.0
        delta = (dtime - vtime) / size

        calendar = self.calendar
        if calendar == "360_day":
            units = _Units["360_day 0-1-1"]
        elif calendar == "gregorian":
            units = _Units["gregorian 1752-09-13"]
        elif calendar == "365_day":
            units = _Units["365_day 1752-09-13"]
        else:
            units = None

        array = np.arange(vtime, vtime + delta * size, size, dtype=float)

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(dc, array, units=units)
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)
        self.implementation.set_dimension_coordinate(
            self.field,
            dc,
            axes=[_axis[axis]],
            copy=False,
            autocyclic=_autocyclic_false,
        )
        return dc

    def vtime(self, rec):
        """Return the elapsed time since the validity time of the given
        record.

        :Parameters:

            rec:

        :Returns:

            `float`

        **Examples**

        >>> u.vtime(rec)
        31.5

        """
        units = self.refunits
        calendar = self.calendar

        LBVTIME = tuple(self.header_vtime(rec))

        key = (LBVTIME, units, calendar)

        time = _cached_date2num.get(key, None)
        if time is None:
            # It is important to use the same time_units as dtime
            try:
                time = cftime.date2num(
                    cftime.datetime(*LBVTIME, calendar=self.calendar),
                    units,
                    calendar,
                )

                _cached_date2num[key] = time
            except ValueError:
                time = np.nan  # ppp

        return time

    #    def dddd(self):
    #        """TODO."""
    #        for axis_code, extra_type in zip((11, 10), ("x", "y")):
    #            coord_type = extra_type + "_domain_bounds"
    #
    #            if coord_type in p.extra:
    #                p.extra[coord_type]
    #                # Create, from extra data, an auxiliary coordinate
    #                # with 1) data and bounds, if the upper and lower
    #                # bounds have no missing values; or 2) data but no
    #                # bounds, if the upper bound has missing values
    #                # but the lower bound does not.
    #
    #                # Should be the axis which has axis_code 13
    #                file_position = ppfile.tell()
    #                bounds = p.extra[coord_type][...]
    #
    #                # Reset the file pointer after reading the extra
    #                # data into a numpy array
    #                ppfile.seek(file_position, os.SEEK_SET)
    #                data = None
    #                # dch also test in bmdi?:
    #                if np.any(bounds[..., 1] == _pp_rmdi):
    #                    # dch also test in bmdi?:
    #                    if not np.any(bounds[..., 0] == _pp_rmdi):
    #                        data = bounds[..., 0]
    #                    bounds = None
    #                else:
    #                    data = np.mean(bounds, axis=1)
    #
    #                if (data, bounds) != (None, None):
    #                    aux = "aux%(auxN)d" % locals()
    #                    auxN += 1  # Increment auxiliary number
    #
    #                    coord = _create_Coordinate(
    #                        domain,
    #                        aux,
    #                        axis_code,
    #                        p=p,
    #                        array=data,
    #                        aux=True,
    #                        bounds_array=bounds,
    #                        pubattr={"axis": None},
    #                        # DCH xdim? should be the axis which has axis_code 13:
    #                        dimensions=[xdim],
    #                    )
    #            else:
    #                coord_type = "{0}_domain_lower_bound".format(extra_type)
    #                if coord_type in p.extra:
    #                    # Create, from extra data, an auxiliary
    #                    # coordinate with data but no bounds, if the
    #                    # data noes not contain any missing values
    #                    file_position = ppfile.tell()
    #                    data = p.extra[coord_type][...]
    #                    # Reset the file pointer after reading the
    #                    # extra data into a numpy array
    #                    ppfile.seek(file_position, os.SEEK_SET)
    #                    if not np.any(data == _pp_rmdi):  # dch + test in bmdi
    #                        aux = "aux%(auxN)d" % locals()
    #                        auxN += 1  # Increment auxiliary number
    #                        coord = _create_Coordinate(
    #                            domain,
    #                            aux,
    #                            axis_code,
    #                            p=p,
    #                            aux=True,
    #                            array=np.array(data),
    #                            pubattr={"axis": None},
    #                            dimensions=[xdim],
    #                        )  # DCH xdim?

    def unrotated_latlon(self, rotated_lat, rotated_lon, pole_lat, pole_lon):
        """Create 2-d arrays of unrotated latitudes and longitudes.

        :Parameters:

            rotated_lat: `numpy.ndarray`

            rotated_lon: `numpy.ndarray`

            pole_lat: `float`

            pole_lon: `float`

        :Returns:

            lat, lon: `numpy.ndarray`, `numpy.ndarray`

        """
        # Make sure rotated_lon and pole_lon is in [0, 360)
        pole_lon = pole_lon % 360.0

        # Convert everything to radians
        pole_lon *= _pi_over_180
        pole_lat *= _pi_over_180

        cos_pole_lat = np.cos(pole_lat)
        sin_pole_lat = np.sin(pole_lat)

        # Create appropriate copies of the input rotated arrays
        rot_lon = rotated_lon.copy()
        rot_lat = rotated_lat.view()

        # Make sure rotated longitudes are between -180 and 180
        rot_lon %= 360.0
        rot_lon = np.where(rot_lon < 180.0, rot_lon, rot_lon - 360)

        # Create 2-d arrays of rotated latitudes and longitudes in radians
        nlat = rot_lat.size
        nlon = rot_lon.size
        rot_lon = np.resize(np.deg2rad(rot_lon), (nlat, nlon))
        rot_lat = np.resize(np.deg2rad(rot_lat), (nlon, nlat))
        rot_lat = np.transpose(rot_lat, axes=(1, 0))

        # Find unrotated latitudes
        CPART = np.cos(rot_lon) * np.cos(rot_lat)
        sin_rot_lat = np.sin(rot_lat)
        x = cos_pole_lat * CPART + sin_pole_lat * sin_rot_lat
        x = np.clip(x, -1.0, 1.0)
        unrotated_lat = np.arcsin(x)

        # Find unrotated longitudes
        x = -cos_pole_lat * sin_rot_lat + sin_pole_lat * CPART
        x /= np.cos(unrotated_lat)
        # dch /0 or overflow here? surely lat could be ~+-pi/2? if so,
        # does x ~ cos(lat)?
        x = np.clip(x, -1.0, 1.0)
        unrotated_lon = -np.arccos(x)

        unrotated_lon = np.where(rot_lon > 0.0, -unrotated_lon, unrotated_lon)
        if pole_lon >= self.atol:
            SOCK = pole_lon - np.pi
        else:
            SOCK = 0
        unrotated_lon += SOCK

        # Convert unrotated latitudes and longitudes to degrees
        unrotated_lat = np.rad2deg(unrotated_lat)
        unrotated_lon = np.rad2deg(unrotated_lon)

        # Return unrotated latitudes and longitudes
        return (unrotated_lat, unrotated_lon)

    def xy_coordinate(self, axiscode, axis):
        """Create an X or Y dimension coordinate from header entries or
        extra data.

        :Parameters:

            axiscode: `int`

            axis: `str`
                Which type of coordinate to create: ``'x'`` or
                ``'y'``.

        :Returns:

            (`str`, `DimensionCoordinate`)

        """
        X = axiscode in (11, -11)

        if X:
            autocyclic = {"X": True}
        else:
            autocyclic = _autocyclic_false

        if axis == "x":
            delta = self.bdx
            origin = self.real_hdr[bzx]
            size = self.lbnpt

            da = self.implementation.initialise_DomainAxis(size=size)
            axis_key = self.implementation.set_domain_axis(
                self.field, da, copy=False
            )
            _axis["x"] = axis_key
        else:
            delta = self.bdy
            origin = self.real_hdr[bzy]
            size = self.lbrow

            da = self.implementation.initialise_DomainAxis(size=size)
            axis_key = self.implementation.set_domain_axis(
                self.field, da, copy=False
            )
            _axis["y"] = axis_key

            autocyclic = _autocyclic_false

        if abs(delta) > self.atol:
            # Create regular coordinates from header items
            if axiscode == 11 or axiscode == -11:
                origin -= divmod(origin + delta * size, 360.0)[0] * 360
                while origin + delta * size > 360.0:
                    origin -= 360.0
                while origin + delta * size < -360.0:
                    origin += 360.0

            array = _cached_regular_array.get((origin, delta, size))
            if array is None:
                array = np.arange(
                    origin + delta,
                    origin + delta * (size + 0.5),
                    delta,
                    dtype=float,
                )
                _cached_regular_array[(origin, delta, size)] = array

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
                bounds = _cached_regular_bounds.get((origin, delta, size))
                if bounds is None:
                    delta_by_2 = 0.5 * delta
                    bounds = self.create_bounds_array(
                        array - delta_by_2, array + delta_by_2
                    )
                    _cached_regular_bounds[(origin, delta, size)] = bounds
        else:
            # Create coordinate from extra data
            array = self.extra.get(axis, None)
            lower_bounds = self.extra.get(axis + "_lower_bound", None)
            upper_bounds = self.extra.get(axis + "_upper_bound", None)
            if lower_bounds is not None and upper_bounds is not None:
                bounds = self.create_bounds_array(lower_bounds, upper_bounds)
            else:
                bounds = None

        units = _axiscode_to_Units.setdefault(axiscode, None)

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(dc, array, bounds, units=units)
        dc = self.coord_positive(dc, axiscode, axis_key)
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        if X and bounds is not None:
            autocyclic["cyclic"] = abs(bounds[0, 0] - bounds[-1, -1]) == 360.0
            autocyclic["period"] = self.get_data(np.array(360.0), units)
            autocyclic["axis"] = axis_key
            autocyclic["coord"] = dc

        key = self.implementation.set_dimension_coordinate(
            self.field, dc, axes=[axis_key], copy=False, autocyclic=autocyclic
        )

        return key, dc, axis_key

    def get_data(self, array, units, fill_value=None, bounds=False):
        """Create data, or get it from the cache.

        .. versionadded:: 3.15.0

        :Parameters:

            array: `np.ndarray`
                The data.

            units: `Units
                The units.

            fill_value: scalar
                The fill value.

            bounds: `bool`
                Whether or not the data are bounds of 1-d coordinates.

        :Returns:

            `Data`
                An independent copy of the new data.

        """
        token = tokenize(array, units)
        data = _cached_data.get(token)
        if data is None:
            data = Data(array, units=units, fill_value=fill_value)
            if not bounds:
                if array.size == 1:
                    value = array.item(0)
                    data._set_cached_elements({0: value, -1: value})
                else:
                    data._set_cached_elements(
                        {
                            0: array.item(0),
                            1: array.item(1),
                            -1: array.item(-1),
                        }
                    )
            else:
                data._set_cached_elements(
                    {
                        0: array.item(0),
                        1: array.item(1),
                        -2: array.item(-2),
                        -1: array.item(-1),
                    }
                )

            _cached_data[token] = data

        return data.copy()

    def site_coordinates_from_extra_data(self):
        """Create site-related coordinates from extra data.

        :Returns:

            `None`

        """
        # Create coordinate from extra data
        for axis, standard_name, units in zip(
            ("x", "y"),
            ("longitude", "latitude"),
            (_Units["degrees_east"], _Units["degrees_north"]),
        ):
            lower_bounds = self.extra.get(axis + "_domain_lower_bound", None)
            upper_bounds = self.extra.get(axis + "_domain_upper_bound", None)
            if lower_bounds is None or upper_bounds is None:
                continue

            # Still here?
            bounds = self.create_bounds_array(lower_bounds, upper_bounds)
            array = np.average(bounds, axis=1)

            ac = self.implementation.initialise_AuxiliaryCoordinate()
            ac = self.coord_data(ac, array, bounds, units=units)

            ac.standard_name = standard_name
            ac.long_name = "region limit"
            self.implementation.set_auxiliary_coordinate(
                self.field,
                ac,
                axes=[_axis["site_axis"]],
                copy=False,
                autocyclic=_autocyclic_false,
            )

        array = self.extra.get("domain_title", None)
        if array is not None:
            ac = self.implementation.initialise_AuxiliaryCoordinate()
            ac = self.coord_data(ac, array, None, units=None)

            ac.standard_name = "region"
            self.implementation.set_auxiliary_coordinate(
                self.field,
                ac,
                axes=[_axis["site_axis"]],
                copy=False,
                autocyclic=_autocyclic_false,
            )

    @_manage_log_level_via_verbose_attr
    def z_coordinate(self, axiscode):
        """Create a Z dimension coordinate from BLEV.

        :Parameters:

            axiscode: `int`

        :Returns:

            `DimensionCoordinate`

        """
        if self.info:
            logger.info(
                "Creating Z coordinates and bounds from BLEV, BRLEV and "
                "BRSVD1:"
            )  # pragma: no cover

        z_recs = self.z_recs
        array = tuple([rec.real_hdr.item(blev) for rec in z_recs])
        bounds0 = tuple(
            [rec.real_hdr[brlev] for rec in z_recs]
        )  # lower level boundary
        bounds1 = tuple([rec.real_hdr[brsvd1] for rec in z_recs])  # bulev
        if _coord_positive.get(axiscode, None) == "down":
            bounds0, bounds1 = bounds1, bounds0

        array = np.array(array, dtype=self.real_hdr_dtype)
        bounds0 = np.array(bounds0, dtype=self.real_hdr_dtype)
        bounds1 = np.array(bounds1, dtype=self.real_hdr_dtype)
        bounds = self.create_bounds_array(bounds0, bounds1)

        if (bounds0 == bounds1).all() or np.allclose(bounds.min(), _pp_rmdi):
            bounds = None
        else:
            bounds = self.create_bounds_array(bounds0, bounds1)

        da = self.implementation.initialise_DomainAxis(size=array.size)
        axisZ = self.implementation.set_domain_axis(self.field, da, copy=False)
        _axis["z"] = axisZ

        dc = self.implementation.initialise_DimensionCoordinate()
        dc = self.coord_data(
            dc,
            array,
            bounds=bounds,
            units=_axiscode_to_Units.setdefault(axiscode, None),
        )
        dc = self.coord_positive(dc, axiscode, _axis["z"])
        dc = self.coord_axis(dc, axiscode)
        dc = self.coord_names(dc, axiscode)

        self.implementation.set_dimension_coordinate(
            self.field,
            dc,
            axes=[_axis["z"]],
            copy=False,
            autocyclic=_autocyclic_false,
        )

        return dc


class UMRead(cfdm.read_write.IORead):
    """A container for instantiating Fields from a UM fields file."""

    @_manage_log_level_via_verbosity
    def read(
        self,
        filename,
        um_version=405,
        aggregate=True,
        endian=None,
        word_size=None,
        set_standard_name=True,
        height_at_top_of_model=None,
        fmt=None,
        chunk=True,
        verbose=None,
        select=None,
    ):
        """Read fields from a PP file or UM fields file.

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

            verbose: `int` or `str` or `None`, optional
                If an integer from ``-1`` to ``3``, or an equivalent string
                equal ignoring case to one of:

                * ``'DISABLE'`` (``0``)
                * ``'WARNING'`` (``1``)
                * ``'INFO'`` (``2``)
                * ``'DETAIL'`` (``3``)
                * ``'DEBUG'`` (``-1``)

                set for the duration of the method call only as the minimum
                cut-off for the verboseness level of displayed output (log)
                messages, regardless of the globally-configured `cf.log_level`.
                Note that increasing numerical value corresponds to increasing
                verbosity, with the exception of ``-1`` as a special case of
                maximal and extreme verbosity.

                Otherwise, if `None` (the default value), output messages will
                be shown according to the value of the `cf.log_level` setting.

                Overall, the higher a non-negative integer or equivalent string
                that is set (up to a maximum of ``3``/``'DETAIL'``) for
                increasing verbosity, the more description that is printed
                about the read process.

            set_standard_name: `bool`, optional

        select: (sequence of) `str` or `Query` or `re.Pattern`, optional
            Only return field constructs whose identities match the
            given values(s), i.e. those fields ``f`` for which
            ``f.match_by_identity(*select)`` is `True`. See
            `cf.Field.match_by_identity` for details.

            This is equivalent to, but faster than, not using the
            *select* parameter but applying its value to the returned
            field list with its `cf.FieldList.select_by_identity`
            method. For example, ``fl = cf.read(file,
            select='stash_code=3236')`` is equivalent to ``fl =
            cf.read(file).select_by_identity('stash_code=3236')``.

        :Returns:

            `list`
                The fields in the file.

        **Examples**

        >>> f = read('file.pp')
        >>> f = read('*/file[0-9].pp', um_version=708)

        """
        if not _stash2standard_name:
            # --------------------------------------------------------
            # Create the STASH code to standard_name conversion
            # dictionary
            # --------------------------------------------------------
            load_stash2standard_name()

        if endian:
            byte_ordering = endian + "_endian"
        else:
            byte_ordering = None

        self.read_vars = {
            "filename": filename,
            "byte_ordering": byte_ordering,
            "word_size": word_size,
            "fmt": fmt,
        }

        history = f"Converted from UM/PP by cf-python v{__version__}"

        if endian:
            byte_ordering = endian + "_endian"
        else:
            byte_ordering = None

        f = self.file_open(filename, parse=True)

        info = is_log_level_info(logger)

        um = [
            UMField(
                var,
                f.fmt,
                f.byte_ordering,
                f.word_size,
                um_version,
                set_standard_name,
                history=history,
                height_at_top_of_model=height_at_top_of_model,
                verbose=verbose,
                implementation=self.implementation,
                select=select,
                info=info,
            )
            for var in f.vars
        ]

        self.file_close()

        return [field for x in um for field in x.fields if field]

    def _open_um_file(
        self,
        filename,
        aggregate=True,
        fmt=None,
        word_size=None,
        byte_ordering=None,
        parse=True,
    ):
        """Open a UM fields file or PP file.

        :Parameters:

            filename: `str`
                The file to be opened.

            parse: `bool`, optional
                If True, the default, then parse the contents. If
                False then the contents are not parsed, which can be
                considerably faster in cases when the contents are not
                required.

                .. versionadded:: 3.16.2

        :Returns:

            `umread_lib.umfile.File`
                The open PP or FF file object.

        """
        self.file_close()
        try:
            f = File(
                filename,
                byte_ordering=byte_ordering,
                word_size=word_size,
                fmt=fmt,
                parse=parse,
            )
        except Exception as error:
            try:
                f.close_fd()
            except Exception:
                pass

            raise Exception(error)

        self._um_file = f
        return f

    def is_um_file(self, filename):
        """Whether or not a file is a PP file or UM fields file.

        Note that the file type is determined by inspecting the file's
        content and any file suffix is not not considered.

        :Parameters:

            filename: `str`
                The file.

        :Returns:

            `bool`

        **Examples**

        >>> r.is_um_file('ppfile')
        True

        """
        try:
            # Note: No need to completely parse the file to ascertain
            #       if it's PP or FF.
            self.file_open(filename, parse=False)
        except Exception:
            self.file_close()
            return False
        else:
            self.file_close()
            return True

    def file_close(self):
        """Close the file that has been read.

        :Returns:

            `None`

        """
        f = getattr(self, "_um_file", None)
        if f is not None:
            f.close_fd()

        self._um_file = None

    def file_open(self, filename, parse=True):
        """Open the file for reading.

        :Paramters:

            filename: `str`
                The file to be read.

            parse: `bool`, optional
                If True, the default, then parse the contents. If
                False then the contents are not parsed, which can be
                considerably faster in cases when the contents are not
                required.

                .. versionadded:: 3.16.2

        :Returns:

            `umread_lib.umfile.File`
                The open PP or FF file object.

        """
        g = getattr(self, "read_vars", {})

        return self._open_um_file(
            filename,
            byte_ordering=g.get("byte_ordering"),
            word_size=g.get("word_size"),
            fmt=g.get("fmt"),
            parse=parse,
        )


"""
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

"""
