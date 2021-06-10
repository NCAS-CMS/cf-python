"""The Python `cf` package is an Earth Science data analysis library
that is built on a complete implementation of the CF data model.

The `cf` package implements the CF data model for its internal data
structures and so is able to process any CF-compliant dataset. It is
not strict about CF-compliance, however, so that partially conformant
datasets may be ingested from existing datasets and written to new
datasets. This is so that datasets which are partially conformant may
nonetheless be modified in memory.

The `cf` package can:

* read field constructs from netCDF, CDL, PP and UM datasets,

* create new field constructs in memory,

* inspect field constructs,

* test whether two field constructs are the same,

* modify field construct metadata and data,

* create subspaces of field constructs,

* write and append field constructs to netCDF datasets on disk,

* incorporate, and create, metadata stored in external files (*new in
  version 3.0.0*),

* read, write, and create data that have been compressed by convention
  (i.e. ragged or gathered arrays), whilst presenting a view of the
  data in its uncompressed form,

* read, write, and create coordinates defined by geometry cells (*new
  in version 3.2.0*),

* combine field constructs arithmetically,

* manipulate field construct data by arithmetical and trigonometrical
  operations,

* perform statistical collapses on field constructs,

* perform histogram, percentile and binning operations on field
  constructs (*new in version 3.0.3*),

* regrid field constructs with (multi-)linear, nearest neighbour,
  first- and second-order conservative and higher order patch recovery
  methods,

* apply convolution filters to field constructs,

* calculate derivatives of field constructs,

* create field constructs to create derived quantities (such as
  vorticity).

All of the above use LAMA functionality, which allows multiple fields
larger than the available memory to exist and be manipulated.


**Hierarchical groups**

Hierarchical groups provide a powerful mechanism to structure
variables within datasets. A future release of `cf` will include
support for netCDF4 files containing data organised in hierarchical
groups, but this is not available in version 3.2.0 (even though it is
allowed in CF-1.8).


**Visualization**

Powerful, flexible, and very simple to produce visualizations of field
constructs uses the `cfplot` package
(http://ajheaps.github.io/cf-plot), that is automatically installed
along with with `cf`.

See the :ref:`cf-python home page <cf-python-home>` for documentation,
installation and source code.

"""

__Conventions__ = "CF-1.8"
__date__ = "2021-06-10"
__version__ = "3.10.0"

_requires = (
    "numpy",
    "netCDF4",
    "cftime",
    "cfunits",
    "cfdm",
    "psutil",
)

x = ", ".join(_requires)
_error0 = f"cf v{ __version__} requires the modules {x}. "

try:
    import cfdm
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

__cf_version__ = cfdm.core.__cf_version__

from distutils.version import LooseVersion
import importlib.util
import platform

# Check the version of Python
_minimum_vn = "3.6.0"
if LooseVersion(platform.python_version()) < LooseVersion(_minimum_vn):
    raise ValueError(
        f"Bad python version: cf requires python version {_minimum_vn} "
        f"or later. Got {platform.python_version()}"
    )

if LooseVersion(platform.python_version()) < LooseVersion("3.7.0"):
    print(
        "\nDeprecation Warning: Python 3.6 support will be removed at "
        "the next version of cf\n"
    )

_found_ESMF = bool(importlib.util.find_spec("ESMF"))

# TODODASK - Remove the next 2 lines when the move to dask is complete
mpi_on = False
mpi_size = 1

try:
    import netCDF4
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

try:
    import numpy
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

try:
    import cftime
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

try:
    import cfunits
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

try:
    import psutil
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

# Check the version of psutil
_minimum_vn = "0.6.0"
if LooseVersion(psutil.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad psutil version: cf requires psutil>={_minimum_vn}. "
        f"Got {psutil.__version__} at {psutil.__file__}"
    )

# Check the version of netCDF4
_minimum_vn = "1.5.4"
if LooseVersion(netCDF4.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad netCDF4 version: cf requires netCDF4>={_minimum_vn}. "
        f"Got {netCDF4.__version__} at {netCDF4.__file__}"
    )

# Check the version of cftime
_minimum_vn = "1.5.0"
if LooseVersion(cftime.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad cftime version: cf requires cftime>={_minimum_vn}. "
        f"Got {cftime.__version__} at {cftime.__file__}"
    )

# Check the version of numpy
_minimum_vn = "1.15"
if LooseVersion(numpy.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad numpy version: cf requires numpy>={_minimum_vn}. "
        f"Got {numpy.__version__} at {numpy.__file__}"
    )

# Check the version of cfunits
_minimum_vn = "3.3.3"
if LooseVersion(cfunits.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad cfunits version: cf requires cfunits>={_minimum_vn}. "
        f"Got {cfunits.__version__} at {cfunits.__file__}"
    )

# Check the version of cfdm
_minimum_vn = "1.8.9.0"
_maximum_vn = "1.8.10.0"
_cfdm_version = LooseVersion(cfdm.__version__)
if not LooseVersion(_minimum_vn) <= _cfdm_version < LooseVersion(_maximum_vn):
    raise RuntimeError(
        f"Bad cfdm version: cf requires {_minimum_vn}<=cfdm<{_maximum_vn}. "
        f"Got {_cfdm_version} at {cfdm.__file__}"
    )

from .constructs import Constructs

from .mixin import Coordinate

from .count import Count
from .index import Index
from .list import List
from .nodecountproperties import NodeCountProperties
from .partnodecountproperties import PartNodeCountProperties
from .interiorring import InteriorRing

from .bounds import Bounds
from .domain import Domain
from .datum import Datum
from .coordinateconversion import CoordinateConversion

from .cfdatetime import dt, dt_vector
from .flags import Flags
from .timeduration import TimeDuration, Y, M, D, h, m, s
from .units import Units

from .constructlist import ConstructList
from .fieldlist import FieldList

from .dimensioncoordinate import DimensionCoordinate
from .auxiliarycoordinate import AuxiliaryCoordinate
from .coordinatereference import CoordinateReference
from .cellmethod import CellMethod
from .cellmeasure import CellMeasure
from .domainancillary import DomainAncillary
from .domainaxis import DomainAxis
from .fieldancillary import FieldAncillary
from .field import Field
from .data import (
    Data,
    FilledArray,
    GatheredArray,
    NetCDFArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
)

from .aggregate import aggregate
from .query import (
    Query,
    lt,
    le,
    gt,
    ge,
    eq,
    ne,
    contain,
    contains,
    wi,
    wo,
    set,
    year,
    month,
    day,
    hour,
    minute,
    second,
    dtlt,
    dtle,
    dtgt,
    dtge,
    dteq,
    dtne,
    cellsize,
    cellge,
    cellgt,
    cellle,
    celllt,
    cellwi,
    cellwo,
    djf,
    mam,
    jja,
    son,
    seasons,
)
from .constants import *  # noqa: F403
from .functions import *  # noqa: F403
from .maths import relative_vorticity, histogram
from .examplefield import example_field, example_fields, example_domain

from .cfimplementation import CFImplementation, implementation

from .read_write import read, write

from .regrid import RegridOperator


# Set up basic logging for the full project with a root logger
import logging
import sys

# Configure the root logger which all module loggers inherit from:
logging.basicConfig(
    stream=sys.stdout,
    style="{",  # default is old style ('%') string formatting
    format="{message}",  # no module names or datetimes etc. for basic case
    level=logging.WARNING,  # default but change level via log_level()
)

# And create custom level inbetween 'INFO' & 'DEBUG', to understand value see:
# https://docs.python.org/3.8/howto/logging.html#logging-levels
logging.DETAIL = 15  # set value as an attribute as done for built-in levels
logging.addLevelName(logging.DETAIL, "DETAIL")


def detail(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.DETAIL):
        self._log(logging.DETAIL, message, args, **kwargs)


logging.Logger.detail = detail


# Also create special, secret level below even 'DEBUG'. It will not be
# advertised to users. The user-facing cf.log_level() can set all but this
# one level; we deliberately have not set up:
#     cf.log_level('PARTITIONING')
# to work to change the level to logging.PARTITIONING. Instead, to set this
# manipulate the cf root logger directly via a built-in method, i.e call:
#     cf.logging.getLogger().setLevel('PARTITIONING')
logging.PARTITIONING = 5
logging.addLevelName(logging.PARTITIONING, "PARTITIONING")


def partitioning(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.PARTITIONING):
        self._log(logging.PARTITIONING, message, args, **kwargs)


logging.Logger.partitioning = partitioning
