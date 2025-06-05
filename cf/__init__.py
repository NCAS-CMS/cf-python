"""The Python `cf` package is an Earth Science data analysis library
that is built on a complete implementation of the CF data model.

The `cf` package implements the CF data model for its internal data
structures and so is able to process any CF-compliant dataset. It is
not strict about CF-compliance, however, so that partially conformant
datasets may be ingested from existing datasets and written to new
datasets. This is so that datasets which are partially conformant may
nonetheless be modified in memory.

The `cf` package uses `dask` for all of its array manipulation and
can:

* read field constructs from netCDF, CDL, Zarr, PP and UM datasets,

* read field constructs and domain constructs from netCDF, CDL, PP and
  UM datasets with a choice of netCDF backends,

* read files from OPeNDAP servers and S3 object stores,

* create new field constructs in memory,

* write and append field constructs to netCDF datasets on disk,

* read, write, and manipulate UGRID mesh topologies,

* read, write, and create coordinates defined by geometry cells,

* read netCDF and CDL datasets containing hierarchical groups,

* inspect field constructs,

* test whether two field constructs are the same,

* modify field construct metadata and data,

* create subspaces of field constructs,

* write field constructs to netCDF datasets on disk,

* incorporate, and create, metadata stored in external files,

* read, write, and create data that have been compressed by convention
  (i.e. ragged or gathered arrays, or coordinate arrays compressed by
  subsampling), whilst presenting a view of the data in its
  uncompressed form,

* combine field constructs arithmetically,

* manipulate field construct data by arithmetical and trigonometrical
  operations,

* perform statistical collapses on field constructs,

* perform histogram, percentile and binning operations on field
  constructs,

* regrid field constructs with (multi-)linear, nearest neighbour,
  first- and second-order conservative and higher order patch recovery
  methods,

* apply convolution filters to field constructs,

* create running means from field constructs,

* apply differential operators to field constructs,

* create derived quantities (such as relative vorticity).


**Visualisation**

Powerful, flexible, and very simple to produce visualisations of field
constructs uses the `cfplot` package
(https://ncas-cms.github.io/cf-plot/build/), that is automatically installed
along with with `cf`.

See the :ref:`cf-python home page <cf-python-home>` for documentation,
installation and source code.

"""

__date__ = "2025-06-05"
__version__ = "3.18.0"

_requires = (
    "numpy",
    "netCDF4",
    "cftime",
    "cfunits",
    "cfdm",
    "psutil",
    "dask",
    "packaging",
    "scipy",
)
x = ", ".join(_requires)
_error0 = f"cf v{__version__} requires the modules {x}. "

import importlib.util
from platform import python_version

_found_esmpy = bool(importlib.util.find_spec("esmpy"))

try:
    import packaging
    from packaging.version import Version
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    _minimum_vn = "20.0"
    if Version(packaging.__version__) < Version(_minimum_vn):
        raise RuntimeError(
            f"Bad packaging version: cf requires packaging>={_minimum_vn}. "
            f"Got {packaging.__version__} at {packaging.__file__}"
        )

try:
    import cfdm
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    # Check the version of cfdm
    _minimum_vn = "1.12.2.0"
    _maximum_vn = "1.12.3.0"
    _cfdm_version = Version(cfdm.__version__)
    if _cfdm_version < Version(_minimum_vn) or _cfdm_version >= Version(
        _maximum_vn
    ):
        raise RuntimeError(
            "Bad cfdm version: cf requires "
            f"{_minimum_vn}<=cfdm<{_maximum_vn}. "
            f"Got {_cfdm_version} at {cfdm.__file__}"
        )

__cf_version__ = cfdm.__cf_version__
__Conventions__ = f"CF-{__cf_version__}"

try:
    import netCDF4
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    _minimum_vn = "1.7.2"
    if Version(netCDF4.__version__) < Version(_minimum_vn):
        raise RuntimeError(
            f"Bad netCDF4 version: cf requires netCDF4>={_minimum_vn}. "
            f"Got {netCDF4.__version__} at {netCDF4.__file__}"
        )

try:
    import numpy as np
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    _minimum_vn = "2.0.0"
    if Version(np.__version__) < Version(_minimum_vn):
        raise ValueError(
            f"Bad numpy version: cf requires numpy>={_minimum_vn} "
            f"Got {np.__version__} at {np.__file__}"
        )

try:
    import cftime
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    _minimum_vn = "1.6.4"
    if Version(cftime.__version__) < Version(_minimum_vn):
        raise RuntimeError(
            f"Bad cftime version: cf requires cftime>={_minimum_vn}. "
            f"Got {cftime.__version__} at {cftime.__file__}"
        )

try:
    import cfunits
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    _minimum_vn = "3.3.7"
    if Version(cfunits.__version__) < Version(_minimum_vn):
        raise RuntimeError(
            f"Bad cfunits version: cf requires cfunits>={_minimum_vn}. "
            f"Got {cfunits.__version__} at {cfunits.__file__}"
        )

try:
    import psutil
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    _minimum_vn = "0.6.0"
    if Version(psutil.__version__) < Version(_minimum_vn):
        raise RuntimeError(
            f"Bad psutil version: cf requires psutil>={_minimum_vn}. "
            f"Got {psutil.__version__} at {psutil.__file__}"
        )

# Check the version of dask
try:
    import dask
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    _minimum_vn = "2025.5.1"
    if Version(dask.__version__) < Version(_minimum_vn):
        raise ValueError(
            f"Bad dask version: cf requires dask>={_minimum_vn}. "
            f"Got {dask.__version__} at {dask.__file__}"
        )

try:
    import scipy
except ImportError as error1:
    raise ImportError(_error0 + str(error1))
else:
    _minimum_vn = "1.10.0"
    if Version(scipy.__version__) < Version(_minimum_vn):
        raise RuntimeError(
            f"Bad scipy version: cf requires scipy>={_minimum_vn}. "
            f"Got {scipy.__version__} at {scipy.__file__}"
        )

_minimum_vn = "3.9.0"
if Version(python_version()) < Version(_minimum_vn):
    raise ValueError(
        f"Bad python version: cf requires python>={_minimum_vn}. "
        f"Got {python_version()}"
    )

del _minimum_vn, _maximum_vn

from .constructs import Constructs

from .mixin import Coordinate

from .count import Count
from .index import Index
from .interpolationparameter import InterpolationParameter
from .list import List
from .nodecountproperties import NodeCountProperties
from .partnodecountproperties import PartNodeCountProperties
from .interiorring import InteriorRing
from .quantization import Quantization
from .tiepointindex import TiePointIndex

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
from .domainlist import DomainList

from .dimensioncoordinate import DimensionCoordinate
from .auxiliarycoordinate import AuxiliaryCoordinate
from .coordinatereference import CoordinateReference
from .cellconnectivity import CellConnectivity
from .cellmethod import CellMethod
from .cellmeasure import CellMeasure
from .domainancillary import DomainAncillary
from .domainaxis import DomainAxis
from .domaintopology import DomainTopology
from .fieldancillary import FieldAncillary
from .field import Field
from .data import Data
from .data.array import (
    AggregatedArray,
    BoundsFromNodesArray,
    CellConnectivityArray,
    FullArray,
    GatheredArray,
    H5netcdfArray,
    NetCDFArray,
    NetCDF4Array,
    PointTopologyArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
    SubsampledArray,
    UMArray,
    ZarrArray,
)

from .aggregate import aggregate, climatology_cells
from .query import (
    Query,
    lt,
    le,
    gt,
    ge,
    eq,
    ne,
    isclose,
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
from .maths import curl_xy, div_xy, relative_vorticity, histogram
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
