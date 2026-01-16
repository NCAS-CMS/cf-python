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

import cfdm

from packaging.version import Version


__date__ = "2026-01-16"
__version__ = "3.19.0"
__cf_version__ = cfdm.__cf_version__
__Conventions__ = f"CF-{__cf_version__}"

# Check the version of cfdm (this is worth doing because of the very
# tight coupling between cf and cfdm, and the risk of bad things
# happening at run time if the versions are mismatched).
_minimum_vn = "1.13.0.0"
_maximum_vn = "1.13.1.0"
_cfdm_vn = Version(cfdm.__version__)
if _cfdm_vn < Version(_minimum_vn) or _cfdm_vn >= Version(_maximum_vn):
    raise RuntimeError(
        f"cf v{__version__} requires {_minimum_vn}<=cfdm<{_maximum_vn}. "
        f"Got {_cfdm_vn} at {cfdm.__file__}"
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
