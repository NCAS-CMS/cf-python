'''The Python `cf` package is an Earth Science data analysis library
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

* write field constructs to netCDF datasets on disk,

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

'''
__Conventions__ = 'CF-1.8'
__author__ = 'David Hassell'
__date__ = '2020-06-10'
__version__ = '3.5.1'

_requires = (
    'numpy',
    'netCDF4',
    'cftime',
    'cfunits',
    'cfdm',
    'psutil',
)

_error0 = "cf v{} requires the modules {}. ".format(
    __version__, ', '.join(_requires))

try:
    import cfdm
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

__cf_version__ = cfdm.core.__cf_version__

from distutils.version import LooseVersion
import importlib
import platform

# Check the version of python
_minimum_vn = '3.5.0'
if LooseVersion(platform.python_version()) < LooseVersion(_minimum_vn):
    raise ValueError(
        "Bad python version: cf requires python version {} or later. "
        "Got {}".format(_minimum_vn, platform.python_version())
    )

_found_ESMF = bool(importlib.util.find_spec('ESMF'))

if importlib.util.find_spec('mpi4py'):
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    if mpi_size > 1:
        mpi_on = True
        if mpi_rank == 0:
            print('===============================================')
            print('WARNING: MPI support is an experimental feature')
            print('  and is not recommended for operational use.')
            print('===============================================')
    else:
        mpi_on = False
else:
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
_minimum_vn = '0.6.0'
if LooseVersion(psutil.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        "Bad psutil version: cf requires psutil>={}. "
        "Got {} at {}".format(
            _minimum_vn, psutil.__version__, psutil.__file__))

# Check the version of netCDF4
_minimum_vn = '1.5.3'
if LooseVersion(netCDF4.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        "Bad netCDF4 version: cf requires netCDF4>={}. Got {} "
        "at {}".format(_minimum_vn, netCDF4.__version__, netCDF4.__file__)
    )

# Check the version of cftime
_minimum_vn = '1.1.3'
if LooseVersion(cftime.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        "Bad cftime version: cf requires cftime>={}. "
        "Got {} at {}".format(
            _minimum_vn, cftime.__version__, cftime.__file__))

# Check the version of numpy
_minimum_vn = '1.15'
if LooseVersion(numpy.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        "Bad numpy version: cf requires numpy>={}. Got {} "
        "at {}".format(_minimum_vn, numpy.__version__, numpy.__file__)
    )

# Check the version of cfunits
_minimum_vn = '3.2.6'
if LooseVersion(cfunits.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        "Bad cfunits version: cf requires cfunits>={}. Got {} "
        "at {}".format(_minimum_vn, cfunits.__version__, cfunits.__file__)
    )

# Check the version of cfdm
_minimum_vn = '1.8.5'
_maximum_vn = '1.9'
_cfdm_version = LooseVersion(cfdm.__version__)
# if (_cfdm_version < LooseVersion(_minimum_vn)
#     or _cfdm_version >= LooseVersion(_maximum_vn)):
if not (LooseVersion(_minimum_vn) <= _cfdm_version < LooseVersion(_maximum_vn)):
    raise RuntimeError(
        "Bad cfdm version: cf requires {}<=cfdm<{}. Got {} "
        "at {}".format(_minimum_vn, _maximum_vn,
                       _cfdm_version, cfdm.__file__))

from .constructs import Constructs

# from .abstract import Coordinate
from .mixin import Coordinate

from .count                   import Count
from .index                   import Index
from .list                    import List
from .nodecountproperties     import NodeCountProperties
from .partnodecountproperties import PartNodeCountProperties
from .interiorring            import InteriorRing

from .bounds               import Bounds
from .domain               import Domain
from .datum                import Datum
from .coordinateconversion import CoordinateConversion

from .cfdatetime   import dt, dt_vector
from .flags        import Flags
from .timeduration import TimeDuration, Y, M, D, h, m, s
from .units        import Units

from .fieldlist import FieldList

from .dimensioncoordinate import DimensionCoordinate
from .auxiliarycoordinate import AuxiliaryCoordinate
from .coordinatereference import CoordinateReference
from .cellmethod          import CellMethod
from .cellmeasure         import CellMeasure
from .domainancillary     import DomainAncillary
from .domainaxis          import DomainAxis
from .fieldancillary      import FieldAncillary
from .field               import Field
from .data                import (Data,
                                  FilledArray,
                                  GatheredArray,
                                  NetCDFArray,
                                  RaggedContiguousArray,
                                  RaggedIndexedArray,
                                  RaggedIndexedContiguousArray)

from .aggregate    import aggregate
from .query        import (Query, lt, le, gt, ge, eq, ne, contain, contains,
                           wi, wo, set, year, month, day, hour,
                           minute, second, dtlt, dtle, dtgt, dtge,
                           dteq, dtne, cellsize, cellge, cellgt,
                           cellle, celllt, cellwi, cellwo, djf, mam,
                           jja, son, seasons)
from .constants    import *
from .functions    import *
from .maths        import relative_vorticity, histogram
from .examplefield import example_field


from .cfimplementation import (CFImplementation,
                               implementation)

from .read_write import (read,
                         write)


# Set up basic logging for the full project with a root logger
import logging
import sys

# Configure the root logger which all module loggers inherit from:
logging.basicConfig(
    stream=sys.stdout,
    style='{',              # default is old style ('%') string formatting
    format='{message}',     # no module names or datetimes etc. for basic case
    level=logging.WARNING,  # default but change level via LOG_LEVEL()
)

# And create custom level inbetween 'INFO' & 'DEBUG', to understand value see:
# https://docs.python.org/3.8/howto/logging.html#logging-levels
logging.DETAIL = 15  # set value as an attribute as done for built-in levels
logging.addLevelName(logging.DETAIL, 'DETAIL')


def detail(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.DETAIL):
        self._log(logging.DETAIL, message, args, **kwargs)


logging.Logger.detail = detail
