'''The cf package is an Earth science data analysis library that is
built on a complete implementation of the CF data model.

The cf package implements the CF data model for its internal data
structures and so is able to process any CF-compliant dataset. It is
not strict about CF-compliance, however, so that partially conformant
datasets may be ingested from existing datasets and written to new
datasets. This is so that datasets which are partially conformant may
nonetheless be modified in memory.

The cf package can:

* read field constructs from netCDF and UM datasets,

* create new field constructs in memory,

* inspect field constructs,

* test whether two field constructs are the same,

* modify field construct metadata and data,

* create subspaces of field constructs,

* write field constructs to netCDF datasets on disk,

* incorporate, and create, metadata stored in external files,

* read, write, and create data that have been compressed by convention
  (i.e. ragged or gathered arrays), whilst presenting a vi ew of the
  data in its uncompressed form,

* Combine field constructs arithmetically,

* Manipulate field construct data by arithmetical and trigonometrical
  operations,

* Perform statistical collapses on field constructs,

* Regrid field constructs,

* Apply convolution filters to field constructs,

* Calculate derivatives of field constructs,

* Create field constructs to create derived quantities (such as
  vorticity).

All of the above use :ref:`LAMA` functionality, which allows multiple
fields larger than the available memory to exist and be manipulated.


**Visualization**

Powerful, flexible, and very simple to produce visualizations of field
constructs uses the `cfplot` package
(http://ajheaps.github.io/cf-plot), that is automatically installed
along with with cf.

See the cf-python home page (https://ncas-cms.github.io/cf-python) for
documentation, installation and source code.

'''

_requires = ('numpy',
             'netCDF4',
             'cftime',
             'cfunits',
             'cfdm',
             'psutil',
)

_error0 = 'cf requires the modules {}. '.format(', '.join(_requires))

try:
    import cfdm
except ImportError as error1:
    raise ImportError(_error0+str(error1))

__Conventions__  = 'CF-1.7'
__author__       = 'David Hassell'
__date__         = '2019-11-08'
__version__      = '3.0.4'
__cf_version__   = cfdm.core.__cf_version__

from distutils.version import LooseVersion
import importlib
import platform

# Check the version of python
_minimum_vn = '3.5.0'
if LooseVersion(platform.python_version()) < LooseVersion(_minimum_vn):
    raise ValueError(
        "Bad python version: cf requires python version {} or later. Got {}".format(
            _minimum_vn, platform.python_version()))

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
    raise ImportError(_error0+str(error1))

try:
    import numpy
except ImportError as error1:
    raise ImportError(_error0+str(error1))

try:
    import cftime
except ImportError as error1:
    raise ImportError(_error0+str(error1))

try:
    import cfunits
except ImportError as error1:
    raise ImportError(_error0+str(error1))

try:
    import psutil
except ImportError as error1:
    raise ImportError(_error0+str(error1))

# Check the version of netCDF4
_minimum_vn = '1.4.0'
if LooseVersion(netCDF4.__version__) < LooseVersion(_minimum_vn):
    raise ValueError(
        "Bad netCDF4 version: cf requires netCDF4 version {} or later. Got {} at {}".format(
            _minimum_vn, netCDF4.__version__, netCDF4.__file__))

# Check the version of numpy
_minimum_vn = '1.15'
if LooseVersion(numpy.__version__) < LooseVersion(_minimum_vn):
    raise ValueError(
        "Bad numpy version: cf requires numpy version {} or later. Got {} at {}".format(
            _minimum_vn, numpy.__version__, numpy.__file__))

# Check the version of cfunits
_minimum_vn = '3.1.1'
if LooseVersion(cfunits.__version__) < LooseVersion(_minimum_vn):
    raise ValueError(
        "Bad cfunits version: cf requires cfunits version {} or later. Got {} at {}".format(
            _minimum_vn, cfunits.__version__, cfunits.__file__))

from .constructs import Constructs

from .abstract             import Coordinate

from .count                import Count
from .index                import Index
from .list                 import List

from .bounds               import Bounds
from .domain               import Domain
from .datum                import Datum
from .coordinateconversion import CoordinateConversion

from .cfdatetime           import dt, dt_vector
from .flags                import Flags
from .timeduration         import TimeDuration, Y, M, D, h, m, s
from .units                import Units

from .fieldlist            import FieldList

from .dimensioncoordinate  import DimensionCoordinate
from .auxiliarycoordinate  import AuxiliaryCoordinate
from .coordinatereference  import CoordinateReference
from .cellmethod           import CellMethod
from .cellmeasure          import CellMeasure
from .domainancillary      import DomainAncillary
from .domainaxis           import DomainAxis
from .fieldancillary       import FieldAncillary
from .field                import Field
from .data                 import (Data,
                                   FilledArray,
                                   GatheredArray,
                                   NetCDFArray,
                                   RaggedContiguousArray,
                                   RaggedIndexedArray,
                                   RaggedIndexedContiguousArray)

from .aggregate            import aggregate
from .query                import (Query, lt, le, gt, ge, eq, ne, contain, contains,
                                   wi, wo, set, year, month, day, hour, minute,
                                   second, dtlt, dtle, dtgt, dtge, dteq, dtne,
                                   cellsize, cellge, cellgt, cellle, celllt,
                                   cellwi, cellwo, djf, mam, jja, son, seasons)
from .constants            import *
from .functions            import *
from .maths                import relative_vorticity, histogram


from .read_write import (read,
                         write,
                         CFImplementation)

