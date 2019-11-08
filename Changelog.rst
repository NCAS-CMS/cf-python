version 3.0.4
-------------
----

**2019-11-08**

* New methods: `cf.Field.percentile`, `cf.Field.example_field`,
  `cf.Field.creation_commands`.
* New field construct collapse methods: ``median``,
  ``mean_of_upper_decile``.
* New method: `cf.FieldList.select_field`.
* New methods: `cf.Data.median`, `cf.Data.mean_of_upper_decile`,
  `cf.Data.percentile`, `cf.Data.filled`, `cf.Data.creation_commands`.
* New keyword parameter to `cf.Data`: ``dtype``.
* Changed default ``ddof`` *back* to 1 in `cf.Data.var` and
  `cf.Data.sd` (see version 3.0.3 and
  https://github.com/NCAS-CMS/cf-python/issues/8)
* Fixed bug that sometimes caused an exception to be raised when
  metadata constructs were selected by a property value that
  legitimately contained a colon.
* Changed minimum cfdm dependency to version 1.7.9.

version 3.0.3
-------------
----

**2019-11-01**

* Fixed bug (introduced at v3.0.2) that caused ``mean_absolute_value``
  collapses by `cf.Field.collapse` to be not weighted when they should
  be (https://github.com/NCAS-CMS/cf-python/issues/9)
* Changed default ``ddof`` from 0 to 1 in `cf.Data.var` and
  `cf.Data.sd` (https://github.com/NCAS-CMS/cf-python/issues/8)
   
version 3.0.2
-------------
----

**2019-10-31**

* Now reads CDL files (https://github.com/NCAS-CMS/cf-python/issues/1)
* New methods: `cf.Field.cumsum`, `cf.Field.digitize`, `cf.Field.bin`,
  `cf.Field.swapaxes`, `cf.Field.flatten`, `cf.Field.radius`.
* New function: `cf.histogram`.
* New field construct collapse methods: ``integral``,
  ``mean_absolute_value``, ``maximum_absolute_value``,
  ``minimum_absolute_value``, ``sum_of_squares``,
  ``root_mean_square``.
* New keyword parameters to `cf.Field.collapse` and
  `cf.Field.weights`: ``measure``, ``scale``, ``radius``
* New methods: `cf.Data.cumsum`, `cf.Data.digitize`,
  `cf.Data.masked_all`, `cf.Data.mean_absolute_value`,
  `cf.Data.maximum_absolute_value`, `cf.Data.minimum_absolute_value`,
  `cf.Data.sum_of_squares`, `cf.Data.root_mean_square`,
  `cf.Data.flatten`.
* Renamed `cf.default_fillvals` to `cf.default_netCDF_fillvals`.
* Changed minimum cfdm dependency to version 1.7.8. This fixes a bug
  that sometimes occurs when writing to disk and the _FillValue and
  data have different data types.
* Changed minimum cfunits dependency to version 3.2.2.
* Changed minimum cftime dependency to version 1.0.4.2.
* Fixed occasional failure to delete all temporary directories at
  exit.
* Fixed bug in `cf.Data.func` when overriding units. Affects all
  methods that call `cf.Data.func`, such as `cf.Data.tan` and
  `cf.Field.tan`.
* Fixed "relaxed units" behaviour in `cf.aggregate` and field
  construct arithmetic.
* Fixed bug that led to incorrect persistent entries in output of
  `cf.Field.properties`.
* Fixed bug in `cf.Data.squeeze` that sometimes created
  inconsistencies with the cyclic dimensions.
* Fixed bug in `cf.Field.mask` that assigned incorrect units to the
  result.

version 3.0.1
-------------
----

**2019-10-01**

* Updated description in ``setup.py``

version 3.0.0 (*first Python 3 version*)
----------------------------------------
----

**2019-10-01**

* Complete refactor for Python 3, including some API changes.

  Scripts written for version 2.x but running under version 3.x should
  either work as expected, or provide informative error messages on
  the new API usage. However, it is advised that the outputs of older
  scripts be checked when running with Python 3 versions of the cf
  library.

* Deprecated ``cfdump`` (its functionality is now included in
  ``cfa``).
  

version 2.3.8 (*latest Python 2 version*)
-----------------------------------------
----

**2019-10-07**

* In `cf.write`, can set ``single=False`` to mean ``double=True``, and
  vice versa.
* Fixed bug in `cf.aggregate` - removed overly strict test on
  dimension coordinate bounds.
* Fixed bug in `cf.read` that set the climatology attribute to True
  when there are no bounds.
* Fixed bug in `cf.write` when writing missing values (set_fill was
  off, now on)

version 2.3.5
-------------
----

**2019-04-04**

* Changed calculation of chunksize in parallel case to avoid potential
  problems and introduced a new method `cf.SET_PERFORMANCE` to tune
  the chunksize and the fraction of memory to keep free.

version 2.3.4
-------------
----

**2019-03-27**

* Fix bug in creating a during cell method during a field collapse.
	
version 2.3.3
-------------
----

**2019-03-05**

* Allow failure to compile to go through with a warning, rather than
  failing to install. if this happens, reading a PP/UM file will
  result in "Exception: Can't determine format of file test2.pp"
* Fixed bug in `cf.Field.convolution_filter` giving false error over
  units.
	
version 2.3.2
-------------
----

**2018-12-10**

* `cf.Field.regridc` now compares the units of the source and
  destination grids and converts between them if possible or raises an
  error if they are not equivalent.
	
version 2.3.1
-------------
----

**2018-11-07**

* Fixed bug in `cf.Field.regridc` that caused it to fail when
  regridding a multidimensional field along only one dimension.
* Fixed bug which in which the default logarithm is base 10, rather
  than base e
	
version 2.3.0
-------------
----

**2018-10-22**

* The collapse method can now be parallelised by running any cf-python
  script with mpirun if mpi4py is installed. This is an experimental
  feature and is not recommended for operational use. None of the
  parallel code is executed when a script is run in serial.
	
version 2.2.8
-------------
----

**2018-08-28**

* Bug fix: better handle subspacing by multiple multidimensional items
	
version 2.2.7
-------------
----

**2018-07-25**

* Bug fix: correctly set units of bounds when the `cf.Data` object
  inserted with insert_bounds has units of ''. In this case the bounds
  of the parent coordinate are now inherited.
	
version 2.2.6
-------------
----

**2018-07-24**

* Improved error messages
* Changed behaviour when printing reference times with a calendar of
  ``'none'`` - no longer attempts a to create a date-time
  representation
	
version 2.2.5
-------------
----

**2018-7-02**

* Fixed bug with HDF chunk sizes that prevented the writing of large
  files
	
version 2.2.4
-------------
----

**2018-06-29**

* Interim fix for with HDF chunk sizes that prevented the writing of
  large files
	
version 2.2.3
--------------
----

**2018-06-21**

* During writing, disallow the creation of netCDF variable names that
  contain characters other than letters, digits, and underscores.
	
version 2.2.3
-------------
----

**2018-06-21**

* During writing, disallow the creation of netCDF variable names that
  contain characters other than letters, digits, and underscores.
	
version 2.2.2
-------------
----

**2018-06-06**


* Fix for removing duplicated netCDF dimensions when writing data on
  (e.g.) tripolar grids.
	
version 2.2.1
-------------
----

**2018-06-05**

* Fix for calculating are weights from projection coordinates
			
version 2.2.0
-------------
----

**2018-06-04**

* Updated for `netCDF4` v1.4 `cftime` API changes
	
version 2.1.9
-------------
----

**2018-05-31**

* Allowed invalid units through. Can test with `cf.Units.isvalid`.
	
version 2.1.8
-------------
----

**2018-03-08**

* Fixed bug when weights parameter is a string in `cf.Field.collapse`
		
version 2.1.7
-------------
----

**2018-02-13**

* Fixed bug in `cf.Field.collapse` when doing climatological time
  collapse with only one period per year/day
		
version 2.1.6
-------------
----

**2018-02-09**

* Fixed bug in Variable.mask
		
version 2.1.4
-------------
----

**2018-02-09**

* Added override_calendar method to coordinates and domain ancillaries
  that changes the calendar of the bounds, too.
* Fixed bug in `cf.Data.where` when the condition is a `cf.Query`
  object.
* Fixed bug in `cf.Variable.mask`
		
version 2.1.3
-------------
----

**2018-02-07**

* Allowed `scipy` and `matplotlib` imports to be optional
	
version 2.1.2
-------------
----

**2017-11-28**

* Added ``group_span`` and ``contiguous_group`` options to
  `cf.Field.collapse`
	
version 2.1.1
-------------
----

**2017-11-10**

* Disallowed raising offset units to a power (e.g. taking the square
  of data in units of K @ 273.15).
* Removed len() of `cf.Field` (previously always, and misleadingly,
  returned 1)
* Fixed setting of cell methods after climatological time collapses
* Added printing of ncvar in `cf.Field.__str__` and `cf.Field.dump`
* Added user stash table option to ``cfa`` script
	
version 2.1
-----------
----

**2017-10-30**

* Misc. bug fixes

version 2.0.6
-------------
----

**2017-09-28**

* Removed error when `cf.read` finds no fields - an empty field list
  is now returned
* New method `cf.Field.count`

version 2.0.5
-------------
----

**2017-09-19**

* Bug fix when creating wrap-around subspaces from cyclic fields
* Fix (partial?) for memory leak when reading UM PP and fields files

version 2.0.4
-------------
----

**2017-09-15**

* submodel property for PP files
* API change for `cf.Field.axis`: now returns a `cf.DomainAxis` object
  by default
* Bug fix in `cf.Field.where`
* Bug fix when initializing a field with the source parameter
* Changed default output format to NETCDF4 (from NETCDF3_CLASSIC)

version 2.0.1.post1
-------------------
----

**2017-07-12**

* Bug fix for reading DSG ragged arrays

version 2.0.1
-------------
----

**2017-07-11**

* Updated `cf.FieldList` behaviour (with reduced methods)

version 2.0
-----------
----

**2017-07-07**

* First release with full CF data model and full CF-1.6 compliance
  (including DSG)

version 1.5.4.post4
-------------------
----

**2017-07-07**

* Bug fixes to `cf.Field.regridc`

version 1.5.4.post1
-------------------
----

**2017-06-13**

* removed errant scikit import

version 1.5.4
-------------
----

**2017-06-09**

* Tripolar regridding
	
version 1.5.3 
-------------
----

**2017-05-10**

* Updated STASH code to standard_name table (with thanks to Jeff Cole)
* Fixed bug when comparing masked arrays for equality

version 1.5.2 
-------------
----

**2017-03-17**

* Fixed bug when accessing PP file whose format/endian/word-size has
  been specified

version 1.5.1 
-------------
----

**2017-03-14**

* Can specify 'pp' or 'PP' in um option to `cf.read`

version 1.5
-----------
----

**2017-02-24**

* Changed weights in calculation of variance to reliability weights
  (from frequency weights). This not only scientifically better, but
  faster, too.

version 1.4
-----------
----

**2017-02-22**

* Rounded datetime to time-since conversions to the nearest
  microsecond, to reflect the accuracy of netCDF4.netcdftime
* Removed import tests from setup.py
* New option --um to ``cfa``, ``cfdump``
* New parameter um to `cf.read`

version 1.3.3
-------------
----

**2017-01-31**

* Rounded datetime to time-since conversions to the nearest
  microsecond, to reflect the accuracy of netCDF4.netcdftime
* Fix for netCDF4.__version__ > 1.2.4 do to with datetime.calendar
  *handle with care*

version 1.3.2
-------------
----

**2016-09-21**

* Added --build-id to LDFLAGS in umread Makefile, for sake of RPM
  builds (otherwise fails when building debuginfo RPM). Pull request
  #16, thanks to Klaus Zimmerman.
* Improved test handling. Pull request #21, thanks to Klaus
  Zimmerman.
* Removed udunits2 database. This removes the modified version of the
  udunits2 database in order to avoid redundancies, possible version
  incompatibilities, and license questions. The modifications are
  instead carried out programmatically in units.py. Pull request #20,
  thanks to Klaus Zimmerman.

version 1.3.1
-------------
----

**2016-09-09**

* New method: `cf.Field.unlimited`, and new 'unlimited' parameter to
  `cf.write` and ``cfa``

version 1.3
-----------
----

**2016-09-05**

* Removed asreftime, asdatetime and dtvarray methods
* New method: `convert_reference_time` for converting reference time
  data values to have new units.

version 1.2.3
-------------
----

**2016-08-23**

* Fixed bug in `cf.Data.equals`

version 1.2.2
-------------
----

**2016-08-22**

* Fixed bug in binary operations to do with the setting of
  `Partition.part`
* Added `cf.TimeDuration` functionality to get_bounds cellsizes
  parameter. Also new parameter flt ("fraction less than") to position
  the coordinate within the cell.

version 1.2
-----------
----

**2016-07-05**

* Added HDF_chunks methods

version 1.1.11
--------------
----

**2016-07-01**

* Added cellsize option to `cf.Coordinate.get_bounds`, and fixed bugs.
* Added variable_attributes option to `cf.write`
* Added `cf.ENVIRONMENT` method

version 1.1.10
--------------
----

**2016-06-23**

* Added reference_datetime option to cf.write	
* Fixed bug in `cf.um.read.read` which incorrectly ordered vertical
  coordinates
  	
version 1.1.9
-------------
----

**2016-06-17**

* New methods `cf.Variable.files` and `cf.Data.files`,
  `cf.Field.files` which report which files are referenced by the data
  array.
* Fix to stop partitions return `numpy.bool_` instead of
  `numpy.ndarray`
* Fix to determining cyclicity of regridded fields.
* Functionality to recursively read directories in `cf.read`, ``cfa``
  and ``cfump``
* Print warning but carry on when ESMF import fails
* Fixed bug in `cf.Field.subspace` when accessing axes derived from UM
  format files
	
version 1.1.8
-------------
----

**2016-05-18**

* Slightly changed the compression API to `cf.write`
* Added compression support to the ``cfa`` command line script
* Added functionality to change data type on writing to `cf.write` and
  ``cfa`` - both in general and for with extra convenience for the
  common case of double to single (and vice versa).
* Removed annoying debug print statements from `cf.um.read.read`

version 1.1.7
-------------
----

**2016-05-04**

* Added fix for change in numpy behaviour (`numpy.number` types do not
  support assignment)
* Added capability to load in a user STASH to standard name table:
  `cf.um.read.load_stash2standard_name`
	
	
version 1.1.6
-------------
----

**2016-04-27**

* Added --reference_datetime option to ``cfa``
* Bug fix to `cf.Field.collapse` when providing `cf.Query` objects via
  the group parameter
* Added auto regridding method, which is now the default
	
version 1.1.5 
-------------
----

**2016-03-03**

* Bug fix in `cf.Field.where` when using `cf.masked`
* conda installation (with thanks to Andy Heaps)
* Bug fix for type casting in `cf.Field.collapse`
* Display long_name if it exists and there is no standard_name
* Fix for compiling the UM C code on certain OSs (with thanks to Simon Wilson)
* Fixed incorrect assignment of cyclicity in `cf.Field.regrids`
* Nearest neighbour regridding in `cf.Field.regrids`
	
version 1.1.4 
-------------
----

**2016-02-09**

* Bug fix to `cf.Field.autocyclic`
* Bug fix to `cf.Field.clip` - now works when limit units are supplied
* New methods: `cf.Data.round`, `cf.Field.Round`
* Added ``lbtim`` as a `cf.Field` property when reading UM files
* Fixed coordinate creation for UM atmosphere_hybrid_height_coordinate
* Bug fix to handling of cyclic fields by `cf.Field.regrids`
* Added nearest neighbour field regridding
* Changed keyword ignore_dst_mask in `cf.Field.regrids` to
  use_dst_mask, which is false by default
	
version 1.1.3 
-------------
----

**2015-12-10**

* Bug fixes to `cf.Field.collapse` when the "group" parameter is used
* Correct setting of cyclic axes on regridded fields
* Updates to STASH_to_CF.txt table: 3209, 3210
	
version 1.1.2 
-------------
----

**2015-12-01**

* Updates to STASH_to_CF.txt table
* Fixed bug in decoding UM version in `cf.um.read.read`
* Fixed bug in `cf.units.Utime.num2date`
* Fixed go-slow behaviour for silly BZX, BDX in PP and fields file
  lookup headers

version 1.1.1
-------------
----

**2015-11-05**

* Fixed bug in decoding UM version in `cf.read`
	
version 1.1
-----------
----

**2015-10-28**

* Fixed bug in `cf.Units.conform`
* Changed `cf.Field.__init__` so that it works with just a data object
* Added `cf.Field.regrids` for lat-lon regridding using ESMF library
* Removed support for netCDF4-python versions < 1.1.1
* Fixed bug which made certain types of coordinate bounds
  non-contiguous after transpose
* Fixed bug with i=True in `cf.Field.where` and in
  `cf.Field.mask_invalid`
* cyclic methods now return a set, rather than a list
* Fixed bug in _write_attributes which might have slowed down some
  writes to netCDF files.
* Reduced annoying redirection in the documentation
* Added `cf.Field.field` method and added top_level keyword to
  `cf.read`
* Fixed bug in calculation of standard deviation and variance (the bug
  caused occasional crashes - no incorrect results were calculated)
* In items method (and friends), removed strict_axes keyword and added
  axes_all, axes_superset and axes_subset keywords

version 1.0.3
-------------
----

**2015-06-23**

* Added default keyword to fill_value() and fixed bugs when doing
  delattr on _fillValue and missing_value properties.

version 1.0.2 - 05 June 2015
----------------------------

* PyPI release

version 1.0.1
-------------
----

**2015-06-01**

* Fixed bug in when using the select keyword to `cf.read`

version 1.0
-----------
----

**2015-05-27**

* Mac OS support
* Limited Nd functionality to `cf.Field.indices`
* Correct treatment of add_offset and scale_factor
* Replaced -a with -x in ``cfa`` and ``cfdump`` scripts
* added ncvar_identities parameter to `cf.aggregate`
* Performance improvements to field subspacing
* Documentation
* Improved API to match, select, items, axes, etc.
* Reads UM fields files
* Optimised reading PP and UM fields files
* `cf.collapse` replaced by `cf.Field.collapse`
* `cf.Field.collapse` includes CF climatological time statistics

version 0.9.9.1
---------------
----

**2015-01-09**

* Fixed bug for changes to netCDF4-python library versions >= 1.1.2
* Miscellaneous bug fixes

version 0.9.9
-------------
----

**2015-01-05**

* Added netCDF4 compression options to `cf.write`.
* Added `__mod__`, `__imod__`, `__rmod__`, `ceil`, `floor`, `trunc`,
  `rint` methods to `cf.Data` and `cf.Variable`
* Added ceil, floor, trunc, rint to `cf.Data` and `cf.Variable`
* Fixed bug in which array `cf.Data.array` sometimes behaved like
  `cf.Data.varray`
* Fixed bug in `cf.netcdf.read.read` which affected reading fields
  with formula_terms.
* Refactored the test suite to use the unittest package
* Cyclic axes functionality
* Documentation updates

version 0.9.8.3
---------------
----

**2014-07-14**

* Implemented multiple grid_mappings (CF trac ticket #70)
* Improved functionality and speed of field aggregation and ``cfa``
  and ``cfdump`` command line utilities.
* Collapse methods on `cf.Data` object (min, max, mean, var, sd,
  sum, range, mid_range).
* Improved match/select functionality

version 0.9.8.2
---------------
----

**2014-03-13**

* Copes with PP fields with 365_day calendars
* Revamped CFA files in line with the evolving standard. CFA files
  from PP data created with a previous version will no longer work.

version 0.9.8
-------------
----

**2013-12-06**

* Improved API.
* Plenty of speed and memory optimizations.
* A proper treatment of datetimes.
* WGDOS-packed PP fields are now unpacked on demand.
* Fixed bug in functions.py for numpy v1.7. Fixed bug when deleting
  the 'id' attribute.
* Assign a standard name to aggregated PP fields after aggregation
  rather than before (because some stash codes are too similar,
  e.g. 407 and 408).
* New subclasses of `cf.Coordinate`: `cf.DimensionCoordinate` and
  `cf.AuxiliaryCoordinate`.
* A `cf.Units` object is now immutable.

version 0.9.7.1
---------------
----

**2013-04-26**

* Fixed endian bug in CFA-netCDF files referring to PP files
* Changed default output format to NETCDF3_CLASSIC and trap error when
  when writing unsigned integer types and the 64-bit integer type to
  file formats other than NETCDF4.
* Changed unhelpful history created when aggregating

version 0.9.7
-------------
----

**2013-04-24**

* Read and write CFA-netCDF files
* `cf.Field` creation interface
* New command line utilities: ``cfa``, ``cfdump``
* Redesigned repr, str and dump() output (which is shared with ``cfa``
  and ``cfdump``)
* Removed superseded (by ``cfa``) command line utilities ``pp2cf``,
  ``cf2cf``
* Renamed the 'subset' method to 'select'
* Now needs netCDF4-python 0.9.7 or later (and numpy 1.6 or later)

version 0.9.6.2
---------------
----

**2013-03-27**

* Fixed bug in ``cf/pp.py`` which caused the creation of incorrect
  latitude coordinate arrays.

version 0.9.6.1
---------------
----

**2013-02-20**

* Fixed bug in ``cf/netcdf.py`` which caused a failure when a file
  with badly formatted units was encountered.

version 0.9.6
-------------
----

**2012-11-27**

* Assignment to a field's data array with metadata-aware broadcasting,
  assigning to subspaces, assignment where data meets conditions,
  assignment to unmasked elements, etc. (setitem method)
* Proper treatment of the missing data mask, including metadata-aware
  assignment (setmask method)
* Proper treatment of ancillary data.
* Ancillary data and transforms are subspaced with their parent field.
* Much faster aggregation algorithm (with thanks to Jonathan
  Gregory). Also aggregates fields transforms, ancillary variables and
  flags.

version 0.9.5
-------------
----

**2012-10-01**

* Restructured documentation and package code files.
* Large Amounts of Massive Arrays (LAMA) functionality.
* Metadata-aware field manipulation and combination with
  metadata-aware broadcasting.
* Better treatment of cell measures.
* Slightly faster aggregation algorithm (a much improved one is in
  development).
* API changes for clarity.
* Bug fixes.
* Added 'TEMPDIR' to the `cf.CONSTANTS` dictionary
* This is a snapshot of the trunk at revision r195.

version 0.9.5.dev
-----------------
----

**2012-09-19**

* Loads of exciting improvements - mainly LAMA functionality,
  metadata-aware field manipulation and documentation.
* This is a snapshot of the trunk at revision r185. A proper vn0.9.5
  release is imminent.

version 0.9.4.2
---------------
----

**2012-04-17**

* General bug fixes and code restructure

version 0.9.4
-------------
----

**2012-03-15**

* A proper treatment of units using the Udunits C library and the
  extra time functionality provided by the netCDF4 package.
* A command line script to do CF-netCDF to CF-netCDF via cf-python.

version 0.9.3.3
---------------
----

**2018-02-08**

* Objects renamed in line with the CF data model: `cf.Space` becomes
  `cf.Field` and `cf.Grid` becomes `cf.Space`.
* Field aggregation using the CF aggregation rules is available when
  reading fields from disk and on fields in memory. The data of a
  field resulting from aggregation are stored as a collection of the
  data from the component fields and so, as before, may be file
  pointers, arrays in memory or a mixture of these two forms.
* Units, missing data flags, dimension order, dimension direction and
  packing flags may all be different between data components and are
  conformed at the time of data access.
* Files in UK Met Office PP format may now be read into CF fields.
* A command line script for PP to CF-netCDF file conversion is
  provided.

version 0.9.3
-------------
----

**2012-01-05**

* A more consistent treatment of spaces and lists of spaces
  (`cf.Space` and `cf.SpaceList` objects respectively).
* A corrected treatment of scalar or 1-d, size 1 dimensions in the
  space and its grid.
* Data stored in `cf.Data` objects which contain metadata need to
  correctly interpret and manipulate the data. This will be
  particularly useful when data arrays spanning many files/arrays is
  implemented.

version 0.9.2
-------------
----

**2011-08-26**

* Created a ``setup.py`` script for easier installation (with thanks
  to Jeff Whitaker).
* Added support for reading OPeNDAP-hosted datasets given by URLs.
* Restructured the documentation.
* Created a test directory with scripts and sample output.
* No longer fails for unknown calendar types (such as ``'360d'``).

version 0.9.1
-------------
----

**2011-08-06**

* First release.
