version 3.0.0 (*first Python 3 version*)
----------------------------------------
----

**2019-10-01**

* Complete refactor for Python 3, including some API changes.

  Version 2 scripts should either work as expected, or provide
  informative error mesages on the new API usage. However, it is
  advised that the outputs of older scripts be checked when running
  with Python 3 versions of the cf library.

version 2.3.5 (*last Python 2 version*)
---------------------------------------
----

**2019-04-04**

* Changed calculation of chunksize in parallel case to avoid
  potential problems and introduced a new method cf.SET_PERFORMANCE
  to tune the chunksize and the fraction of memory to keep free.

version 2.3.4
-------------
----

**2019-03-27**

* Fix bug in creating a during cell method during a field
  collapse.
	
version 2.3.3
-------------
----

**2019-03-05**

* Allow failure to compile to go through with a warning, rather
  than failing to install. if this happens, reading a PP/UM file
  will result in "Exception: Can't determine format of file
  test2.pp"
	
* Fixed bug in Field.convolution_filter giving false error over
  units.
	
version 2.3.2
-------------
----

**2018-12-10**

* regridc now compares the units of the source and destination
  grids and converts between them if possible or raises an error
  if they are not equivalent.
	
version 2.3.1
-------------
----

**2018-11-07**

* Fixed bug in regridc that caused it to fail when regridding a
  multidimensional field along only one dimension.
	
* Fixed bug which in which the default logarithm base 10, rather
  than base e
	
version 2.3.0
-------------
----

**2018-10-22**

* The collapse method can now be parallelised by running any
  cf-python script with mpirun if mpi4py is installed. This is an
  experimental feature and is not recommended for operational
  use. None of the parallel code is executed when a script is run in
  serial.
	
version 2.2.8
-------------
----

**2018-08-28**

* Bug fix: better handle subspacing by multiple multidimensional
  items
	
	
Version 2.2.7 - 25 July 2018
----------------------------

* Bug fix: correctly set units of bounds when the Data object
  inserted with insert_bounds has units of ''. In this case the
  bounds of the parent coordinate are now inherited.
	
Version 2.2.6 - 24 July 2018
----------------------------

* Improved error messages

* Changed behaviour when printing reference times with a calendar
  of "none" - no longer attemps a to create a date-time
  representation
	
Version 2.2.5 - 02 July 2018
----------------------------

* Fixed bug with HDF chunk sizes that prevented the writing of large files
	
sion 2.2.4 - 29 June 2018
----------------------------

* Interim fix for with HDF chunk sizes that prevented the writing of large files
	
Version 2.2.3 - 21 June 2018
----------------------------

* During writing, disallow the creation of netCDF variable names
  that contain characters other than letters, digits, and
  underscores.
	
Version 2.2.3 - 21 June 2018
----------------------------

* During writing, disallow the creation of netCDF variable names
  that contain characters other than letters, digits, and
  underscores.
	
Version 2.2.2 - 06 June 2018
----------------------------

* Fix for removing duplicated netCDF dimensions when writing data
  on (e.g.) tripolar grids. 
	
Version 2.2.1 - 05 June 2018
----------------------------

* Fix for calculating are weights from projection coordinates
	
		
Version 2.2.0 - 04 June 2018
----------------------------

* Updated for netCDF4 v1.4 cftime API changes
	
		
Version 2.1.9 - 31 May 2018
---------------------------

* Allowed invalid units through. Can test with cf.Units.isvalid.
	
		
Version 2.1.8 - 08 March 2018
-----------------------------

* Fixed bug when weights parameter is a string in Field.collapse
		
Version 2.1.7 - 13 February 2018
--------------------------------

* Fixed bug in Field.collapse when doing climatological time
  collapse with only one persiod per year/day
		
Version 2.1.6 - 09 February 2018
--------------------------------

* Fixed bug in Variable.mask
		
Version 2.1.4 - 09 February 2018
--------------------------------

* Added override_calendar method to coordinates and domain
  ancillaries that changes the calendar of the bounds, too

* Fixed bug in cf.Data.where when the condition is a cf.Query
  object (f5e0d43)

* Fixed bug in Variable.mask
		
Version 2.1.3 - 07 February 2018
--------------------------------

* Allowed scipy and matplotlib imorts to be optional (23a4347)
	
Version 2.1.2 - 28 November 2017
--------------------------------

* Added group_span and contiguous_group options to Field.collapse
	
VVersion 2.1.1 - 10 November 2017
--------------------------------

* Dissallowed raising offset units to a power (e.g. taking the
  square of data in units of K @ 273.15). 
	
* Removed len() of Field (previously always, and misleadingly,
  returned 1)

* Fixed setting of cell methods after climatological time collapses

* Added printing of ncvar in Field.__str__ and Field.dump

* Added user stash table option to cfa script
	
Version 2.1 - 30 October 2017
-----------------------------

* Misc. bug fixes

Version 2.0.6 - 28 September 2017
---------------------------------

* Removed error when cf.read finds no fields - an empty field list is now returned

* Allowed the count method on a field (it was only on a data object before)

Version 2.0.5 - 19 September 2017
---------------------------------

* Bug fix when creating wrap-around subspaces from cyclic fields

* Fix (partial?) for memory leak when reading UM PP and fields files

Version 2.0.4 - 15 September 2017
---------------------------------

* submodel property for PP files

* API change for cf.Field.axis: now returns a cf.DomainAxis object
	by default

* Bug fix in cf.Field.where

* Bug fix when initializing a field with the source parameter	

* Changed default ouptut format to NETCDF4 (from NETCDF3_CLASSIC)

Vesion 2.0.1.post1 - 12 July 2017
---------------------------------

* Bug fix for reading DSG ragged arrays

Vesion 2.0.1 - 11 July 2017
---------------------------

* Updated cf.FieldList behaviour (with reduced methods)

Vesion 2.0 - 07 July 2017
-------------------------

* First release with full CF data model and full CF-1.6 compliance
  (including DSG)

Version 1.5.4.post4 - 07 July 2017
----------------------------------

* Bug fixes to regridc

Version 1.5.4.post1 - 13 June 2017
----------------------------------

* removed errant scikit import

Version 1.5.4 - 09 June 2017 
----------------------------

* Tripolar regridding
	
Version 1.5.3 - 
-----------------------------

* Updated STASH code to standard_name table (with thanks to Jeff Cole)

* Fixed bug when comparing masked arrays for equality

Version 1.5.2 - 17 March 2017
-----------------------------

* Fixed bug when accessing PP file whose format/endian/word-size
  has been specified

Version 1.5.1 - 14 March 2017
-----------------------------

* Can specify 'pp' or 'PP' in um option to cf.read

Version 1.5 - 24 February 2017
------------------------------

* Changed weights in calculation of variance to reliability
  weights (from frequency weights). This not only scientifically
  better, but faster, too.

Version 1.4 - 22 February 2017
------------------------------

* Rounded datetime to time-since conversions to the nearest
  microsecond, to reflect the accuracy of netCDF4.netcdftime

* Removed import tests from setup.py

* New option --um to cfa, cfdump

* New parameter um to cf.read

Version 1.3.3 - 31 January 2017
-------------------------------

* Rounded datetime to time-since conversions to the nearest
  microsecond, to reflect the accuracy of netCDF4.netcdftime

* Fix for netCDF4.__version__ > 1.2.4 do to with datetime.calendar *handle with care*

Version 1.3.2 - 21 September 2016
---------------------------------

* Added --build-id to LDFLAGS in umread Makefile, for sake of RPM
  builds (otherwise fails when building debuginfo RPM). Pull request
  #16, thanks to Klaus Zimmerman.

* Improved test handling. Pull request #21, thanks to Klaus
  Zimmerman.

* Removed udunits2 database. This removes the modified version of
  the udunits2 database in order to avoid redundancies, possible
  version incompatibilities, and license questions. The
  modifications are instead carried out programmatically in
  units.py. Pull request #20, thanks to Klaus Zimmerman.

Version 1.3.1 - 09 September 2016
---------------------------------

* New method: cf.Field.unlimited, and new 'unlimited' parameter to
  cf.write and cfa

Version 1.3 - 05 September 2016
-------------------------------

* Removed asreftime, asdatetime and dtvarray methods

* New method: convert_reference_time for converting reference time
  data values to have new units.

Version 1.2.3 - 23 August 2016
------------------------------

* Fixed bug in Data.equals

Version 1.2.2 - 22 August 2016
------------------------------

* Fixed bug in binary operations to do with the setting of
  Partition.part

* Added TimeDuration functionality to get_bounds cellsizes
  parameter. Also new parameter flt ("fraction less than") to
  position the coordinate within the cell.

Version 1.2 - 05 July 2016
--------------------------

* Added HDF_chunks methods

Version 1.1.11 - 01 July 2016
-----------------------------

* Added cellsize option to cf.Coordinate.get_bounds, and fixed bugs

* Added variable_attributes option to cf.write
	
* Added cf.ENVIRONMENT method

Version 1.1.10 - 23 June 2016
-----------------------------

* Added reference_datetime option to cf.write	

* Fixed bug in cf.um.read.read which incorrectly ordered vertical
  coordinates
  	
ersion 1.1.9 - 17 June 2016
----------------------------

* New methods cf.Variable.files and cf.Data.files, cf.Field.files
  which report which files are referenced by the data array.

* Fix to stop partitions return numpy.bool_ instead of
  numy.ndarray
	
* Fix to determining cyclicity of regridded fields.

* Functionality to recursively read directories in
  cf.read, cfa and cfump

* Print warning but carry on when ESMF import fails
	
* Fixed bug in cf.Field.subspace when accessing axes derived
  from UM format files
	
Version 1.1.8 - 18 May 2016
---------------------------

* Slightly changed the compression API to cf.write
	
* Added compression support to the cfa command line script

* Added functionality to change data type on writing to cf.write
  and cfa - both in general and for with extra convienience for the
  common case of double to single (and vice versa).

* Removed annoying debug print statements from cf.um.read.read

Version 1.1.7 - 04 May 2016
---------------------------

* Added fix for change in numpy behaviour (numpy.number types do
  not support assingment)
	
* Added capability to load in a user STASH to standard name table:
  cf.um.read.load_stash2standard_name
	
	
Version 1.1.6 - 27 April 2016
-----------------------------

* Added --reference_datetime option to cfa

* Bug fix to cf.Field.collapse when providing cf.Query objects via
  the group parameter

* Added auto regridding method, which is now the default
	
Version 1.1.5 - 03 March 2016
-----------------------------

* Bug fix in cf.Field.where() when using cf.masked
	
* conda installation (with thanks to Andy Heaps)
	
* Bug fix for type casting in cf.Field.collapse

* Dispay long_name if it exists and there is no standard_name
	
* Fix for compiling the UM C code on certiain OSs (with thanks to Simon Wilson)
	
* Fixed incorrect assignment of cyclicity in cf.Field.regrids
	
* Nearest neighbour regridding in cf.Field.regrids
	
Version 1.1.4 - 09 February 2016
--------------------------------

* Bug fix to cf.Field.autocyclic
	
* Bug fix to cf.Field.clip - now works when limit units are supplied
	
* New methods: cf.Data.round, cf.Field.Round

* Added lbtim as a Field property when reading UM files

* Fixed coordinate creation for UM atmosphere_hybrid_height_coordinate

* Bug fix to handling of cyclic fields by cf.Field.regrids

* Added nearest neighbour field regridding

* Changed keyword ignore_dst_mask in regrids to use_dst_mask, which is
  false by default
	
Version 1.1.3 - 10 December 2015
--------------------------------

* Bug fixes to cf.Field.collapse when the "group" parameter is
  used
	
* Correct setting of cyclic axes on regridded fields

* Updates to STASH_to_CF.txt table: 3209, 3210
	
Version 1.1.2 - 01 December 2015
--------------------------------

* Updates to STASH_to_CF.txt table
	
* Fixed bug in decoding UM version in cf.um.read.read
	
* Fixed bug in cf.units.Utime.num2date
	
* Fixed go-slow behaviour for silly BZX, BDX in PP and fields file
  lookup headers
	
Version 1.1.1 - 05 November 2015
--------------------------------

* Fixed bug in decoding UM version in cf.read
	
Version 1.1 - 28 October 2015
-----------------------------

* Fixed bug in cf.Units.conform

* Changed cf.Field.__init__ so that it works with just a data object
	
* Added cf.Field.regrids for lat-lon regridding using ESMF library
	
* Removed support for netCDF4-python versions < 1.1.1
	
* Fixed bug which made certain types of coordinate bounds
  non-contiguous after transpose

* Fixed bug with i=True in cf.Field.where and in
  cf.Field.mask_invalid

* cyclic methods now return a set, rather than a list

* Fixed bug in _write_attributes which might have slowed down some
  writes to netCDF files.

* Reduced annoying redirection in the documentation

* Added cf.Field.field method and added top_level keyword to
  cf.read

* Fixed bug in calculation of standard deviation and
  variance (the bug caused occasional crashes - no incorrect results
  were calculated)

* In items method (and friends), removed strict_axes keyword and
  added axes_all, axes_superset and axes_subset keywords

Version 1.0.3 - 23 June 2015
----------------------------

* Added default keyword to fill_value() and fixed bugs when doing
  delattr on _fillValue and missinge_value properties.

Version 1.0.2 - 05 June 2015
----------------------------

* PyPI release

Version 1.0.1 - 01 June 2015
----------------------------

* Fixed bug in when using the select keyword to cf.read

Version 1.0 - 27 May 2015
-------------------------

* Max OS support

* Limited Nd funtionality to Field.indices

* Correct treatment of add_offset and scale_factor

* Replaced -a with -x in cfa and cfdump scripts

* added ncvar_identities parameter to cf.aggregate

* Performance improvements to field subspacing

* Documentation

* Improved API to match, select, items, axes, etc.

* Reads UM fields files

* Optimised readin PP and UM fields files

* cf.collapse replaced by cf.Field.collapse

* cf.Field.collapse includes CF climatological time statistics

Version 0.9.9.1 - 09 January 2015
---------------------------------

* Fixed bug for changes to netCDF4-python library versions >= 1.1.2

* Miscellaneous bug fixes

Version 0.9.9 - 05 January 2015
-------------------------------

* Added netCDF4 compression options to cf.write.

* Added __mod__, __imod__, __rmod__, ceil, floor, trunc, rint
  methods to cf.Data and cf.Variable

* Added ceil, floor, trunc, rint to cf.Data and cf.Variable

* Fixed bug in which array cf.Data.array sometimes behaved like
  cf.Data.varray

* Fixed bug in cf.netcdf.read.read which affected reading fields
  with formula_terms.

* Refactored the test suite to use the unittest package

* Cyclic axes functionality

* Documentation updates

Version 0.9.8.3 - 14 July 2014
------------------------------

* Implemented multiple grid_mappings (CF trac ticket #70)

* Improved functionality and speed of field aggregation and cfa
  and cfdump command line utilities.

* Collapse methods on cf.Data object (min, max, mean, var, sd,
  sum, range, mid_range).

* Improved match/select functionality

Version 0.9.8.2 - 13 March 2014
-------------------------------

* Copes with PP fields with 365_day calendars

* Revamped CFA files in line with the evolving standard. CFA files
  from PP data created with a previous version will no longer work.

Version 0.9.8.1 -  December 2013
--------------------------------

Version 0.9.8 - 06 December 2013
--------------------------------

* Improved API.

* Plenty of speed and memory optimizations.

* A proper treatment of datetimes.

* WGDOS-packed PP fields are now unpacked on demand.

* Fixed bug in functions.py for numpy v1.7. Fixed bug when deleting
  the 'id' attribute.

* Assign a standard name to aggregated PP fields after aggregation
  rather than before (because some stash codes are too similar,
  e.g. 407 and 408).

* New subclasses of cf.Coordinate: cf.DimensionCoordinate and
  cf.AuxiliaryCoordinate.

* A cf.Units object is now immutable.

Version 0.9.7.1 - 26 April 2013
-------------------------------

* Fixed endian bug in CFA-netCDF files referring to PP files

* Changed default output format to NETCDF3_CLASSIC and trap error when
  when writing unsigned integer types and the 64-bit integer type to
  file formats other than NETCDF4.

* Changed unhelpful history created when aggregating

Version 0.9.7 - 24 April 2013
-----------------------------

* Read and write CFA-netCDF files

* Field creation interface

* New command line utilities: cfa, cfdump

* Redesigned repr, str and dump() output (which is shared with cfa and
  cfdump)

* Removed superceded (by cfa) command line utilities pp2cf, cf2cf

* Renamed the 'subset' method to 'select'

* Now needs netCDF4-python 0.9.7 or later (and numpy 1.6 or later)

Version 0.9.6.2 - 27 March 2013
-------------------------------

* Fixed bug in cf/pp.py which caused the creation of incorrect
  latitude coordinate arrays.

Version 0.9.6.1 - 20 February 2013
----------------------------------

* Fixed bug in cf/netcdf.py which caused a failure when a file with
  badly formatted units was encountered.

Version 0.9.6 - 27 November 2012
--------------------------------

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

Version 0.9.5 - 01 October 2012
-------------------------------

* Restructured documentation and package code files.

* Large Amounts of Massive Arrays (LAMA) functionality.

* Metadata-aware field manipulation and combination with
  metadata-aware broadcasting.

* Better treatment of cell measures.

* Slightly faster aggregation algorithm (a much improved one is in
  development).

* API changes for clarity.

* Bug fixes.

* Added 'TEMPDIR' to the cf.CONSTANTS dictionary

* This is a snapshot of the trunk at revision r195.

Version 0.9.5.dev - 19 September 2012
-------------------------------------

* Loads of exciting improvements - mainly LAMA functionality,
  metadata-aware field manipulation and documentation.

* This is a snapshot of the trunk at revision r185. A proper vn0.9.5
  release is imminent.

Version 0.9.4.2 - 17 April 2012
-------------------------------

* General bug fixes and code restructure

Version 0.9.4 - 15 March 2012
-----------------------------

* A proper treatment of units using the Udunits C library and the
  extra time functionality provided by the netCDF4 package.

* A command line script to do CF-netCDF to CF-netCDF via cf-python.

Version 0.9.3.3 - 08 February 2012
----------------------------------

* Objects renamed in line with the CF data model: 'Space' becomes
  'Field' and 'Grid' becomes 'Space'.

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

Version 0.9.3 - 05 January 2012
-------------------------------

* A more consistent treatment of spaces and lists of spaces (Space and
  SpaceList objects respectively).

* A corrected treatment of scalar or 1-d, size 1 dimensions in the
  space and its grid.

* Data stored in Data objects which contain metadata need to correctly
  interpret and manipulate the data. This will be particularly useful
  when data arrays spanning many files/arrays is implemented

Version 0.9.2 - 26 August 2011
-------------------------------

* Created a setup.py script for easier installation (with thanks to
  Jeff Whitaker).

* Added support for reading OPeNDAP-hosted datasets given by URLs.

* Restructured the documentation.

* Created a test directory with scripts and sample output.

* No longer fails for unknown calendar types (such as '360d').

Version 0.9.1 - 06 August 2011
------------------------------

* First release.
