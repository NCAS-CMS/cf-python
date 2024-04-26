version 3.16.2
--------------

**2024-04-26**

* Improve the performance of reading and accessing the data of PP and
  UM fields files (https://github.com/NCAS-CMS/cf-python/issues/746)
* Improve `cf.Field.collapse` performance by lazily computing reduced
  axis coordinates (https://github.com/NCAS-CMS/cf-python/issues/741)
* Improve `cf.Field.__getitem__` performance by not re-calculating
  axis cyclicity (https://github.com/NCAS-CMS/cf-python/issues/744)
* Reduce output CFA netCDF file size by setting the HDF5 chunksizes of
  CFA variables to be no larger than required
  (https://github.com/NCAS-CMS/cf-python/issues/739)
* Allow a halo to be added by `cf.Field.indices` and
  `cf.Field.subspace`
  (https://github.com/NCAS-CMS/cf-python/issues/759)
* Added spherical regridding to discrete sampling geometry destination
  grids (https://github.com/NCAS-CMS/cf-python/issues/716)
* Added 3-d spherical regridding to `cf.Field.regrids`, and the option
  to regrid the vertical axis in logarithmic coordinates to
  `cf.Field.regrids` and `cf.Field.regridc`
  (https://github.com/NCAS-CMS/cf-python/issues/715)
* New keyword parameter to `cf.Field.regrids` and `cf.Field.regridc`:
  ``return_esmpy_regrid_operator``
  (https://github.com/NCAS-CMS/cf-python/issues/766)
* New keyword parameters to `cf.wi`: ``open_lower`` and ``open_upper``
  (https://github.com/NCAS-CMS/cf-python/issues/740)
* Fix misleading error message when it is not possible to create area
  weights requested from `cf.Field.collapse`
  (https://github.com/NCAS-CMS/cf-python/issues/731)
* Fix bug in `cf.read` when reading UM files that caused LBPROC value
  131072 (Mean over an ensemble of parallel runs) to be ignored
  (https://github.com/NCAS-CMS/cf-python/issues/737)
* Fix bug in `cf.aggregate` that sometimes put a null transpose
  operation into the Dask graph when one was not needed
  (https://github.com/NCAS-CMS/cf-python/issues/754)
* Fix bug in `cf.aggregate` that caused a failure when property values
  were `numpy` arrays with two or more elements
  (https://github.com/NCAS-CMS/cf-python/issues/764)
* Fix bug in `cf.aggregate` that didn't correctly handle the
  "actual_range" CF attribute
  (https://github.com/NCAS-CMS/cf-python/issues/764)
* Fix bug whereby `Field.cyclic` is not updated after a
  `Field.del_construct` operation
  (https://github.com/NCAS-CMS/cf-python/issues/758)
* Fix bug that meant `cyclic()` always returned an empty
  set for domains produced by `cf.Field.domain`
  (https://github.com/NCAS-CMS/cf-python/issues/762)
* Changed dependency: ``cfunits>=3.3.7``
* Changed dependency: ``netCDF4>=1.6.5``

----

version 3.16.1
--------------

**2024-03-01**

* Allow DSG trajectories with identical `trajectory_id` values to be
  aggregated (https://github.com/NCAS-CMS/cf-python/issues/723)
* New methods: `cf.Field.pad_missing` and `cf.Data.pad_missing`
  (https://github.com/NCAS-CMS/cf-python/issues/717)
* Fix occasional bug when calculating UGRID cell areas when
  non-spatial coordinates span the discrete axis
  (https://github.com/NCAS-CMS/cf-python/issues/721)
* New keyword parameter to `cf.Field.insert_dimension`:
  ``constructs`` (https://github.com/NCAS-CMS/cf-python/issues/719)
* Added the ``cell_measures`` and ``coordinates`` keyword arguments to
  `cf.Field.weights`
  (https://github.com/NCAS-CMS/cf-python/issues/709)
* Added the ``cell_measures``, ``coordinates``,
  ``return_cell_measure``, and ``methods`` keyword arguments to
  `cf.Field.cell_area`
  (https://github.com/NCAS-CMS/cf-python/issues/709)
* Allow `cf.Data` to be initialised with `xarray.DataAarray`
  (https://github.com/NCAS-CMS/cf-python/issues/706)
* Fix bug that caused `cf.Field.del_file_location` to fail when
  updating its metadata constructs
  (https://github.com/NCAS-CMS/cf-python/issues/707)
* Fix bug that caused incorrect data arrays in some cyclic subspaces
  created by `cf.Field.subspace` and `cf.Field.__getitem__`
  (https://github.com/NCAS-CMS/cf-python/issues/713)
* Changed dependency: ``1.11.1.0<=cfdm<1.11.2.0``

----

version 3.16.0
--------------

**2023-12-06**

* Implemented the reading and manipulation of UGRID mesh topologies
  (https://github.com/NCAS-CMS/cf-python/issues/696)
* New methods: `cf.Field.cell_connectivity`,
  `cf.Field.cell_connectivities`
  (https://github.com/NCAS-CMS/cf-python/issues/696)
* New methods: `cf.Field.domain_topology`,
  `cf.Field.domain_topologies`
  (https://github.com/NCAS-CMS/cf-python/issues/696)
* New method: `cf.Data.masked_values`
  (https://github.com/NCAS-CMS/cf-python/issues/696)
* New method: `cf.Data.arctan2`
  (https://github.com/NCAS-CMS/cf-python/issues/38)
* Fix bug that caused `cf.Field.collapse` to give incorrect results
  for the "sum", "sum_of_weights" and "sum_of_weights2" methods, only
  in the case that weights have been requested
  (https://github.com/NCAS-CMS/cf-python/issues/701)
* Changed dependency: ``1.11.0.0<=cfdm<1.11.1.0``
* New dependency: ``scipy>=1.10.0``

version 3.15.4
--------------

**2023-10-10**

* Record dimension coordinate cell characteristics
  (https://github.com/NCAS-CMS/cf-python/issues/692)
* New set of methods to query, set, del, and `get_cell_characteristics`
* Fix bug in `cf.Field.match_by_construct` that always returned True for
  1-d constructs whose axis is not in the data, even when the
  criterion was not matched
  (https://github.com/NCAS-CMS/cf-python/issues/691)
* Fix bug that prevented "https://" netCDF files from being read
  (https://github.com/NCAS-CMS/cf-python/issues/699)

----

version 3.15.3
--------------

**2023-08-31**

* Prevent unlimited dimensions from being written to CFA-netCDF files
  (https://github.com/NCAS-CMS/cf-python/issues/689)

version 3.15.2
--------------

**2023-07-21**

* Performance improvements to `cf.Field.regrids` and
  `cf.Field.regridc`
  (https://github.com/NCAS-CMS/cf-python/issues/675)
* Performance improvements to `cf.read` when reading datasets with
  compression-by-convention
  (https://github.com/NCAS-CMS/cf-python/pull/671)
* New keyword parameter to `cf.Field.regrids` and `cf.Field.regridc`:
  ``weights_file`` (https://github.com/NCAS-CMS/cf-python/issues/675)
* New keyword parameter to `cf.aggregate`: ``cells``
  (https://github.com/NCAS-CMS/cf-python/issues/452)
* New function: `cf.isclose`
  (https://github.com/NCAS-CMS/cf-python/issues/661)
* Allow `cf.Data` and other objects that have data (such as
  `cf.DimensionCoordinate`) to be used as indices
  (https://github.com/NCAS-CMS/cf-python/issues/677)
* Fix bug that caused `cf.Field.regrids` and `cf.Field.regridc` to
  fail when the regrid axes are chunked with more than one chunk
  (https://github.com/NCAS-CMS/cf-python/issues/665)
* Fix bug that caused `cf.read` to fail for some PP data with a single
  vertical level (https://github.com/NCAS-CMS/cf-python/issues/667)
* Changed dependency: ``1.10.1.1<=cfdm<1.10.2.0``

----

version 3.15.1
--------------

**2023-06-09**

* New documentation cheat sheet
  (https://github.com/NCAS-CMS/cf-python/issues/591)
* Performance improvements to `cf.aggregate`
  (https://github.com/NCAS-CMS/cf-python/issues/640)
* New methods: `cf.Data.argmin`, `cf.Field.argmin`
  (https://github.com/NCAS-CMS/cf-python/issues/577)
* Deprecate `cf.relative_vorticity` in favour of `cf.curl_xy`
  (https://github.com/NCAS-CMS/cf-python/pull/643)
* Fix bug when using the ``-d`` option to the `cfa` script 
  (https://github.com/NCAS-CMS/cf-python/issues/649)
* Fix the return type from `cf.example_fields`
  (https://github.com/NCAS-CMS/cf-python/pull/654)
* Fix `set_data` when the units are invalid
  (https://github.com/NCAS-CMS/cf-python/pull/646)
* Fix `cf.Field.laplacian_xy`, `cf.Field.grad_xy`, `cf.curl_xy` and
  `cf.div_xy` to work in cases when the dimension coordinates are
  missing standard names
  (https://github.com/NCAS-CMS/cf-python/pull/643)
* Changed dependency: ``cfunits>=3.3.6``
* Changed dependency: ``cftime>=1.6.2``

----

version 3.15.0
--------------

**2023-04-27**

* Re-introduction of CFA-netCDF functionality for CFA-0.6
  (https://github.com/NCAS-CMS/cf-python/issues/451,
  https://github.com/NCAS-CMS/cf-python/issues/475,
  https://github.com/NCAS-CMS/cf-python/issues/637)
* New function: `cf.CFA`
* New method: `cf.Data.get_cfa_write`
* New method: `cf.Data.set_cfa_write`
* Fix excessive memory use arising from `cf.Field.regrids` and
  `cf.Field.regridc`
  (https://github.com/NCAS-CMS/cf-python/issues/623)
* Fixed bug in `cf.Field.read` when reading UM/PP data that are
  run-length encoded
  (https://github.com/NCAS-CMS/cf-python/issues/621)
* Fixed bug in `cf.Field.compute_vertical_coordinates` that retained
  incorrect bounds properties
  (https://github.com/NCAS-CMS/cf-python/issues/626)
* Removed benign UserWarning from `cf.Field.percentile`
  (https://github.com/NCAS-CMS/cf-python/issues/619)
* Handled the renaming of the ESMF Python interface from `ESMF` to
  `esmpy` at version 8.4.0. Both module names are accepted for now.
* Changed dependency: ``1.10.1.0<=cfdm<1.10.2.0``
* Changed (optional) dependency: ``8.0.0<=esmpy``
* Changed (optional) dependency: ``1.10.0<=scipy``

----

version 3.14.1
--------------

**2023-03-10**

* Python 3.7 support removed
  (https://github.com/NCAS-CMS/cf-python/issues/611)
* New method: `cf.Field.del_properties`
  (https://github.com/NCAS-CMS/cf-python/issues/598)
* New keyword parameter to `cf.Data.concatenate`: ``relaxed_units``
  (https://github.com/NCAS-CMS/cf-python/issues/602)
* New keyword parameter to `cf.unique_constructs`:
  ``ignore_properties``
  (https://github.com/NCAS-CMS/cf-python/issues/597)
* Allow `cf.aggregate` and to retain external cell measures
  (https://github.com/NCAS-CMS/cf-python/issues/605)
* Fixed bug in `cf.Field.collapse` that always retained invalid
  vertical coordinate reference systems
  (https://github.com/NCAS-CMS/cf-python/issues/599)
* Changed dependency: ``1.10.0.3<=cfdm<1.10.1.0``
* Changed dependency: ``cfunits>=3.3.5``
* Changed dependency: ``8.0.0<=ESMPY<8.2.0``

----

version 3.14.0 (*first Dask release*)
-------------------------------------

**2023-01-31**

* Introduction of `dask` for all data manipulations, replacing the old
  LAMA methodology
  (e.g. https://github.com/NCAS-CMS/cf-python/issues/295,
  https://github.com/NCAS-CMS/cf-python/issues/428)
* Backwards incompatible API changes to facilitate the use of Dask
  (https://github.com/NCAS-CMS/cf-python/issues/579)
* Removal of CFA-0.4 functionality (CFA-0.6 will introduced at a later
  version).
* New method: `cf.Field.get_original_filenames`
  (https://github.com/NCAS-CMS/cf-python/issues/448)
* New method: `cf.Field.to_dask_array`
* Fixed bug when reading UM stash codes that imply an area mean cell
  method (https://github.com/NCAS-CMS/cf-python/issues/487)
* New keyword parameter to `cf.write`: ``omit_data``
  (https://github.com/NCAS-CMS/cf-python/issues/477)
* New keyword parameter to `cf.read`: ``chunks``
  (https://github.com/NCAS-CMS/cf-python/pull/522)
* Extend functionality of `cf.Data.roll` and `cf.Field.roll` to allow
  multiple axes to be rolled simultaneously.
* Fixed bug that raised an exception when using the ``equal`` or
  ``exist`` keyword of `cf.aggregate`
  (https://github.com/NCAS-CMS/cf-python/issues/499)
* Fixed bug that created incorrect cyclic Voronoi bounds from
  `DimensionCoordinate.create_bounds`
  (https://github.com/NCAS-CMS/cf-python/issues/514)
* Changed dependency: ``1.10.0.2<=cfdm<1.10.1.0``
* New dependency: ``dask>=2022.12.1``

----

version 3.13.1 (*last LAMA release*)
------------------------------------

**2022-10-17**

* Upgrade to allow cf to work with Python 3.10
  (https://github.com/NCAS-CMS/cf-python/issues/419)
* Fixed bug when setting coordinate bounds with different units to
  their parent coordinates
  (https://github.com/NCAS-CMS/cf-python/issues/417)
* Fixed bug that created incorrect hybrid height levels when reading
  UM fields that also have pseudolevels
  (https://github.com/NCAS-CMS/cf-python/issues/468)
* Fixed bug when reading zero length netCDF or PP files caused
  cf-python to hang (https://github.com/NCAS-CMS/cf-python/issues/422)
* Changed dependency: ``1.9.0.4<=cfdm<1.9.1.0``

----

version 3.13.0
--------------

**2022-06-23**

* Fixed bug in `cf.read` when reading PP or fields files for which the
  ``um`` keyword has been set, but without the ``'version'`` key
  (https://github.com/NCAS-CMS/cf-python/issues/306)
* Fixed bug when setting the CFA "base" option to an empty string in
  `cf.write` (https://github.com/NCAS-CMS/cf-python/issues/346)
* Fixed failure from `cf.write` when writing identical (auxiliary)
  coordinates to different data variables in different groups
  (https://github.com/NCAS-CMS/cf-python/issues/336)
* Removed ``-g`` from the ``umread_lib/c-lib`` compiler options
  (https://github.com/NCAS-CMS/cf-python/pull/305)
* Updated the deprecated `distutils.version` to `packaging.version`
  (https://github.com/NCAS-CMS/cf-python/pull/324)
* Changed dependency: ``1.9.0.3<=cfdm<1.9.1.0``
* New dependency: ``packaging>=20.0``

----

version 3.12.0
--------------

**2022-01-18**

* New method: `cf.Field.grad_xy`
  (https://github.com/NCAS-CMS/cf-python/issues/292)
* New method: `cf.Field.laplacian_xy`
  (https://github.com/NCAS-CMS/cf-python/issues/292)
* New function: `cf.div_xy`
  (https://github.com/NCAS-CMS/cf-python/issues/292)
* New function: `cf.curl_xy`
  (https://github.com/NCAS-CMS/cf-python/issues/292)
* Allow element-wise comparisons with `None` on `cf.Data` and
  `cf.Field` objects (https://github.com/NCAS-CMS/cf-python/issues/281)
* Add missing closing parenthesis in `cf.Query` string representations
  (https://github.com/NCAS-CMS/cf-python/issues/279)
* Fix `cf.aggregate` failures when using the ``dimension`` keyword
  parameter (https://github.com/NCAS-CMS/cf-python/issues/283)
* Fix bug that raised error with subtraction of a `cf.TimeDuration`
  (https://github.com/NCAS-CMS/cf-python/issues/287)
* Fix bug in `cf.Field.derivative` when wrapping with periodic
  coordinates (https://github.com/NCAS-CMS/cf-python/issues/289)
* Changed dependency: ``1.9.0.1<=cfdm<1.9.1.0``
* Changed dependency: ``cfunits>=3.3.4``

----

version 3.11.0
--------------

**2021-10-08**

* Python 3.6 support removed
  (https://github.com/NCAS-CMS/cf-python/issues/208)
* Conversion of `cf.Domain` to a non-abstract that may be read from
  and written to a netCDF dataset
  (https://github.com/NCAS-CMS/cf-python/issues/220)
* New method: `cf.Domain.creation_commands`
* New method: `cf.Domain.climatological_time_axes`
* New method: `cf.AuxiliaryCoordinate.del_climatology`
* New method: `cf.AuxiliaryCoordinate.get_climatology`
* New method: `cf.AuxiliaryCoordinate.is_climatology`
* New method: `cf.AuxiliaryCoordinate.set_climatology`
* New method: `cf.DimensionCoordinate.del_climatology`
* New method: `cf.DimensionCoordinate.get_climatology`
* New method: `cf.DimensionCoordinate.is_climatology`
* New method: `cf.DimensionCoordinate.set_climatology`
* New function: `cf.unique_constructs`
* New function: `cf.example_fields`
  (https://github.com/NCAS-CMS/cf-python/issues/220)
* New keyword parameter to `cf.read`: ``cdl_string``
  (https://github.com/NCAS-CMS/cf-python/issues/171)
* Improved verbose information output from `cf.aggregate`
  (https://github.com/NCAS-CMS/cf-python/issues/228)
* Fix bug that causes a failure in `cf.aggregate` when otherwise
  aggregatable fields have non-valid units
  (https://github.com/NCAS-CMS/cf-python/issues/229)
* Fix for `cf.aggregate` failures when a datum or coordinate
  conversion parameter has an array value
  (https://github.com/NCAS-CMS/cf-python/issues/230)
* Allow for regridding using a destination field featuring size 1
  dimension(s) (https://github.com/NCAS-CMS/cf-python/issues/250)
* Fix bug that sometimes caused `cf.Field.autocyclic` to fail when
  setting a construct that is cyclic and has a defined period
* Fix bug that sometimes caused a failure when reading PP extra data
  (https://github.com/NCAS-CMS/cf-python/issues/241)
* Changed dependency: ``1.9.0.0<=cfdm<1.9.1.0``

----
  
version 3.10.0
--------------

**2021-06-10**

* Improve performance by allowing a saved `RegridOperator` instance to
  define the regrid operator in `cf.Field.regridc` and
  `cf.Field.regrids` (https://github.com/NCAS-CMS/cf-python/issues/222)
* Fix for incorrectly formatted `logging.info` statement sometimes
  causing a hang in `cf.Field.collapse`
  (https://github.com/NCAS-CMS/cf-python/issues/217)

----

version 3.9.0
-------------

**2021-05-25**

* Construct access API changes
  (https://github.com/NCAS-CMS/cf-python/issues/201,
  https://github.com/NCAS-CMS/cf-python/issues/202,
  https://github.com/NCAS-CMS/cf-python/issues/203,
  https://github.com/NCAS-CMS/cf-python/issues/204)
* Performance enhancements
  (https://github.com/NCAS-CMS/cf-python/issues/201,
  https://github.com/NCAS-CMS/cf-python/issues/202)
* New write mode ``mode='a'`` for appending to, rather than over-writing,
  a netCDF file on disk (https://github.com/NCAS-CMS/cf-python/issues/30)
* Temporarily removed the experimental ability to parallelise the
  collapse operation with MPI
  (https://github.com/NCAS-CMS/cf-python/issues/207)
* Improved docstrings
* Fix for unlimited dimensions read from a netCDF4 sub-group having
  zero size (https://github.com/NCAS-CMS/cfdm/issues/113)
* Fixes for changes in behaviour in cftime==1.4.0
  (https://github.com/NCAS-CMS/cf-python/issues/184)
* Better error message in the case of a `numpy.ma.core.MaskError` occurring
  upon reading of CDL files with only header or coordinate information
  (https://github.com/NCAS-CMS/cf-python/issues/197)
* Changed dependency: ``1.8.9.0<=cfdm<1.8.10.0``
* Changed dependency: ``cftime>=1.5.0``
* Changed dependency: ``cfunits>=3.3.3``
* Changed dependency: ``netCDF4>=1.5.4``

----

version 3.8.0
-------------

**2020-12-18**

* The setting of global constants can now be controlled by a context
  manager (https://github.com/NCAS-CMS/cf-python/issues/154)
* Changed the behaviour of binary operations for constructs that have
  bounds (https://github.com/NCAS-CMS/cf-python/issues/146)
* Changed the behaviour of unary operations for constructs that have
  bounds (https://github.com/NCAS-CMS/cf-python/issues/147)
* New function: `cf.bounds_combination_mode`
  (https://github.com/NCAS-CMS/cf-python/issues/146)
* New method: `cf.Field.compute_vertical_coordinates`
  (https://github.com/NCAS-CMS/cf-python/issues/142)
* Fixed bug that prevented the verbosity from changing to any value
  specified as a ``verbose`` keyword parameter to `cf.aggregate` (only).
* Fixed bug that caused a failure when writing a dataset that contains
  a scalar domain ancillary construct
  (https://github.com/NCAS-CMS/cf-python/issues/152)
* Fixed bug that prevented aggregation of fields with external cell measures
  (https://github.com/NCAS-CMS/cf-python/issues/150#issuecomment-729747867)
* Fixed bug that caused rows full of zeros to appear in WGDOS packed
  UM data that contain masked points
  (https://github.com/NCAS-CMS/cf-python/issues/161)
* Changed dependency: ``1.8.8.0<=cfdm<1.8.9.0``
* Changed dependency: ``cftime>=1.3.0``
* Changed dependency: ``cfunits>=3.3.1``

----

version 3.7.0
-------------

**2020-10-15**

* Python 3.5 support deprecated (3.5 was retired on 2020-09-13)
* New method: `cf.Field.del_domain_axis`
* New method: `cf.Field._docstring_special_substitutions`
* New method: `cf.Field._docstring_substitutions`
* New method: `cf.Field._docstring_package_depth`
* New method: `cf.Field._docstring_method_exclusions`
* New keyword parameter to `cf.Field.set_data`: ``inplace``
* New keyword parameter to `cf.write`: ``coordinates``
  (https://github.com/NCAS-CMS/cf-python/issues/125)
* New keyword parameter to `cf.aggregate`: ``ignore``
  (https://github.com/NCAS-CMS/cf-python/issues/115)
* Fixed bug that caused a failure when reading a dataset with
  incompatible bounds units. Now a warning is given (controllable by
  the logging level) and the offending bounds are returned as a
  separate field construct.
* Fixed bug in `cf.aggregate` that caused it to error if either the
  `equal_all` or `exist_all` parameter were set to `True`.
* Fixed bug in `Data.percentile` that caused it to error for non-singular
  ranks if the squeeze parameter was set to `True`.
* ``cfa`` now prints error messages to the stderr stream rather than
  stdout.
* Changed dependency: ``1.8.7.0<=cfdm<1.8.8.0``
* Changed dependency: ``cfunits>=3.3.0``

----

version 3.6.0
-------------

**2020-07-24**

* Implemented the reading and writing of netCDF4 group hierarchies for
  CF-1.8 (https://github.com/NCAS-CMS/cf-python/issues/33)
* New method: `cf.Field.nc_variable_groups`
* New method: `cf.Field.nc_set_variable_groups`
* New method: `cf.Field.nc_clear_variable_groups`
* New method: `cf.Field.nc_group_attributes`
* New method: `cf.Field.nc_set_group_attribute`
* New method: `cf.Field.nc_set_group_attributes`
* New method: `cf.Field.nc_clear_group_attributes`
* New method: `cf.Field.nc_geometry_variable_groups`
* New method: `cf.Field.nc_set_geometry_variable_groups`
* New method: `cf.Field.nc_clear_geometry_variable_groups`
* New method: `cf.DomainAxis.nc_dimension_groups`
* New method: `cf.DomainAxis.nc_set_dimension_groups`
* New method: `cf.DomainAxis.nc_clear_dimension_groups`
* New keyword parameter to `cf.write`: ``group``
* Keyword parameter ``verbose`` to multiple methods now accepts named
  strings, not just the equivalent integer levels, to set verbosity.
* New function: `cf.configuration`
* Renamed to lower-case (but otherwise identical) names all functions which
  get and/or set global constants: `cf.atol`, `cf.rtol`, `cf.log_level`,
  `cf.chunksize`, `cf.collapse_parallel_mode`, `cf.free_memory`,
  `cf.free_memory_factor`, `cf.fm_threshold`, `cf.of_fraction`,
  `cf.regrid_logging`, `cf.set_performance`, `cf.tempdir`, `cf.total_memory`,
  `cf.relaxed_identities`. The upper-case names remain functional as aliases.
* Changed dependency: ``cftime>=1.2.1``
* Changed dependency: ``1.8.6.0<=cfdm<1.8.7.0``
* Changed dependency: ``cfunits>=3.2.9``

----

version 3.5.1
-------------

**2020-06-10**

* Changed dependency: ``1.8.5<=cfdm<1.9.0``
* Fixed bug (emerging from the cfdm library) that prevented the
  reading of certain netCDF files, such as those with at least one
  external variable.

----

version 3.5.0
-------------

**2020-06-09**

* Changed the API to `cf.Field.period`: Now sets and reports on the
  period of the field construct data, rather than that of its metadata
  constructs.
* Enabled configuration of the extent and nature of informational and
  warning messages output by `cf` using a logging framework (see
  points below and also https://github.com/NCAS-CMS/cf-python/issues/37)
* Changed behaviour and default of ``verbose`` keyword argument when
  available to a function/method so it interfaces with the new logging
  functionality.
* Renamed and re-mapped all ``info`` keyword arguments available to any
  function/method to ``verbose``, with equal granularity but a different
  numbering system: ``V = I + 1`` maps ``info=I`` to ``verbose=V`` except
  for the ``debug`` case of ``I=3`` mapping to ``V=-1`` (``V=0`` disables).
* New function `cf.LOG_LEVEL` to set the minimum log level for which
  messages are displayed globally, i.e. to change the project-wide
  verbosity.
* New method: `cf.Field.halo`
* New method: `cf.Data.halo`
* New keyword parameter to `cf.Data.empty`: ``fill_value``
* Changed dependency: ``1.8.4<=cfdm<1.9.0``
* Changed dependency: ``cfunits>=3.2.7``
* Changed dependency: ``cftime>=1.1.3``
* When assessing coordinate constructs for contiguousness with
  `cf.Bounds.contiguous`, allow periodic values that differ by the
  period to be considered the same
  (https://github.com/NCAS-CMS/cf-python/issues/75).
* Fixed bug in `cf.Field.regrids` that caused a failure when
  regridding from latitude-longitude to tripolar domains
  (https://github.com/NCAS-CMS/cf-python/issues/73).
* Fixed bug in `cf.Field.regrids` that caused a failure when
  regridding to tripolar domains the do not have dimension coordinate
  constructs (https://github.com/NCAS-CMS/cf-python/issues/73).
* Fixed bug in `cf.Field.regrids` and `cf.Field.regridc` that caused a
  failure when applying the destination mask to the regridded fields
  (https://github.com/NCAS-CMS/cf-python/issues/73).
* Fixed bug that caused `cf.FieldList.select_by_ncvar` to always fail
  (https://github.com/NCAS-CMS/cf-python/issues/76).
* Fixed bug that stopped 'integral' collapses working for grouped
  collapses (https://github.com/NCAS-CMS/cf-python/issues/81).
* Fixed bug that wouldn't allow the reading of a netCDF file which
  specifies Conventions other than CF
  (https://github.com/NCAS-CMS/cf-python/issues/78).

----

version 3.4.0
-------------

**2020-04-30**

* New method: `cf.Field.apply_masking`
* New method: `cf.Data.apply_masking`
* New method: `cf.Field.get_filenames` (replaces deprecated
  `cf.Field.files`)
* New method: `cf.Data.get_filenames` (replaces deprecated
  `cf.Data.files`)
* New keyword parameter to `cf.read`: ``mask``
* New keyword parameter to `cf.read`: ``warn_valid``
  (https://github.com/NCAS-CMS/cfdm/issues/30)
* New keyword parameter to `cf.write`: ``warn_valid``
  (https://github.com/NCAS-CMS/cfdm/issues/30)
* New keyword parameter to `cf.Field.nc_global_attributes`: ``values``
* Added time coordinate bounds to the polygon geometry example field
  ``6`` returned by `cf.example_field`.
* Changed dependency: ``cfdm==1.8.3``
* Changed dependency: ``cfunits>=3.2.6``
* Fixed bug in `cf.write` that caused (what are effectively)
  string-valued scalar auxiliary coordinates to not be written to disk
  as such, or even an exception to be raised.
* Fixed bug in `cf.write` that caused the ``single`` and ``double``
  keyword parameters to have no effect. This bug was introduced at
  version 3.0.0 (https://github.com/NCAS-CMS/cf-python/issues/65).
* Fixed bug in `cf.Field.has_construct` that caused it to always
  return `False` unless a construct key was used as the construct
  identity (https://github.com/NCAS-CMS/cf-python/issues/67).
  
----

version 3.3.0
-------------

**2020-04-20**

* Changed the API to `cf.Field.convolution_filter`: renamed the
  ``weights`` parameter to ``window``.
* Reinstated `True` as a permitted value of the ``weights`` keyword of
  `cf.Field.collapse` (which was deprecated at version 3.2.0).
* New method: `cf.Field.moving_window`
  (https://github.com/NCAS-CMS/cf-python/issues/44)
* New method: `cf.Data.convolution_filter`
* New keyword parameter to `cf.Field.weights`: ``axes``
* New permitted values to ``coordinate`` keyword parameter of
  `cf.Field.collapse` and `cf.Field.cumsum`: ``'minimum'``,
  ``'maximum'``
* New keyword parameter to `cf.Data.cumsum`: ``inplace``
* Fixed bug that prevented omitted the geometry type when creating
  creation commands (https://github.com/NCAS-CMS/cf-python/issues/59).
* Fixed bug that caused a failure when rolling a dimension coordinate
  construct without bounds.
  
----

version 3.2.0
-------------

**2020-04-01**

* First release for CF-1.8 (does not include netCDF hierarchical
  groups functionality)
  (https://github.com/NCAS-CMS/cf-python/issues/33)
* Deprecated `True` as a permitted value of the ``weights`` keyword of
  `cf.Field.collapse`.
* New methods: `cf.Data.compressed`, `cf.Data.diff`
* New function: `cf.implementation`
* New methods completing coverage of the inverse trigonometric and
  hyperbolic operations: `cf.Data.arccos`, `cf.Data.arccosh`,
  `cf.Data.arcsin`, `cf.Data.arctanh`.
* New keyword parameters to `cf.Field.collapse`, `cf.Field.cell_area`,
  `cf.Field.weights`: ``radius``, ``great_circle``.
* Implemented simple geometries for CF-1.8.
* Implemented string data-types for CF-1.8.
* Changed dependency: ``cfdm>=1.8.0``
* Changed dependency: ``cfunits>=3.2.5``
* Changed dependency: ``netCDF4>=1.5.3``
* Changed dependency: ``cftime>=1.1.1``
* Renamed the regridding method, i.e. option for the ``method``
  parameter to `cf.Field.regridc` and `cf.Field.regrids`, ``bilinear``
  to ``linear``, though ``bilinear`` is still supported (use of it
  gives a message as such).
* Made documentation of available `cf.Field.regridc` and
  `cf.Field.regrids` ``method`` parameters clearer & documented
  second-order conservative method.
* Fixed bug that prevented writing to ``'NETCDF3_64BIT_OFFSET'`` and
  ``'NETCDF3_64BIT_DATA'`` format files
  (https://github.com/NCAS-CMS/cfdm/issues/9).
* Fixed bug that prevented the ``select`` keyword of `cf.read` from
  working with PP and UM files
  (https://github.com/NCAS-CMS/cf-python/issues/40).
* Fixed bug that prevented the reading of PP and UM files with "zero"
  data or validity times.
* Fixed broken API reference 'source' links to code in `cfdm`.
* Fixed bug in `cf.Field.weights` with the parameter ``methods`` set
  to ``True`` where it would always error before returning dictionary
  of methods.
* Fixed bug in `cf.Data.where` that meant the units were not taken
  into account when the condition was a `cf.Query` object with
  specified units.
* Addressed many 'TODO' placeholders in the documentation.

----

version 3.1.0
-------------

**2020-01-17**

* Changed the API to `cf.Field.match_by_construct` and
  `cf.FieldList.select_by_construct`.
* Changed the default value of the `cf.Field.collapse` ``group_span``
  parameter to `True` and default value of the ``group_contiguous``
  parameter to ``1``
  (https://github.com/NCAS-CMS/cf-python/issues/28).
* Changed the default values of the `cf.Field.collapse` ``group_by``
  and ``coordinate`` parameters to `None`.
* Changed the default value of the ``identity`` parameter to `None`
  for `cf.Field.coordinate`, `cf.Field.dimension_coordinate`,
  `cf.Field.auxiliary_coordinate`, `cf.Field.field_ancillary`,
  `cf.Field.domain_ancillary`, `cf.Field.cell_method`,
  `cf.Field.cell_measure`, `cf.Field.coordinate_reference`,
  `cf.Field.domain_axis`.
* New keyword parameter to `cf.Field.weights`: ``data``.
* New keyword parameter to `cf.aggregate`: ``field_identity``
  (https://github.com/NCAS-CMS/cf-python/issues/29).
* New example field (``5``) available from `cf.example_field`.
* New regridding option: ``'conservative_2nd'``.
* Fixed bug that didn't change the units of bounds when the units of
  the coordinates were changed.
* Fixed bug in `cf.Field.domain_axis` that caused an error when no
  unique domain axis construct could be identified.
* Changed dependency:``cfunits>=3.2.4``. This fixes a bug that raised
  an exception for units specified by non-strings
  (https://github.com/NCAS-CMS/cfunits/issues/1).
* Changed dependency: ``ESMF>=to 8.0.0``. This fixes an issue with
  second-order conservative regridding, which is now fully documented
  and available.
* Converted all remaining instances of Python 2 print statements in the
  documentation API reference examples to Python 3.
* Corrected aspects of the API documentation for trigonometric functions.
* Fixed bug whereby `cf.Data.arctan` would not process bounds.
* New methods for hyperbolic operations: `cf.Data.sinh`, `cf.Data.cosh`,
  `cf.Data.tanh`, `cf.Data.arcsinh`.

----

version 3.0.6
-------------

**2019-11-27**

* New method: `cf.Field.uncompress`.
* New method: `cf.Data.uncompress`.
* New keyword parameter to `cf.environment`: ``paths``.
* Can now insert a size 1 data dimension for a new, previously
  non-existent domain axis with `cf.Field.insert_dimension`.
* Changed the default value of the ``ignore_compression`` parameter to
  `True`.
* Fixed bug that sometimes gave incorrect cell sizes from the
  `cellsize` attribute when used on multidimensional coordinates
  (https://github.com/NCAS-CMS/cf-python/issues/15).
* Fixed bug that sometimes gave an error when the LHS and RHS operands
  are swapped in field construct arithmetic
  (https://github.com/NCAS-CMS/cf-python/issues/16).
* Changed dependency: ``cfdm>=1.7.11``

----

version 3.0.5
-------------

**2019-11-14**

* New method: `cf.Field.compress`.
* New function: `cf.example_field`
* New keyword parameter to `cf.Data`: ``mask``.
* Deprecated method: `cf.Field.example_field`
* Fixed bug that didn't allow `cf.Field.cell_area` to work with
  dimension coordinates with units equivalent to metres
  (https://github.com/NCAS-CMS/cf-python/issues/12)
* Fixed bug that omitted bounds having their units changed by
  `override_units` and `override calendar`
  (https://github.com/NCAS-CMS/cf-python/issues/13).
* Removed specific user shebang from ``cfa`` script
  (https://github.com/NCAS-CMS/cf-python/pull/14).
* Changed dependency: ``cfdm>=1.7.10``. This fixes a bug that didn't
  allow CDL files to start with comments or blank lines
  (https://github.com/NCAS-CMS/cfdm/issues/5).
* Changed dependency: ``cftime>=1.0.4.2``

----

version 3.0.4
-------------

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
* Changed dependency: ``cfdm>=1.7.9``

----

version 3.0.3
-------------

**2019-11-01**

* Fixed bug (introduced at v3.0.2) that caused ``mean_absolute_value``
  collapses by `cf.Field.collapse` to be not weighted when they should
  be (https://github.com/NCAS-CMS/cf-python/issues/9)
* Changed default ``ddof`` from 0 to 1 in `cf.Data.var` and
  `cf.Data.sd` (https://github.com/NCAS-CMS/cf-python/issues/8)
   
----

version 3.0.2
-------------

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
* Changed dependency: ``cfdm>=1.7.8``. This fixes a bug that sometimes
  occurs when writing to disk and the _FillValue and data have
  different data types.
* Changed dependency: ``cfunits>=3.2.2``
* Changed dependency: ``cftime>=1.0.4.2``
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

----

version 3.0.1
-------------

**2019-10-01**

* Updated description in ``setup.py``

----

version 3.0.0 (*first Python 3 release*)
----------------------------------------

**2019-10-01**

* Complete refactor for Python 3, including some API changes.

  Scripts written for version 2.x but running under version 3.x should
  either work as expected, or provide informative error messages on
  the new API usage. However, it is advised that the outputs of older
  scripts be checked when running with Python 3 versions of the cf
  library.
* Deprecated ``cfdump`` (its functionality is now included in
  ``cfa``).
  
----

version 2.3.8 (*last Python 2 release*)
---------------------------------------

**2019-10-07**

* In `cf.write`, can set ``single=False`` to mean ``double=True``, and
  vice versa.
* Fixed bug in `cf.aggregate` - removed overly strict test on
  dimension coordinate bounds.
* Fixed bug in `cf.read` that set the climatology attribute to True
  when there are no bounds.
* Fixed bug in `cf.write` when writing missing values (set_fill was
  off, now on)

----

version 2.3.5
-------------

**2019-04-04**

* Changed calculation of chunksize in parallel case to avoid potential
  problems and introduced a new method `cf.SET_PERFORMANCE` to tune
  the chunksize and the fraction of memory to keep free.

----

version 2.3.4
-------------

**2019-03-27**

* Fix bug in creating a during cell method during a field collapse.
	
----

version 2.3.3
-------------

**2019-03-05**

* Allow failure to compile to go through with a warning, rather than
  failing to install. if this happens, reading a PP/UM file will
  result in "Exception: Can't determine format of file test2.pp"
* Fixed bug in `cf.Field.convolution_filter` giving false error over
  units.

----
	
version 2.3.2
-------------

**2018-12-10**

* `cf.Field.regridc` now compares the units of the source and
  destination grids and converts between them if possible or raises an
  error if they are not equivalent.
	
----

version 2.3.1
-------------

**2018-11-07**

* Fixed bug in `cf.Field.regridc` that caused it to fail when
  regridding a multidimensional field along only one dimension.
* Fixed bug which in which the default logarithm is base 10, rather
  than base e
	
version 2.3.0
-------------

**2018-10-22**

* The collapse method can now be parallelised by running any cf-python
  script with mpirun if mpi4py is installed. This is an experimental
  feature and is not recommended for operational use. None of the
  parallel code is executed when a script is run in serial.
	
----

version 2.2.8
-------------

**2018-08-28**

* Bug fix: better handle subspacing by multiple multidimensional items
	
----

version 2.2.7
-------------

**2018-07-25**

* Bug fix: correctly set units of bounds when the `cf.Data` object
  inserted with insert_bounds has units of ''. In this case the bounds
  of the parent coordinate are now inherited.
	
----

version 2.2.6
-------------

**2018-07-24**

* Improved error messages
* Changed behaviour when printing reference times with a calendar of
  ``'none'`` - no longer attempts a to create a date-time
  representation
	
----

version 2.2.5
-------------

**2018-07-02**

* Fixed bug with HDF chunk sizes that prevented the writing of large
  files
	
----

version 2.2.4
-------------

**2018-06-29**

* Interim fix for with HDF chunk sizes that prevented the writing of
  large files
	
version 2.2.3
--------------
----

**2018-06-21**

* During writing, disallow the creation of netCDF variable names that
  contain characters other than letters, digits, and underscores.
	
----

version 2.2.3
-------------

**2018-06-21**

* During writing, disallow the creation of netCDF variable names that
  contain characters other than letters, digits, and underscores.
	
----

version 2.2.2
-------------

**2018-06-06**


* Fix for removing duplicated netCDF dimensions when writing data on
  (e.g.) tripolar grids.
	
----

version 2.2.1
-------------

**2018-06-05**

* Fix for calculating are weights from projection coordinates
			
version 2.2.0
-------------
----

**2018-06-04**

* Updated for `netCDF4` v1.4 `cftime` API changes
	
----

version 2.1.9
-------------

**2018-05-31**

* Allowed invalid units through. Can test with `cf.Units.isvalid`.
	
----

version 2.1.8
-------------

**2018-03-08**

* Fixed bug when weights parameter is a string in `cf.Field.collapse`
	
----

version 2.1.7
-------------

**2018-02-13**

* Fixed bug in `cf.Field.collapse` when doing climatological time
  collapse with only one period per year/day
		
----

version 2.1.6
-------------

**2018-02-09**

* Fixed bug in Variable.mask
	
----

version 2.1.4
-------------

**2018-02-09**

* Added override_calendar method to coordinates and domain ancillaries
  that changes the calendar of the bounds, too.
* Fixed bug in `cf.Data.where` when the condition is a `cf.Query`
  object.
* Fixed bug in `cf.Variable.mask`
	
----

version 2.1.3
-------------

**2018-02-07**

* Allowed `scipy` and `matplotlib` imports to be optional
	
version 2.1.2
-------------
----

**2017-11-28**

* Added ``group_span`` and ``contiguous_group`` options to
  `cf.Field.collapse`
	
----

version 2.1.1
-------------

**2017-11-10**

* Disallowed raising offset units to a power (e.g. taking the square
  of data in units of K @ 273.15).
* Removed len() of `cf.Field` (previously always, and misleadingly,
  returned 1)
* Fixed setting of cell methods after climatological time collapses
* Added printing of ncvar in `cf.Field.__str__` and `cf.Field.dump`
* Added user stash table option to ``cfa`` script
	
----

version 2.1
-----------

**2017-10-30**

* Misc. bug fixes

version 2.0.6
-------------
----

**2017-09-28**

* Removed error when `cf.read` finds no fields - an empty field list
  is now returned
* New method `cf.Field.count`

----

version 2.0.5
-------------

**2017-09-19**

* Bug fix when creating wrap-around subspaces from cyclic fields
* Fix (partial?) for memory leak when reading UM PP and fields files

----

version 2.0.4
-------------

**2017-09-15**

* submodel property for PP files
* API change for `cf.Field.axis`: now returns a `cf.DomainAxis` object
  by default
* Bug fix in `cf.Field.where`
* Bug fix when initialising a field with the source parameter
* Changed default output format to NETCDF4 (from NETCDF3_CLASSIC)

----

version 2.0.3
-------------

**2017-08-01**

----

version 2.0.1.post1
-------------------

**2017-07-12**

* Bug fix for reading DSG ragged arrays

----

version 2.0.1
-------------

**2017-07-11**

* Updated `cf.FieldList` behaviour (with reduced methods)

----

version 2.0
-----------

**2017-07-07**

* First release with full CF data model and full CF-1.6 compliance
  (including DSG)

----

version 1.5.4.post4
-------------------

**2017-07-07**

* Bug fixes to `cf.Field.regridc`

----

version 1.5.4.post1
-------------------

**2017-06-13**

* removed errant scikit import

----

version 1.5.4
-------------

**2017-06-09**

* Tripolar regridding
	
----

version 1.5.3 
-------------

**2017-05-10**

* Updated STASH code to standard_name table (with thanks to Jeff Cole)
* Fixed bug when comparing masked arrays for equality

----

version 1.5.2 
-------------

**2017-03-17**

* Fixed bug when accessing PP file whose format/endian/word-size has
  been specified

----

version 1.5.1 
-------------

**2017-03-14**

* Can specify 'pp' or 'PP' in um option to `cf.read`

----

version 1.5
-----------

**2017-02-24**

* Changed weights in calculation of variance to reliability weights
  (from frequency weights). This not only scientifically better, but
  faster, too.

----

version 1.4
-----------

**2017-02-22**

* Rounded datetime to time-since conversions to the nearest
  microsecond, to reflect the accuracy of netCDF4.netcdftime
* Removed import tests from setup.py
* New option --um to ``cfa``, ``cfdump``
* New parameter um to `cf.read`

----

version 1.3.3
-------------

**2017-01-31**

* Rounded datetime to time-since conversions to the nearest
  microsecond, to reflect the accuracy of netCDF4.netcdftime
* Fix for netCDF4.__version__ > 1.2.4 do to with datetime.calendar
  *handle with care*

----

version 1.3.2
-------------

**2016-09-21**

* Added --build-id to LDFLAGS in umread Makefile, for sake of RPM
  builds (otherwise fails when building debuginfo RPM). Pull request
  #16, thanks to Klaus Zimmermann.
* Improved test handling. Pull request #21, thanks to Klaus
  Zimmermann.
* Removed udunits2 database. This removes the modified version of the
  udunits2 database in order to avoid redundancies, possible version
  incompatibilities, and license questions. The modifications are
  instead carried out programmatically in units.py. Pull request #20,
  thanks to Klaus Zimmermann.

----

version 1.3.1
-------------

**2016-09-09**

* New method: `cf.Field.unlimited`, and new 'unlimited' parameter to
  `cf.write` and ``cfa``

----

version 1.3
-----------

**2016-09-05**

* Removed asreftime, asdatetime and dtvarray methods
* New method: `convert_reference_time` for converting reference time
  data values to have new units.

----

version 1.2.3
-------------

**2016-08-23**

* Fixed bug in `cf.Data.equals`

----

version 1.2.2
-------------

**2016-08-22**

* Fixed bug in binary operations to do with the setting of
  `Partition.part`
* Added `cf.TimeDuration` functionality to get_bounds cellsizes
  parameter. Also new parameter flt ("fraction less than") to position
  the coordinate within the cell.

----

version 1.2
-----------

**2016-07-05**

* Added HDF_chunks methods

----

version 1.1.11
--------------

**2016-07-01**

* Added cellsize option to `cf.Coordinate.get_bounds`, and fixed bugs.
* Added variable_attributes option to `cf.write`
* Added `cf.ENVIRONMENT` method

----

version 1.1.10
--------------

**2016-06-23**

* Added reference_datetime option to cf.write	
* Fixed bug in `cf.um.read.read` which incorrectly ordered vertical
  coordinates

----

version 1.1.9
-------------

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
	
----

version 1.1.8
-------------

**2016-05-18**

* Slightly changed the compression API to `cf.write`
* Added compression support to the ``cfa`` command line script
* Added functionality to change data type on writing to `cf.write` and
  ``cfa`` - both in general and for with extra convenience for the
  common case of double to single (and vice versa).
* Removed annoying debug print statements from `cf.um.read.read`

----

version 1.1.7
-------------

**2016-05-04**

* Added fix for change in numpy behaviour (`numpy.number` types do not
  support assignment)
* Added capability to load in a user STASH to standard name table:
  `cf.um.read.load_stash2standard_name`
	
----

version 1.1.6
-------------

**2016-04-27**

* Added --reference_datetime option to ``cfa``
* Bug fix to `cf.Field.collapse` when providing `cf.Query` objects via
  the group parameter
* Added auto regridding method, which is now the default
	

----

version 1.1.5 
-------------

**2016-03-03**

* Bug fix in `cf.Field.where` when using `cf.masked`
* conda installation (with thanks to Andy Heaps)
* Bug fix for type casting in `cf.Field.collapse`
* Display long_name if it exists and there is no standard_name
* Fix for compiling the UM C code on certain OSs (with thanks to Simon Wilson)
* Fixed incorrect assignment of cyclicity in `cf.Field.regrids`
* Nearest neighbour regridding in `cf.Field.regrids`
	
----

version 1.1.4 
-------------

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

----

version 1.1.3 
-------------

**2015-12-10**

* Bug fixes to `cf.Field.collapse` when the "group" parameter is used
* Correct setting of cyclic axes on regridded fields
* Updates to STASH_to_CF.txt table: 3209, 3210
	
----

version 1.1.2 
-------------

**2015-12-01**

* Updates to STASH_to_CF.txt table
* Fixed bug in decoding UM version in `cf.um.read.read`
* Fixed bug in `cf.units.Utime.num2date`
* Fixed go-slow behaviour for silly BZX, BDX in PP and fields file
  lookup headers

----

version 1.1.1
-------------

**2015-11-05**

* Fixed bug in decoding UM version in `cf.read`
	
----

version 1.1
-----------

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

----

version 1.0.3
-------------

**2015-06-23**

* Added default keyword to fill_value() and fixed bugs when doing
  delattr on _fillValue and missing_value properties.

version 1.0.2
-------------

**2015-06-05**

* PyPI release

----

version 1.0.1
-------------

**2015-06-01**

* Fixed bug in when using the select keyword to `cf.read`

----

version 1.0
-----------

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

----

version 0.9.9.1
---------------

**2015-01-09**

* Fixed bug for changes to netCDF4-python library versions >= 1.1.2
* Miscellaneous bug fixes

----

version 0.9.9
-------------

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

----

version 0.9.8.3
---------------

**2014-07-14**

* Implemented multiple grid_mappings (CF trac ticket #70)
* Improved functionality and speed of field aggregation and ``cfa``
  and ``cfdump`` command line utilities.
* Collapse methods on `cf.Data` object (min, max, mean, var, sd,
  sum, range, mid_range).
* Improved match/select functionality

----

version 0.9.8.2
---------------

**2014-03-13**

* Copes with PP fields with 365_day calendars
* Revamped CFA files in line with the evolving standard. CFA files
  from PP data created with a previous version will no longer work.

----

version 0.9.8
-------------

**2013-12-06**

* Improved API.
* Plenty of speed and memory optimisations.
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

----

version 0.9.7.1
---------------

**2013-04-26**

* Fixed endian bug in CFA-netCDF files referring to PP files
* Changed default output format to NETCDF3_CLASSIC and trap error when
  when writing unsigned integer types and the 64-bit integer type to
  file formats other than NETCDF4.
* Changed unhelpful history created when aggregating

----

version 0.9.7
-------------

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

----

version 0.9.6.2
---------------

**2013-03-27**

* Fixed bug in ``cf/pp.py`` which caused the creation of incorrect
  latitude coordinate arrays.

----

version 0.9.6.1
---------------

**2013-02-20**

* Fixed bug in ``cf/netcdf.py`` which caused a failure when a file
  with badly formatted units was encountered.

----

version 0.9.6
-------------

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

----

version 0.9.5 (*first LAMA release*)
------------------------------------

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

----

version 0.9.5.dev 
-----------------

**2012-09-19**

* Loads of exciting improvements - mainly LAMA functionality,
  metadata-aware field manipulation and documentation.
* This is a snapshot of the trunk at revision r185. A proper vn0.9.5
  release is imminent.

----

version 0.9.4.2
---------------

**2012-04-17**

* General bug fixes and code restructure

----

version 0.9.4
-------------

**2012-03-15**

* A proper treatment of units using the Udunits C library and the
  extra time functionality provided by the netCDF4 package.
* A command line script to do CF-netCDF to CF-netCDF via cf-python.

----

version 0.9.3.3
---------------

**2012-02-08**

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

----

version 0.9.3
-------------

**2012-01-05**

* A more consistent treatment of spaces and lists of spaces
  (`cf.Space` and `cf.SpaceList` objects respectively).
* A corrected treatment of scalar or 1-d, size 1 dimensions in the
  space and its grid.
* Data stored in `cf.Data` objects which contain metadata need to
  correctly interpret and manipulate the data. This will be
  particularly useful when data arrays spanning many files/arrays is
  implemented.

----

version 0.9.2
-------------

**2011-08-26**

* Created a ``setup.py`` script for easier installation (with thanks
  to Jeff Whitaker).
* Added support for reading OPeNDAP-hosted datasets given by URLs.
* Restructured the documentation.
* Created a test directory with scripts and sample output.
* No longer fails for unknown calendar types (such as ``'360d'``).

----

version 0.9.1
-------------

**2011-08-06**

* First release.
