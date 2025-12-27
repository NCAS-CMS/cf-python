.. currentmodule:: cf
.. default-role:: obj

.. _function:

**cf functions**
================

----

Version |release| for version |version| of the CF conventions.

Reading and writing
-------------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.read 
   cf.write
   cf.netcdf_lock

Aggregation
-----------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.aggregate
   cf.climatology_cells

.. _functions-mathematical-operations:

Mathematical operations
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.atol
   cf.rtol
   cf.bounds_combination_mode
   cf.default_netCDF_fillvals
   cf.curl_xy
   cf.div_xy
   cf.histogram
   cf.ATOL
   cf.RTOL
   
Condition constructors
----------------------

.. rubric:: General conditions
   
.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.eq
   cf.ge
   cf.gt
   cf.le
   cf.lt
   cf.ne
   cf.wi
   cf.wo
   cf.set
   cf.isclose

.. rubric:: Date-time conditions
   
.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.year 	
   cf.month 	
   cf.day 	
   cf.hour 	
   cf.minute 	
   cf.second 	
   cf.jja 	
   cf.son 	
   cf.djf 	
   cf.mam
   cf.seasons 	

.. rubric:: Coordinate cell conditions

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.contains
   cf.cellsize
   cf.cellgt
   cf.cellge
   cf.cellle
   cf.celllt
   cf.cellwi
   cf.cellwo

Date-time and time duration
---------------------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.dt
   cf.dt_vector
   cf.Y
   cf.M
   cf.D
   cf.h
   cf.m
   cf.s

Resource management
-------------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.configuration
   cf.chunksize
   cf.free_memory
   cf.regrid_logging
   cf.tempdir
   cf.total_memory
   cf.CHUNKSIZE
   cf.FREE_MEMORY
   cf.REGRID_LOGGING
   cf.TEMPDIR
   cf.TOTAL_MEMORY

Active storage reductions
-------------------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.active_storage
   cf.active_storage_url
   cf.active_storage_max_requests
   cf.netcdf_lock

Miscellaneous
-------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.CF
   cf.abspath
   cf.dirname
   cf.dump
   cf.environment
   cf.example_field
   cf.example_fields
   cf.example_domain
   cf.flat
   cf.implementation
   cf.indices_shape
   cf.inspect
   cf.log_level
   cf.pathjoin
   cf.relaxed_identities
   cf.relpath
   cf.load_stash2standard_name
   cf.stash2standard_name 
   cf.unique_constructs
   cf.LOG_LEVEL
   cf.RELAXED_IDENTITIES

Deprecated
----------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.close_files
   cf.close_one_file   
   cf.collapse_parallel_mode
   cf.fm_threshold
   cf.free_memory_factor
   cf.hash_array
   cf.min_total_memory
   cf.of_fraction
   cf.open_files
   cf.open_files_threshold_exceeded
   cf.relative_vorticity
   cf.set_performance
   cf.COLLAPSE_PARALLEL_MODE
   cf.FM_THRESHOLD
   cf.FREE_MEMORY_FACTOR
   cf.MIN_TOTAL_MEMORY
   cf.OF_FRACTION
   cf.SET_PERFORMANCE
   
