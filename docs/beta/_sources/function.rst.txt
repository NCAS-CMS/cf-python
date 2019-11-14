.. currentmodule:: cf
.. default-role:: obj

.. _function:

**cf functions**
================

----

Version |release| for version |version| of the CF conventions.

**Reading and writing**
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.read 
   cf.write
   cf.load_stash2standard_name 

**Aggregation**
---------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.aggregate

.. _functions-mathematical-operations:

**Mathematical operations**
---------------------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.ATOL
   cf.RTOL
   cf.default_netCDF_fillvals
   cf.histogram
   cf.relative_vorticity

**Condition constructors**
--------------------------

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

**Date-time and time duration**
-------------------------------

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

**Resource management**
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.CHUNKSIZE
   cf.COLLAPSE_PARALLEL_MODE
   cf.FREE_MEMORY
   cf.FREE_MEMORY_FACTOR
   cf.FM_THRESHOLD
   cf.MINNCFM
   cf.OF_FRACTION
   cf.REGRID_LOGGING
   cf.SET_PERFORMANCE
   cf.TEMPDIR
   cf.TOTAL_MEMORY
   cf.close_files
   cf.close_one_file
   cf.open_files
   cf.open_files_threshold_exceeded

**Miscellaneous**
-----------------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.CF
   cf.RELAXED_IDENTITIES
   cf.abspath
   cf.dirname
   cf.dump
   cf.environment
   cf.example_field
   cf.flat
   cf.hash_array
   cf.implementation
   cf.inspect
   cf.pathjoin
   cf.pickle
   cf.relpath
   cf.unpickle

