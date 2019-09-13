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
   cf.load_stash2standard_name 

Aggregation
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   cf.aggregate

.. _functions-mathematical-operations:

Mathematical operations
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   cf.relative_vorticity
   
Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: function/
   :template: function.rst

   cf.CF
   cf.ATOL
   cf.RTOLa

Condition constructors
----------------------

.. rubric:: General conditions
   
.. autosummary::
   :nosignatures:
   :toctree: generated/
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
   :toctree: generated/
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
   :toctree: generated/
   :template: function.rst

   cf.constains
   cf.cellsize
   cf.cellgt
   cf.cellge
   cf.cellle
   cf.celllt
   cf.cellwi
   cf.cellwo

Date-time
---------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   cf.dt
   cf.Y
   cf.M
   cf.D
   cf.h
   cf.m
   cf.s

Constants
---------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   cf.ATOL
   cf.RTOL
   cf.TEMPDIR
   cf.CHUNKSIZE
   cf.FM_THRESHOLD
   cf.MINNCFM
   cf.OF_FRACTION
   cf.REGRID_LOGGING

  
Miscellaneous
-------------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   cf.abspath
   cf.close_files
   cf.close_one_file
   cf.default_fillvals
   cf.dirname
   cf.dump
   cf.environment
   cf.flat
   cf.implementation
   cf.open_files
   cf.open_files_threshold_exceeded
   cf.pathjoin
   cf.pickle
   cf.relpath
   cf.unpickle
