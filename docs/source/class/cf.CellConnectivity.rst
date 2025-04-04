.. currentmodule:: cf
.. default-role:: obj

.. _cf-CellConnectivity:

cf.CellConnectivity
=====================

----

.. autoclass:: cf.CellConnectivity
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.dump
   ~cf.CellConnectivity.identity  
   ~cf.CellConnectivity.identities
   ~cf.CellConnectivity.inspect

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellConnectivity.construct_type

Topology
--------
 
.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.del_connectivity
   ~cf.CellConnectivity.get_connectivity
   ~cf.CellConnectivity.has_connectivity
   ~cf.CellConnectivity.set_connectivity
   ~cf.CellConnectivity.normalise

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellConnectivity.connectivity

Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.match_by_identity
   ~cf.CellConnectivity.match_by_naxes
   ~cf.CellConnectivity.match_by_ncvar
   ~cf.CellConnectivity.match_by_property
   ~cf.CellConnectivity.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.del_property
   ~cf.CellConnectivity.get_property
   ~cf.CellConnectivity.has_property
   ~cf.CellConnectivity.set_property
   ~cf.CellConnectivity.properties
   ~cf.CellConnectivity.clear_properties
   ~cf.CellConnectivity.del_properties
   ~cf.CellConnectivity.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellConnectivity.add_offset
   ~cf.CellConnectivity.calendar
   ~cf.CellConnectivity.comment
   ~cf.CellConnectivity._FillValue
   ~cf.CellConnectivity.history
   ~cf.CellConnectivity.leap_month
   ~cf.CellConnectivity.leap_year
   ~cf.CellConnectivity.long_name
   ~cf.CellConnectivity.missing_value
   ~cf.CellConnectivity.month_lengths
   ~cf.CellConnectivity.scale_factor
   ~cf.CellConnectivity.standard_name
   ~cf.CellConnectivity.units
   ~cf.CellConnectivity.valid_max
   ~cf.CellConnectivity.valid_min
   ~cf.CellConnectivity.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.override_units
   ~cf.CellConnectivity.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellConnectivity.Units

Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellConnectivity.array
   ~cf.CellConnectivity.Data
   ~cf.CellConnectivity.data
   ~cf.CellConnectivity.datetime_array
   ~cf.CellConnectivity.datum
   ~cf.CellConnectivity.dtype
   ~cf.CellConnectivity.isscalar
   ~cf.CellConnectivity.ndim
   ~cf.CellConnectivity.shape
   ~cf.CellConnectivity.size
   ~cf.CellConnectivity.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.to_dask_array
   ~cf.CellConnectivity.__getitem__
   ~cf.CellConnectivity.del_data
   ~cf.CellConnectivity.get_data
   ~cf.CellConnectivity.has_data
   ~cf.CellConnectivity.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.flatten
   ~cf.CellConnectivity.flip
   ~cf.CellConnectivity.insert_dimension
   ~cf.CellConnectivity.roll
   ~cf.CellConnectivity.squeeze
   ~cf.CellConnectivity.swapaxes
   ~cf.CellConnectivity.transpose
   
.. rubric:: *Expanding the data*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.halo
   ~cf.CellConnectivity.pad_missing

.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.apply_masking
   ~cf.CellConnectivity.count
   ~cf.CellConnectivity.count_masked
   ~cf.CellConnectivity.fill_value
   ~cf.CellConnectivity.filled
   ~cf.CellConnectivity.masked_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellConnectivity.binary_mask
   ~cf.CellConnectivity.hardmask
   ~cf.CellConnectivity.mask

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.__setitem__
   ~cf.CellConnectivity.masked_invalid
   ~cf.CellConnectivity.subspace
   ~cf.CellConnectivity.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.CellConnectivity.rechunk
   ~cf.CellConnectivity.close
   ~cf.CellConnectivity.convert_reference_time
   ~cf.CellConnectivity.cyclic
   ~cf.CellConnectivity.period
   ~cf.CellConnectivity.iscyclic
   ~cf.CellConnectivity.isperiodic
   ~cf.CellConnectivity.get_original_filenames
   ~cf.CellConnectivity.has_bounds
   ~cf.CellConnectivity.persist

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.concatenate
   ~cf.CellConnectivity.copy
   ~cf.CellConnectivity.creation_commands
   ~cf.CellConnectivity.equals
   ~cf.CellConnectivity.to_memory
   ~cf.CellConnectivity.uncompress

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: attribute.rst

   ~cf.CellConnectivity.T
   ~cf.CellConnectivity.X
   ~cf.CellConnectivity.Y
   ~cf.CellConnectivity.Z
   ~cf.CellConnectivity.id

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical and hyperbolic functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.arccos
   ~cf.CellConnectivity.arccosh
   ~cf.CellConnectivity.arcsin
   ~cf.CellConnectivity.arcsinh
   ~cf.CellConnectivity.arctan
   .. ~cf.CellConnectivity.arctan2  [AT2]
   ~cf.CellConnectivity.arctanh
   ~cf.CellConnectivity.cos
   ~cf.CellConnectivity.cosh
   ~cf.CellConnectivity.sin
   ~cf.CellConnectivity.sinh
   ~cf.CellConnectivity.tan
   ~cf.CellConnectivity.tanh

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.ceil  
   ~cf.CellConnectivity.clip
   ~cf.CellConnectivity.floor
   ~cf.CellConnectivity.rint
   ~cf.CellConnectivity.round
   ~cf.CellConnectivity.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.max
   ~cf.CellConnectivity.mean
   ~cf.CellConnectivity.mid_range
   ~cf.CellConnectivity.min
   ~cf.CellConnectivity.range
   ~cf.CellConnectivity.sample_size
   ~cf.CellConnectivity.sum  
   ~cf.CellConnectivity.sd
   ~cf.CellConnectivity.var
   ~cf.CellConnectivity.standard_deviation
   ~cf.CellConnectivity.variance
   ~cf.CellConnectivity.maximum
   ~cf.CellConnectivity.minimum

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.exp
   ~cf.CellConnectivity.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellConnectivity.day
   ~cf.CellConnectivity.datetime_array
   ~cf.CellConnectivity.hour
   ~cf.CellConnectivity.minute
   ~cf.CellConnectivity.month
   ~cf.CellConnectivity.reference_datetime   
   ~cf.CellConnectivity.second
   ~cf.CellConnectivity.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.all
   ~cf.CellConnectivity.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.allclose
   ~cf.CellConnectivity.equals
   ~cf.CellConnectivity.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.nc_del_variable
   ~cf.CellConnectivity.nc_get_variable
   ~cf.CellConnectivity.nc_has_variable
   ~cf.CellConnectivity.nc_set_variable 
   ~cf.CellConnectivity.nc_clear_hdf5_chunksizes
   ~cf.CellConnectivity.nc_hdf5_chunksizes
   ~cf.CellConnectivity.nc_set_hdf5_chunksizes

Aggregation
-----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.CellConnectivity.file_directories
   ~cf.CellConnectivity.replace_directory

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.nc_del_variable
   ~cf.CellConnectivity.nc_get_variable
   ~cf.CellConnectivity.nc_has_variable
   ~cf.CellConnectivity.nc_set_variable

Groups
^^^^^^

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.nc_variable_groups
   ~cf.CellConnectivity.nc_clear_variable_groups
   ~cf.CellConnectivity.nc_set_variable_groups

HDF5 chunks
^^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.nc_hdf5_chunksizes
   ~cf.CellConnectivity.nc_set_hdf5_chunksizes
   ~cf.CellConnectivity.nc_clear_hdf5_chunksizes

Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.CellConnectivity.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellConnectivity.dtarray
   
Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.__deepcopy__
   ~cf.CellConnectivity.__getitem__
   ~cf.CellConnectivity.__repr__
   ~cf.CellConnectivity.__str__

Docstring substitutions
-----------------------                   
                                          
.. rubric:: Methods                       
                                          
.. autosummary::                          
   :nosignatures:                         
   :toctree: ../method/                   
   :template: method.rst                  
                                          
   ~cf.CellConnectivity._docstring_special_substitutions
   ~cf.CellConnectivity._docstring_substitutions        
   ~cf.CellConnectivity._docstring_package_depth        
   ~cf.CellConnectivity._docstring_method_exclusions    
	 
Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellConnectivity.add_file_location
   ~cf.CellConnectivity.asdatetime
   ~cf.CellConnectivity.asreftime
   ~cf.CellConnectivity.attributes
   ~cf.CellConnectivity.cfa_clear_file_substitutions
   ~cf.CellConnectivity.cfa_del_file_substitution
   ~cf.CellConnectivity.cfa_file_substitutions
   ~cf.CellConnectivity.cfa_update_file_substitutions
   ~cf.CellConnectivity.chunk
   ~cf.CellConnectivity.del_file_location
   ~cf.CellConnectivity.delprop
   ~cf.CellConnectivity.dtvarray
   ~cf.CellConnectivity.expand_dims
   ~cf.CellConnectivity.get_filenames
   ~cf.CellConnectivity.file_locations
   ~cf.CellConnectivity.getprop
   ~cf.CellConnectivity.hasbounds
   ~cf.CellConnectivity.hasdata
   ~cf.CellConnectivity.hasprop
   ~cf.CellConnectivity.insert_data
   ~cf.CellConnectivity.isauxiliary
   ~cf.CellConnectivity.isdimension
   ~cf.CellConnectivity.isdomainancillary
   ~cf.CellConnectivity.isfieldancillary
   ~cf.CellConnectivity.ismeasure
   ~cf.CellConnectivity.mask_invalid
   ~cf.CellConnectivity.name
   ~cf.CellConnectivity.remove_data
   ~cf.CellConnectivity.select
   ~cf.CellConnectivity.setprop
   ~cf.CellConnectivity.unsafe_array
