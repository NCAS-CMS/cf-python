.. currentmodule:: cf
.. default-role:: obj

.. _cf-DomainTopology:

cf.DomainTopology
===================

----

.. autoclass:: cf.DomainTopology
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.dump
   ~cf.DomainTopology.identity  
   ~cf.DomainTopology.identities
   ~cf.DomainTopology.inspect
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainTopology.construct_type
   ~cf.DomainTopology.id

Topology
--------
 
.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.del_cell
   ~cf.DomainTopology.get_cell
   ~cf.DomainTopology.has_cell
   ~cf.DomainTopology.set_cell
   ~cf.DomainTopology.normalise
 
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainTopology.cell

Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.match_by_identity
   ~cf.DomainTopology.match_by_naxes
   ~cf.DomainTopology.match_by_ncvar
   ~cf.DomainTopology.match_by_property
   ~cf.DomainTopology.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.del_property
   ~cf.DomainTopology.get_property
   ~cf.DomainTopology.has_property
   ~cf.DomainTopology.set_property
   ~cf.DomainTopology.properties
   ~cf.DomainTopology.clear_properties
   ~cf.DomainTopology.del_properties
   ~cf.DomainTopology.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainTopology.add_offset
   ~cf.DomainTopology.calendar
   ~cf.DomainTopology.comment
   ~cf.DomainTopology._FillValue
   ~cf.DomainTopology.history
   ~cf.DomainTopology.leap_month
   ~cf.DomainTopology.leap_year
   ~cf.DomainTopology.long_name
   ~cf.DomainTopology.missing_value
   ~cf.DomainTopology.month_lengths
   ~cf.DomainTopology.scale_factor
   ~cf.DomainTopology.standard_name
   ~cf.DomainTopology.units
   ~cf.DomainTopology.valid_max
   ~cf.DomainTopology.valid_min
   ~cf.DomainTopology.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.override_units
   ~cf.DomainTopology.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainTopology.Units

Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainTopology.array
   ~cf.DomainTopology.Data
   ~cf.DomainTopology.data
   ~cf.DomainTopology.datetime_array
   ~cf.DomainTopology.datum
   ~cf.DomainTopology.dtype
   ~cf.DomainTopology.isscalar
   ~cf.DomainTopology.ndim
   ~cf.DomainTopology.shape
   ~cf.DomainTopology.size
   ~cf.DomainTopology.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.to_dask_array
   ~cf.DomainTopology.__getitem__
   ~cf.DomainTopology.del_data
   ~cf.DomainTopology.get_data
   ~cf.DomainTopology.has_data
   ~cf.DomainTopology.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.flatten
   ~cf.DomainTopology.flip
   ~cf.DomainTopology.insert_dimension
   ~cf.DomainTopology.roll
   ~cf.DomainTopology.squeeze
   ~cf.DomainTopology.swapaxes
   ~cf.DomainTopology.transpose
   
.. rubric:: *Expanding the data*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.halo
   ~cf.DomainTopology.pad_missing

.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.apply_masking
   ~cf.DomainTopology.count
   ~cf.DomainTopology.count_masked
   ~cf.DomainTopology.fill_value
   ~cf.DomainTopology.filled
   ~cf.DomainTopology.masked_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainTopology.binary_mask
   ~cf.DomainTopology.hardmask
   ~cf.DomainTopology.mask

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.__setitem__
   ~cf.DomainTopology.masked_invalid
   ~cf.DomainTopology.subspace
   ~cf.DomainTopology.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.DomainTopology.rechunk
   ~cf.DomainTopology.close
   ~cf.DomainTopology.convert_reference_time
   ~cf.DomainTopology.cyclic
   ~cf.DomainTopology.period
   ~cf.DomainTopology.iscyclic
   ~cf.DomainTopology.isperiodic
   ~cf.DomainTopology.get_original_filenames
   ~cf.DomainTopology.has_bounds
   ~cf.DomainTopology.persist

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.concatenate
   ~cf.DomainTopology.copy
   ~cf.DomainTopology.creation_commands
   ~cf.DomainTopology.equals
   ~cf.DomainTopology.to_memory
   ~cf.DomainTopology.uncompress

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: attribute.rst

   ~cf.DomainTopology.T
   ~cf.DomainTopology.X
   ~cf.DomainTopology.Y
   ~cf.DomainTopology.Z
   ~cf.DomainTopology.id

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical and hyperbolic functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.arccos
   ~cf.DomainTopology.arccosh
   ~cf.DomainTopology.arcsin
   ~cf.DomainTopology.arcsinh
   ~cf.DomainTopology.arctan
   .. ~cf.DomainTopology.arctan2  [AT2]
   ~cf.DomainTopology.arctanh
   ~cf.DomainTopology.cos
   ~cf.DomainTopology.cosh
   ~cf.DomainTopology.sin
   ~cf.DomainTopology.sinh
   ~cf.DomainTopology.tan
   ~cf.DomainTopology.tanh

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.ceil  
   ~cf.DomainTopology.clip
   ~cf.DomainTopology.floor
   ~cf.DomainTopology.rint
   ~cf.DomainTopology.round
   ~cf.DomainTopology.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.max
   ~cf.DomainTopology.mean
   ~cf.DomainTopology.mid_range
   ~cf.DomainTopology.min
   ~cf.DomainTopology.range
   ~cf.DomainTopology.sample_size
   ~cf.DomainTopology.sum  
   ~cf.DomainTopology.sd
   ~cf.DomainTopology.var
   ~cf.DomainTopology.standard_deviation
   ~cf.DomainTopology.variance
   ~cf.DomainTopology.maximum
   ~cf.DomainTopology.minimum

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.exp
   ~cf.DomainTopology.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainTopology.day
   ~cf.DomainTopology.datetime_array
   ~cf.DomainTopology.hour
   ~cf.DomainTopology.minute
   ~cf.DomainTopology.month
   ~cf.DomainTopology.reference_datetime   
   ~cf.DomainTopology.second
   ~cf.DomainTopology.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.all
   ~cf.DomainTopology.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.allclose
   ~cf.DomainTopology.equals
   ~cf.DomainTopology.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.nc_del_variable
   ~cf.DomainTopology.nc_get_variable
   ~cf.DomainTopology.nc_has_variable
   ~cf.DomainTopology.nc_set_variable 
   ~cf.DomainTopology.nc_clear_hdf5_chunksizes
   ~cf.DomainTopology.nc_hdf5_chunksizes
   ~cf.DomainTopology.nc_set_hdf5_chunksizes

Aggregation
-----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.DomainTopology.file_directories
   ~cf.DomainTopology.replace_directory

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.nc_del_variable
   ~cf.DomainTopology.nc_get_variable
   ~cf.DomainTopology.nc_has_variable
   ~cf.DomainTopology.nc_set_variable

Groups
^^^^^^

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.nc_variable_groups
   ~cf.DomainTopology.nc_clear_variable_groups
   ~cf.DomainTopology.nc_set_variable_groups

HDF5 chunks
^^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.nc_hdf5_chunksizes
   ~cf.DomainTopology.nc_set_hdf5_chunksizes
   ~cf.DomainTopology.nc_clear_hdf5_chunksizes

Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.DomainTopology.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainTopology.dtarray
   
Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.__deepcopy__
   ~cf.DomainTopology.__getitem__
   ~cf.DomainTopology.__repr__
   ~cf.DomainTopology.__str__

Docstring substitutions
-----------------------                   
                                          
.. rubric:: Methods                       
                                          
.. autosummary::                          
   :nosignatures:                         
   :toctree: ../method/                   
   :template: method.rst                  
                                          
   ~cf.DomainTopology._docstring_special_substitutions
   ~cf.DomainTopology._docstring_substitutions        
   ~cf.DomainTopology._docstring_package_depth        
   ~cf.DomainTopology._docstring_method_exclusions    
	 
Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainTopology.add_file_location
   ~cf.DomainTopology.asdatetime
   ~cf.DomainTopology.asreftime
   ~cf.DomainTopology.attributes
   ~cf.DomainTopology.cfa_clear_file_substitutions
   ~cf.DomainTopology.cfa_del_file_substitution
   ~cf.DomainTopology.cfa_file_substitutions
   ~cf.DomainTopology.cfa_update_file_substitutions
   ~cf.DomainTopology.chunk
   ~cf.DomainTopology.del_file_location
   ~cf.DomainTopology.delprop
   ~cf.DomainTopology.dtvarray
   ~cf.DomainTopology.expand_dims
   ~cf.DomainTopology.get_filenames
   ~cf.DomainTopology.file_locations
   ~cf.DomainTopology.getprop
   ~cf.DomainTopology.hasbounds
   ~cf.DomainTopology.hasdata
   ~cf.DomainTopology.hasprop
   ~cf.DomainTopology.insert_data
   ~cf.DomainTopology.isauxiliary
   ~cf.DomainTopology.isdimension
   ~cf.DomainTopology.isdomainancillary
   ~cf.DomainTopology.isfieldancillary
   ~cf.DomainTopology.ismeasure
   ~cf.DomainTopology.mask_invalid
   ~cf.DomainTopology.name
   ~cf.DomainTopology.remove_data
   ~cf.DomainTopology.select
   ~cf.DomainTopology.setprop
   ~cf.DomainTopology.unsafe_array
