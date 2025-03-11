.. currentmodule:: cf
.. default-role:: obj

cf.List
========

----

.. autoclass:: cf.List
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::   
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.dump
   ~cf.List.identity  
   ~cf.List.identities
   ~cf.List.inspect
   
.. rubric:: Attributes
	    
.. autosummary::   
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.List.id
   
Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.match_by_identity
   ~cf.List.match_by_naxes
   ~cf.List.match_by_ncvar
   ~cf.List.match_by_property
   ~cf.List.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.del_property
   ~cf.List.get_property
   ~cf.List.has_property
   ~cf.List.set_property
   ~cf.List.properties
   ~cf.List.clear_properties
   ~cf.List.del_properties
   ~cf.List.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.List.add_offset
   ~cf.List.calendar
   ~cf.List.comment
   ~cf.List._FillValue
   ~cf.List.history
   ~cf.List.leap_month
   ~cf.List.leap_year
   ~cf.List.long_name
   ~cf.List.missing_value
   ~cf.List.month_lengths
   ~cf.List.scale_factor
   ~cf.List.standard_name
   ~cf.List.units
   ~cf.List.valid_max
   ~cf.List.valid_min
   ~cf.List.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.override_units
   ~cf.List.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.List.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.List.array
   ~cf.List.data
   ~cf.List.datetime_array
   ~cf.List.datum
   ~cf.List.dtype
   ~cf.List.isscalar
   ~cf.List.ndim
   ~cf.List.shape
   ~cf.List.size
   ~cf.List.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.to_dask_array
   ~cf.List.__getitem__
   ~cf.List.del_data
   ~cf.List.get_data
   ~cf.List.has_data
   ~cf.List.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.flatten
   ~cf.List.flip
   ~cf.List.insert_dimension
   ~cf.List.roll
   ~cf.List.squeeze
   ~cf.List.swapaxes
   ~cf.List.transpose
   
.. rubric:: *Expanding the data*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.halo
   ~cf.List.pad_missing

.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.apply_masking
   ~cf.List.count
   ~cf.List.count_masked
   ~cf.List.fill_value
   ~cf.List.filled
   ~cf.List.masked_invalid
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.List.binary_mask
   ~cf.List.hardmask
   ~cf.List.mask

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__setitem__
   ~cf.List.masked_invalid
   ~cf.List.subspace
   ~cf.List.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.List.rechunk
   ~cf.List.close
   ~cf.List.convert_reference_time
   ~cf.List.cyclic
   ~cf.List.period
   ~cf.List.iscyclic
   ~cf.List.isperiodic
   ~cf.List.get_original_filenames
   ~cf.List.has_bounds
   ~cf.List.persist

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.concatenate
   ~cf.List.copy
   ~cf.List.creation_commands
   ~cf.List.equals
   ~cf.List.to_memory
   ~cf.List.uncompress

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: attribute.rst

   ~cf.List.T
   ~cf.List.X
   ~cf.List.Y
   ~cf.List.Z
   ~cf.List.id

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical and hyperbolic functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.arccos
   ~cf.List.arccosh
   ~cf.List.arcsin
   ~cf.List.arcsinh
   ~cf.List.arctan
   .. ~cf.List.arctan2  [AT2]
   ~cf.List.arctanh
   ~cf.List.cos
   ~cf.List.cosh
   ~cf.List.sin
   ~cf.List.sinh
   ~cf.List.tan
   ~cf.List.tanh

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.ceil  
   ~cf.List.clip
   ~cf.List.floor
   ~cf.List.rint
   ~cf.List.round
   ~cf.List.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.max
   ~cf.List.mean
   ~cf.List.mid_range
   ~cf.List.min
   ~cf.List.range
   ~cf.List.sample_size
   ~cf.List.sum  
   ~cf.List.sd
   ~cf.List.var
   ~cf.List.standard_deviation
   ~cf.List.variance
   ~cf.List.maximum
   ~cf.List.minimum

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.exp
   ~cf.List.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.List.day
   ~cf.List.datetime_array
   ~cf.List.hour
   ~cf.List.minute
   ~cf.List.month
   ~cf.List.reference_datetime   
   ~cf.List.second
   ~cf.List.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.all
   ~cf.List.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.allclose
   ~cf.List.equals
   ~cf.List.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.nc_del_variable
   ~cf.List.nc_get_variable
   ~cf.List.nc_has_variable
   ~cf.List.nc_set_variable
   ~cf.List.nc_clear_hdf5_chunksizes
   ~cf.List.nc_hdf5_chunksizes
   ~cf.List.nc_set_hdf5_chunksizes

Aggregation
-----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.List.file_directories
   ~cf.List.replace_directory

Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.List.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.List.dtarray
   
Arithmetic and comparison operations
------------------------------------

Arithmetic, bitwise and comparison operations are defined as
element-wise operations on the data, which yield a new construct or,
for augmented assignments, modify the construct's data in-place.

.. rubric:: Comparison operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__lt__
   ~cf.List.__le__
   ~cf.List.__eq__
   ~cf.List.__ne__
   ~cf.List.__gt__
   ~cf.List.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__add__     
   ~cf.List.__sub__     
   ~cf.List.__mul__     
   ~cf.List.__div__     
   ~cf.List.__truediv__ 
   ~cf.List.__floordiv__
   ~cf.List.__pow__     
   ~cf.List.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__radd__     
   ~cf.List.__rsub__     
   ~cf.List.__rmul__     
   ~cf.List.__rdiv__     
   ~cf.List.__rtruediv__ 
   ~cf.List.__rfloordiv__
   ~cf.List.__rpow__   
   ~cf.List.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__iadd__ 
   ~cf.List.__isub__ 
   ~cf.List.__imul__ 
   ~cf.List.__idiv__ 
   ~cf.List.__itruediv__
   ~cf.List.__ifloordiv__
   ~cf.List.__ipow__ 
   ~cf.List.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__neg__    
   ~cf.List.__pos__    
   ~cf.List.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__and__     
   ~cf.List.__or__
   ~cf.List.__xor__     
   ~cf.List.__lshift__
   ~cf.List.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__rand__     
   ~cf.List.__ror__
   ~cf.List.__rxor__     
   ~cf.List.__rlshift__
   ~cf.List.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__iand__     
   ~cf.List.__ior__
   ~cf.List.__ixor__     
   ~cf.List.__ilshift__
   ~cf.List.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__invert__ 

Groups
^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.nc_variable_groups
   ~cf.List.nc_clear_variable_groups
   ~cf.List.nc_set_variable_groups

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.List.__contains__
   ~cf.List.__deepcopy__
   ~cf.List.__getitem__
   ~cf.List.__repr__
   ~cf.List.__setitem__
   ~cf.List.__str__
   ~cf.List.__array__
   ~cf.List.__data__
   ~cf.List.__query_isclose__

Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst


   ~cf.List.add_file_location
   ~cf.List.asdatetime
   ~cf.List.asreftime
   ~cf.List.attributes
   ~cf.List.cfa_clear_file_substitutions
   ~cf.List.cfa_del_file_substitution
   ~cf.List.cfa_file_substitutions
   ~cf.List.cfa_update_file_substitutions
   ~cf.List.chunk
   ~cf.List.Data
   ~cf.List.del_file_location
   ~cf.List.delprop
   ~cf.List.dtvarray
   ~cf.List.expand_dims
   ~cf.List.file_locations
   ~cf.List.get_filenames
   ~cf.List.getprop
   ~cf.List.hasbounds
   ~cf.List.hasdata
   ~cf.List.hasprop
   ~cf.List.insert_data
   ~cf.List.isauxiliary
   ~cf.List.isdimension
   ~cf.List.isdomainancillary
   ~cf.List.isfieldancillary
   ~cf.List.ismeasure
   ~cf.List.mask_invalid
   ~cf.List.name
   ~cf.List.remove_data
   ~cf.List.select
   ~cf.List.setprop
   ~cf.List.unsafe_array
