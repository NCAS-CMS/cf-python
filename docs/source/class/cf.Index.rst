.. Currentmodule:: cf
.. default-role:: obj

cf.Index
========

----

.. autoclass:: cf.Index
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.dump
   ~cf.Index.identity  
   ~cf.Index.identities
   ~cf.Index.inspect

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Index.id
   
Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.match_by_identity
   ~cf.Index.match_by_naxes
   ~cf.Index.match_by_ncvar
   ~cf.Index.match_by_property
   ~cf.Index.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.del_property
   ~cf.Index.get_property
   ~cf.Index.has_property
   ~cf.Index.set_property
   ~cf.Index.properties
   ~cf.Index.clear_properties
   ~cf.Index.del_properties
   ~cf.Index.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Index.add_offset
   ~cf.Index.calendar
   ~cf.Index.comment
   ~cf.Index._FillValue
   ~cf.Index.history
   ~cf.Index.leap_month
   ~cf.Index.leap_year
   ~cf.Index.long_name
   ~cf.Index.missing_value
   ~cf.Index.month_lengths
   ~cf.Index.scale_factor
   ~cf.Index.standard_name
   ~cf.Index.units
   ~cf.Index.valid_max
   ~cf.Index.valid_min
   ~cf.Index.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.override_units
   ~cf.Index.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Index.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Index.array
   ~cf.Index.data
   ~cf.Index.datetime_array
   ~cf.Index.datum
   ~cf.Index.dtype
   ~cf.Index.isscalar
   ~cf.Index.ndim
   ~cf.Index.shape
   ~cf.Index.size
   ~cf.Index.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.to_dask_array
   ~cf.Index.__getitem__
   ~cf.Index.del_data
   ~cf.Index.get_data
   ~cf.Index.has_data
   ~cf.Index.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.flatten
   ~cf.Index.flip
   ~cf.Index.insert_dimension
   ~cf.Index.roll
   ~cf.Index.squeeze
   ~cf.Index.swapaxes
   ~cf.Index.transpose
   
.. rubric:: *Expanding the data*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.halo
   ~cf.Index.pad_missing

.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.apply_masking
   ~cf.Index.count
   ~cf.Index.count_masked
   ~cf.Index.fill_value
   ~cf.Index.filled
   ~cf.Index.masked_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Index.binary_mask
   ~cf.Index.hardmask
   ~cf.Index.mask

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__setitem__
   ~cf.Index.masked_invalid
   ~cf.Index.subspace
   ~cf.Index.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.Index.rechunk
   ~cf.Index.close
   ~cf.Index.convert_reference_time
   ~cf.Index.cyclic
   ~cf.Index.period
   ~cf.Index.iscyclic
   ~cf.Index.isperiodic
   ~cf.Index.get_original_filenames
   ~cf.Index.has_bounds
   ~cf.Index.persist

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.concatenate
   ~cf.Index.copy
   ~cf.Index.creation_commands
   ~cf.Index.equals
   ~cf.Index.to_memory
   ~cf.Index.uncompress

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: attribute.rst

   ~cf.Index.T
   ~cf.Index.X
   ~cf.Index.Y
   ~cf.Index.Z
   ~cf.Index.id

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical and hyperbolic functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.arccos
   ~cf.Index.arccosh
   ~cf.Index.arcsin
   ~cf.Index.arcsinh
   ~cf.Index.arctan
   .. ~cf.Index.arctan2  [AT2]
   ~cf.Index.arctanh
   ~cf.Index.cos
   ~cf.Index.cosh
   ~cf.Index.sin
   ~cf.Index.sinh
   ~cf.Index.tan
   ~cf.Index.tanh

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.ceil  
   ~cf.Index.clip
   ~cf.Index.floor
   ~cf.Index.rint
   ~cf.Index.round
   ~cf.Index.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.max
   ~cf.Index.mean
   ~cf.Index.mid_range
   ~cf.Index.min
   ~cf.Index.range
   ~cf.Index.sample_size
   ~cf.Index.sum  
   ~cf.Index.sd
   ~cf.Index.var
   ~cf.Index.standard_deviation
   ~cf.Index.variance
   ~cf.Index.maximum
   ~cf.Index.minimum

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.exp
   ~cf.Index.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Index.day
   ~cf.Index.datetime_array
   ~cf.Index.hour
   ~cf.Index.minute
   ~cf.Index.month
   ~cf.Index.reference_datetime   
   ~cf.Index.second
   ~cf.Index.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.all
   ~cf.Index.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.allclose
   ~cf.Index.equals
   ~cf.Index.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.nc_del_variable
   ~cf.Index.nc_get_variable
   ~cf.Index.nc_has_variable
   ~cf.Index.nc_set_variable 
   ~cf.Index.nc_del_dimension
   ~cf.Index.nc_get_dimension
   ~cf.Index.nc_has_dimension
   ~cf.Index.nc_set_dimension
   ~cf.Index.nc_del_sample_dimension
   ~cf.Index.nc_get_sample_dimension
   ~cf.Index.nc_has_sample_dimension
   ~cf.Index.nc_set_sample_dimension
   ~cf.Index.nc_clear_hdf5_chunksizes
   ~cf.Index.nc_hdf5_chunksizes
   ~cf.Index.nc_set_hdf5_chunksizes

Aggregation
-----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Index.file_directories
   ~cf.Index.replace_directory

Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.Index.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Index.dtarray
   
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

   ~cf.Index.__lt__
   ~cf.Index.__le__
   ~cf.Index.__eq__
   ~cf.Index.__ne__
   ~cf.Index.__gt__
   ~cf.Index.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__add__     
   ~cf.Index.__sub__     
   ~cf.Index.__mul__     
   ~cf.Index.__div__     
   ~cf.Index.__truediv__ 
   ~cf.Index.__floordiv__
   ~cf.Index.__pow__     
   ~cf.Index.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__radd__     
   ~cf.Index.__rsub__     
   ~cf.Index.__rmul__     
   ~cf.Index.__rdiv__     
   ~cf.Index.__rtruediv__ 
   ~cf.Index.__rfloordiv__
   ~cf.Index.__rpow__   
   ~cf.Index.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__iadd__ 
   ~cf.Index.__isub__ 
   ~cf.Index.__imul__ 
   ~cf.Index.__idiv__ 
   ~cf.Index.__itruediv__
   ~cf.Index.__ifloordiv__
   ~cf.Index.__ipow__ 
   ~cf.Index.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__neg__    
   ~cf.Index.__pos__    
   ~cf.Index.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__and__     
   ~cf.Index.__or__
   ~cf.Index.__xor__     
   ~cf.Index.__lshift__
   ~cf.Index.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__rand__     
   ~cf.Index.__ror__
   ~cf.Index.__rxor__     
   ~cf.Index.__rlshift__
   ~cf.Index.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__iand__     
   ~cf.Index.__ior__
   ~cf.Index.__ixor__     
   ~cf.Index.__ilshift__
   ~cf.Index.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__invert__ 

Groups
^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.nc_variable_groups
   ~cf.Index.nc_clear_variable_groups
   ~cf.Index.nc_set_variable_groups
   ~cf.Index.nc_dimension_groups
   ~cf.Index.nc_clear_dimension_groups
   ~cf.Index.nc_set_dimension_groups
   ~cf.Index.nc_clear_sample_dimension_groups
   ~cf.Index.nc_sample_dimension_groups
   ~cf.Index.nc_set_sample_dimension_groups

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Index.__contains__
   ~cf.Index.__deepcopy__
   ~cf.Index.__getitem__
   ~cf.Index.__repr__
   ~cf.Index.__setitem__
   ~cf.Index.__str__
   ~cf.Index.__array__
   ~cf.Index.__data__
   ~cf.Index.__query_isclose__

Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst


   ~cf.Index.add_file_location
   ~cf.Index.asdatetime
   ~cf.Index.asreftime
   ~cf.Index.attributes
   ~cf.Index.cfa_clear_file_substitutions
   ~cf.Index.cfa_del_file_substitution
   ~cf.Index.cfa_file_substitutions
   ~cf.Index.cfa_update_file_substitutions
   ~cf.Index.chunk
   ~cf.Index.Data
   ~cf.Index.del_file_location
   ~cf.Index.delprop
   ~cf.Index.dtvarray
   ~cf.Index.expand_dims
   ~cf.Index.file_locations   
   ~cf.Index.get_filenames
   ~cf.Index.getprop
   ~cf.Index.hasbounds
   ~cf.Index.hasdata
   ~cf.Index.hasprop
   ~cf.Index.insert_data
   ~cf.Index.isauxiliary
   ~cf.Index.isdimension
   ~cf.Index.isdomainancillary
   ~cf.Index.isfieldancillary
   ~cf.Index.ismeasure
   ~cf.Index.mask_invalid
   ~cf.Index.name
   ~cf.Index.remove_data
   ~cf.Index.select
   ~cf.Index.setprop
   ~cf.Index.unsafe_array
