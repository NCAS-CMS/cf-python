.. currentmodule:: cf
.. default-role:: obj

cf.Count
========

----

.. autoclass:: cf.Count
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.dump
   ~cf.Count.identity  
   ~cf.Count.identities
   ~cf.Count.inspect

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Count.id
   
Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.match_by_identity
   ~cf.Count.match_by_naxes
   ~cf.Count.match_by_ncvar
   ~cf.Count.match_by_property
   ~cf.Count.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.del_property
   ~cf.Count.get_property
   ~cf.Count.has_property
   ~cf.Count.set_property
   ~cf.Count.properties
   ~cf.Count.clear_properties
   ~cf.Count.del_properties
   ~cf.Count.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Count.add_offset
   ~cf.Count.calendar
   ~cf.Count.comment
   ~cf.Count._FillValue
   ~cf.Count.history
   ~cf.Count.leap_month
   ~cf.Count.leap_year
   ~cf.Count.long_name
   ~cf.Count.missing_value
   ~cf.Count.month_lengths
   ~cf.Count.scale_factor
   ~cf.Count.standard_name
   ~cf.Count.units
   ~cf.Count.valid_max
   ~cf.Count.valid_min
   ~cf.Count.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.override_units
   ~cf.Count.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Count.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Count.array
   ~cf.Count.to_dask_array
   ~cf.Count.data
   ~cf.Count.datetime_array
   ~cf.Count.datum
   ~cf.Count.dtype
   ~cf.Count.isscalar
   ~cf.Count.ndim
   ~cf.Count.shape
   ~cf.Count.size
   ~cf.Count.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__getitem__
   ~cf.Count.del_data
   ~cf.Count.get_data
   ~cf.Count.has_data
   ~cf.Count.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.flatten
   ~cf.Count.flip
   ~cf.Count.insert_dimension
   ~cf.Count.roll
   ~cf.Count.squeeze
   ~cf.Count.swapaxes
   ~cf.Count.transpose
   
.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.apply_masking
   ~cf.Count.count
   ~cf.Count.count_masked
   ~cf.Count.fill_value
   ~cf.Count.masked_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Count.binary_mask
   ~cf.Count.hardmask
   ~cf.Count.mask

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__setitem__
   ~cf.Count.masked_invalid
   ~cf.Count.subspace
   ~cf.Count.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.Count.rechunk
   ~cf.Count.close
   ~cf.Count.convert_reference_time
   ~cf.Count.cyclic
   ~cf.Count.period
   ~cf.Count.iscyclic
   ~cf.Count.isperiodic
   ~cf.Count.get_original_filenames
   ~cf.Count.has_bounds
   ~cf.Count.persist

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.concatenate
   ~cf.Count.copy
   ~cf.Count.creation_commands
   ~cf.Count.equals
   ~cf.Count.to_memory
   ~cf.Count.uncompress

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: attribute.rst

   ~cf.Count.T
   ~cf.Count.X
   ~cf.Count.Y
   ~cf.Count.Z
   ~cf.Count.id

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical and hyperbolic functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.arccos
   ~cf.Count.arccosh
   ~cf.Count.arcsin
   ~cf.Count.arcsinh
   ~cf.Count.arctan
   .. ~cf.Count.arctan2  [AT2]
   ~cf.Count.arctanh
   ~cf.Count.cos
   ~cf.Count.cosh
   ~cf.Count.sin
   ~cf.Count.sinh
   ~cf.Count.tan
   ~cf.Count.tanh

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.ceil  
   ~cf.Count.clip
   ~cf.Count.floor
   ~cf.Count.rint
   ~cf.Count.round
   ~cf.Count.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.max
   ~cf.Count.mean
   ~cf.Count.mid_range
   ~cf.Count.min
   ~cf.Count.range
   ~cf.Count.sample_size
   ~cf.Count.sum  
   ~cf.Count.sd
   ~cf.Count.var
   ~cf.Count.standard_deviation
   ~cf.Count.variance
   ~cf.Count.maximum
   ~cf.Count.minimum

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.exp
   ~cf.Count.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Count.day
   ~cf.Count.datetime_array
   ~cf.Count.hour
   ~cf.Count.minute
   ~cf.Count.month
   ~cf.Count.reference_datetime   
   ~cf.Count.second
   ~cf.Count.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.all
   ~cf.Count.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.allclose
   ~cf.Count.equals
   ~cf.Count.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.nc_del_variable
   ~cf.Count.nc_get_variable
   ~cf.Count.nc_has_variable
   ~cf.Count.nc_set_variable 
   ~cf.Count.nc_del_dimension
   ~cf.Count.nc_get_dimension
   ~cf.Count.nc_has_dimension
   ~cf.Count.nc_set_dimension
   ~cf.Count.nc_del_sample_dimension
   ~cf.Count.nc_get_sample_dimension
   ~cf.Count.nc_has_sample_dimension
   ~cf.Count.nc_set_sample_dimension

CFA
---

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.add_file_location
   ~cf.Count.cfa_clear_file_substitutions
   ~cf.Count.cfa_del_file_substitution
   ~cf.Count.cfa_file_substitutions
   ~cf.Count.cfa_update_file_substitutions
   ~cf.Count.del_file_location
   ~cf.Count.file_locations

Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.Count.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Count.dtarray
   
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

   ~cf.Count.__lt__
   ~cf.Count.__le__
   ~cf.Count.__eq__
   ~cf.Count.__ne__
   ~cf.Count.__gt__
   ~cf.Count.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__add__     
   ~cf.Count.__sub__     
   ~cf.Count.__mul__     
   ~cf.Count.__div__     
   ~cf.Count.__truediv__ 
   ~cf.Count.__floordiv__
   ~cf.Count.__pow__     
   ~cf.Count.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__radd__     
   ~cf.Count.__rsub__     
   ~cf.Count.__rmul__     
   ~cf.Count.__rdiv__     
   ~cf.Count.__rtruediv__ 
   ~cf.Count.__rfloordiv__
   ~cf.Count.__rpow__   
   ~cf.Count.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__iadd__ 
   ~cf.Count.__isub__ 
   ~cf.Count.__imul__ 
   ~cf.Count.__idiv__ 
   ~cf.Count.__itruediv__
   ~cf.Count.__ifloordiv__
   ~cf.Count.__ipow__ 
   ~cf.Count.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__neg__    
   ~cf.Count.__pos__    
   ~cf.Count.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__and__     
   ~cf.Count.__or__
   ~cf.Count.__xor__     
   ~cf.Count.__lshift__
   ~cf.Count.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__rand__     
   ~cf.Count.__ror__
   ~cf.Count.__rxor__     
   ~cf.Count.__rlshift__
   ~cf.Count.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__iand__     
   ~cf.Count.__ior__
   ~cf.Count.__ixor__     
   ~cf.Count.__ilshift__
   ~cf.Count.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__invert__ 

Groups
^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.nc_variable_groups
   ~cf.Count.nc_clear_variable_groups
   ~cf.Count.nc_set_variable_groups
   ~cf.Count.nc_dimension_groups
   ~cf.Count.nc_clear_dimension_groups
   ~cf.Count.nc_set_dimension_groups
   ~cf.Count.nc_clear_sample_dimension_groups
   ~cf.Count.nc_sample_dimension_groups
   ~cf.Count.nc_set_sample_dimension_groups

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.__contains__
   ~cf.Count.__deepcopy__
   ~cf.Count.__getitem__
   ~cf.Count.__repr__
   ~cf.Count.__setitem__
   ~cf.Count.__str__
   ~cf.Count.__array__
   ~cf.Count.__data__
   ~cf.Count.__query_isclose__
   
Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Count.asdatetime
   ~cf.Count.asreftime
   ~cf.Count.attributes
   ~cf.Count.chunk
   ~cf.Count.Data
   ~cf.Count.delprop
   ~cf.Count.dtvarray
   ~cf.Count.expand_dims
   ~cf.Count.get_filenames
   ~cf.Count.getprop
   ~cf.Count.halo
   ~cf.Count.hasbounds
   ~cf.Count.hasdata
   ~cf.Count.hasprop
   ~cf.Count.insert_data
   ~cf.Count.isauxiliary
   ~cf.Count.isdimension
   ~cf.Count.isdomainancillary
   ~cf.Count.isfieldancillary
   ~cf.Count.ismeasure
   ~cf.Count.mask_invalid
   ~cf.Count.name
   ~cf.Count.remove_data
   ~cf.Count.select
   ~cf.Count.setprop
   ~cf.Count.unsafe_array
