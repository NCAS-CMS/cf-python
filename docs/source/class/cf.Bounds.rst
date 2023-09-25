.. currentmodule:: cf
.. default-role:: obj

cf.Bounds
=========

----

.. autoclass:: cf.Bounds
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.contiguous
   ~cf.Bounds.dump
   ~cf.Bounds.identity  
   ~cf.Bounds.identities
   ~cf.Bounds.inspect

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Bounds.id

Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.match_by_identity
   ~cf.Bounds.match_by_naxes
   ~cf.Bounds.match_by_ncvar
   ~cf.Bounds.match_by_property
   ~cf.Bounds.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.del_property
   ~cf.Bounds.get_property
   ~cf.Bounds.has_property
   ~cf.Bounds.set_property
   ~cf.Bounds.properties
   ~cf.Bounds.clear_properties
   ~cf.Bounds.del_properties
   ~cf.Bounds.set_properties
   ~cf.Bounds.inherited_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Bounds.add_offset
   ~cf.Bounds.axis
   ~cf.Bounds.calendar
   ~cf.Bounds.comment
   ~cf.Bounds._FillValue
   ~cf.Bounds.history
   ~cf.Bounds.leap_month
   ~cf.Bounds.leap_year
   ~cf.Bounds.long_name
   ~cf.Bounds.missing_value
   ~cf.Bounds.month_lengths
   ~cf.Bounds.positive
   ~cf.Bounds.scale_factor
   ~cf.Bounds.standard_name
   ~cf.Bounds.units
   ~cf.Bounds.valid_max
   ~cf.Bounds.valid_min
   ~cf.Bounds.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.override_units
   ~cf.Bounds.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Bounds.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Bounds.array
   ~cf.Bounds.Data
   ~cf.Bounds.data
   ~cf.Bounds.datetime_array
   ~cf.Bounds.datum
   ~cf.Bounds.dtype
   ~cf.Bounds.isscalar
   ~cf.Bounds.ndim
   ~cf.Bounds.shape
   ~cf.Bounds.size
   ~cf.Bounds.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.to_dask_array
   ~cf.Bounds.__getitem__
   ~cf.Bounds.del_data
   ~cf.Bounds.get_data
   ~cf.Bounds.has_data
   ~cf.Bounds.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.flatten
   ~cf.Bounds.flip
   ~cf.Bounds.insert_dimension
   ~cf.Bounds.roll
   ~cf.Bounds.squeeze
   ~cf.Bounds.swapaxes
   ~cf.Bounds.transpose
   
.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.apply_masking
   ~cf.Bounds.count
   ~cf.Bounds.count_masked
   ~cf.Bounds.fill_value
   ~cf.Bounds.masked_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Bounds.binary_mask
   ~cf.Bounds.hardmask
   ~cf.Bounds.mask

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__setitem__
   ~cf.Bounds.halo
   ~cf.Bounds.masked_invalid
   ~cf.Bounds.subspace
   ~cf.Bounds.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Bounds.autoperiod
   ~cf.Bounds.rechunk
   ~cf.Bounds.close
   ~cf.Bounds.convert_reference_time
   ~cf.Bounds.cyclic
   ~cf.Bounds.period
   ~cf.Bounds.iscyclic
   ~cf.Bounds.isperiodic
   ~cf.Bounds.get_original_filenames
   ~cf.Bounds.has_bounds
   ~cf.Bounds.persist

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.concatenate
   ~cf.Bounds.copy
   ~cf.Bounds.creation_commands
   ~cf.Bounds.equals
   ~cf.Bounds.to_memory
   ~cf.Bounds.uncompress

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: attribute.rst

   ~cf.Bounds.T
   ~cf.Bounds.X
   ~cf.Bounds.Y
   ~cf.Bounds.Z
   ~cf.Bounds.ctype
   ~cf.Bounds.id

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical and hyperbolic functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.arccos
   ~cf.Bounds.arccosh
   ~cf.Bounds.arcsin
   ~cf.Bounds.arcsinh
   ~cf.Bounds.arctan
   .. ~cf.Bounds.arctan2  [AT2]
   ~cf.Bounds.arctanh
   ~cf.Bounds.cos
   ~cf.Bounds.cosh
   ~cf.Bounds.sin
   ~cf.Bounds.sinh
   ~cf.Bounds.tan
   ~cf.Bounds.tanh

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.ceil  
   ~cf.Bounds.clip
   ~cf.Bounds.floor
   ~cf.Bounds.rint
   ~cf.Bounds.round
   ~cf.Bounds.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.max
   ~cf.Bounds.mean
   ~cf.Bounds.mid_range
   ~cf.Bounds.min
   ~cf.Bounds.range
   ~cf.Bounds.sample_size
   ~cf.Bounds.sum  
   ~cf.Bounds.sd
   ~cf.Bounds.var
   ~cf.Bounds.standard_deviation
   ~cf.Bounds.variance
   ~cf.Bounds.maximum
   ~cf.Bounds.minimum

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.exp
   ~cf.Bounds.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Bounds.day
   ~cf.Bounds.datetime_array
   ~cf.Bounds.hour
   ~cf.Bounds.minute
   ~cf.Bounds.month
   ~cf.Bounds.reference_datetime   
   ~cf.Bounds.second
   ~cf.Bounds.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.all
   ~cf.Bounds.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.allclose
   ~cf.Bounds.equals
   ~cf.Bounds.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.nc_del_variable
   ~cf.Bounds.nc_get_variable
   ~cf.Bounds.nc_has_variable
   ~cf.Bounds.nc_set_variable 
   ~cf.Bounds.nc_del_dimension
   ~cf.Bounds.nc_get_dimension
   ~cf.Bounds.nc_has_dimension
   ~cf.Bounds.nc_set_dimension

CFA
---

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.add_file_location
   ~cf.Bounds.cfa_clear_file_substitutions
   ~cf.Bounds.cfa_del_file_substitution
   ~cf.Bounds.cfa_file_substitutions
   ~cf.Bounds.cfa_update_file_substitutions
   ~cf.Bounds.del_file_location
   ~cf.Bounds.file_locations

Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.Bounds.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Bounds.dtarray
   
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

   ~cf.Bounds.__lt__
   ~cf.Bounds.__le__
   ~cf.Bounds.__eq__
   ~cf.Bounds.__ne__
   ~cf.Bounds.__gt__
   ~cf.Bounds.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__add__     
   ~cf.Bounds.__sub__     
   ~cf.Bounds.__mul__     
   ~cf.Bounds.__div__     
   ~cf.Bounds.__truediv__ 
   ~cf.Bounds.__floordiv__
   ~cf.Bounds.__pow__     
   ~cf.Bounds.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__radd__     
   ~cf.Bounds.__rsub__     
   ~cf.Bounds.__rmul__     
   ~cf.Bounds.__rdiv__     
   ~cf.Bounds.__rtruediv__ 
   ~cf.Bounds.__rfloordiv__
   ~cf.Bounds.__rpow__   
   ~cf.Bounds.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__iadd__ 
   ~cf.Bounds.__isub__ 
   ~cf.Bounds.__imul__ 
   ~cf.Bounds.__idiv__ 
   ~cf.Bounds.__itruediv__
   ~cf.Bounds.__ifloordiv__
   ~cf.Bounds.__ipow__ 
   ~cf.Bounds.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__neg__    
   ~cf.Bounds.__pos__    
   ~cf.Bounds.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__and__     
   ~cf.Bounds.__or__
   ~cf.Bounds.__xor__     
   ~cf.Bounds.__lshift__
   ~cf.Bounds.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__rand__     
   ~cf.Bounds.__ror__
   ~cf.Bounds.__rxor__     
   ~cf.Bounds.__rlshift__
   ~cf.Bounds.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__iand__     
   ~cf.Bounds.__ior__
   ~cf.Bounds.__ixor__     
   ~cf.Bounds.__ilshift__
   ~cf.Bounds.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__invert__ 

Groups
^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.nc_variable_groups
   ~cf.Bounds.nc_clear_variable_groups
   ~cf.Bounds.nc_set_variable_groups
   ~cf.Bounds.nc_dimension_groups
   ~cf.Bounds.nc_clear_dimension_groups
   ~cf.Bounds.nc_set_dimension_groups

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.__contains__
   ~cf.Bounds.__deepcopy__
   ~cf.Bounds.__getitem__
   ~cf.Bounds.__repr__
   ~cf.Bounds.__setitem__
   ~cf.Bounds.__str__
   ~cf.Bounds.__array__
   ~cf.Bounds.__data__ 
   ~cf.Bounds.__query_isclose__

Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Bounds.asdatetime
   ~cf.Bounds.asreftime
   ~cf.Bounds.attributes
   ~cf.Bounds.chunk
   ~cf.Bounds.delprop
   ~cf.Bounds.dtvarray
   ~cf.Bounds.expand_dims
   ~cf.Bounds.getprop
   ~cf.Bounds.get_filenames
   ~cf.Bounds.hasbounds
   ~cf.Bounds.hasdata
   ~cf.Bounds.hasprop
   ~cf.Bounds.insert_data
   ~cf.Bounds.isdimension
   ~cf.Bounds.isdomainancillary
   ~cf.Bounds.isfieldancillary
   ~cf.Bounds.ismeasure
   ~cf.Bounds.isauxiliary
   ~cf.Bounds.mask_invalid
   ~cf.Bounds.name
   ~cf.Bounds.remove_data
   ~cf.Bounds.select
   ~cf.Bounds.setprop
   ~cf.Bounds.unsafe_array
