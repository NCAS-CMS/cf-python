.. currentmodule:: cf
.. default-role:: obj

.. _cf-DimensionCoordinate:

cf.DimensionCoordinate
======================

----

.. autoclass:: cf.DimensionCoordinate
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.dump
   ~cf.DimensionCoordinate.identity  
   ~cf.DimensionCoordinate.identities
   ~cf.DimensionCoordinate.direction
   ~cf.DimensionCoordinate.inspect
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.construct_type
   ~cf.DimensionCoordinate.id
   ~cf.DimensionCoordinate.ctype
   ~cf.DimensionCoordinate.T
   ~cf.DimensionCoordinate.X
   ~cf.DimensionCoordinate.Y
   ~cf.DimensionCoordinate.Z
   ~cf.DimensionCoordinate.decreasing
   ~cf.DimensionCoordinate.increasing

Bounds
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.del_bounds
   ~cf.DimensionCoordinate.get_bounds
   ~cf.DimensionCoordinate.has_bounds
   ~cf.DimensionCoordinate.set_bounds  
   ~cf.DimensionCoordinate.get_bounds_data
   ~cf.DimensionCoordinate.create_bounds
   ~cf.DimensionCoordinate.contiguous
      
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.bounds
   ~cf.DimensionCoordinate.cellsize
   ~cf.DimensionCoordinate.lower_bounds  
   ~cf.DimensionCoordinate.upper_bounds

Geometries
^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.del_geometry
   ~cf.DimensionCoordinate.get_geometry
   ~cf.DimensionCoordinate.has_geometry
   ~cf.DimensionCoordinate.set_geometry
   ~cf.DimensionCoordinate.del_interior_ring
   ~cf.DimensionCoordinate.get_interior_ring
   ~cf.DimensionCoordinate.has_interior_ring
   ~cf.DimensionCoordinate.set_interior_ring
   ~cf.DimensionCoordinate.del_node_count
   ~cf.DimensionCoordinate.get_node_count
   ~cf.DimensionCoordinate.has_node_count
   ~cf.DimensionCoordinate.set_node_count
   ~cf.DimensionCoordinate.del_part_node_count
   ~cf.DimensionCoordinate.get_part_node_count
   ~cf.DimensionCoordinate.has_part_node_count
   ~cf.DimensionCoordinate.set_part_node_count

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.interior_ring

Climatology
^^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.del_climatology
   ~cf.DimensionCoordinate.get_climatology
   ~cf.DimensionCoordinate.is_climatology
   ~cf.DimensionCoordinate.set_climatology

Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.match_by_identity
   ~cf.DimensionCoordinate.match_by_naxes
   ~cf.DimensionCoordinate.match_by_ncvar
   ~cf.DimensionCoordinate.match_by_property
   ~cf.DimensionCoordinate.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.del_property
   ~cf.DimensionCoordinate.get_property
   ~cf.DimensionCoordinate.has_property
   ~cf.DimensionCoordinate.set_property
   ~cf.DimensionCoordinate.properties
   ~cf.DimensionCoordinate.clear_properties
   ~cf.DimensionCoordinate.del_properties
   ~cf.DimensionCoordinate.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.add_offset
   ~cf.DimensionCoordinate.axis
   ~cf.DimensionCoordinate.calendar
   ~cf.DimensionCoordinate.comment
   ~cf.DimensionCoordinate._FillValue
   ~cf.DimensionCoordinate.history
   ~cf.DimensionCoordinate.leap_month
   ~cf.DimensionCoordinate.leap_year
   ~cf.DimensionCoordinate.long_name
   ~cf.DimensionCoordinate.missing_value
   ~cf.DimensionCoordinate.month_lengths
   ~cf.DimensionCoordinate.positive
   ~cf.DimensionCoordinate.scale_factor
   ~cf.DimensionCoordinate.standard_name
   ~cf.DimensionCoordinate.units
   ~cf.DimensionCoordinate.valid_max
   ~cf.DimensionCoordinate.valid_min
   ~cf.DimensionCoordinate.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.override_units
   ~cf.DimensionCoordinate.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.array
   ~cf.DimensionCoordinate.Data
   ~cf.DimensionCoordinate.data
   ~cf.DimensionCoordinate.datetime_array
   ~cf.DimensionCoordinate.datum
   ~cf.DimensionCoordinate.dtype
   ~cf.DimensionCoordinate.isscalar
   ~cf.DimensionCoordinate.ndim
   ~cf.DimensionCoordinate.shape
   ~cf.DimensionCoordinate.size
   ~cf.DimensionCoordinate.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.to_dask_array
   ~cf.DimensionCoordinate.__getitem__
   ~cf.DimensionCoordinate.del_data
   ~cf.DimensionCoordinate.get_data
   ~cf.DimensionCoordinate.has_data
   ~cf.DimensionCoordinate.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.flatten
   ~cf.DimensionCoordinate.flip
   ~cf.DimensionCoordinate.insert_dimension
   ~cf.DimensionCoordinate.roll
   ~cf.DimensionCoordinate.squeeze
   ~cf.DimensionCoordinate.swapaxes
   ~cf.DimensionCoordinate.transpose
      
.. rubric:: *Expanding the data*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.halo

.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.apply_masking
   ~cf.DimensionCoordinate.count
   ~cf.DimensionCoordinate.count_masked
   ~cf.DimensionCoordinate.fill_value
   ~cf.DimensionCoordinate.masked_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.binary_mask
   ~cf.DimensionCoordinate.hardmask
   ~cf.DimensionCoordinate.mask

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__setitem__
   ~cf.DimensionCoordinate.halo
   ~cf.DimensionCoordinate.masked_invalid
   ~cf.DimensionCoordinate.subspace
   ~cf.DimensionCoordinate.where

Cyclicity
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.DimensionCoordinate.autoperiod

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst
	      
   ~cf.DimensionCoordinate.iscyclic
   ~cf.DimensionCoordinate.isperiodic

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.DimensionCoordinate.rechunk
   ~cf.DimensionCoordinate.close
   ~cf.DimensionCoordinate.convert_reference_time
   ~cf.DimensionCoordinate.cyclic
   ~cf.DimensionCoordinate.get_original_filenames
   ~cf.DimensionCoordinate.period
   ~cf.DimensionCoordinate.persist

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.concatenate
   ~cf.DimensionCoordinate.copy
   ~cf.DimensionCoordinate.create_regular
   ~cf.DimensionCoordinate.creation_commands
   ~cf.DimensionCoordinate.equals
   ~cf.DimensionCoordinate.inspect
   ~cf.DimensionCoordinate.to_memory
   ~cf.DimensionCoordinate.uncompress
   
Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical and hyperbolic functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.arccos
   ~cf.DimensionCoordinate.arccosh
   ~cf.DimensionCoordinate.arcsin
   ~cf.DimensionCoordinate.arcsinh
   ~cf.DimensionCoordinate.arctan
   .. ~cf.DimensionCoordinate.arctan2  [AT2]
   ~cf.DimensionCoordinate.arctanh
   ~cf.DimensionCoordinate.cos
   ~cf.DimensionCoordinate.cosh
   ~cf.DimensionCoordinate.sin
   ~cf.DimensionCoordinate.sinh
   ~cf.DimensionCoordinate.tan
   ~cf.DimensionCoordinate.tanh

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.ceil  
   ~cf.DimensionCoordinate.clip
   ~cf.DimensionCoordinate.floor
   ~cf.DimensionCoordinate.rint
   ~cf.DimensionCoordinate.round
   ~cf.DimensionCoordinate.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst


   ~cf.DimensionCoordinate.maximum
   ~cf.DimensionCoordinate.mean
   ~cf.DimensionCoordinate.mid_range
   ~cf.DimensionCoordinate.minimum
   ~cf.DimensionCoordinate.range
   ~cf.DimensionCoordinate.sample_size
   ~cf.DimensionCoordinate.sum  
   ~cf.DimensionCoordinate.standard_deviation
   ~cf.DimensionCoordinate.variance

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.exp
   ~cf.DimensionCoordinate.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.day
   ~cf.DimensionCoordinate.datetime_array
   ~cf.DimensionCoordinate.hour
   ~cf.DimensionCoordinate.minute
   ~cf.DimensionCoordinate.month
   ~cf.DimensionCoordinate.reference_datetime   
   ~cf.DimensionCoordinate.second
   ~cf.DimensionCoordinate.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.all
   ~cf.DimensionCoordinate.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.allclose
   ~cf.DimensionCoordinate.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.unique
   
NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.nc_del_variable
   ~cf.DimensionCoordinate.nc_get_variable
   ~cf.DimensionCoordinate.nc_has_variable
   ~cf.DimensionCoordinate.nc_set_variable 
   
Groups
^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.DimensionCoordinate.nc_variable_groups
   ~cf.DimensionCoordinate.nc_set_variable_groups
   ~cf.DimensionCoordinate.nc_clear_variable_groups
  
CFA
---

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.add_file_location
   ~cf.DimensionCoordinate.cfa_clear_file_substitutions
   ~cf.DimensionCoordinate.cfa_del_file_substitution
   ~cf.DimensionCoordinate.cfa_file_substitutions
   ~cf.DimensionCoordinate.cfa_update_file_substitutions
   ~cf.DimensionCoordinate.del_file_location
   ~cf.DimensionCoordinate.file_locations

Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.DimensionCoordinate.match
   ~cf.DimensionCoordinate.max
   ~cf.DimensionCoordinate.min
   ~cf.DimensionCoordinate.sd
   ~cf.DimensionCoordinate.var
	      
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.dtarray
   
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

   ~cf.DimensionCoordinate.__lt__
   ~cf.DimensionCoordinate.__le__
   ~cf.DimensionCoordinate.__eq__
   ~cf.DimensionCoordinate.__ne__
   ~cf.DimensionCoordinate.__gt__
   ~cf.DimensionCoordinate.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__add__     
   ~cf.DimensionCoordinate.__sub__     
   ~cf.DimensionCoordinate.__mul__     
   ~cf.DimensionCoordinate.__div__     
   ~cf.DimensionCoordinate.__truediv__ 
   ~cf.DimensionCoordinate.__floordiv__
   ~cf.DimensionCoordinate.__pow__     
   ~cf.DimensionCoordinate.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__radd__     
   ~cf.DimensionCoordinate.__rsub__     
   ~cf.DimensionCoordinate.__rmul__     
   ~cf.DimensionCoordinate.__rdiv__     
   ~cf.DimensionCoordinate.__rtruediv__ 
   ~cf.DimensionCoordinate.__rfloordiv__
   ~cf.DimensionCoordinate.__rpow__   
   ~cf.DimensionCoordinate.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__iadd__ 
   ~cf.DimensionCoordinate.__isub__ 
   ~cf.DimensionCoordinate.__imul__ 
   ~cf.DimensionCoordinate.__idiv__ 
   ~cf.DimensionCoordinate.__itruediv__
   ~cf.DimensionCoordinate.__ifloordiv__
   ~cf.DimensionCoordinate.__ipow__ 
   ~cf.DimensionCoordinate.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__neg__    
   ~cf.DimensionCoordinate.__pos__    
   ~cf.DimensionCoordinate.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__and__     
   ~cf.DimensionCoordinate.__or__
   ~cf.DimensionCoordinate.__xor__     
   ~cf.DimensionCoordinate.__lshift__
   ~cf.DimensionCoordinate.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__rand__     
   ~cf.DimensionCoordinate.__ror__
   ~cf.DimensionCoordinate.__rxor__     
   ~cf.DimensionCoordinate.__rlshift__
   ~cf.DimensionCoordinate.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__iand__     
   ~cf.DimensionCoordinate.__ior__
   ~cf.DimensionCoordinate.__ixor__     
   ~cf.DimensionCoordinate.__ilshift__
   ~cf.DimensionCoordinate.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__invert__ 

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.__contains__
   ~cf.DimensionCoordinate.__deepcopy__
   ~cf.DimensionCoordinate.__getitem__
   ~cf.DimensionCoordinate.__repr__
   ~cf.DimensionCoordinate.__setitem__
   ~cf.DimensionCoordinate.__str__
   ~cf.DimensionCoordinate.__array__
   ~cf.DimensionCoordinate.__data__ 
   ~cf.DimensionCoordinate.__query_isclose__

Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DimensionCoordinate.asdatetime
   ~cf.DimensionCoordinate.asreftime
   ~cf.DimensionCoordinate.chunk
   ~cf.DimensionCoordinate.delprop
   ~cf.DimensionCoordinate.expand_dims
   ~cf.DimensionCoordinate.files
   ~cf.DimensionCoordinate.get_filenames
   ~cf.DimensionCoordinate.getprop
   ~cf.DimensionCoordinate.hasprop
   ~cf.DimensionCoordinate.insert_data
   ~cf.DimensionCoordinate.mask_invalid
   ~cf.DimensionCoordinate.name
   ~cf.DimensionCoordinate.remove_data
   ~cf.DimensionCoordinate.select
   ~cf.DimensionCoordinate.setprop

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DimensionCoordinate.attributes
   ~cf.DimensionCoordinate.Data
   ~cf.DimensionCoordinate.dtvarray
   ~cf.DimensionCoordinate.hasbounds
   ~cf.DimensionCoordinate.hasdata
   ~cf.DimensionCoordinate.isauxiliary
   ~cf.DimensionCoordinate.isdimension
   ~cf.DimensionCoordinate.isdomainancillary
   ~cf.DimensionCoordinate.isfieldancillary
   ~cf.DimensionCoordinate.ismeasure
   ~cf.DimensionCoordinate.role
   ~cf.DimensionCoordinate.unsafe_array
