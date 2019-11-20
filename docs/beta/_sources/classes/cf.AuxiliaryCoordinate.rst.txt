.. currentmodule:: cf
.. default-role:: obj

cf.AuxiliaryCoordinate
======================

----

.. autoclass:: cf.AuxiliaryCoordinate
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.dump
   ~cf.AuxiliaryCoordinate.identity  
   ~cf.AuxiliaryCoordinate.identities
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.AuxiliaryCoordinate.construct_type
   ~cf.AuxiliaryCoordinate.id
   ~cf.AuxiliaryCoordinate.ctype
   ~cf.AuxiliaryCoordinate.T
   ~cf.AuxiliaryCoordinate.X
   ~cf.AuxiliaryCoordinate.Y
   ~cf.AuxiliaryCoordinate.Z

Bounds
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.del_bounds
   ~cf.AuxiliaryCoordinate.get_bounds
   ~cf.AuxiliaryCoordinate.has_bounds
   ~cf.AuxiliaryCoordinate.set_bounds  
   ~cf.AuxiliaryCoordinate.get_bounds_data
   ~cf.AuxiliaryCoordinate.contiguous
      
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.AuxiliaryCoordinate.bounds
   ~cf.AuxiliaryCoordinate.cellsize
   ~cf.AuxiliaryCoordinate.lower_bounds  
   ~cf.AuxiliaryCoordinate.upper_bounds

Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.match_by_identity
   ~cf.AuxiliaryCoordinate.match_by_naxes
   ~cf.AuxiliaryCoordinate.match_by_ncvar
   ~cf.AuxiliaryCoordinate.match_by_property
   ~cf.AuxiliaryCoordinate.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.del_property
   ~cf.AuxiliaryCoordinate.get_property
   ~cf.AuxiliaryCoordinate.has_property
   ~cf.AuxiliaryCoordinate.set_property
   ~cf.AuxiliaryCoordinate.properties
   ~cf.AuxiliaryCoordinate.clear_properties
   ~cf.AuxiliaryCoordinate.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.AuxiliaryCoordinate.add_offset
   ~cf.AuxiliaryCoordinate.axis
   ~cf.AuxiliaryCoordinate.calendar
   ~cf.AuxiliaryCoordinate.comment
   ~cf.AuxiliaryCoordinate._FillValue
   ~cf.AuxiliaryCoordinate.history
   ~cf.AuxiliaryCoordinate.leap_month
   ~cf.AuxiliaryCoordinate.leap_year
   ~cf.AuxiliaryCoordinate.long_name
   ~cf.AuxiliaryCoordinate.missing_value
   ~cf.AuxiliaryCoordinate.month_lengths
   ~cf.AuxiliaryCoordinate.positive
   ~cf.AuxiliaryCoordinate.scale_factor
   ~cf.AuxiliaryCoordinate.standard_name
   ~cf.AuxiliaryCoordinate.units
   ~cf.AuxiliaryCoordinate.valid_max
   ~cf.AuxiliaryCoordinate.valid_min
   ~cf.AuxiliaryCoordinate.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.override_units
   ~cf.AuxiliaryCoordinate.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.AuxiliaryCoordinate.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.AuxiliaryCoordinate.array
   ~cf.AuxiliaryCoordinate.data
   ~cf.AuxiliaryCoordinate.datetime_array
   ~cf.AuxiliaryCoordinate.datum
   ~cf.AuxiliaryCoordinate.dtype
   ~cf.AuxiliaryCoordinate.isscalar
   ~cf.AuxiliaryCoordinate.ndim
   ~cf.AuxiliaryCoordinate.shape
   ~cf.AuxiliaryCoordinate.size
   ~cf.AuxiliaryCoordinate.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__getitem__
   ~cf.AuxiliaryCoordinate.del_data
   ~cf.AuxiliaryCoordinate.get_data
   ~cf.AuxiliaryCoordinate.has_data
   ~cf.AuxiliaryCoordinate.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.flatten
   ~cf.AuxiliaryCoordinate.flip
   ~cf.AuxiliaryCoordinate.insert_dimension
   ~cf.AuxiliaryCoordinate.roll
   ~cf.AuxiliaryCoordinate.squeeze
   ~cf.AuxiliaryCoordinate.swapaxes
   ~cf.AuxiliaryCoordinate.transpose
   
.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.fill_value

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.AuxiliaryCoordinate.binary_mask
   ~cf.AuxiliaryCoordinate.count
   ~cf.AuxiliaryCoordinate.count_masked
   ~cf.AuxiliaryCoordinate.hardmask
   ~cf.AuxiliaryCoordinate.mask
   ~cf.AuxiliaryCoordinate.mask_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__setitem__
   ~cf.AuxiliaryCoordinate.mask_invalid
   ~cf.AuxiliaryCoordinate.subspace
   ~cf.AuxiliaryCoordinate.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.AuxiliaryCoordinate.chunk
   ~cf.AuxiliaryCoordinate.close
   ~cf.AuxiliaryCoordinate.convert_reference_time
   ~cf.AuxiliaryCoordinate.cyclic
   ~cf.AuxiliaryCoordinate.files

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.concatenate
   ~cf.AuxiliaryCoordinate.copy
   ~cf.AuxiliaryCoordinate.equals
   ~cf.AuxiliaryCoordinate.inspect
   ~cf.AuxiliaryCoordinate.uncompress
   
Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.cos
   ~cf.AuxiliaryCoordinate.sin
   ~cf.AuxiliaryCoordinate.tan

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.ceil  
   ~cf.AuxiliaryCoordinate.clip
   ~cf.AuxiliaryCoordinate.floor
   ~cf.AuxiliaryCoordinate.rint
   ~cf.AuxiliaryCoordinate.round
   ~cf.AuxiliaryCoordinate.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.max
   ~cf.AuxiliaryCoordinate.mean
   ~cf.AuxiliaryCoordinate.mid_range
   ~cf.AuxiliaryCoordinate.min
   ~cf.AuxiliaryCoordinate.range
   ~cf.AuxiliaryCoordinate.sample_size
   ~cf.AuxiliaryCoordinate.sum  
   ~cf.AuxiliaryCoordinate.sd
   ~cf.AuxiliaryCoordinate.var

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.exp
   ~cf.AuxiliaryCoordinate.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.AuxiliaryCoordinate.day
   ~cf.AuxiliaryCoordinate.datetime_array
   ~cf.AuxiliaryCoordinate.hour
   ~cf.AuxiliaryCoordinate.minute
   ~cf.AuxiliaryCoordinate.month
   ~cf.AuxiliaryCoordinate.reference_datetime   
   ~cf.AuxiliaryCoordinate.second
   ~cf.AuxiliaryCoordinate.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.all
   ~cf.AuxiliaryCoordinate.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.allclose
   ~cf.AuxiliaryCoordinate.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.nc_del_variable
   ~cf.AuxiliaryCoordinate.nc_get_variable
   ~cf.AuxiliaryCoordinate.nc_has_variable
   ~cf.AuxiliaryCoordinate.nc_set_variable 
   
Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.AuxiliaryCoordinate.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.AuxiliaryCoordinate.dtarray
   
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

   ~cf.AuxiliaryCoordinate.__lt__
   ~cf.AuxiliaryCoordinate.__le__
   ~cf.AuxiliaryCoordinate.__eq__
   ~cf.AuxiliaryCoordinate.__ne__
   ~cf.AuxiliaryCoordinate.__gt__
   ~cf.AuxiliaryCoordinate.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__add__     
   ~cf.AuxiliaryCoordinate.__sub__     
   ~cf.AuxiliaryCoordinate.__mul__     
   ~cf.AuxiliaryCoordinate.__div__     
   ~cf.AuxiliaryCoordinate.__truediv__ 
   ~cf.AuxiliaryCoordinate.__floordiv__
   ~cf.AuxiliaryCoordinate.__pow__     
   ~cf.AuxiliaryCoordinate.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__radd__     
   ~cf.AuxiliaryCoordinate.__rsub__     
   ~cf.AuxiliaryCoordinate.__rmul__     
   ~cf.AuxiliaryCoordinate.__rdiv__     
   ~cf.AuxiliaryCoordinate.__rtruediv__ 
   ~cf.AuxiliaryCoordinate.__rfloordiv__
   ~cf.AuxiliaryCoordinate.__rpow__   
   ~cf.AuxiliaryCoordinate.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__iadd__ 
   ~cf.AuxiliaryCoordinate.__isub__ 
   ~cf.AuxiliaryCoordinate.__imul__ 
   ~cf.AuxiliaryCoordinate.__idiv__ 
   ~cf.AuxiliaryCoordinate.__itruediv__
   ~cf.AuxiliaryCoordinate.__ifloordiv__
   ~cf.AuxiliaryCoordinate.__ipow__ 
   ~cf.AuxiliaryCoordinate.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__neg__    
   ~cf.AuxiliaryCoordinate.__pos__    
   ~cf.AuxiliaryCoordinate.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__and__     
   ~cf.AuxiliaryCoordinate.__or__
   ~cf.AuxiliaryCoordinate.__xor__     
   ~cf.AuxiliaryCoordinate.__lshift__
   ~cf.AuxiliaryCoordinate.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__rand__     
   ~cf.AuxiliaryCoordinate.__ror__
   ~cf.AuxiliaryCoordinate.__rxor__     
   ~cf.AuxiliaryCoordinate.__rlshift__
   ~cf.AuxiliaryCoordinate.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__iand__     
   ~cf.AuxiliaryCoordinate.__ior__
   ~cf.AuxiliaryCoordinate.__ixor__     
   ~cf.AuxiliaryCoordinate.__ilshift__
   ~cf.AuxiliaryCoordinate.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__invert__ 
 
Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.AuxiliaryCoordinate.__contains__
   ~cf.AuxiliaryCoordinate.__deepcopy__
   ~cf.AuxiliaryCoordinate.__getitem__
   ~cf.AuxiliaryCoordinate.__repr__
   ~cf.AuxiliaryCoordinate.__setitem__
   ~cf.AuxiliaryCoordinate.__str__
   ~cf.AuxiliaryCoordinate.__array__
   ~cf.AuxiliaryCoordinate.__data__
   ~cf.AuxiliaryCoordinate.__query_set__
   ~cf.AuxiliaryCoordinate.__query_wi__
   ~cf.AuxiliaryCoordinate.__query_wo__




.. todo for CF-1.8  
      ~AuxiliaryCoordinate.del_geometry
      ~AuxiliaryCoordinate.get_geometry
      ~AuxiliaryCoordinate.has_geometry
      ~AuxiliaryCoordinate.set_geometry
      
      ~AuxiliaryCoordinate.del_node_count
      ~AuxiliaryCoordinate.del_part_node_count

      ~AuxiliaryCoordinate.get_interior_ring
      ~AuxiliaryCoordinate.get_node_count
      ~AuxiliaryCoordinate.get_part_node_count

      ~AuxiliaryCoordinate.has_interior_ring
      ~AuxiliaryCoordinate.has_node_count
      ~AuxiliaryCoordinate.has_part_node_count

      ~AuxiliaryCoordinate.set_interior_ring
      ~AuxiliaryCoordinate.set_node_count
      ~AuxiliaryCoordinate.set_part_node_count
   
      ~AuxiliaryCoordinate.interior_ring
   
   
