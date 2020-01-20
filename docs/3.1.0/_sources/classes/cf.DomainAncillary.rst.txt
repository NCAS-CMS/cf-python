.. currentmodule:: cf
.. default-role:: obj

cf.DomainAncillary
==================

----

.. autoclass:: cf.DomainAncillary
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.dump
   ~cf.DomainAncillary.identity  
   ~cf.DomainAncillary.identities
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAncillary.construct_type
   ~cf.DomainAncillary.id
   ~cf.DomainAncillary.T
   ~cf.DomainAncillary.X
   ~cf.DomainAncillary.Y
   ~cf.DomainAncillary.Z

Bounds
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.del_bounds
   ~cf.DomainAncillary.get_bounds
   ~cf.DomainAncillary.has_bounds
   ~cf.DomainAncillary.set_bounds  
   ~cf.DomainAncillary.get_bounds_data
   ~cf.DomainAncillary.contiguous
      
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAncillary.bounds
   ~cf.DomainAncillary.cellsize
   ~cf.DomainAncillary.lower_bounds  
   ~cf.DomainAncillary.upper_bounds

Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.match_by_identity
   ~cf.DomainAncillary.match_by_naxes
   ~cf.DomainAncillary.match_by_ncvar
   ~cf.DomainAncillary.match_by_property
   ~cf.DomainAncillary.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.del_property
   ~cf.DomainAncillary.get_property
   ~cf.DomainAncillary.has_property
   ~cf.DomainAncillary.set_property
   ~cf.DomainAncillary.properties
   ~cf.DomainAncillary.clear_properties
   ~cf.DomainAncillary.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAncillary.add_offset
   ~cf.DomainAncillary.calendar
   ~cf.DomainAncillary.comment
   ~cf.DomainAncillary._FillValue
   ~cf.DomainAncillary.history
   ~cf.DomainAncillary.leap_month
   ~cf.DomainAncillary.leap_year
   ~cf.DomainAncillary.long_name
   ~cf.DomainAncillary.missing_value
   ~cf.DomainAncillary.month_lengths
   ~cf.DomainAncillary.scale_factor
   ~cf.DomainAncillary.standard_name
   ~cf.DomainAncillary.units
   ~cf.DomainAncillary.valid_max
   ~cf.DomainAncillary.valid_min
   ~cf.DomainAncillary.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.override_units
   ~cf.DomainAncillary.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAncillary.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAncillary.array
   ~cf.DomainAncillary.data
   ~cf.DomainAncillary.datetime_array
   ~cf.DomainAncillary.datum
   ~cf.DomainAncillary.dtype
   ~cf.DomainAncillary.isscalar
   ~cf.DomainAncillary.ndim
   ~cf.DomainAncillary.shape
   ~cf.DomainAncillary.size
   ~cf.DomainAncillary.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__getitem__
   ~cf.DomainAncillary.del_data
   ~cf.DomainAncillary.get_data
   ~cf.DomainAncillary.has_data
   ~cf.DomainAncillary.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.flatten
   ~cf.DomainAncillary.flip
   ~cf.DomainAncillary.insert_dimension
   ~cf.DomainAncillary.roll
   ~cf.DomainAncillary.squeeze
   ~cf.DomainAncillary.swapaxes
   ~cf.DomainAncillary.transpose
   
.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.fill_value

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAncillary.binary_mask
   ~cf.DomainAncillary.count
   ~cf.DomainAncillary.count_masked
   ~cf.DomainAncillary.hardmask
   ~cf.DomainAncillary.mask
   ~cf.DomainAncillary.mask_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__setitem__
   ~cf.DomainAncillary.mask_invalid
   ~cf.DomainAncillary.subspace
   ~cf.DomainAncillary.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.DomainAncillary.chunk
   ~cf.DomainAncillary.close
   ~cf.DomainAncillary.convert_reference_time
   ~cf.DomainAncillary.cyclic
   ~cf.DomainAncillary.files

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.concatenate
   ~cf.DomainAncillary.copy
   ~cf.DomainAncillary.equals
   ~cf.DomainAncillary.inspect
   ~cf.DomainAncillary.uncompress
   
Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical and hyperbolic functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.arcsinh
   ~cf.DomainAncillary.arctan
   ~cf.DomainAncillary.cos
   ~cf.DomainAncillary.cosh
   ~cf.DomainAncillary.sin
   ~cf.DomainAncillary.sinh
   ~cf.DomainAncillary.tan
   ~cf.DomainAncillary.tanh

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.ceil  
   ~cf.DomainAncillary.clip
   ~cf.DomainAncillary.floor
   ~cf.DomainAncillary.rint
   ~cf.DomainAncillary.round
   ~cf.DomainAncillary.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.max
   ~cf.DomainAncillary.mean
   ~cf.DomainAncillary.mid_range
   ~cf.DomainAncillary.min
   ~cf.DomainAncillary.range
   ~cf.DomainAncillary.sample_size
   ~cf.DomainAncillary.sum  
   ~cf.DomainAncillary.sd
   ~cf.DomainAncillary.var

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.exp
   ~cf.DomainAncillary.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAncillary.day
   ~cf.DomainAncillary.datetime_array
   ~cf.DomainAncillary.hour
   ~cf.DomainAncillary.minute
   ~cf.DomainAncillary.month
   ~cf.DomainAncillary.reference_datetime   
   ~cf.DomainAncillary.second
   ~cf.DomainAncillary.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.all
   ~cf.DomainAncillary.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.allclose
   ~cf.DomainAncillary.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.nc_del_variable
   ~cf.DomainAncillary.nc_get_variable
   ~cf.DomainAncillary.nc_has_variable
   ~cf.DomainAncillary.nc_set_variable 
   
Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.DomainAncillary.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAncillary.dtarray
   
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

   ~cf.DomainAncillary.__lt__
   ~cf.DomainAncillary.__le__
   ~cf.DomainAncillary.__eq__
   ~cf.DomainAncillary.__ne__
   ~cf.DomainAncillary.__gt__
   ~cf.DomainAncillary.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__add__     
   ~cf.DomainAncillary.__sub__     
   ~cf.DomainAncillary.__mul__     
   ~cf.DomainAncillary.__div__     
   ~cf.DomainAncillary.__truediv__ 
   ~cf.DomainAncillary.__floordiv__
   ~cf.DomainAncillary.__pow__     
   ~cf.DomainAncillary.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__radd__     
   ~cf.DomainAncillary.__rsub__     
   ~cf.DomainAncillary.__rmul__     
   ~cf.DomainAncillary.__rdiv__     
   ~cf.DomainAncillary.__rtruediv__ 
   ~cf.DomainAncillary.__rfloordiv__
   ~cf.DomainAncillary.__rpow__   
   ~cf.DomainAncillary.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__iadd__ 
   ~cf.DomainAncillary.__isub__ 
   ~cf.DomainAncillary.__imul__ 
   ~cf.DomainAncillary.__idiv__ 
   ~cf.DomainAncillary.__itruediv__
   ~cf.DomainAncillary.__ifloordiv__
   ~cf.DomainAncillary.__ipow__ 
   ~cf.DomainAncillary.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__neg__    
   ~cf.DomainAncillary.__pos__    
   ~cf.DomainAncillary.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__and__     
   ~cf.DomainAncillary.__or__
   ~cf.DomainAncillary.__xor__     
   ~cf.DomainAncillary.__lshift__
   ~cf.DomainAncillary.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__rand__     
   ~cf.DomainAncillary.__ror__
   ~cf.DomainAncillary.__rxor__     
   ~cf.DomainAncillary.__rlshift__
   ~cf.DomainAncillary.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__iand__     
   ~cf.DomainAncillary.__ior__
   ~cf.DomainAncillary.__ixor__     
   ~cf.DomainAncillary.__ilshift__
   ~cf.DomainAncillary.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__invert__ 
 
Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAncillary.__contains__
   ~cf.DomainAncillary.__deepcopy__
   ~cf.DomainAncillary.__getitem__
   ~cf.DomainAncillary.__repr__
   ~cf.DomainAncillary.__setitem__
   ~cf.DomainAncillary.__str__
   ~cf.DomainAncillary.__array__
   ~cf.DomainAncillary.__data__
   ~cf.DomainAncillary.__query_set__
   ~cf.DomainAncillary.__query_wi__
   ~cf.DomainAncillary.__query_wo__




.. todo for CF-1.8  
      ~DomainAncillary.del_geometry
      ~DomainAncillary.get_geometry
      ~DomainAncillary.has_geometry
      ~DomainAncillary.set_geometry
      
      ~DomainAncillary.del_node_count
      ~DomainAncillary.del_part_node_count

      ~DomainAncillary.get_interior_ring
      ~DomainAncillary.get_node_count
      ~DomainAncillary.get_part_node_count

      ~DomainAncillary.has_interior_ring
      ~DomainAncillary.has_node_count
      ~DomainAncillary.has_part_node_count

      ~DomainAncillary.set_interior_ring
      ~DomainAncillary.set_node_count
      ~DomainAncillary.set_part_node_count
   
      ~DomainAncillary.interior_ring
   
   
