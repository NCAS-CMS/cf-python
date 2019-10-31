.. currentmodule:: cf
.. default-role:: obj

cf.FieldAncillary
=================

----

.. autoclass:: cf.FieldAncillary
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.dump
   ~cf.FieldAncillary.identity  
   ~cf.FieldAncillary.identities
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.FieldAncillary.construct_type
   ~cf.FieldAncillary.id

Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.match_by_identity
   ~cf.FieldAncillary.match_by_naxes
   ~cf.FieldAncillary.match_by_ncvar
   ~cf.FieldAncillary.match_by_property
   ~cf.FieldAncillary.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.del_property
   ~cf.FieldAncillary.get_property
   ~cf.FieldAncillary.has_property
   ~cf.FieldAncillary.set_property
   ~cf.FieldAncillary.properties
   ~cf.FieldAncillary.clear_properties
   ~cf.FieldAncillary.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.FieldAncillary.add_offset
   ~cf.FieldAncillary.calendar
   ~cf.FieldAncillary.comment
   ~cf.FieldAncillary._FillValue
   ~cf.FieldAncillary.history
   ~cf.FieldAncillary.leap_month
   ~cf.FieldAncillary.leap_year
   ~cf.FieldAncillary.long_name
   ~cf.FieldAncillary.missing_value
   ~cf.FieldAncillary.month_lengths
   ~cf.FieldAncillary.scale_factor
   ~cf.FieldAncillary.standard_name
   ~cf.FieldAncillary.units
   ~cf.FieldAncillary.valid_max
   ~cf.FieldAncillary.valid_min
   ~cf.FieldAncillary.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.override_units
   ~cf.FieldAncillary.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.FieldAncillary.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.FieldAncillary.array
   ~cf.FieldAncillary.data
   ~cf.FieldAncillary.datetime_array
   ~cf.FieldAncillary.datum
   ~cf.FieldAncillary.dtype
   ~cf.FieldAncillary.isscalar
   ~cf.FieldAncillary.ndim
   ~cf.FieldAncillary.shape
   ~cf.FieldAncillary.size
   ~cf.FieldAncillary.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__getitem__
   ~cf.FieldAncillary.del_data
   ~cf.FieldAncillary.get_data
   ~cf.FieldAncillary.has_data
   ~cf.FieldAncillary.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.flatten
   ~cf.FieldAncillary.flip
   ~cf.FieldAncillary.insert_dimension
   ~cf.FieldAncillary.roll
   ~cf.FieldAncillary.squeeze
   ~cf.FieldAncillary.swapaxes
   ~cf.FieldAncillary.transpose
   
.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.fill_value

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.FieldAncillary.binary_mask
   ~cf.FieldAncillary.count
   ~cf.FieldAncillary.count_masked
   ~cf.FieldAncillary.hardmask
   ~cf.FieldAncillary.mask
   ~cf.FieldAncillary.mask_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__setitem__
   ~cf.FieldAncillary.mask_invalid
   ~cf.FieldAncillary.subspace
   ~cf.FieldAncillary.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.FieldAncillary.chunk
   ~cf.FieldAncillary.close
   ~cf.FieldAncillary.convert_reference_time
   ~cf.FieldAncillary.cyclic
   ~cf.FieldAncillary.files
   ~cf.FieldAncillary.has_bounds

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.concatenate
   ~cf.FieldAncillary.copy
   ~cf.FieldAncillary.equals

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: attribute.rst

   ~cf.FieldAncillary.T
   ~cf.FieldAncillary.X
   ~cf.FieldAncillary.Y
   ~cf.FieldAncillary.Z
   ~cf.FieldAncillary.id

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.cos
   ~cf.FieldAncillary.sin
   ~cf.FieldAncillary.tan

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.ceil  
   ~cf.FieldAncillary.clip
   ~cf.FieldAncillary.floor
   ~cf.FieldAncillary.rint
   ~cf.FieldAncillary.round
   ~cf.FieldAncillary.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.max
   ~cf.FieldAncillary.mean
   ~cf.FieldAncillary.mid_range
   ~cf.FieldAncillary.min
   ~cf.FieldAncillary.range
   ~cf.FieldAncillary.sample_size
   ~cf.FieldAncillary.sum  
   ~cf.FieldAncillary.sd
   ~cf.FieldAncillary.var

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.exp
   ~cf.FieldAncillary.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.FieldAncillary.day
   ~cf.FieldAncillary.datetime_array
   ~cf.FieldAncillary.hour
   ~cf.FieldAncillary.minute
   ~cf.FieldAncillary.month
   ~cf.FieldAncillary.reference_datetime   
   ~cf.FieldAncillary.second
   ~cf.FieldAncillary.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.all
   ~cf.FieldAncillary.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.allclose
   ~cf.FieldAncillary.equals
   ~cf.FieldAncillary.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.nc_del_variable
   ~cf.FieldAncillary.nc_get_variable
   ~cf.FieldAncillary.nc_has_variable
   ~cf.FieldAncillary.nc_set_variable 


Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.FieldAncillary.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.FieldAncillary.dtarray
   
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

   ~cf.FieldAncillary.__lt__
   ~cf.FieldAncillary.__le__
   ~cf.FieldAncillary.__eq__
   ~cf.FieldAncillary.__ne__
   ~cf.FieldAncillary.__gt__
   ~cf.FieldAncillary.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__add__     
   ~cf.FieldAncillary.__sub__     
   ~cf.FieldAncillary.__mul__     
   ~cf.FieldAncillary.__div__     
   ~cf.FieldAncillary.__truediv__ 
   ~cf.FieldAncillary.__floordiv__
   ~cf.FieldAncillary.__pow__     
   ~cf.FieldAncillary.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__radd__     
   ~cf.FieldAncillary.__rsub__     
   ~cf.FieldAncillary.__rmul__     
   ~cf.FieldAncillary.__rdiv__     
   ~cf.FieldAncillary.__rtruediv__ 
   ~cf.FieldAncillary.__rfloordiv__
   ~cf.FieldAncillary.__rpow__   
   ~cf.FieldAncillary.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__iadd__ 
   ~cf.FieldAncillary.__isub__ 
   ~cf.FieldAncillary.__imul__ 
   ~cf.FieldAncillary.__idiv__ 
   ~cf.FieldAncillary.__itruediv__
   ~cf.FieldAncillary.__ifloordiv__
   ~cf.FieldAncillary.__ipow__ 
   ~cf.FieldAncillary.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__neg__    
   ~cf.FieldAncillary.__pos__    
   ~cf.FieldAncillary.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__and__     
   ~cf.FieldAncillary.__or__
   ~cf.FieldAncillary.__xor__     
   ~cf.FieldAncillary.__lshift__
   ~cf.FieldAncillary.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__rand__     
   ~cf.FieldAncillary.__ror__
   ~cf.FieldAncillary.__rxor__     
   ~cf.FieldAncillary.__rlshift__
   ~cf.FieldAncillary.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__iand__     
   ~cf.FieldAncillary.__ior__
   ~cf.FieldAncillary.__ixor__     
   ~cf.FieldAncillary.__ilshift__
   ~cf.FieldAncillary.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__invert__ 
 
Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__contains__
   ~cf.FieldAncillary.__deepcopy__
   ~cf.FieldAncillary.__getitem__
   ~cf.FieldAncillary.__repr__
   ~cf.FieldAncillary.__setitem__
   ~cf.FieldAncillary.__str__
   ~cf.FieldAncillary.__array__
   ~cf.FieldAncillary.__data__
   ~cf.FieldAncillary.__query_set__
   ~cf.FieldAncillary.__query_wi__
   ~cf.FieldAncillary.__query_wo__
   

   
