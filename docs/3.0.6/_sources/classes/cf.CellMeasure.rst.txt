.. currentmodule:: cf
.. default-role:: obj

cf.CellMeasure
==============

----

.. autoclass:: cf.CellMeasure
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.dump
   ~cf.CellMeasure.identity  
   ~cf.CellMeasure.identities
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellMeasure.construct_type
   ~cf.CellMeasure.id

Measure
-------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.del_measure
   ~cf.CellMeasure.get_measure
   ~cf.CellMeasure.has_measure
   ~cf.CellMeasure.set_measure
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellMeasure.measure

Selection
---------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.match_by_identity
   ~cf.CellMeasure.match_by_naxes
   ~cf.CellMeasure.match_by_ncvar
   ~cf.CellMeasure.match_by_property
   ~cf.CellMeasure.match_by_units
 
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.del_property
   ~cf.CellMeasure.get_property
   ~cf.CellMeasure.has_property
   ~cf.CellMeasure.set_property
   ~cf.CellMeasure.properties
   ~cf.CellMeasure.clear_properties
   ~cf.CellMeasure.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellMeasure.add_offset
   ~cf.CellMeasure.calendar
   ~cf.CellMeasure.comment
   ~cf.CellMeasure._FillValue
   ~cf.CellMeasure.history
   ~cf.CellMeasure.leap_month
   ~cf.CellMeasure.leap_year
   ~cf.CellMeasure.long_name
   ~cf.CellMeasure.missing_value
   ~cf.CellMeasure.month_lengths
   ~cf.CellMeasure.scale_factor
   ~cf.CellMeasure.standard_name
   ~cf.CellMeasure.units
   ~cf.CellMeasure.valid_max
   ~cf.CellMeasure.valid_min
   ~cf.CellMeasure.valid_range

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.override_units
   ~cf.CellMeasure.override_calendar

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellMeasure.Units


Data
----

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellMeasure.array
   ~cf.CellMeasure.data
   ~cf.CellMeasure.datetime_array
   ~cf.CellMeasure.datum
   ~cf.CellMeasure.dtype
   ~cf.CellMeasure.isscalar
   ~cf.CellMeasure.ndim
   ~cf.CellMeasure.shape
   ~cf.CellMeasure.size
   ~cf.CellMeasure.varray

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__getitem__
   ~cf.CellMeasure.del_data
   ~cf.CellMeasure.get_data
   ~cf.CellMeasure.has_data
   ~cf.CellMeasure.set_data
 
.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.flatten
   ~cf.CellMeasure.flip
   ~cf.CellMeasure.insert_dimension
   ~cf.CellMeasure.roll
   ~cf.CellMeasure.squeeze
   ~cf.CellMeasure.swapaxes
   ~cf.CellMeasure.transpose
   
.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.fill_value

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellMeasure.binary_mask
   ~cf.CellMeasure.count
   ~cf.CellMeasure.count_masked
   ~cf.CellMeasure.hardmask
   ~cf.CellMeasure.mask
   ~cf.CellMeasure.mask_invalid

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__setitem__
   ~cf.CellMeasure.mask_invalid
   ~cf.CellMeasure.subspace
   ~cf.CellMeasure.where

.. rubric:: *Miscellaneous*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      

   ~cf.CellMeasure.chunk
   ~cf.CellMeasure.close
   ~cf.CellMeasure.convert_reference_time
   ~cf.CellMeasure.cyclic
   ~cf.CellMeasure.files
   ~cf.CellMeasure.has_bounds

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.concatenate
   ~cf.CellMeasure.copy
   ~cf.CellMeasure.equals
   ~cf.CellMeasure.uncompress

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: attribute.rst

   ~cf.CellMeasure.T
   ~cf.CellMeasure.X
   ~cf.CellMeasure.Y
   ~cf.CellMeasure.Z
   ~cf.CellMeasure.id

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Methods

.. rubric:: *Trigonometrical functions*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.cos
   ~cf.CellMeasure.sin
   ~cf.CellMeasure.tan

.. rubric:: *Rounding and truncation*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.ceil  
   ~cf.CellMeasure.clip
   ~cf.CellMeasure.floor
   ~cf.CellMeasure.rint
   ~cf.CellMeasure.round
   ~cf.CellMeasure.trunc

.. rubric:: *Statistical collapses*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.max
   ~cf.CellMeasure.mean
   ~cf.CellMeasure.mid_range
   ~cf.CellMeasure.min
   ~cf.CellMeasure.range
   ~cf.CellMeasure.sample_size
   ~cf.CellMeasure.sum  
   ~cf.CellMeasure.sd
   ~cf.CellMeasure.var

.. rubric:: *Exponential and logarithmic functions*
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.exp
   ~cf.CellMeasure.log

Date-time operations
--------------------

.. rubric:: Attributes
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellMeasure.day
   ~cf.CellMeasure.datetime_array
   ~cf.CellMeasure.hour
   ~cf.CellMeasure.minute
   ~cf.CellMeasure.month
   ~cf.CellMeasure.reference_datetime   
   ~cf.CellMeasure.second
   ~cf.CellMeasure.year

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.all
   ~cf.CellMeasure.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.allclose
   ~cf.CellMeasure.equals
   ~cf.CellMeasure.equivalent

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.unique

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.nc_del_variable
   ~cf.CellMeasure.nc_get_variable
   ~cf.CellMeasure.nc_has_variable
   ~cf.CellMeasure.nc_set_variable 
   ~cf.CellMeasure.nc_get_external
   ~cf.CellMeasure.nc_set_external 


Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.CellMeasure.match

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.CellMeasure.dtarray
   
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

   ~cf.CellMeasure.__lt__
   ~cf.CellMeasure.__le__
   ~cf.CellMeasure.__eq__
   ~cf.CellMeasure.__ne__
   ~cf.CellMeasure.__gt__
   ~cf.CellMeasure.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__add__     
   ~cf.CellMeasure.__sub__     
   ~cf.CellMeasure.__mul__     
   ~cf.CellMeasure.__div__     
   ~cf.CellMeasure.__truediv__ 
   ~cf.CellMeasure.__floordiv__
   ~cf.CellMeasure.__pow__     
   ~cf.CellMeasure.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__radd__     
   ~cf.CellMeasure.__rsub__     
   ~cf.CellMeasure.__rmul__     
   ~cf.CellMeasure.__rdiv__     
   ~cf.CellMeasure.__rtruediv__ 
   ~cf.CellMeasure.__rfloordiv__
   ~cf.CellMeasure.__rpow__   
   ~cf.CellMeasure.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__iadd__ 
   ~cf.CellMeasure.__isub__ 
   ~cf.CellMeasure.__imul__ 
   ~cf.CellMeasure.__idiv__ 
   ~cf.CellMeasure.__itruediv__
   ~cf.CellMeasure.__ifloordiv__
   ~cf.CellMeasure.__ipow__ 
   ~cf.CellMeasure.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__neg__    
   ~cf.CellMeasure.__pos__    
   ~cf.CellMeasure.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__and__     
   ~cf.CellMeasure.__or__
   ~cf.CellMeasure.__xor__     
   ~cf.CellMeasure.__lshift__
   ~cf.CellMeasure.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__rand__     
   ~cf.CellMeasure.__ror__
   ~cf.CellMeasure.__rxor__     
   ~cf.CellMeasure.__rlshift__
   ~cf.CellMeasure.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__iand__     
   ~cf.CellMeasure.__ior__
   ~cf.CellMeasure.__ixor__     
   ~cf.CellMeasure.__ilshift__
   ~cf.CellMeasure.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__invert__ 
 
Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.CellMeasure.__contains__
   ~cf.CellMeasure.__deepcopy__
   ~cf.CellMeasure.__getitem__
   ~cf.CellMeasure.__repr__
   ~cf.CellMeasure.__setitem__
   ~cf.CellMeasure.__str__
   ~cf.CellMeasure.__array__
   ~cf.CellMeasure.__data__
   ~cf.CellMeasure.__query_set__
   ~cf.CellMeasure.__query_wi__
   ~cf.CellMeasure.__query_wo__
   
