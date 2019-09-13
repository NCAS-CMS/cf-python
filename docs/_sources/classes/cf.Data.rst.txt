.. currentmodule:: cf
.. default-role:: obj

cf.Data
=======

.. autoclass:: cf.Data
   :no-members:
   :no-inherited-members:

Data attributes
---------------

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: attribute.rst

   ~cf.Data.array
   ~cf.Data.binary_mask
   ~cf.Data.data
   ~cf.Data.day
   ~cf.Data.dtarray
   ~cf.Data.dtype
   ~cf.Data.fill_value
   ~cf.Data.hardmask
   ~cf.Data.hour
   ~cf.Data.ismasked
   ~cf.Data.isscalar
   ~cf.Data.mask
   ~cf.Data.minute
   ~cf.Data.month
   ~cf.Data.nbytes
   ~cf.Data.ndim
   ~cf.Data.second
   ~cf.Data.shape
   ~cf.Data.size
   ~cf.Data.Units 
   ~cf.Data.varray
   ~cf.Data.year
 
Data methods
------------
   
.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.Data.all
   ~cf.Data.allclose
   ~cf.Data.argmax
   ~cf.Data.max
   ~cf.Data.min
   ~cf.Data.any
   ~cf.Data.chunk
   ~cf.Data.ceil
   ~cf.Data.clip
   ~cf.Data.close
   ~cf.Data.copy
   ~cf.Data.cos
   ~cf.Data.datum
   ~cf.Data.dump
   ~cf.Data.dumpd
   ~cf.Data.equals
   ~cf.Data.equivalent
   ~cf.Data.expand_dims
   ~cf.Data.files
   ~cf.Data.flat
   ~cf.Data.flip
   ~cf.Data.floor
   ~cf.Data.func 
   ~cf.Data.HDF_chunks
   ~cf.Data.isclose
   ~cf.Data.loadd
   ~cf.Data.mask_invalid
   ~cf.Data.mean
   ~cf.Data.mid_range
   ~cf.Data.ndindex
   ~cf.Data.outerproduct
   ~cf.Data.override_calendar
   ~cf.Data.override_units
   ~cf.Data.partition_boundaries
   ~cf.Data.range
   ~cf.Data.rint
   ~cf.Data.roll
   ~cf.Data.save_to_disk
   ~cf.Data.sample_size
   ~cf.Data.sd
   ~cf.Data.sin
   ~cf.Data.squeeze
   ~cf.Data.sum
   ~cf.Data.sum_of_weights
   ~cf.Data.sum_of_weights2
   ~cf.Data.swapaxes
   ~cf.Data.tan
   ~cf.Data.to_disk
   ~cf.Data.to_memory
   ~cf.Data.transpose
   ~cf.Data.trunc
   ~cf.Data.unique
   ~cf.Data.var
   ~cf.Data.where


Data static methods
-------------------

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.Data.mask_fpe
   ~cf.Data.seterr
 
Data arithmetic and comparison operations
-----------------------------------------

Arithmetic, bitwise and comparison operations are defined as
element-wise data array operations which yield a new `cf.Data` object
or, for augmented assignments, modify the data array in-place.


**Comparison operators**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__lt__
   ~cf.Data.__le__
   ~cf.Data.__eq__
   ~cf.Data.__ne__
   ~cf.Data.__gt__
   ~cf.Data.__ge__

**Truth value of an array**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__nonzero__

**Binary arithmetic operators**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__add__     
   ~cf.Data.__sub__     
   ~cf.Data.__mul__     
   ~cf.Data.__div__     
   ~cf.Data.__truediv__ 
   ~cf.Data.__floordiv__
   ~cf.Data.__pow__     
   ~cf.Data.__mod__     

**Binary arithmetic operators with reflected (swapped) operands**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__radd__     
   ~cf.Data.__rsub__     
   ~cf.Data.__rmul__     
   ~cf.Data.__rdiv__     
   ~cf.Data.__rtruediv__ 
   ~cf.Data.__rfloordiv__
   ~cf.Data.__rpow__
   ~cf.Data.__rmod__

**Augmented arithmetic assignments**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__iadd__ 
   ~cf.Data.__isub__ 
   ~cf.Data.__imul__ 
   ~cf.Data.__idiv__ 
   ~cf.Data.__itruediv__
   ~cf.Data.__ifloordiv__
   ~cf.Data.__ipow__ 
   ~cf.Data.__imod__ 

**Unary arithmetic operators**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__neg__    
   ~cf.Data.__pos__    
   ~cf.Data.__abs__    

**Binary bitwise operators**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__and__     
   ~cf.Data.__or__
   ~cf.Data.__xor__     
   ~cf.Data.__lshift__
   ~cf.Data.__rshift__     

**Binary bitwise operators with reflected (swapped) operands**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__rand__     
   ~cf.Data.__ror__
   ~cf.Data.__rxor__     
   ~cf.Data.__rlshift__
   ~cf.Data.__rrshift__     

**Augmented bitwise assignments**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__iand__     
   ~cf.Data.__ior__
   ~cf.Data.__ixor__     
   ~cf.Data.__ilshift__
   ~cf.Data.__irshift__     

**Unary bitwise operators**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__invert__ 
 
Data special methods
--------------------

**Standard library functions**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__deepcopy__
   ~cf.Data.__hash__

**Container customization**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__len__
   ~cf.Data.__getitem__ 
   ~cf.Data.__iter__ 
   ~cf.Data.__setitem__ 
   ~cf.Data.__contains__

**String representations**

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.Data.__repr__
   ~cf.Data.__str__
