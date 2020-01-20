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
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Data.array
   ~cf.Data.binary_mask
   ~cf.Data.data
   ~cf.Data.day
   ~cf.Data.datetime_array
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
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.add_partitions
   ~cf.Data.all
   ~cf.Data.allclose
   ~cf.Data.any
   ~cf.Data.arcsinh
   ~cf.Data.arctan
   ~cf.Data.argmax
   ~cf.Data.asdata
   ~cf.Data.ceil
   ~cf.Data.change_calendar
   ~cf.Data.chunk
   ~cf.Data.clip
   ~cf.Data.close
   ~cf.Data.concatenate
   ~cf.Data.concatenate_data
   ~cf.Data.copy
   ~cf.Data.cos
   ~cf.Data.cosh
   ~cf.Data.count
   ~cf.Data.count_masked
   ~cf.Data.creation_commands
   ~cf.Data.cumsum
   ~cf.Data.cyclic
   ~cf.Data.datum
   ~cf.Data.del_calendar
   ~cf.Data.del_fill_value
   ~cf.Data.del_units
   ~cf.Data.dump
   ~cf.Data.digitize
   ~cf.Data.dumpd
   ~cf.Data.dumps
   ~cf.Data.empty
   ~cf.Data.equals
   ~cf.Data.exp
   ~cf.Data.expand_dims
   ~cf.Data.files
   ~cf.Data.filled
   ~cf.Data.first_element
   ~cf.Data.fits_in_memory
   ~cf.Data.fits_in_one_chunk_in_memory
   ~cf.Data.flat
   ~cf.Data.flatten
   ~cf.Data.flip
   ~cf.Data.floor
   ~cf.Data.full
   ~cf.Data.func
   ~cf.Data.get_calendar
   ~cf.Data.get_compressed_axes
   ~cf.Data.get_compressed_dimension
   ~cf.Data.get_compression_type
   ~cf.Data.get_count
   ~cf.Data.get_data
   ~cf.Data.get_fill_value
   ~cf.Data.get_index
   ~cf.Data.get_list
   ~cf.Data.get_units
   ~cf.Data.has_calendar
   ~cf.Data.has_fill_value
   ~cf.Data.has_units
   ~cf.Data.insert_dimension
   ~cf.Data.inspect
   ~cf.Data.integral
   ~cf.Data.isclose
   ~cf.Data.last_element
   ~cf.Data.loadd
   ~cf.Data.loads
   ~cf.Data.log
   ~cf.Data.mask_fpe
   ~cf.Data.mask_invalid
   ~cf.Data.max
   ~cf.Data.maximum
   ~cf.Data.maximum_absolute_value
   ~cf.Data.mean
   ~cf.Data.mean_absolute_value
   ~cf.Data.mean_of_upper_decile
   ~cf.Data.median
   ~cf.Data.mid_range
   ~cf.Data.min
   ~cf.Data.minimum
   ~cf.Data.minimum_absolute_value
   ~cf.Data.nc_clear_hdf5_chunksizes
   ~cf.Data.nc_hdf5_chunksizes
   ~cf.Data.nc_set_hdf5_chunksizes
   ~cf.Data.ndindex
   ~cf.Data.ones
   ~cf.Data.outerproduct
   ~cf.Data.override_calendar
   ~cf.Data.override_units
   ~cf.Data.partition_boundaries
   ~cf.Data.partition_configuration
   ~cf.Data.percentile
   ~cf.Data.range
   ~cf.Data.reconstruct_sectioned_data
   ~cf.Data.rint
   ~cf.Data.roll
   ~cf.Data.root_mean_square
   ~cf.Data.round
   ~cf.Data.sample_size
   ~cf.Data.save_to_disk
   ~cf.Data.sd
   ~cf.Data.standard_deviation
   ~cf.Data.second_element
   ~cf.Data.section
   ~cf.Data.set_calendar
   ~cf.Data.set_fill_value
   ~cf.Data.set_units
   ~cf.Data.seterr
   ~cf.Data.sin
   ~cf.Data.sinh
   ~cf.Data.source
   ~cf.Data.squeeze
   ~cf.Data.standard_deviation      
   ~cf.Data.stats
   ~cf.Data.sum
   ~cf.Data.sum_of_squares
   ~cf.Data.sum_of_weights
   ~cf.Data.sum_of_weights2
   ~cf.Data.swapaxes
   ~cf.Data.tan
   ~cf.Data.tanh
   ~cf.Data.to_disk
   ~cf.Data.to_memory
   ~cf.Data.tolist
   ~cf.Data.transpose
   ~cf.Data.trunc
   ~cf.Data.uncompress
   ~cf.Data.unique
   ~cf.Data.var
   ~cf.Data.variance
   ~cf.Data.where
   ~cf.Data.zeros

Data static methods
-------------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.mask_fpe
   ~cf.Data.seterr
 
Data arithmetic and comparison operations
-----------------------------------------

Arithmetic, bitwise and comparison operations are defined as
element-wise data array operations which yield a new `cf.Data` object
or, for augmented assignments, modify the data in-place.

.. rubric:: Comparison operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__lt__
   ~cf.Data.__le__
   ~cf.Data.__eq__
   ~cf.Data.__ne__
   ~cf.Data.__gt__
   ~cf.Data.__ge__

.. rubric:: Truth value of an array

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__bool__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__add__     
   ~cf.Data.__sub__     
   ~cf.Data.__mul__     
   ~cf.Data.__div__     
   ~cf.Data.__truediv__ 
   ~cf.Data.__floordiv__
   ~cf.Data.__pow__     
   ~cf.Data.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__radd__     
   ~cf.Data.__rsub__     
   ~cf.Data.__rmul__     
   ~cf.Data.__rdiv__     
   ~cf.Data.__rtruediv__ 
   ~cf.Data.__rfloordiv__
   ~cf.Data.__rpow__
   ~cf.Data.__rmod__

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__iadd__ 
   ~cf.Data.__isub__ 
   ~cf.Data.__imul__ 
   ~cf.Data.__idiv__ 
   ~cf.Data.__itruediv__
   ~cf.Data.__ifloordiv__
   ~cf.Data.__ipow__ 
   ~cf.Data.__imod__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__neg__    
   ~cf.Data.__pos__    
   ~cf.Data.__abs__    

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__and__     
   ~cf.Data.__or__
   ~cf.Data.__xor__     
   ~cf.Data.__lshift__
   ~cf.Data.__rshift__     

..rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__rand__     
   ~cf.Data.__ror__
   ~cf.Data.__rxor__     
   ~cf.Data.__rlshift__
   ~cf.Data.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__iand__     
   ~cf.Data.__ior__
   ~cf.Data.__ixor__     
   ~cf.Data.__ilshift__
   ~cf.Data.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__invert__ 
 
Special
-------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.__array__
   ~cf.Data.__contains__
   ~cf.Data.__data__     
   ~cf.Data.__deepcopy__
   ~cf.Data.__getitem__ 
   ~cf.Data.__hash__
   ~cf.Data.__iter__ 
   ~cf.Data.__len__
   ~cf.Data.__query_set__
   ~cf.Data.__query_wi__
   ~cf.Data.__query_wo__
   ~cf.Data.__repr__
   ~cf.Data.__setitem__ 
   ~cf.Data.__str__
