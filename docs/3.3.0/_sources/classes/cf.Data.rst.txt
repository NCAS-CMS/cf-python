.. currentmodule:: cf
.. default-role:: obj

cf.Data
=======

.. autoclass:: cf.Data
   :no-members:
   :no-inherited-members:


Inspection
----------

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Data.array
   ~cf.Data.varray
   ~cf.Data.dtype
   ~cf.Data.ndim
   ~cf.Data.shape
   ~cf.Data.size
   ~cf.Data.nbytes
   ~cf.Data.dump
   ~cf.Data.inspect
   ~cf.Data.isscalar
   
Units
-----

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.del_units
   ~cf.Data.get_units
   ~cf.Data.has_units
   ~cf.Data.set_units
   ~cf.Data.override_units
   ~cf.Data.del_calendar
   ~cf.Data.get_calendar
   ~cf.Data.has_calendar
   ~cf.Data.set_calendar
   ~cf.Data.override_calendar
   ~cf.Data.change_calendar

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Data.Units 

Data creation routines
----------------------

Ones and zeros
^^^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.empty
   ~cf.Data.full
   ~cf.Data.ones
   ~cf.Data.zeros

From existing data
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.copy
   ~cf.Data.asdata
   ~cf.Data.loadd
   ~cf.Data.loads

Data manipulation routines
--------------------------

Changing data shape
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.flatten

Transpose-like operations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.swapaxes		 
   ~cf.Data.transpose

Changing number of dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.insert_dimension
   ~cf.Data.squeeze
    
Joining data
^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.concatenate
   
Adding and removing elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.unique
	    
Rearranging elements
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.flip
   ~cf.Data.roll
	    
Binary operations
^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

Date-time support
-----------------

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Data.datetime_array
   ~cf.Data.datetime_as_string
   ~cf.Data.day
   ~cf.Data.hour
   ~cf.Data.minute
   ~cf.Data.month
   ~cf.Data.second
   ~cf.Data.year
 
Indexing routines
-----------------

Single value selection
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.datum
   ~cf.Data.first_element
   ~cf.Data.second_element
   ~cf.Data.last_element

Iterating over data
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.flat
   ~cf.Data.ndindex

Cyclic axes
-----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.cyclic
   
Input and output
----------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.dumpd
   ~cf.Data.dumps
   ~cf.Data.tolist

Linear algebra
--------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.outerproduct

Logic functions
---------------

Truth value testing
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.all
   ~cf.Data.any

Comparison
^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.allclose
   ~cf.Data.isclose
   ~cf.Data.equals

Mask support
------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.count
   ~cf.Data.count_masked
   ~cf.Data.compressed
   ~cf.Data.filled
   ~cf.Data.mask_fpe
   ~cf.Data.mask_invalid
   ~cf.Data.del_fill_value
   ~cf.Data.get_fill_value
   ~cf.Data.has_fill_value
   ~cf.Data.set_fill_value
   
.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Data.binary_mask
   ~cf.Data.hardmask
   ~cf.Data.ismasked
   ~cf.Data.mask
   ~cf.Data.fill_value

Mathematical functions
----------------------

Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^
  
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.sin
   ~cf.Data.cos 
   ~cf.Data.tan 
   ~cf.Data.arcsin
   ~cf.Data.arccos
   ~cf.Data.arctan
..  ~cf.Data.arctan2  [AT2]
   
Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^

  
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.sinh
   ~cf.Data.cosh
   ~cf.Data.tanh 
   ~cf.Data.arcsinh
   ~cf.Data.arccosh
   ~cf.Data.arctanh

Rounding
^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.ceil
   ~cf.Data.floor
   ~cf.Data.rint
   ~cf.Data.round
   ~cf.Data.trunc

Sums, products, differences
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.cumsum
   ~cf.Data.diff
   ~cf.Data.sum

.. rubric:: Convolution filters

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.convolution_filter

Exponents and logarithms
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.exp
   ~cf.Data.log
		 
Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.clip

Set routines
-------------

Making proper sets
^^^^^^^^^^^^^^^^^^    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.unique
	    
Sorting, searching, and counting
--------------------------------

Searching
^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.argmax
   ~cf.Data.where
	      
Counting
^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.count
   ~cf.Data.count_masked

Statistics
----------

Order statistics
^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.maximum
   ~cf.Data.maximum_absolute_value
   ~cf.Data.minimum
   ~cf.Data.minimum_absolute_value
   ~cf.Data.percentile

Averages and variances
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.mean
   ~cf.Data.mean_absolute_value
   ~cf.Data.mean_of_upper_decile
   ~cf.Data.median
   ~cf.Data.mid_range
   ~cf.Data.range
   ~cf.Data.root_mean_square
   ~cf.Data.standard_deviation
   ~cf.Data.variance

Sums
^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.integral
   ~cf.Data.sum
   ~cf.Data.sum_of_squares

Histograms
^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.digitize
     
Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.sample_size
   ~cf.Data.stats
   ~cf.Data.sum_of_weights
   ~cf.Data.sum_of_weights2


Error handling
--------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.seterr

Compression by convention
-------------------------
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.get_compressed_axes
   ~cf.Data.get_compressed_dimension
   ~cf.Data.get_compression_type
   ~cf.Data.get_count
   ~cf.Data.get_index
   ~cf.Data.get_list
   ~cf.Data.uncompress

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Data.compressed_array

Miscellaneous
-------------
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Data.creation_commands
   ~cf.Data.files
   ~cf.Data.get_data
   ~cf.Data.source

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Data.data

Performance
-----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Data.nc_clear_hdf5_chunksizes
   ~cf.Data.nc_hdf5_chunksizes
   ~cf.Data.nc_set_hdf5_chunksizes
   ~cf.Data.close
   ~cf.Data.chunk
   ~cf.Data.add_partitions
   ~cf.Data.partition_boundaries
   ~cf.Data.partition_configuration
   ~cf.Data.to_disk
   ~cf.Data.to_memory
   ~cf.Data.fits_in_memory
   ~cf.Data.fits_in_one_chunk_in_memory
   ~cf.Data.section
   ~cf.Data.reconstruct_sectioned_data
 
Element-wise arithmetic, bit and comparison operations
------------------------------------------------------

Arithmetic, bit and comparison operations are defined as element-wise
data array operations which yield a new `cf.Data` object or, for
augmented assignments, modify the data in-place.

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
   ~cf.Data.__repr__
   ~cf.Data.__setitem__ 
   ~cf.Data.__str__
   ~cf.Data.__query_set__
   ~cf.Data.__query_wi__
   ~cf.Data.__query_wo__
