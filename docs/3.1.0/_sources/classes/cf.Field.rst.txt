.. currentmodule:: cf
.. default-role:: obj

cf.Field
========

----


.. autoclass:: cf.Field
   :no-members:
   :no-inherited-members:

.. _Field-Inspection:

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.dump
   ~cf.Field.identity  
   ~cf.Field.identities

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.construct_type
   ~cf.Field.id

Selection
---------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.match_by_identity
   ~cf.Field.match_by_naxes
   ~cf.Field.match_by_ncvar
   ~cf.Field.match_by_property
   ~cf.Field.match_by_rank
   ~cf.Field.match_by_units
   ~cf.Field.match_by_construct
 
.. _Field-Properties:

Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.del_property
   ~cf.Field.get_property
   ~cf.Field.has_property
   ~cf.Field.set_property
   ~cf.Field.properties
   ~cf.Field.clear_properties
   ~cf.Field.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.add_offset
   ~cf.Field.calendar
   ~cf.Field.cell_methods
   ~cf.Field.comment
   ~cf.Field.Conventions
   ~cf.Field._FillValue
   ~cf.Field.flag_masks
   ~cf.Field.flag_meanings
   ~cf.Field.flag_values
   ~cf.Field.history
   ~cf.Field.institution
   ~cf.Field.leap_month
   ~cf.Field.leap_year
   ~cf.Field.long_name
   ~cf.Field.missing_value
   ~cf.Field.month_lengths
   ~cf.Field.references
   ~cf.Field.scale_factor
   ~cf.Field.source
   ~cf.Field.standard_error_multiplier
   ~cf.Field.standard_name
   ~cf.Field.title
   ~cf.Field.units
   ~cf.Field.valid_max
   ~cf.Field.valid_min
   ~cf.Field.valid_range

Units
-----

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.override_units
   ~cf.Field.override_calendar

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.Units

.. _Field-Data:

Data
----

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.del_data
   ~cf.Field.get_data
   ~cf.Field.has_data
   ~cf.Field.set_data
   ~cf.Field.del_data_axes
   ~cf.Field.get_data_axes
   ~cf.Field.has_data_axes
   ~cf.Field.set_data_axes
 
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.array
   ~cf.Field.data
   ~cf.Field.datetime_array
   ~cf.Field.datum
   ~cf.Field.dtype
   ~cf.Field.ndim
   ~cf.Field.shape
   ~cf.Field.size
   ~cf.Field.varray

.. rubric:: *Rearranging elements*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.anchor
   ~cf.Field.flatten
   ~cf.Field.flip
   ~cf.Field.insert_dimension
   ~cf.Field.roll
   ~cf.Field.squeeze
   ~cf.Field.swapaxes
   ~cf.Field.transpose
   ~cf.Field.unsqueeze

.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.count
   ~cf.Field.count_masked
   ~cf.Field.fill_value
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.binary_mask
   ~cf.Field.hardmask
   ~cf.Field.mask
   ~cf.Field.mask_invalid
   
.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__setitem__
   ~cf.Field.indices
   ~cf.Field.mask_invalid
   ~cf.Field.subspace
   ~cf.Field.where

Miscellaneous data operations
-----------------------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Field.chunk
   ~cf.Field.close
   ~cf.Field.cyclic
   ~cf.Field.files

Metadata constructs
-------------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.construct
   ~cf.Field.construct_key
   ~cf.Field.del_construct
   ~cf.Field.get_construct
   ~cf.Field.has_construct
   ~cf.Field.set_construct
   ~cf.Field.replace_construct
   ~cf.Field.del_data_axes
   ~cf.Field.get_data_axes
   ~cf.Field.has_data_axes
   ~cf.Field.set_data_axes
   ~cf.Field.auxiliary_coordinate
   ~cf.Field.cell_measure
   ~cf.Field.cell_method
   ~cf.Field.coordinate
   ~cf.Field.coordinate_reference
   ~cf.Field.dimension_coordinate
   ~cf.Field.domain_ancillary
   ~cf.Field.domain_axis
   ~cf.Field.domain_axis_key
   ~cf.Field.domain_axis_position
   ~cf.Field.field_ancillary
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.constructs
   ~cf.Field.auxiliary_coordinates
   ~cf.Field.cell_measures
   ~cf.Field.cell_methods
   ~cf.Field.coordinates
   ~cf.Field.coordinate_references
   ~cf.Field.dimension_coordinates
   ~cf.Field.domain_ancillaries
   ~cf.Field.domain_axes
   ~cf.Field.axes
   ~cf.Field.field_ancillaries

.. _Field-Domain:

Domain
------


.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.get_domain
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.domain

.. _Field-Miscellaneous:

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.copy
   ~cf.Field.equals
   ~cf.Field.compress
   ~cf.Field.convert
   ~cf.Field.creation_commands
   ~cf.Field.radius
   ~cf.Field.uncompress
   cf.Field.concatenate
   
.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.Flags
   ~cf.Field.has_bounds
   ~cf.Field.isauxiliary
   ~cf.Field.isdimension
   ~cf.Field.ismeasure
   ~cf.Field.rank
   ~cf.Field.T
   ~cf.Field.X
   ~cf.Field.Y
   ~cf.Field.Z

.. _Field-NetCDF:
   
NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.nc_del_variable
   ~cf.Field.nc_get_variable
   ~cf.Field.nc_has_variable
   ~cf.Field.nc_set_variable 
   ~cf.Field.nc_global_attributes
   ~cf.Field.nc_clear_global_attributes
   ~cf.Field.nc_set_global_attribute
   ~cf.Field.nc_set_global_attributes
   ~cf.Field.dataset_compliance

.. _field_methods:

Domain axes
-----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.autocyclic
   ~cf.Field.axes
   ~cf.Field.axes_sizes
   ~cf.Field.axis
   ~cf.Field.axis_name
   ~cf.Field.axis_size
   ~cf.Field.cyclic
   ~cf.Field.data_axes
   ~cf.Field.iscyclic 
   ~cf.Field.item_axes
   ~cf.Field.items_axes
   ~cf.Field.period

Subspacing
----------

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.__getitem__
   ~cf.Field.subspace
   ~cf.Field.indices

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Trigonometrical and hyperbolic functions

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.arcsinh
   ~cf.Field.arctan
   ~cf.Field.cos
   ~cf.Field.cosh
   ~cf.Field.sin
   ~cf.Field.sinh
   ~cf.Field.tan
   ~cf.Field.tanh


.. rubric:: Rounding and truncation

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.ceil  
   ~cf.Field.clip
   ~cf.Field.floor
   ~cf.Field.rint
   ~cf.Field.round
   ~cf.Field.trunc

.. rubric:: Statistical collapses

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.collapse
   ~cf.Field.cell_area
   ~cf.Field.max
   ~cf.Field.mean
   ~cf.Field.mid_range
   ~cf.Field.min
   ~cf.Field.range
   ~cf.Field.sample_size
   ~cf.Field.sum  
   ~cf.Field.sd
   ~cf.Field.var
   ~cf.Field.weights

.. rubric:: Exponential and logarithmic functions
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.exp
   ~cf.Field.log

.. rubric:: Derivatives

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.derivative

.. rubric:: Convolution filters

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.convolution_filter

.. rubric:: Cumulative sums

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.cumsum

.. rubric:: Binning operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.bin
   ~cf.Field.digitize
   ~cf.Field.percentile

Data array operations
---------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html

.. _field_data_array_access:



.. rubric:: Adding and removing elements

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.unique

.. rubric:: Miscellaneous data array operations

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.chunk
   ~cf.Field.isscalar

Regridding operations
---------------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.regridc
   ~cf.Field.regrids

Date-time operations
--------------------

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.day
   ~cf.Field.datetime_array
   ~cf.Field.hour
   ~cf.Field.minute
   ~cf.Field.month
   ~cf.Field.second
   ~cf.Field.year

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.convert_reference_time

Logic functions
---------------

.. http://docs.scipy.org/doc/numpy/reference/routines.logic.html#truth-value-testing

.. rubric:: Truth value testing

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.all
   ~cf.Field.any
 
.. rubric:: Comparison

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.allclose
   ~cf.Field.equals
   ~cf.Field.equivalent
   ~cf.Field.equivalent_data
   ~cf.Field.equivalent_domain

.. rubric:: Set operations

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.unique


Aliases
-------

.. rubric:: Methods
   
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.Field.aux
   ~cf.Field.auxs
   ~cf.Field.axis
   ~cf.Field.coord
   ~cf.Field.coords
   ~cf.Field.dim
   ~cf.Field.dims
   ~cf.Field.domain_anc
   ~cf.Field.domain_ancs
   ~cf.Field.field_anc
   ~cf.Field.field_ancs
   ~cf.Field.item
   ~cf.Field.items
   ~cf.Field.key
   ~cf.Field.match
   ~cf.Field.measure
   ~cf.Field.measures
   ~cf.Field.ref
   ~cf.Field.refs

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.dtarray

.. _Field-arithmetic:

Arithmetic and comparison operations
------------------------------------

Arithmetic, bitwise and comparison operations are defined on a field
construct as element-wise operations on its data which yield a new
field construct or, for augmented assignments, modify the field
construct's data in-place.


.. _Field-comparison:

.. rubric:: Relational operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__lt__
   ~cf.Field.__le__
   ~cf.Field.__eq__
   ~cf.Field.__ne__
   ~cf.Field.__gt__
   ~cf.Field.__ge__

.. _Field-binary-arithmetic:

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__add__     
   ~cf.Field.__sub__     
   ~cf.Field.__mul__     
   ~cf.Field.__div__     
   ~cf.Field.__truediv__ 
   ~cf.Field.__floordiv__
   ~cf.Field.__pow__     
   ~cf.Field.__mod__     

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__radd__     
   ~cf.Field.__rsub__     
   ~cf.Field.__rmul__     
   ~cf.Field.__rdiv__     
   ~cf.Field.__rtruediv__ 
   ~cf.Field.__rfloordiv__
   ~cf.Field.__rpow__   
   ~cf.Field.__rmod__   

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__iadd__ 
   ~cf.Field.__isub__ 
   ~cf.Field.__imul__ 
   ~cf.Field.__idiv__ 
   ~cf.Field.__itruediv__
   ~cf.Field.__ifloordiv__
   ~cf.Field.__ipow__ 
   ~cf.Field.__imod__ 

.. _Field-unary-arithmetic:
   
.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__neg__    
   ~cf.Field.__pos__    
   ~cf.Field.__abs__    

.. _Field-bitwise:

.. rubric:: Binary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__and__     
   ~cf.Field.__or__
   ~cf.Field.__xor__     
   ~cf.Field.__lshift__
   ~cf.Field.__rshift__     

.. rubric:: Binary bitwise operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__rand__     
   ~cf.Field.__ror__
   ~cf.Field.__rxor__     
   ~cf.Field.__rlshift__
   ~cf.Field.__rrshift__     

.. rubric:: Augmented bitwise assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__iand__     
   ~cf.Field.__ior__
   ~cf.Field.__ixor__     
   ~cf.Field.__ilshift__
   ~cf.Field.__irshift__     

.. rubric:: Unary bitwise operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__invert__ 
 
.. _Field-Special:

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__deepcopy__
   ~cf.Field.__getitem__
   ~cf.Field.__repr__
   ~cf.Field.__str__

.. 'add_offset',
   'all',
   'allclose',
   'analyse_items',
   'any',
   'array',
   'asdatetime',
   'asreftime',
   'attributes',
   'autocyclic',
   'auxiliary_coordinate',
   'auxiliary_coordinates',
   'axes_names',
   'axis_name',
   'axis_size',
   'binary_mask',
   'calendar',
   'cell_area',
   'cell_measure',
   'cell_measures',
   'cell_method',
   'cell_methods',
   'chunk',
   'climatological_time_axes',
   'close',
   'collapse',
   'comment',
   'concatenate',
   'construct',
   'construct_key',
   'construct_type',
   'constructs',
   'convert',
   'convert_reference_time',
   'convolution_filter',
   'coord',
   'coordinate',
   'coordinate_reference',
   'coordinate_reference_domain_axes',
   'coordinate_references',
   'coordinates',
   'coords',
   'copy',
   'count',
   'count_masked',
   'cyclic',
   'data',
   'data_axes',
   'dataset_compliance',
   'datetime_array',
   'datum',
   'day',
   'del_construct',
   'del_coordinate_reference',
   'del_data',
   'del_data_axes',
   'derivative',
   'dimension_coordinate',
   'dimension_coordinates',
   'direction',
   'directions',
   'domain',
   'domain_ancillaries',
   'domain_ancillary',
   'domain_axes',
   'domain_axis',
   'domain_axis_key',
   'domain_mask',
   'dump',
   'equals',
   'equivalent',
   'featureType',
   'field',
   'field_ancillaries',
   'field_ancillary',
   'fill_value',
   'flag_masks',
   'flag_meanings',
   'flag_values',
   'floor',
   'get_construct',
   'get_data',
   'get_data_axes',
   'get_domain',
   'hardmask',
   'has_bounds',
   'has_construct',
   'has_data',
   'has_data_axes',
   'history',
   'hour',
   'id',
   'inspect',
   'isauxiliary',
   'iscyclic',
   'isdimension',
   'isdomainancillary',
   'isfieldancillary',
   'ismeasure',
   'isscalar',
   'item',
   'item_axes',
   'items',
   'items_axes',
   'key_item',
   'map_axes',
   'mask',
   'mask_invalid',
   'minute',
   'missing_value',
   'name',
   'ncdimensions',
   'new_identifier',
   'period',
   'properties',
   'reference_datetime',
   'second',
   'section',
   'select',
   'set_construct',
   'set_coordinate_reference',
   'set_data',
   'set_data_axes',
   'shape',
   'size',
   'source',
   'standard_error_multiplier',
   'standard_name',
   'unique',
   'valid_max',
   'valid_min',
   'valid_range',
   'year']

