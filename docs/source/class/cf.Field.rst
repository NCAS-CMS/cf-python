.. currentmodule:: cf
.. default-role:: obj

.. _cf-Field:
   
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
   ~cf.Field.inspect

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
   ~cf.Field.del_properties
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
   ~cf.Field.DSG
   ~cf.Field._FillValue
   ~cf.Field.featureType
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
   ~cf.Field.reference_datetime
   
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
   ~cf.Field.to_dask_array
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

.. rubric:: *Expanding the data*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.halo
   ~cf.Field.pad_missing

.. rubric:: *Data array mask*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.apply_masking
   ~cf.Field.count
   ~cf.Field.count_masked
   ~cf.Field.fill_value
   ~cf.Field.filled
   ~cf.Field.masked_invalid
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.binary_mask
   ~cf.Field.hardmask
   ~cf.Field.mask
   
.. rubric:: *Changing data values*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__setitem__
   ~cf.Field.subspace
   ~cf.Field.indices
   ~cf.Field.where
   ~cf.Field.apply_masking
   ~cf.Field.masked_invalid

.. rubric:: *Searching and counting*

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.argmax
   ~cf.Field.argmin
   ~cf.Field.where

Miscellaneous data operations
-----------------------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Field.cyclic
   ~cf.Field.period
   ~cf.Field.get_original_filenames
   ~cf.Field.close
   ~cf.Field.rechunk
   ~cf.Field.persist
   ~cf.Field.persist_metadata
 
Metadata constructs
-------------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.auxiliary_coordinates
   ~cf.Field.auxiliary_coordinate
   ~cf.Field.cell_connectivities
   ~cf.Field.cell_connectivity
   ~cf.Field.cell_measures
   ~cf.Field.cell_measure
   ~cf.Field.cell_methods
   ~cf.Field.cell_method
   ~cf.Field.coordinates
   ~cf.Field.coordinate
   ~cf.Field.coordinate_references
   ~cf.Field.coordinate_reference
   ~cf.Field.dimension_coordinates
   ~cf.Field.dimension_coordinate
   ~cf.Field.domain_ancillaries
   ~cf.Field.domain_ancillary
   ~cf.Field.domain_axes
   ~cf.Field.domain_axis
   ~cf.Field.domain_topologies
   ~cf.Field.domain_topology
   ~cf.Field.field_ancillaries
   ~cf.Field.field_ancillary
   ~cf.Field.construct
   ~cf.Field.construct_item
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
   ~cf.Field.auxiliary_to_dimension
   ~cf.Field.dimension_to_auxiliary
   ~cf.Field.coordinate_reference_domain_axes
   ~cf.Field.get_coordinate_reference
   ~cf.Field.set_coordinate_reference
   ~cf.Field.del_coordinate_reference
   ~cf.Field.domain_axis_key
   ~cf.Field.domain_axis_position
   ~cf.Field.domain_mask
   ~cf.Field.del_domain_axis
   ~cf.Field.map_axes
   ~cf.Field.climatological_time_axes

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.constructs

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
   ~cf.Field.compute_vertical_coordinates
   ~cf.Field.dataset_compliance
   ~cf.Field.equals
   ~cf.Field.compress
   ~cf.Field.convert
   ~cf.Field.creation_commands
   ~cf.Field.radius
   ~cf.Field.to_memory
   ~cf.Field.uncompress
   ~cf.Field.concatenate
   ~cf.Field.section

.. rubric:: Attributes
	    
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.Flags
   ~cf.Field.has_bounds
   ~cf.Field.has_geometry
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
   ~cf.Field.ncdimensions
   ~cf.Field.nc_clear_hdf5_chunksizes
   ~cf.Field.nc_hdf5_chunksizes
   ~cf.Field.nc_set_hdf5_chunksizes

Groups
^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Field.nc_variable_groups
   ~cf.Field.nc_set_variable_groups
   ~cf.Field.nc_clear_variable_groups
   ~cf.Field.nc_group_attributes
   ~cf.Field.nc_clear_group_attributes
   ~cf.Field.nc_set_group_attribute
   ~cf.Field.nc_set_group_attributes

Aggregation
^^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.file_directories
   ~cf.Field.replace_directory

Geometries
^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Field.nc_del_geometry_variable
   ~cf.Field.nc_get_geometry_variable
   ~cf.Field.nc_has_geometry_variable
   ~cf.Field.nc_set_geometry_variable 
   ~cf.Field.nc_geometry_variable_groups
   ~cf.Field.nc_set_geometry_variable_groups
   ~cf.Field.nc_clear_geometry_variable_groups

Components
^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Field.nc_del_component_variable
   ~cf.Field.nc_set_component_variable
   ~cf.Field.nc_set_component_variable_groups
   ~cf.Field.nc_clear_component_variable_groups      
   ~cf.Field.nc_del_component_dimension
   ~cf.Field.nc_set_component_dimension
   ~cf.Field.nc_set_component_dimension_groups
   ~cf.Field.nc_clear_component_dimension_groups
   ~cf.Field.nc_del_component_sample_dimension
   ~cf.Field.nc_set_component_sample_dimension   
   ~cf.Field.nc_set_component_sample_dimension_groups
   ~cf.Field.nc_clear_component_sample_dimension_groups

.. _field_methods:

Domain axes
-----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.domain_axis
   ~cf.Field.domain_axes
   ~cf.Field.analyse_items
   ~cf.Field.autocyclic
   ~cf.Field.axes
   ~cf.Field.axes_names
   ~cf.Field.axes_sizes
   ~cf.Field.axis
   ~cf.Field.axis_name
   ~cf.Field.axis_size
   ~cf.Field.cyclic
   ~cf.Field.data_axes
   ~cf.Field.direction
   ~cf.Field.directions
   ~cf.Field.iscyclic
   ~cf.Field.is_discrete_axis
   ~cf.Field.isperiodic
   ~cf.Field.item_axes
   ~cf.Field.items_axes

Subspacing
----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__getitem__
   ~cf.Field.indices

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.subspace

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Trigonometrical and hyperbolic functions

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.arccos
   ~cf.Field.arccosh
   ~cf.Field.arcsin
   ~cf.Field.arcsinh
   ~cf.Field.arctan
   .. ~cf.Field.arctan2  [AT2]
   ~cf.Field.arctanh
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
   ~cf.Field.moving_window
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
   ~cf.Field.standard_deviation
   ~cf.Field.variance
   ~cf.Field.maximum
   ~cf.Field.minimum

.. rubric:: Exponential and logarithmic functions
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.exp
   ~cf.Field.log

.. rubric:: Differential operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.derivative
   ~cf.Field.grad_xy
   ~cf.Field.laplacian_xy

.. rubric:: Convolution filters

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.convolution_filter
   ~cf.Field.moving_window

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
   ~cf.Field.histogram
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

Groups
^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.nc_variable_groups
   ~cf.Field.nc_set_variable_groups
   ~cf.Field.nc_clear_variable_groups
   ~cf.Field.nc_group_attributes
   ~cf.Field.nc_set_group_attribute
   ~cf.Field.nc_set_group_attributes
   ~cf.Field.nc_clear_group_attributes
   ~cf.Field.nc_geometry_variable_groups
   ~cf.Field.nc_set_geometry_variable_groups
   ~cf.Field.nc_clear_geometry_variable_groups

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
   ~cf.Field.__array__
   ~cf.Field.__data__
   ~cf.Field.__query_isclose__

Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst


   ~cf.Field.add_file_location
   ~cf.Field.asdatetime
   ~cf.Field.asreftime
   ~cf.Field.axis_name
   ~cf.Field.cfa_clear_file_substitutions
   ~cf.Field.cfa_del_file_substitution
   ~cf.Field.cfa_file_substitutions
   ~cf.Field.cfa_update_file_substitutions
   ~cf.Field.chunk
   ~cf.Field.data_axes
   ~cf.Field.del_file_location
   ~cf.Field.delprop
   ~cf.Field.equivalent
   ~cf.Field.example_field
   ~cf.Field.expand_dims
   ~cf.Field.field
   ~cf.Field.file_locations
   ~cf.Field.get_filenames
   ~cf.Field.getprop
   ~cf.Field.HDF_chunks
   ~cf.Field.hasprop
   ~cf.Field.insert_aux
   ~cf.Field.insert_axis
   ~cf.Field.insert_cell_methods
   ~cf.Field.insert_data
   ~cf.Field.insert_domain_anc
   ~cf.Field.insert_field_anc
   ~cf.Field.insert_item
   ~cf.Field.insert_measure
   ~cf.Field.insert_ref
   ~cf.Field.isauxiliary
   ~cf.Field.isdimension
   ~cf.Field.isdomainancillary
   ~cf.Field.isfieldancillary
   ~cf.Field.ismeasure
   ~cf.Field.item_axes
   ~cf.Field.key_item
   ~cf.Field.mask_invalid
   ~cf.Field.name
   ~cf.Field.new_identifier
   ~cf.Field.remove_axes
   ~cf.Field.remove_axis
   ~cf.Field.remove_data
   ~cf.Field.remove_item
   ~cf.Field.remove_items
   ~cf.Field.select
   ~cf.Field.setprop
   ~cf.Field.transpose_item
   ~cf.Field.unlimited

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Field.attributes
   ~cf.Field._Axes
   ~cf.Field.CellMethods
   ~cf.Field.CM
   ~cf.Field.Data
   ~cf.Field.dtvarray
   ~cf.Field.hasbounds
   ~cf.Field.hasdata
   ~cf.Field.Items
   ~cf.Field.unsafe_array
