.. currentmodule:: cf
.. default-role:: obj

cf.FieldAncillary
=================

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
   :toctree: ../generated/
   :template: attribute.rst

   ~cf.FieldAncillary.add_offset
   ~cf.FieldAncillary.calendar
   ~cf.FieldAncillary.cell_methods
   ~cf.FieldAncillary.comment
   ~cf.FieldAncillary.Conventions
   ~cf.FieldAncillary._FillValue
   ~cf.FieldAncillary.flag_masks
   ~cf.FieldAncillary.flag_meanings
   ~cf.FieldAncillary.flag_values
   ~cf.FieldAncillary.history
   ~cf.FieldAncillary.institution
   ~cf.FieldAncillary.leap_month
   ~cf.FieldAncillary.leap_year
   ~cf.FieldAncillary.long_name
   ~cf.FieldAncillary.missing_value
   ~cf.FieldAncillary.month_lengths
   ~cf.FieldAncillary.references
   ~cf.FieldAncillary.scale_factor
   ~cf.FieldAncillary.source
   ~cf.FieldAncillary.standard_error_multiplier
   ~cf.FieldAncillary.standard_name
   ~cf.FieldAncillary.title
   ~cf.FieldAncillary.units
   ~cf.FieldAncillary.valid_max
   ~cf.FieldAncillary.valid_min
   ~cf.FieldAncillary.valid_range

Units
-----

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldAncillary.override_units
   ~cf.FieldAncillary.override_calendar

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: attribute.rst

   ~cf.FieldAncillary.Units


Data
----

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.del_data
   ~cf.FieldAncillary.get_data
   ~cf.FieldAncillary.has_data
   ~cf.FieldAncillary.set_data
 
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
   ~cf.FieldAncillary.ndim
   ~cf.FieldAncillary.shape
   ~cf.FieldAncillary.size
   ~cf.FieldAncillary.varray

.. rubric:: Rearranging elements

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldAncillary.flip
   ~cf.FieldAncillary.insert_dimension
   ~cf.FieldAncillary.roll
   ~cf.FieldAncillary.squeeze
   ~cf.FieldAncillary.transpose
   
.. rubric:: Data array mask

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: attribute.rst

   ~cf.FieldAncillary.binary_mask
   ~cf.FieldAncillary.count
   ~cf.FieldAncillary.hardmask
   ~cf.FieldAncillary.mask

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

.. rubric:: Changing data values

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldAncillary.__setitem__
   ~cf.FieldAncillary.mask_invalid
   ~cf.FieldAncillary.subspace
   ~cf.FieldAncillary.where

.. rubric:: Miscellaneous

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst
	      
   ~cf.FieldAncillary.files

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.copy
   ~cf.FieldAncillary.equals

Mathematical operations
-----------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.math.html

.. rubric:: Trigonometrical functions

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldAncillary.cos
   ~cf.FieldAncillary.sin
   ~cf.FieldAncillary.tan

.. rubric:: Rounding and truncation

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldAncillary.ceil  
   ~cf.FieldAncillary.clip
   ~cf.FieldAncillary.floor
   ~cf.FieldAncillary.rint
   ~cf.FieldAncillary.round
   ~cf.FieldAncillary.trunc

.. rubric:: Statistical collapses

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
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

.. rubric:: Exponential and logarithmic functions
	    
.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldAncillary.exp
   ~cf.FieldAncillary.log

Data array operations
---------------------

.. http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html

.. _field_data_array_access:



.. rubric:: Adding and removing elements

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldAncillary.unique

.. rubric:: Miscellaneous data array operations

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: attribute.rst

   ~cf.FieldAncillary.chunk
   ~cf.FieldAncillary.isscalar

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

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldAncillary.__deepcopy__
   ~cf.FieldAncillary.__getitem__
   ~cf.FieldAncillary.__repr__
   ~cf.FieldAncillary.__str__

   
.. 'Data',
    'T',
    'Units',
    'X',
    'Y',
    'Z',
    '_ATOL',
    '_FillValue',
    '_RTOL',
    '_YMDhms',
    '__abs__',
    '__abstractmethods__',
    '__add__',
    '__and__',
    '__array__',
    '__class__',
    '__contains__',
    '__data__',
    '__deepcopy__',
    '__delattr__',
    '__dict__',
    '__dir__',
    '__div__',
    '__doc__',
    '__eq__',
    '__floordiv__',
    '__format__',
    '__ge__',
    '__getattribute__',
    '__getitem__',
    '__gt__',
    '__hash__',
    '__iadd__',
    '__iand__',
    '__idiv__',
    '__ifloordiv__',
    '__ilshift__',
    '__imod__',
    '__imul__',
    '__init__',
    '__init_subclass__',
    '__invert__',
    '__ior__',
    '__ipow__',
    '__irshift__',
    '__isub__',
    '__itruediv__',
    '__ixor__',
    '__le__',
    '__lshift__',
    '__lt__',
    '__mod__',
    '__module__',
    '__mul__',
    '__ne__',
    '__neg__',
    '__new__',
    '__or__',
    '__pos__',
    '__pow__',
    '__query_set__',
    '__query_wi__',
    '__query_wo__',
    '__radd__',
    '__rand__',
    '__rdiv__',
    '__reduce__',
    '__reduce_ex__',
    '__repr__',
    '__rfloordiv__',
    '__rlshift__',
    '__rmod__',
    '__rmul__',
    '__ror__',
    '__rpow__',
    '__rrshift__',
    '__rshift__',
    '__rsub__',
    '__rtruediv__',
    '__rxor__',
    '__setattr__',
    '__setitem__',
    '__sizeof__',
    '__str__',
    '__sub__',
    '__subclasshook__',
    '__truediv__',
    '__weakref__',
    '__xor__',
    '_abc_impl',
    '_binary_operation',
    '_components',
    '_conform_for_assignment',
    '_custom',
    '_default',
    '_del_component',
    '_dump_properties',
    '_equals',
    '_equals_preprocess',
    '_equivalent_data',
    '_get_component',
    '_has_component',
    '_initialise_netcdf',
    '_matching_values',
    '_parse_axes',
    '_parse_match',
    '_set_component',
    '_special_properties',
    '_unary_operation',
    'add_offset',
    'all',
    'allclose',
    'any',
    'asdatetime',
    'asreftime',
    'attributes',
    'binary_mask',
    'calendar',
    'clear_properties',
    'close',
    'comment',
    'concatenate',
    'construct_type',
    'convert_reference_time',
    'count',
    'count_masked',
    'cyclic',
    'datum',
    'day',
    'del_property',
    'dtarray',
    'dump',
    'equivalent',
    'fill_value',
    'get_property',
    'getprop',
    'hardmask',
    'has_bounds',
    'has_property',
    'hasbounds',
    'hasdata',
    'hasprop',
    'history',
    'hour',
    'id',
    'identities',
    'identity',
    'inspect',
    'isauxiliary',
    'isdimension',
    'isdomainancillary',
    'isfieldancillary',
    'ismeasure',
    'leap_month',
    'leap_year',
    'long_name',
    'mask',
    'mask_invalid',
    'match',
    'match_by_identity',
    'match_by_naxes',
    'match_by_ncvar',
    'match_by_property',
    'match_by_units',
    'missing_value',
    'month',
    'month_lengths',
    'name',
    'override_calendar',
    'override_units',
    'properties',
    'reference_datetime',
    'scale_factor',
    'second',
    'select',
    'set_properties',
    'set_property',
    'setprop',
    'standard_name',
    'subspace',
    'units',
    'valid_max',
    'valid_min',
    'valid_range',
   x1 'year'
