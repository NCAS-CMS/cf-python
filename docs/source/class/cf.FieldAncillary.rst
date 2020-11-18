cf.FieldAncillary
=================

.. currentmodule:: cf

.. autoclass:: FieldAncillary

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~FieldAncillary.__init__
      ~FieldAncillary.all
      ~FieldAncillary.allclose
      ~FieldAncillary.any
      ~FieldAncillary.apply_masking
      ~FieldAncillary.arccos
      ~FieldAncillary.arccosh
      ~FieldAncillary.arcsin
      ~FieldAncillary.arcsinh
      ~FieldAncillary.arctan
      ~FieldAncillary.arctanh
      ~FieldAncillary.asdatetime
      ~FieldAncillary.asreftime
      ~FieldAncillary.ceil
      ~FieldAncillary.chunk
      ~FieldAncillary.clear_properties
      ~FieldAncillary.clip
      ~FieldAncillary.close
      ~FieldAncillary.concatenate
      ~FieldAncillary.convert_reference_time
      ~FieldAncillary.copy
      ~FieldAncillary.cos
      ~FieldAncillary.cosh
      ~FieldAncillary.count
      ~FieldAncillary.count_masked
      ~FieldAncillary.creation_commands
      ~FieldAncillary.cyclic
      ~FieldAncillary.datum
      ~FieldAncillary.del_data
      ~FieldAncillary.del_property
      ~FieldAncillary.delprop
      ~FieldAncillary.dump
      ~FieldAncillary.equals
      ~FieldAncillary.equivalent
      ~FieldAncillary.exp
      ~FieldAncillary.expand_dims
      ~FieldAncillary.fill_value
      ~FieldAncillary.flatten
      ~FieldAncillary.flip
      ~FieldAncillary.floor
      ~FieldAncillary.get_data
      ~FieldAncillary.get_filenames
      ~FieldAncillary.get_property
      ~FieldAncillary.getprop
      ~FieldAncillary.halo
      ~FieldAncillary.has_bounds
      ~FieldAncillary.has_data
      ~FieldAncillary.has_property
      ~FieldAncillary.hasprop
      ~FieldAncillary.identities
      ~FieldAncillary.identity
      ~FieldAncillary.insert_data
      ~FieldAncillary.insert_dimension
      ~FieldAncillary.inspect
      ~FieldAncillary.iscyclic
      ~FieldAncillary.log
      ~FieldAncillary.mask_invalid
      ~FieldAncillary.match
      ~FieldAncillary.match_by_identity
      ~FieldAncillary.match_by_naxes
      ~FieldAncillary.match_by_ncvar
      ~FieldAncillary.match_by_property
      ~FieldAncillary.match_by_units
      ~FieldAncillary.max
      ~FieldAncillary.maximum
      ~FieldAncillary.mean
      ~FieldAncillary.mid_range
      ~FieldAncillary.min
      ~FieldAncillary.minimum
      ~FieldAncillary.name
      ~FieldAncillary.nc_clear_variable_groups
      ~FieldAncillary.nc_del_variable
      ~FieldAncillary.nc_get_variable
      ~FieldAncillary.nc_has_variable
      ~FieldAncillary.nc_set_variable
      ~FieldAncillary.nc_set_variable_groups
      ~FieldAncillary.nc_variable_groups
      ~FieldAncillary.override_calendar
      ~FieldAncillary.override_units
      ~FieldAncillary.period
      ~FieldAncillary.properties
      ~FieldAncillary.range
      ~FieldAncillary.remove_data
      ~FieldAncillary.rint
      ~FieldAncillary.roll
      ~FieldAncillary.round
      ~FieldAncillary.sample_size
      ~FieldAncillary.sd
      ~FieldAncillary.select
      ~FieldAncillary.set_data
      ~FieldAncillary.set_properties
      ~FieldAncillary.set_property
      ~FieldAncillary.setprop
      ~FieldAncillary.sin
      ~FieldAncillary.sinh
      ~FieldAncillary.squeeze
      ~FieldAncillary.standard_deviation
      ~FieldAncillary.sum
      ~FieldAncillary.swapaxes
      ~FieldAncillary.tan
      ~FieldAncillary.tanh
      ~FieldAncillary.transpose
      ~FieldAncillary.trunc
      ~FieldAncillary.uncompress
      ~FieldAncillary.unique
      ~FieldAncillary.var
      ~FieldAncillary.variance
      ~FieldAncillary.where
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~FieldAncillary.Data
      ~FieldAncillary.T
      ~FieldAncillary.Units
      ~FieldAncillary.X
      ~FieldAncillary.Y
      ~FieldAncillary.Z
      ~FieldAncillary.add_offset
      ~FieldAncillary.array
      ~FieldAncillary.attributes
      ~FieldAncillary.binary_mask
      ~FieldAncillary.calendar
      ~FieldAncillary.comment
      ~FieldAncillary.construct_type
      ~FieldAncillary.data
      ~FieldAncillary.datetime_array
      ~FieldAncillary.day
      ~FieldAncillary.dtarray
      ~FieldAncillary.dtvarray
      ~FieldAncillary.dtype
      ~FieldAncillary.hardmask
      ~FieldAncillary.hasbounds
      ~FieldAncillary.hasdata
      ~FieldAncillary.history
      ~FieldAncillary.hour
      ~FieldAncillary.id
      ~FieldAncillary.isauxiliary
      ~FieldAncillary.isdimension
      ~FieldAncillary.isdomainancillary
      ~FieldAncillary.isfieldancillary
      ~FieldAncillary.ismeasure
      ~FieldAncillary.isperiodic
      ~FieldAncillary.isscalar
      ~FieldAncillary.leap_month
      ~FieldAncillary.leap_year
      ~FieldAncillary.long_name
      ~FieldAncillary.mask
      ~FieldAncillary.minute
      ~FieldAncillary.missing_value
      ~FieldAncillary.month
      ~FieldAncillary.month_lengths
      ~FieldAncillary.ndim
      ~FieldAncillary.reference_datetime
      ~FieldAncillary.scale_factor
      ~FieldAncillary.second
      ~FieldAncillary.shape
      ~FieldAncillary.size
      ~FieldAncillary.standard_name
      ~FieldAncillary.subspace
      ~FieldAncillary.units
      ~FieldAncillary.unsafe_array
      ~FieldAncillary.valid_max
      ~FieldAncillary.valid_min
      ~FieldAncillary.valid_range
      ~FieldAncillary.varray
      ~FieldAncillary.year
   
   