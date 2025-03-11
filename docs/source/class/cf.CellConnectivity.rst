cf.CellConnectivity
===================

.. currentmodule:: cf

.. autoclass:: CellConnectivity

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~CellConnectivity.__init__
      ~CellConnectivity.all
      ~CellConnectivity.allclose
      ~CellConnectivity.any
      ~CellConnectivity.apply_masking
      ~CellConnectivity.arccos
      ~CellConnectivity.arccosh
      ~CellConnectivity.arcsin
      ~CellConnectivity.arcsinh
      ~CellConnectivity.arctan
      ~CellConnectivity.arctanh
      ~CellConnectivity.asdatetime
      ~CellConnectivity.asreftime
      ~CellConnectivity.ceil
      ~CellConnectivity.chunk
      ~CellConnectivity.clear_properties
      ~CellConnectivity.clip
      ~CellConnectivity.close
      ~CellConnectivity.concatenate
      ~CellConnectivity.convert_reference_time
      ~CellConnectivity.copy
      ~CellConnectivity.cos
      ~CellConnectivity.cosh
      ~CellConnectivity.count
      ~CellConnectivity.count_masked
      ~CellConnectivity.creation_commands
      ~CellConnectivity.cyclic
      ~CellConnectivity.datum
      ~CellConnectivity.del_connectivity
      ~CellConnectivity.del_data
      ~CellConnectivity.del_properties
      ~CellConnectivity.del_property
      ~CellConnectivity.delprop
      ~CellConnectivity.dump
      ~CellConnectivity.equals
      ~CellConnectivity.equivalent
      ~CellConnectivity.exp
      ~CellConnectivity.expand_dims
      ~CellConnectivity.file_directories
      ~CellConnectivity.fill_value
      ~CellConnectivity.filled
      ~CellConnectivity.flatten
      ~CellConnectivity.flip
      ~CellConnectivity.floor
      ~CellConnectivity.get_connectivity
      ~CellConnectivity.get_data
      ~CellConnectivity.get_filenames
      ~CellConnectivity.get_original_filenames
      ~CellConnectivity.get_property
      ~CellConnectivity.getprop
      ~CellConnectivity.halo
      ~CellConnectivity.has_bounds
      ~CellConnectivity.has_connectivity
      ~CellConnectivity.has_data
      ~CellConnectivity.has_property
      ~CellConnectivity.hasprop
      ~CellConnectivity.identities
      ~CellConnectivity.identity
      ~CellConnectivity.insert_data
      ~CellConnectivity.insert_dimension
      ~CellConnectivity.inspect
      ~CellConnectivity.iscyclic
      ~CellConnectivity.log
      ~CellConnectivity.mask_invalid
      ~CellConnectivity.masked_invalid
      ~CellConnectivity.match
      ~CellConnectivity.match_by_identity
      ~CellConnectivity.match_by_naxes
      ~CellConnectivity.match_by_ncvar
      ~CellConnectivity.match_by_property
      ~CellConnectivity.match_by_units
      ~CellConnectivity.max
      ~CellConnectivity.maximum
      ~CellConnectivity.mean
      ~CellConnectivity.mid_range
      ~CellConnectivity.min
      ~CellConnectivity.minimum
      ~CellConnectivity.name
      ~CellConnectivity.nc_clear_hdf5_chunksizes
      ~CellConnectivity.nc_clear_variable_groups
      ~CellConnectivity.nc_del_variable
      ~CellConnectivity.nc_get_variable
      ~CellConnectivity.nc_has_variable
      ~CellConnectivity.nc_hdf5_chunksizes
      ~CellConnectivity.nc_set_hdf5_chunksizes
      ~CellConnectivity.nc_set_variable
      ~CellConnectivity.nc_set_variable_groups
      ~CellConnectivity.nc_variable_groups
      ~CellConnectivity.normalise
      ~CellConnectivity.override_calendar
      ~CellConnectivity.override_units
      ~CellConnectivity.pad_missing
      ~CellConnectivity.period
      ~CellConnectivity.persist
      ~CellConnectivity.properties
      ~CellConnectivity.range
      ~CellConnectivity.rechunk
      ~CellConnectivity.remove_data
      ~CellConnectivity.replace_directory
      ~CellConnectivity.rint
      ~CellConnectivity.roll
      ~CellConnectivity.round
      ~CellConnectivity.sample_size
      ~CellConnectivity.sd
      ~CellConnectivity.select
      ~CellConnectivity.set_connectivity
      ~CellConnectivity.set_data
      ~CellConnectivity.set_properties
      ~CellConnectivity.set_property
      ~CellConnectivity.setprop
      ~CellConnectivity.sin
      ~CellConnectivity.sinh
      ~CellConnectivity.squeeze
      ~CellConnectivity.standard_deviation
      ~CellConnectivity.sum
      ~CellConnectivity.swapaxes
      ~CellConnectivity.tan
      ~CellConnectivity.tanh
      ~CellConnectivity.to_dask_array
      ~CellConnectivity.to_memory
      ~CellConnectivity.transpose
      ~CellConnectivity.trunc
      ~CellConnectivity.uncompress
      ~CellConnectivity.unique
      ~CellConnectivity.var
      ~CellConnectivity.variance
      ~CellConnectivity.where
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~CellConnectivity.Data
      ~CellConnectivity.T
      ~CellConnectivity.Units
      ~CellConnectivity.X
      ~CellConnectivity.Y
      ~CellConnectivity.Z
      ~CellConnectivity.add_offset
      ~CellConnectivity.array
      ~CellConnectivity.attributes
      ~CellConnectivity.binary_mask
      ~CellConnectivity.calendar
      ~CellConnectivity.comment
      ~CellConnectivity.connectivity
      ~CellConnectivity.construct_type
      ~CellConnectivity.data
      ~CellConnectivity.datetime_array
      ~CellConnectivity.day
      ~CellConnectivity.dtarray
      ~CellConnectivity.dtvarray
      ~CellConnectivity.dtype
      ~CellConnectivity.hardmask
      ~CellConnectivity.hasbounds
      ~CellConnectivity.hasdata
      ~CellConnectivity.history
      ~CellConnectivity.hour
      ~CellConnectivity.id
      ~CellConnectivity.isauxiliary
      ~CellConnectivity.isdimension
      ~CellConnectivity.isdomainancillary
      ~CellConnectivity.isfieldancillary
      ~CellConnectivity.ismeasure
      ~CellConnectivity.isperiodic
      ~CellConnectivity.isscalar
      ~CellConnectivity.leap_month
      ~CellConnectivity.leap_year
      ~CellConnectivity.long_name
      ~CellConnectivity.mask
      ~CellConnectivity.minute
      ~CellConnectivity.missing_value
      ~CellConnectivity.month
      ~CellConnectivity.month_lengths
      ~CellConnectivity.ndim
      ~CellConnectivity.reference_datetime
      ~CellConnectivity.scale_factor
      ~CellConnectivity.second
      ~CellConnectivity.shape
      ~CellConnectivity.size
      ~CellConnectivity.standard_name
      ~CellConnectivity.subspace
      ~CellConnectivity.units
      ~CellConnectivity.unsafe_array
      ~CellConnectivity.valid_max
      ~CellConnectivity.valid_min
      ~CellConnectivity.valid_range
      ~CellConnectivity.varray
      ~CellConnectivity.year
   
   