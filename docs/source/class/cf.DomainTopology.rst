cf.DomainTopology
=================

.. currentmodule:: cf

.. autoclass:: DomainTopology

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~DomainTopology.__init__
      ~DomainTopology.all
      ~DomainTopology.allclose
      ~DomainTopology.any
      ~DomainTopology.apply_masking
      ~DomainTopology.arccos
      ~DomainTopology.arccosh
      ~DomainTopology.arcsin
      ~DomainTopology.arcsinh
      ~DomainTopology.arctan
      ~DomainTopology.arctanh
      ~DomainTopology.asdatetime
      ~DomainTopology.asreftime
      ~DomainTopology.ceil
      ~DomainTopology.chunk
      ~DomainTopology.clear_properties
      ~DomainTopology.clip
      ~DomainTopology.close
      ~DomainTopology.concatenate
      ~DomainTopology.convert_reference_time
      ~DomainTopology.copy
      ~DomainTopology.cos
      ~DomainTopology.cosh
      ~DomainTopology.count
      ~DomainTopology.count_masked
      ~DomainTopology.creation_commands
      ~DomainTopology.cyclic
      ~DomainTopology.datum
      ~DomainTopology.del_cell
      ~DomainTopology.del_data
      ~DomainTopology.del_properties
      ~DomainTopology.del_property
      ~DomainTopology.delprop
      ~DomainTopology.dump
      ~DomainTopology.equals
      ~DomainTopology.equivalent
      ~DomainTopology.exp
      ~DomainTopology.expand_dims
      ~DomainTopology.file_directories
      ~DomainTopology.fill_value
      ~DomainTopology.filled
      ~DomainTopology.flatten
      ~DomainTopology.flip
      ~DomainTopology.floor
      ~DomainTopology.get_cell
      ~DomainTopology.get_data
      ~DomainTopology.get_filenames
      ~DomainTopology.get_original_filenames
      ~DomainTopology.get_property
      ~DomainTopology.getprop
      ~DomainTopology.halo
      ~DomainTopology.has_bounds
      ~DomainTopology.has_cell
      ~DomainTopology.has_data
      ~DomainTopology.has_property
      ~DomainTopology.hasprop
      ~DomainTopology.identities
      ~DomainTopology.identity
      ~DomainTopology.insert_data
      ~DomainTopology.insert_dimension
      ~DomainTopology.inspect
      ~DomainTopology.iscyclic
      ~DomainTopology.log
      ~DomainTopology.mask_invalid
      ~DomainTopology.masked_invalid
      ~DomainTopology.match
      ~DomainTopology.match_by_identity
      ~DomainTopology.match_by_naxes
      ~DomainTopology.match_by_ncvar
      ~DomainTopology.match_by_property
      ~DomainTopology.match_by_units
      ~DomainTopology.max
      ~DomainTopology.maximum
      ~DomainTopology.mean
      ~DomainTopology.mid_range
      ~DomainTopology.min
      ~DomainTopology.minimum
      ~DomainTopology.name
      ~DomainTopology.nc_clear_hdf5_chunksizes
      ~DomainTopology.nc_clear_variable_groups
      ~DomainTopology.nc_del_variable
      ~DomainTopology.nc_get_variable
      ~DomainTopology.nc_has_variable
      ~DomainTopology.nc_hdf5_chunksizes
      ~DomainTopology.nc_set_hdf5_chunksizes
      ~DomainTopology.nc_set_variable
      ~DomainTopology.nc_set_variable_groups
      ~DomainTopology.nc_variable_groups
      ~DomainTopology.normalise
      ~DomainTopology.override_calendar
      ~DomainTopology.override_units
      ~DomainTopology.pad_missing
      ~DomainTopology.period
      ~DomainTopology.persist
      ~DomainTopology.properties
      ~DomainTopology.range
      ~DomainTopology.rechunk
      ~DomainTopology.remove_data
      ~DomainTopology.replace_directory
      ~DomainTopology.rint
      ~DomainTopology.roll
      ~DomainTopology.round
      ~DomainTopology.sample_size
      ~DomainTopology.sd
      ~DomainTopology.select
      ~DomainTopology.set_cell
      ~DomainTopology.set_data
      ~DomainTopology.set_properties
      ~DomainTopology.set_property
      ~DomainTopology.setprop
      ~DomainTopology.sin
      ~DomainTopology.sinh
      ~DomainTopology.squeeze
      ~DomainTopology.standard_deviation
      ~DomainTopology.sum
      ~DomainTopology.swapaxes
      ~DomainTopology.tan
      ~DomainTopology.tanh
      ~DomainTopology.to_dask_array
      ~DomainTopology.to_memory
      ~DomainTopology.transpose
      ~DomainTopology.trunc
      ~DomainTopology.uncompress
      ~DomainTopology.unique
      ~DomainTopology.var
      ~DomainTopology.variance
      ~DomainTopology.where
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~DomainTopology.Data
      ~DomainTopology.T
      ~DomainTopology.Units
      ~DomainTopology.X
      ~DomainTopology.Y
      ~DomainTopology.Z
      ~DomainTopology.add_offset
      ~DomainTopology.array
      ~DomainTopology.attributes
      ~DomainTopology.binary_mask
      ~DomainTopology.calendar
      ~DomainTopology.cell
      ~DomainTopology.comment
      ~DomainTopology.construct_type
      ~DomainTopology.data
      ~DomainTopology.datetime_array
      ~DomainTopology.day
      ~DomainTopology.dtarray
      ~DomainTopology.dtvarray
      ~DomainTopology.dtype
      ~DomainTopology.hardmask
      ~DomainTopology.hasbounds
      ~DomainTopology.hasdata
      ~DomainTopology.history
      ~DomainTopology.hour
      ~DomainTopology.id
      ~DomainTopology.isauxiliary
      ~DomainTopology.isdimension
      ~DomainTopology.isdomainancillary
      ~DomainTopology.isfieldancillary
      ~DomainTopology.ismeasure
      ~DomainTopology.isperiodic
      ~DomainTopology.isscalar
      ~DomainTopology.leap_month
      ~DomainTopology.leap_year
      ~DomainTopology.long_name
      ~DomainTopology.mask
      ~DomainTopology.minute
      ~DomainTopology.missing_value
      ~DomainTopology.month
      ~DomainTopology.month_lengths
      ~DomainTopology.ndim
      ~DomainTopology.reference_datetime
      ~DomainTopology.scale_factor
      ~DomainTopology.second
      ~DomainTopology.shape
      ~DomainTopology.size
      ~DomainTopology.standard_name
      ~DomainTopology.subspace
      ~DomainTopology.units
      ~DomainTopology.unsafe_array
      ~DomainTopology.valid_max
      ~DomainTopology.valid_min
      ~DomainTopology.valid_range
      ~DomainTopology.varray
      ~DomainTopology.year
   
   