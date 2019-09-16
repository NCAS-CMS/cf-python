.. currentmodule:: cf
.. default-role:: obj

.. _2_to_3_changes:
		  
Incompatible differences between versions 2.x and 3.x
=====================================================

For those familiar with the cf-python API at version 2.x, some
important, backwards incompatible changes were introduced at version
3.0.0.

Some of these changes could break code written at version 2.x, causing
an exception to be raised with a message on how to change the code to
work at version 3.x. 


Python
------

.. note:: Version 3.x only works for Python 3.5 or later.

Version 2.x only works for python version 2.7.


Deprecated methods and attributes
---------------------------------

===============================  =======================================================================
Version 2.x                      Version 3.x                                     
===============================  =======================================================================
`cf.Field.axis_name`             Use `cf.Field.domain_axis_identity` method.
			         
`cf.Field.CellMethods`           Use `cf.Field.cell_methods.ordered` method.

`cf.Field.data_axes`             Use `cf.Field.get_data_axes` method.
			         
`cf.Field.expand_dims`           Use `cf.Field.insert_dimension` method.
			         
`cf.Field.field`                 Use `cf.Field.convert` method.
			         
`cf.Field.HDF_chunks`            Use
                                 `cf.Field.data.nc_hdf5_chunksizes`,
                                 `cf.Field.data.nc_set_hdf5_chunksizes`,
			         `cf.Field.data.nc_clear_hdf5_chunksizes`
                                 methods.
			         
`cf.Field.insert_axis`           Use `cf.Field.set_construct` method.
			         
`cf.Field.insert_aux`            Use `cf.Field.set_construct` method.
			         
`cf.Field.insert_data`           Use `cf.Field.set_data` method.
			         
`cf.Field.insert_dim`            Use `cf.Field.set_construct` method.

`cf.Field.insert_domain_anc`     Use `cf.Field.set_construct` method.

`cf.Field.insert_field_anc`      Use `cf.Field.set_construct` method.

`cf.Field.insert_item`           Use `cf.Field.set_construct` method.

`cf.Field.item_axes`             Use `cf.Field.get_data_axes` method.

`cf.Field.items_axes`            Use the `~cf.Construct.data_axes` method of the `cf.Field.constructs`
                                 attribute.

`cf.Field.insert_measure`        Use `cf.Field.set_construct` method.
			         
`cf.Field.insert_ref`            Use `cf.Field.set_construct` method.

`cf.Field.remove_item`           Use `cf.Field.del_construct` method.
===============================  =======================================================================
