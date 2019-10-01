.. currentmodule:: cf
.. default-role:: obj

.. _two-to-three-changes:
		  
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

Version 3.x only works for Python 3.5 or later. Version 2.x only works
for python version 2.7.

In-place operations
-------------------

At version 3.x, in-place operations return `None`, rather than the
modified construct. The keyword that defines the operation to be
in-place is now *inplace* (rather than *i*).

.. code:: python

   >>> g = f.tranpose()
   >>> print(type(g))
   <class 'cf.field.Field'>
   >>> x = f.tranpose(inplace=True)
   >>> print(x)
   None
	  

New methods and attributes
--------------------------

Version 3.x methods and the deprecated version 2.x methods and
attributes that they replace:

========================================================  ===============================  
Version 3.x                                      	  Version 2.x                      
========================================================  ===============================  
`cf.Field.cell_methods.ordered`			 	  `cf.Field.CellMethods`           
						 	                                 
`cf.Field.convert`				 	  `cf.Field.field`                 
						 	                                   
`cf.Field.domain_axis_identity`			 	  `cf.Field.axis_name`             
						 		                                   
`cf.Field.del_construct`			 	  `cf.Field.remove_item`

`cf.Field.get_data_axes`			 	  `cf.Field.data_axes`,
                                                          `cf.Field.item_axes`            
						 	                                   
`cf.Field.insert_dimension`			 	  `cf.Field.expand_dims`           
						 	                                   
`cf.Field.set_construct`			 	  `cf.Field.insert_aux`,            
                           			 	  `cf.Field.insert_axis`,          
                           			 	  `cf.Field.insert_dim`,            
                           			 	  `cf.Field.insert_domain_anc`,     
                           			 	  `cf.Field.insert_field_anc`,      
                           			 	  `cf.Field.insert_item`,           
                           			 	  `cf.Field.insert_measure`,        
                           			 	  `cf.Field.insert_ref`            

`cf.Field.set_data`				 	  `cf.Field.insert_data`           

`cf.Constructs.data_axes`	   	 	          `cf.Field.items_axes`            

`cf.Data.nc_hdf5_chunksizes`	    	 	          `cf.Field.HDF_chunks`            
    
`cf.Data.nc_set_hdf5_chunksizes`	           	  `cf.Field.HDF_chunks`    	                           
    
`cf.Data.nc_clear_hdf5_chunksizes`	 	          `cf.Field.HDF_chunks`
========================================================  ===============================  
