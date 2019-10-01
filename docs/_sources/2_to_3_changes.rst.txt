.. currentmodule:: cf
.. default-role:: obj

.. _two-to-three-changes:
		  
Incompatible differences between versions 2.x and 3.x
=====================================================

For those familiar with the cf-python API at version 2.x, some
important, backwards incompatible changes were introduced at version
3.0.0.

Scripts written for version 2.x but running under version 3.x should
either work as expected, or provide informative error mesages on the
new API usage. However, it is advised that the outputs of older
scripts be checked when running with Python 3 versions of the cf
library.


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


Changes to method APIs
----------------------

Methods that have a different API in version 3.x

========================================================  ============================================
Version 3.x                                      	  Changes compared to version 2.x
========================================================  ============================================
`cf.Field.anchor`  			 	          ``inplace`` replaces ``i`` parameter.
			 	                          No ``**kwargs`` parameters.

`cf.Field.aux`  			 	          No ``**kwargs`` parameters.

`cf.Field.auxs`  			 	          No ``**kwargs`` parameters.

`cf.Field.axes`  			 	          No ``**kwargs`` parameters.

`cf.Field.axes_names`  			 	          No ``**kwargs`` parameters.

`cf.Field.axis`  			 	          No ``**kwargs`` parameters.

`cf.Field.axis_size`  			 	          ``identity`` replaces ``axes`` parameter.
			 	                          No ``**kwargs`` parameters.

`cf.Field.cell_area`			 	          No ``insert`` parameter.
						 	                                 
`cf.Field.collapse`			 	          New ``verbose`` parameter.
			 	                          ``inplace`` replaces ``i`` parameter.
			 	                          No ``**kwargs`` parameters.
 
`cf.Field.convolution_filter`		 	          ``inplace`` replaces ``i`` parameter.
			 	                          ``weights`` parameter can not be a string.
						 	                                 
`cf.Field.coord`  			 	          No ``**kwargs`` parameters.

`cf.Field.coords`  			 	          No ``**kwargs`` parameters.

`cf.Field.cyclic`			 	          No ``**kwargs`` parameters.
						 	                                 
`cf.Field.derivative`  			 	          ``inplace`` replaces ``i`` parameter.
  			 	                          ``wrap`` replaces ``cyclic`` parameter.

`cf.Field.dim`  			 	          No ``**kwargs`` parameters.

`cf.Field.dims`  			 	          No ``**kwargs`` parameters.

`cf.Field.direction`			 	          No ``axes`` parameter.
                                                          No ``**kwargs`` parameters.

`cf.Field.domain_anc`  			 	          No ``**kwargs`` parameters.

`cf.Field.domain_ancs`  		 	          No ``**kwargs`` parameters.

`cf.Field.field_anc`  			 	          No ``**kwargs`` parameters.

`cf.Field.field_ancs`  	 	 	                  No ``**kwargs`` parameters.

`cf.Field.flip`  			 	          ``inplace`` replaces ``i`` parameter.
			 	                          No ``**kwargs`` parameters.

`cf.Field.indices`			 	          No ``'exact'`` mode.
						 	                                 
`cf.Field.iscyclic`			 	          No ``**kwargs`` parameters.
						 	                                 
`cf.Field.item`  			 	          No ``**kwargs`` parameters.

`cf.Field.items`  	 	 	                  No ``**kwargs`` parameters.

`cf.Field.key`  			 	          No ``**kwargs`` parameters.

`cf.Field.measure`  			 	          No ``**kwargs`` parameters.

`cf.Field.measures`  	 	 	                  No ``**kwargs`` parameters.

`cf.Field.period`  			 	          No ``**kwargs`` parameters.

`cf.Field.ref`  			 	          No ``**kwargs`` parameters.

`cf.Field.refs`  	 	 	                  No ``**kwargs`` parameters.

`cf.Field.regridc`  			 	          ``inplace`` replaces ``i`` parameter.

`cf.Field.regrids`  			 	          ``inplace`` replaces ``i`` parameter.

`cf.Field.roll`  			 	          ``inplace`` replaces ``i`` parameter.

`cf.Field.squeeze`  			 	          ``inplace`` replaces ``i`` parameter.
			 	                          No ``**kwargs`` parameters.

`cf.Field.tranpose`  			 	          ``inplace`` replaces ``i`` parameter.
			 	                          ``constructs`` replaces ``items`` parameter.
			 	                          No ``**kwargs`` parameters.

`cf.Field.unsqueeze`  			 	          ``inplace`` replaces ``i`` parameter.
			 	                          No ``axes`` parameter.
                                                          No ``**kwargs`` parameters.

`cf.Field.where`  			 	          ``inplace`` replaces ``i`` parameter.
			 	                          ``construct`` replaces ``item`` parameter.
			 	                          No ``axes`` parameter.
                                                          No ``**item_options`` parameters.
========================================================  ============================================  
