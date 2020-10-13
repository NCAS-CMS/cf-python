.. currentmodule:: cf
.. default-role:: obj

.. _two-to-three-changes:
		  
Incompatible differences between versions 2.x and 3.x
=====================================================

For those familiar with the cf-python API at version 2.x, some
important, backwards incompatible changes were introduced at version
3.0.0.

Scripts written for version 2.x but running under version 3.x should
either work as expected, or provide informative error messages on the
new API usage. However, it is advised that the outputs of older
scripts are checked when running with Python 3 versions of the cf
library.

For version 2.x documentation, see the :ref:`releases <Releases>`
page.


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

Field creation
--------------

Field construct creation is has changed in two notable ways:

Firstly, a single method `cf.Field.set_construct` has replaced the
separated methods for each construct type
(e.g. `cf.Field.insert_dim`).

Secondly, domain axis constructs are no longer created on-demand when
data or dimension coordinate constructs are set on the new field
construct. i.e the domain axis constructs have to be created
explicitly before any data or metadata constructs that use them are
set on the field construct:

.. code:: python

   >>> f = cf.Field()
   >>> f.set_construct(cf.DomainAxis(73))
   >>> f.set_construct(cf.DomainAxis(96))
   >>> data = cf.Data(numpy.arange(7008.).reshape(73, 96))
   >>> f.set_data(data)

Note that the `cf.Field.set_data` and `cf.Field.set_construct` methods
are able to match up existing domain axis constructs to the new data
or construct if the mapping can be made unambiguously (as is the case
in the above example).
   
New methods that replace deprecated ones
----------------------------------------

Version 3.x methods that replace deprecated version 2.x methods:

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
						 	                                   
`cf.Field.nc_del_variable`			 	  `cf.Field.ncvar`
						 	                                   
`cf.Field.nc_get_variable`			 	  `cf.Field.ncvar`
						 	                                   
`cf.Field.nc_set_variable`			 	  `cf.Field.ncvar`
						 	                                   
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



Changes to the  API of existing methods
---------------------------------------

Methods that have a different API in version 3.x

==========================================  =====================================================================
Version 3.x                                 API changes compared to version 2.x
==========================================  =====================================================================
`cf.Field.anchor`  			    ``inplace`` replaces ``i`` parameter.
			 	            No ``**kwargs`` parameters.

`cf.Field.aux`  			    No ``**kwargs`` parameters. Construct identity arguments
                                            are no longer assumed to be an abbreviation.

`cf.Field.auxs`  			    No ``**kwargs`` parameters. Construct identity arguments
                                            are no longer assumed to be an abbreviation.

`cf.Field.axes`  			    No ``**kwargs`` parameters. Construct identity arguments
                                            are no longer assumed to be an abbreviation.

`cf.Field.axes_names`  			    No ``**kwargs`` parameters. Construct identity arguments
                                            are no longer assumed to be an abbreviation.

`cf.Field.axis`  			    No ``**kwargs`` parameters. Construct identity arguments
                                            are no longer assumed to be an abbreviation.

`cf.Field.axis_size`  			    ``identity`` replaces ``axes`` parameter.
			 	            No ``**kwargs`` parameters. Construct identity arguments
                                            are no longer assumed to be an abbreviation.

`cf.Field.ceil`    		 	    ``inplace`` replaces ``i`` parameter. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.
					    
`cf.Field.cell_area`			    No ``insert`` parameter.
					                                   
`cf.Field.clip`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.collapse`			    New ``verbose`` parameter.
			 	            ``inplace`` replaces ``i`` parameter.
			 	            No ``**kwargs`` parameters. Construct identity arguments
                                            are no longer assumed to be an abbreviation.
 
`cf.Field.convert_reference_time`	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.convolution_filter`		    ``inplace`` replaces ``i`` parameter.
			 	            ``weights`` parameter can not be a string. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.
					                                   
`cf.Field.coord`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.coords`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.cos`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.cyclic`			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.
					                                   
`cf.Field.derivative`  			    ``inplace`` replaces ``i`` parameter.
  			 	            ``wrap`` replaces ``cyclic`` parameter. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.dim`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.dims`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.direction`			    No ``axes`` parameter.
                                            No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.domain_anc`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.domain_ancs`  		    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.exp`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.field_anc`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.field_ancs`  	 	 	    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.flip`  			    ``inplace`` replaces ``i`` parameter.
			 	            No ``**kwargs`` parameters.

`cf.Field.floor`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.identity`    		 	    No ``relaxed_identity`` parameter.
					    
`cf.Field.indices`			    No ``'exact'`` mode. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.
					                                   
`cf.Field.iscyclic`			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.
					                                   
`cf.Field.item`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.items`  	 	 	    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.key`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.log`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.mask_invalid`  		    ``inplace`` replaces ``i`` parameter.

`cf.Field.measure`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.measures`  	 	 	    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.override_calendar`   	 	    ``inplace`` replaces ``i`` parameter.

`cf.Field.override_units`   	 	    ``inplace`` replaces ``i`` parameter.

`cf.Field.period`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.ref`  			    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.refs`  	 	 	    No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.regridc`  			    ``inplace`` replaces ``i`` parameter. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.regrids`  			    ``inplace`` replaces ``i`` parameter. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.rint`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.roll`  			    ``inplace`` replaces ``i`` parameter. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.round`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.sin`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.squeeze`  			    ``inplace`` replaces ``i`` parameter. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.
			 	            No ``**kwargs`` parameters.

`cf.Field.tan`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.tranpose`  			    ``inplace`` replaces ``i`` parameter.
			 	            ``constructs`` replaces ``items`` parameter.
			 	            No ``**kwargs`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.

`cf.Field.trunc`    		 	    ``inplace`` replaces ``i`` parameter.
					    
`cf.Field.unsqueeze`  			    ``inplace`` replaces ``i`` parameter.
			 	            No ``axes`` parameter.
                                            No ``**kwargs`` parameters.

`cf.Field.where`  			    ``inplace`` replaces ``i`` parameter.
			 	            ``construct`` replaces ``item`` parameter.
			 	            No ``axes`` parameter.
                                            No ``**item_options`` parameters. Construct identity 
                                            arguments are no longer assumed to be an abbreviation.
==========================================  =====================================================================

