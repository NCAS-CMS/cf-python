.. currentmodule:: cf
.. default-role:: obj

.. _cf-Domain:
 
cf.Domain
=========

----

.. autoclass:: cfdm.Domain
   :no-members:
   :no-inherited-members:

.. rubric:: Methods

Inspection
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.dump
   ~cf.Domain.identity  
   ~cf.Domain.identities

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Domain.construct_type
   ~cf.Domain.size
   ~cf.Domain.rank
   
Properties
----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.del_property
   ~cf.Domain.get_property
   ~cf.Domain.has_property
   ~cf.Domain.set_property
   ~cf.Domain.properties
   ~cf.Domain.clear_properties
   ~cf.Domain.del_properties
   ~cf.Domain.set_properties

.. rubric:: Attributes
	    
.. autosummary::
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Domain.calendar
   ~cf.Domain.comment
   ~cf.Domain.history
   ~cf.Domain.leap_month
   ~cf.Domain.leap_year
   ~cf.Domain.long_name
   ~cf.Domain.month_lengths
   ~cf.Domain.standard_name
   ~cf.Domain.units
   ~cf.Domain.valid_max
   ~cf.Domain.valid_min
   ~cf.Domain.valid_range

Selection
---------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.match_by_identity
   ~cf.Domain.match_by_ncvar
   ~cf.Domain.match_by_property
   ~cf.Domain.match_by_rank
   ~cf.Domain.match_by_construct
 
Rearranging
-----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.anchor
   ~cf.Domain.roll
   ~cf.Domain.flip
   ~cf.Domain.transpose

Metadata constructs
-------------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.auxiliary_coordinates
   ~cf.Domain.auxiliary_coordinate
   ~cf.Domain.cell_connectivities
   ~cf.Domain.cell_connectivity
   ~cf.Domain.cell_measures
   ~cf.Domain.cell_measure
   ~cf.Domain.coordinates
   ~cf.Domain.coordinate
   ~cf.Domain.coordinate_references
   ~cf.Domain.coordinate_reference
   ~cf.Domain.dimension_coordinates
   ~cf.Domain.dimension_coordinate
   ~cf.Domain.domain_ancillaries
   ~cf.Domain.domain_ancillary
   ~cf.Domain.domain_axes
   ~cf.Domain.domain_axis
   ~cf.Domain.domain_topologies
   ~cf.Domain.domain_topology
   ~cf.Domain.construct
   ~cf.Domain.construct_item
   ~cf.Domain.construct_key
   ~cf.Domain.del_construct
   ~cf.Domain.get_construct
   ~cf.Domain.has_construct
   ~cf.Domain.set_construct
   ~cf.Domain.replace_construct
   ~cf.Domain.del_data_axes
   ~cf.Domain.get_data_axes
   ~cf.Domain.has_data_axes
   ~cf.Domain.set_data_axes
   ~cf.Domain.auxiliary_to_dimension
   ~cf.Domain.dimension_to_auxiliary
   ~cf.Domain.coordinate_reference_domain_axes
   ~cf.Domain.get_coordinate_reference
   ~cf.Domain.set_coordinate_reference
   ~cf.Domain.del_coordinate_reference
   ~cf.Domain.domain_axis_key
   ~cf.Domain.del_domain_axis
   ~cf.Domain.climatological_time_axes

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.Domain.constructs
 
Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.apply_masking
   ~cf.Domain.climatological_time_axes
   ~cf.Domain.copy
   ~cf.Domain.create_regular   
   ~cf.Domain.creation_commands
   ~cf.Domain.equals
   ~cf.Domain.fromconstructs
   ~cf.Domain.has_bounds
   ~cf.Domain.has_data
   ~cf.Domain.has_geometry
   ~cf.Domain.apply_masking   
   ~cf.Domain.get_original_filenames
   ~cf.Domain.close
   ~cf.Domain.persist
   ~cf.Domain.uncompress

Domain axes
-----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.autocyclic
   ~cf.Domain.axes
   ~cf.Domain.axis
   ~cf.Domain.cyclic
   ~cf.Domain.direction
   ~cf.Domain.directions
   ~cf.Domain.iscyclic
   ~cf.Domain.is_discrete_axis

Subspacing
----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.indices
   ~cf.Domain.subspace

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.nc_del_variable
   ~cf.Domain.nc_get_variable
   ~cf.Domain.nc_has_variable
   ~cf.Domain.nc_set_variable 
   ~cf.Domain.nc_global_attributes
   ~cf.Domain.nc_clear_global_attributes
   ~cf.Domain.nc_set_global_attribute
   ~cf.Domain.nc_set_global_attributes

Groups
^^^^^^

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.nc_variable_groups
   ~cf.Domain.nc_set_variable_groups
   ~cf.Domain.nc_clear_variable_groups
   ~cf.Domain.nc_group_attributes
   ~cf.Domain.nc_clear_group_attributes
   ~cf.Domain.nc_set_group_attribute
   ~cf.Domain.nc_set_group_attributes
  
Aggregation
-----------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

  ~cf.Domain.file_locations

Geometries
^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Domain.nc_del_geometry_variable
   ~cf.Domain.nc_get_geometry_variable
   ~cf.Domain.nc_has_geometry_variable
   ~cf.Domain.nc_set_geometry_variable 
   ~cf.Domain.nc_geometry_variable_groups
   ~cf.Domain.nc_set_geometry_variable_groups
   ~cf.Domain.nc_clear_geometry_variable_groups

Components
^^^^^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
	      
   ~cf.Domain.nc_del_component_variable
   ~cf.Domain.nc_set_component_variable
   ~cf.Domain.nc_set_component_variable_groups
   ~cf.Domain.nc_clear_component_variable_groups      
   ~cf.Domain.nc_del_component_dimension
   ~cf.Domain.nc_set_component_dimension
   ~cf.Domain.nc_set_component_dimension_groups
   ~cf.Domain.nc_clear_component_dimension_groups
   ~cf.Domain.nc_del_component_sample_dimension
   ~cf.Domain.nc_set_component_sample_dimension   
   ~cf.Domain.nc_set_component_sample_dimension_groups
   ~cf.Domain.nc_clear_component_sample_dimension_groups

Dataset compliance
^^^^^^^^^^^^^^^^^^

.. rubric:: Methods


.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.dataset_compliance
   
Aliases
-------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst  

   ~cf.Domain.aux
   ~cf.Domain.auxs
   ~cf.Domain.axis
   ~cf.Domain.coord
   ~cf.Domain.coords
   ~cf.Domain.dim
   ~cf.Domain.dims
   ~cf.Domain.domain_anc
   ~cf.Domain.domain_ancs
   ~cf.Domain.key
   ~cf.Domain.match
   ~cf.Domain.measure
   ~cf.Domain.measures
   ~cf.Domain.ref
   ~cf.Domain.refs

Deprecated
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst


   ~cf.Domain.add_file_location
   ~cf.Domain.cfa_clear_file_substitutions
   ~cf.Domain.cfa_del_file_substitution
   ~cf.Domain.cfa_file_substitutions
   ~cf.Domain.cfa_update_file_substitutions
   ~cf.Domain.del_file_location
   ~cf.Domain.delprop
   ~cf.Domain.get_filenames
   ~cf.Domain.getprop
   ~cf.Domain.hasprop
   ~cf.Domain.setprop

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Domain.__deepcopy__
   ~cf.Domain.__repr__
   ~cf.Domain.__str__

Docstring substitutions
-----------------------                   
                                          
.. rubric:: Methods                       
                                          
.. autosummary::                          
   :nosignatures:                         
   :toctree: ../method/                   
   :template: method.rst                  
                                          
   ~cf.Domain._docstring_special_substitutions
   ~cf.Domain._docstring_substitutions        
   ~cf.Domain._docstring_package_depth        
   ~cf.Domain._docstring_method_exclusions    
