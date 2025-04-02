.. currentmodule:: cf
.. default-role:: obj

cf.UMArray
==========

----

.. autoclass:: cf.UMArray
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
   
   ~cf.UMArray.get_compression_type
   ~cf.UMArray.get_subspace
   ~cf.UMArray.get_attributes
   ~cf.UMArray.get_byte_ordering
   ~cf.UMArray.get_fmt
   ~cf.UMArray.get_word_size
   ~cf.UMArray.get_mask
   ~cf.UMArray.get_unpack
   ~cf.UMArray.is_subspace
   ~cf.UMArray.index

.. rubric:: Attributes

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst
   
   ~cf.UMArray.array
   ~cf.UMArray.astype
   ~cf.UMArray.dtype
   ~cf.UMArray.ndim
   ~cf.UMArray.shape
   ~cf.UMArray.size
   ~cf.UMArray.original_shape
   ~cf.UMArray.reference_shape

Units
-----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
   
   ~cf.UMArray.get_calendar
   ~cf.UMArray.get_units
   ~cf.UMArray.Units
   
File
----
   
.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
   
   ~cf.UMArray.get_address
   ~cf.UMArray.get_addresses
   ~cf.UMArray.close
   ~cf.UMArray.open
   ~cf.UMArray.get_filename
   ~cf.UMArray.get_filenames
   ~cf.UMArray.get_format
   ~cf.UMArray.get_formats
   ~cf.UMArray.get_storage_options
   ~cf.UMArray.add_file_location
   ~cf.UMArray.del_file_location

Miscellaneous
-------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
   
   ~cf.UMArray.copy
   ~cf.UMArray.to_memory
   ~cf.UMArray.file_directory 
   ~cf.UMArray.replace_directory
   ~cf.UMArray.replace_filename

Special
-------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
   
   ~cf.UMArray.__dask_tokenize__
   ~cf.UMArray.__getitem__

Docstring substitutions
-----------------------                   
                                          
.. rubric:: Methods                       
                                          
.. autosummary::                          
   :nosignatures:                         
   :toctree: ../method/                   
   :template: method.rst                  
                                          
   ~cf.UMArray._docstring_special_substitutions
   ~cf.UMArray._docstring_substitutions        
   ~cf.UMArray._docstring_package_depth        
   ~cf.UMArray._docstring_method_exclusions    

Deprecated
----------
                                          
.. rubric:: Methods                       
                                          
.. autosummary::                          
   :nosignatures:                         
   :toctree: ../method/                   
   :template: method.rst                  
                                          
   ~cf.UMArray.get_missing_values
   ~cf.UMArray.byte_ordering
   ~cf.UMArray.data_offset
   ~cf.UMArray.file_locations
   ~cf.UMArray.file_address
   ~cf.UMArray.fmt
   ~cf.UMArray.filename
   ~cf.UMArray.disk_length
   ~cf.UMArray.header_offset
   ~cf.UMArray.word_size
