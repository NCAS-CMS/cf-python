.. currentmodule:: cf
.. default-role:: obj

.. _cf-DomainAxis:

cf.DomainAxis
=============

----

.. autoclass:: cf.DomainAxis
   :no-members:
   :no-inherited-members:

Inspection
----------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
   
   ~cf.DomainAxis.identity  
   ~cf.DomainAxis.identities

.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAxis.construct_type

Size
----

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
   
   ~cf.DomainAxis.del_size
   ~cf.DomainAxis.has_size
   ~cf.DomainAxis.get_size
   ~cf.DomainAxis.set_size
   
.. rubric:: Attributes
   
.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst

   ~cf.DomainAxis.size

Miscellaneous
-------------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAxis.copy
   ~cf.DomainAxis.creation_commands
   ~cf.DomainAxis.equals
   ~cf.DomainAxis.inspect

NetCDF
------

.. rubric:: Methods
	    
.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAxis.nc_del_dimension
   ~cf.DomainAxis.nc_get_dimension
   ~cf.DomainAxis.nc_has_dimension
   ~cf.DomainAxis.nc_set_dimension 
   ~cf.DomainAxis.nc_is_unlimited
   ~cf.DomainAxis.nc_set_unlimited

Arithmetic and comparison operations
------------------------------------

Arithmetic, bitwise and comparison operations are defined on a field
construct as element-wise operations on its data which yield a new
field construct or, for augmented assignments, modify the field
construct’s data in-place.

.. rubric:: Comparison operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__lt__
   ~cf.Field.__le__
   ~cf.Field.__eq__
   ~cf.Field.__ne__
   ~cf.Field.__gt__
   ~cf.Field.__ge__

.. rubric:: Binary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__add__     
   ~cf.Field.__sub__

.. rubric:: Binary arithmetic operators with reflected (swapped) operands

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__radd__     

.. rubric:: Augmented arithmetic assignments

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Field.__iadd__ 
   ~cf.Field.__isub__ 

.. rubric:: Unary arithmetic operators

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAxis.__int__    

Groups
^^^^^^

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAxis.nc_dimension_groups
   ~cf.DomainAxis.nc_clear_dimension_groups
   ~cf.DomainAxis.nc_set_dimension_groups

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainAxis.__deepcopy__
   ~cf.DomainAxis.__hash__
   ~cf.DomainAxis.__repr__
   ~cf.DomainAxis.__str__
