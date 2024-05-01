.. currentmodule:: cf
.. default-role:: obj

cf.Query
========

----

.. autoclass:: cf.Query
   :no-members:
   :no-inherited-members:

Methods
-------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst
   
   ~cf.Query.addattr
   ~cf.Query.copy
   ~cf.Query.dump
   ~cf.Query.equals
   ~cf.Query.equivalent
   ~cf.Query.evaluate
   ~cf.Query.exact
   ~cf.Query.inspect
   ~cf.Query.set_condition_units
   ~cf.Query.setdefault

Attributes
----------

.. autosummary::
   :nosignatures:
   :toctree: ../attribute/
   :template: attribute.rst
   
   ~cf.Query.attr
   ~cf.Query.operator
   ~cf.Query.value
   ~cf.Query.iscontains
   ~cf.Query.isquery
   ~cf.Query.Units
   ~cf.Query.atol
   ~cf.Query.rtol
   ~cf.Query.open_upper
   ~cf.Query.open_lower


.. _Query-Special:

Special
-------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.Query.__deepcopy__
   ~cf.Query.__repr__
   ~cf.Query.__str__
   ~cf.Query.__eq__
   ~cf.Query.__ne__
   ~cf.Query.__and__
   ~cf.Query.__iand__
   ~cf.Query.__or__
   ~cf.Query.__ior__
   ~cf.Query.__dask_tokenize__
