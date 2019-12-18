.. currentmodule:: cf
.. default-role:: obj

.. _fieldlist:

cf.FieldList
============

----

.. autoclass:: cf.FieldList
   :no-members:
   :no-inherited-members:

Selecting
---------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldList.select_by_construct    
   ~cf.FieldList.select_by_identity
   ~cf.FieldList.select_by_naxes
   ~cf.FieldList.select_by_ncvar
   ~cf.FieldList.select_by_property
   ~cf.FieldList.select_by_rank
   ~cf.FieldList.select_by_units
   ~cf.FieldList.select
   ~cf.FieldList.select_field
   ~cf.FieldList.__call__
   
Comparison
----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldList.equals

Miscellaneous
-------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldList.close
   ~cf.FieldList.concatenate
   ~cf.FieldList.copy 

Aliases
-------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldList.select
   ~cf.FieldList.__call__

List-like operations
--------------------

These methods provide functionality identical to that of a Python
`list`, with one exception:

* When a field construct element needs to be assesed for equality, its
  `~cf.Field.equals` method is used, rather than the ``==``
  operator. This affects the `~cf.FieldList.count`,
  `~cf.FieldList.index`, `~cf.FieldList.remove`,
  `~cf.FieldList.__contains__`, `~cf.FieldList.__eq__` and
  `~cf.FieldList.__ne__` methods.

For example

>>> fl.count(x)

is equivalent to

>>> sum(f.equals(x) for f in fl)

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldList.append
   ~cf.FieldList.clear
   ~cf.FieldList.count
   ~cf.FieldList.extend
   ~cf.FieldList.index
   ~cf.FieldList.insert
   ~cf.FieldList.pop
   ~cf.FieldList.remove
   ~cf.FieldList.reverse
   ~cf.FieldList.sort
   ~cf.FieldList.__add__
   ~cf.FieldList.__contains__
   ~cf.FieldList.__eq__
   ~cf.FieldList.__ge__
   ~cf.FieldList.__getitem__ 
   ~cf.FieldList.__getslice__
   ~cf.FieldList.__gt__
   ~cf.FieldList.__iter__
   ~cf.FieldList.__iadd__
   ~cf.FieldList.__imul__
   ~cf.FieldList.__le__
   ~cf.FieldList.__len__
   ~cf.FieldList.__lt__
   ~cf.FieldList.__mul__
   ~cf.FieldList.__ne__
   ~cf.FieldList.__repr__
   ~cf.FieldList.__rmul__
   ~cf.FieldList.__setitem__ 
   ~cf.FieldList.__str__

Special methods
---------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.FieldList.__call__
   ~cf.FieldList.__deepcopy__
   
.. ?? __hash__
