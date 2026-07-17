.. currentmodule:: cf
.. default-role:: obj

.. _domainlist:

cf.DomainList
=============

----

.. autoclass:: cf.DomainList
   :no-members:
   :no-inherited-members:

Selecting
---------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainList.select_by_construct    
   ~cf.DomainList.select_by_identity
   ~cf.DomainList.select_by_ncvar
   ~cf.DomainList.select_by_property
   ~cf.DomainList.select_by_rank
   ~cf.DomainList.select
   ~cf.DomainList.__call__

Comparison
----------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainList.equals

.. _DomainList-xarray:

xarray
------

.. rubric:: Methods

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainList.to_xarray

Miscellaneous
-------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainList.close
   ~cf.DomainList.copy 

Aliases
-------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainList.select
   ~cf.DomainList.__call__

List-like operations
--------------------

These methods provide functionality identical to that of a Python
`list`, with one exception:

* When a field construct element needs to be assessed for equality,
  its `~cf.Domain.equals` method is used, rather than the ``==``
  operator. This affects the `~cf.DomainList.count`,
  `~cf.DomainList.index`, `~cf.DomainList.remove`,
  `~cf.DomainList.__contains__`, `~cf.DomainList.__eq__` and
  `~cf.DomainList.__ne__` methods.

For example

>>> dl.count(x)

is equivalent to

>>> sum(d.equals(x) for d in dl)

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainList.append
   ~cf.DomainList.clear
   ~cf.DomainList.count
   ~cf.DomainList.extend
   ~cf.DomainList.index
   ~cf.DomainList.insert
   ~cf.DomainList.pop
   ~cf.DomainList.remove
   ~cf.DomainList.reverse
   ~cf.DomainList.sort
   ~cf.DomainList.__add__
   ~cf.DomainList.__contains__
   ~cf.DomainList.__eq__
   ~cf.DomainList.__ge__
   ~cf.DomainList.__getitem__ 
   ~cf.DomainList.__getslice__
   ~cf.DomainList.__gt__
   ~cf.DomainList.__iter__
   ~cf.DomainList.__iadd__
   ~cf.DomainList.__imul__
   ~cf.DomainList.__le__
   ~cf.DomainList.__len__
   ~cf.DomainList.__lt__
   ~cf.DomainList.__mul__
   ~cf.DomainList.__ne__
   ~cf.DomainList.__repr__
   ~cf.DomainList.__rmul__
   ~cf.DomainList.__setitem__ 
   ~cf.DomainList.__str__

Special methods
---------------

.. autosummary::
   :nosignatures:
   :toctree: ../method/
   :template: method.rst

   ~cf.DomainList.__call__
   ~cf.DomainList.__deepcopy__
