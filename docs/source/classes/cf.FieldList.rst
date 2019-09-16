.. currentmodule:: cf
.. default-role:: obj

.. _fieldlist:

cf.FieldList
============

----

.. autoclass:: cf.FieldList
   :no-members:
   :no-inherited-members:

Filtering
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst
  
   ~cf.FieldList.filter_by_construct
   ~cf.FieldList.filter_by_identity
   ~cf.FieldList.filter_by_naxes
   ~cf.FieldList.filter_by_ncvar
   ~cf.FieldList.filter_by_property
   ~cf.FieldList.filter_by_rank
   ~cf.FieldList.filter_by_units
    
Comparison
----------

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldList.equals

Miscellaneous
-------------

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldList.close
   ~cf.FieldList.concatenate
   ~cf.FieldList.copy 

Aliases
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldList.filter
   ~cf.FieldList.select

List-like operations
--------------------

These methods provide functionality similar to that of a
:ref:`built-in list <python:tut-morelists>`. The main difference is
that when a field element needs to be assesed for equality its
`~cf.Field.equals` method is used, rather than the ``==``
operator. For example

>>> fl.count(x)

is equivalent to

>>> sum(f.equals(x) for f in fl)

.. autosummary::
   :nosignatures:
   :toctree: ../generated/
   :template: method.rst

   ~cf.FieldList.append
   ~cf.FieldList.clear
   ~cf.FieldList.count
   ~cf.FieldList.extend
   ~cf.FieldList.index
   ~cf.FieldList.insert
   ~cf.FieldList.pop
   ~cf.FieldList.reverse
   ~cf.FieldList.sort
   ~cf.FieldList.__add__
   ~cf.FieldList.__contains__
   ~cf.FieldList.__iadd__
   ~cf.FieldList.__imul__
   ~cf.FieldList.__getitem__ 
   ~cf.FieldList.__len__
   ~cf.FieldList.__mul__
   ~cf.FieldList.__setitem__ 

Special methods
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: method.rst

   ~cf.FieldList.__deepcopy__
   ~cf.FieldList.__repr__
   ~cf.FieldList.__str__
