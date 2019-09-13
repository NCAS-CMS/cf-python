.. currentmodule:: cf
.. default-role:: obj

.. _class:

Classes of the :mod:`cf` module
===============================

Field class
-------------

.. autosummary::
   :nosignatures:
   :toctree: classes/

   cf.Field	              

Field component classes
-----------------------

.. autosummary::
   :nosignatures:
   :toctree: classes/

   cf.AuxiliaryCoordinate
   cf.CellMeasure
   cf.CellMethods
   cf.CoordinateReference
   cf.DimensionCoordinate
   cf.DomainAncillary
   cf.DomainAxis
   cf.FieldAncillary

Miscellaneous classes
---------------------

.. autosummary::
   :nosignatures:
   :toctree: classes/

   cf.Data
   cf.Datetime
   cf.FieldList	              
   cf.Flags
   cf.Query
   cf.TimeDuration
   cf.Units

Base classes
------------

.. autosummary::
   :nosignatures:
   :toctree: classes/
         
   cf.Coordinate
   cf.Variable       
   cf.BoundedVariable       
  

.. comment
   Data component classes
   ----------------------
   
   .. autosummary::
      :nosignatures:
      :toctree: classes/
   
      cf.Partition
      cf.PartitionMatrix


.. _inheritance_diagrams:

.. Inheritance diagrams
   --------------------
   
   The classes defined by the `cf` package inherit as follows:
   
   ----
   
   .. image:: images/inheritance1.png
   
   .. commented out
      .. inheritance-diagram:: cf.Domain
                               cf.Data
                               cf.Flags	
                               cf.Units
                               cf.Datetime
                               cf.TimeDuration
                               cf.Query
         :parts: 1
   
   ----

   .. image:: images/inheritance2.png
   
   .. commented out
      .. inheritance-diagram:: cf.CoordinateBounds
                               cf.AuxiliaryCoordinate
                               cf.DimensionCoordinate
                               cf.FieldList
                               cf.CellMeasure
            :parts: 1
   
   ----
   
   .. image:: images/inheritance3.png
   
   .. commented out
     .. inheritance-diagram:: cf.CellMethods
                              cf.CoordinateReference
            :parts: 1
   
   ----
