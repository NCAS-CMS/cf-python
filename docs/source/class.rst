.. currentmodule:: cf
.. default-role:: obj

.. _class:

**cf classes**
==============

Field construct class
---------------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.Field

Field list class
----------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.FieldList

Domain class
------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.Domain

Domain list class
-----------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.DomainList

Metadata component classes
--------------------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.AuxiliaryCoordinate
   cf.CellConnectivity
   cf.CellMeasure
   cf.CellMethod
   cf.CoordinateReference
   cf.DimensionCoordinate
   cf.DomainAncillary
   cf.DomainAxis
   cf.DomainTopology
   cf.FieldAncillary

Constructs classes
------------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.Constructs
   cf.ConstructList

Coordinate component classes
----------------------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.Bounds
   cf.CoordinateConversion
   cf.Datum
   cf.InteriorRing

Data classes
------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.AggregatedArray
   cf.Data
   cf.H5netcdfArray
   cf.NetCDF4Array
   cf.FullArray
   cf.ScipyNetcdfFileArray
   cf.PyfiveArray
   cf.UMArray
   cf.ZarrArray
   
Data compression classes
------------------------

Classes that support the creation and storage of compressed arrays.

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.Count
   cf.Index
   cf.List
   cf.GatheredArray
   cf.RaggedContiguousArray
   cf.RaggedIndexedArray
   cf.RaggedIndexedContiguousArray
   cf.SubsampledArray
   cf.TiePointIndex
   cf.Quantization

Data UGRID classes
------------------

Classes that support the creation and storage of UGRID-related arrays.

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.BoundsFromNodesArray
   cf.CellConnectivityArray
   cf.PointTopologyArray
   cf.NodeCountProperties
   cf.PartNodeCountProperties

CF Data Model Implementation class
----------------------------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.CFImplementation

Miscellaneous classes
---------------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.Flags
   cf.Query
   cf.TimeDuration
   cf.Units
   cf.RegridOperator
   cf.Constant
   cf.Configuration
   cf.InterpolationParameter

Deprecated classes
------------------

.. autosummary::
   :nosignatures:
   :toctree: class/

   cf.NetCDFArray
