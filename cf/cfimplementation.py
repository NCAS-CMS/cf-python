import cfdm

from .functions import CF

from . import (
    AuxiliaryCoordinate,
    CellMethod,
    CellMeasure,
    CoordinateReference,
    DimensionCoordinate,
    DomainAncillary,
    DomainAxis,
    Field,
    FieldAncillary,
    Bounds,
    InteriorRing,
    CoordinateConversion,
    Datum,
    Count,
    List,
    Index,
    NodeCountProperties,
    PartNodeCountProperties,
)

from .data import (
    Data,
    GatheredArray,
    NetCDFArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
)


class CFImplementation(cfdm.CFDMImplementation):
    '''TODO

    .. versionadded:: 3.0.0

    '''
# --- End: class


_implementation = CFImplementation(
    cf_version=CF(),

    AuxiliaryCoordinate=AuxiliaryCoordinate,
    CellMeasure=CellMeasure,
    CellMethod=CellMethod,
    CoordinateReference=CoordinateReference,
    DimensionCoordinate=DimensionCoordinate,
    DomainAncillary=DomainAncillary,
    DomainAxis=DomainAxis,
    Field=Field,
    FieldAncillary=FieldAncillary,

    Bounds=Bounds,
    InteriorRing=InteriorRing,
    CoordinateConversion=CoordinateConversion,
    Datum=Datum,

    List=List,
    Index=Index,
    Count=Count,
    NodeCountProperties=NodeCountProperties,
    PartNodeCountProperties=PartNodeCountProperties,

    Data=Data,
    GatheredArray=GatheredArray,
    NetCDFArray=NetCDFArray,
    RaggedContiguousArray=RaggedContiguousArray,
    RaggedIndexedArray=RaggedIndexedArray,
    RaggedIndexedContiguousArray=RaggedIndexedContiguousArray,
)


def implementation():
    '''Return a container for the CF data model implementation.

    .. versionadded:: 3.1.0

    .. seealso:: `cf.example_field`, `cf.read`, `cf.write`

    :Returns:

        `CFImplementation`
            A container for the CF data model implementation.

    **Examples:**

    >>> i = cf.implementation()
    >>> i
    <CFDMImplementation: >
    >>> i.classes()
    {'AuxiliaryCoordinate': cf.auxiliarycoordinate.AuxiliaryCoordinate,
     'CellMeasure': cf.cellmeasure.CellMeasure,
     'CellMethod': cf.cellmethod.CellMethod,
     'CoordinateReference': cf.coordinatereference.CoordinateReference,
     'DimensionCoordinate': cf.dimensioncoordinate.DimensionCoordinate,
     'DomainAncillary': cf.domainancillary.DomainAncillary,
     'DomainAxis': cf.domainaxis.DomainAxis,
     'Field': cf.field.Field,
     'FieldAncillary': cf.fieldancillary.FieldAncillary,
     'Bounds': cf.bounds.Bounds,
     'InteriorRing': cf.interiorring.InteriorRing,
     'CoordinateConversion': cf.coordinateconversion.CoordinateConversion,
     'Datum': cf.datum.Datum,
     'Data': cf.data.data.Data,
     'GatheredArray': cf.data.gatheredarray.GatheredArray,
     'NetCDFArray': cf.data.netcdfarray.NetCDFArray,
     'RaggedContiguousArray': cf.data.raggedcontiguousarray.RaggedContiguousArray,
     'RaggedIndexedArray': cf.data.raggedindexedarray.RaggedIndexedArray,
     'RaggedIndexedContiguousArray': cf.data.raggedindexedcontiguousarray.RaggedIndexedContiguousArray,
     'List': cf.list.List,
     'Count': cf.count.Count,
     'Index': cf.index.Index,
     'NodeCountProperties': cf.nodecountproperties.NodeCountProperties,
     'PartNodeCountProperties': cf.partnodecountproperties.PartNodeCountProperties}

    '''
    return _implementation.copy()
