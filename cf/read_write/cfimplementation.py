import cfdm

from ..functions import CF

from .. import (AuxiliaryCoordinate,
                CellMethod,
                CellMeasure,
                CoordinateReference,
                DimensionCoordinate,
                DomainAncillary,
                DomainAxis,
                Field,
                FieldAncillary,
                Bounds,
                CoordinateConversion,
                Datum,
                Count,
                List,
                Index,
)

from ..data import (Data,
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
#--- End: class


_implementation = CFImplementation(
    cf_version = CF(),
    
    AuxiliaryCoordinate = AuxiliaryCoordinate,
    CellMeasure         = CellMeasure,
    CellMethod          = CellMethod,
    CoordinateReference = CoordinateReference,
    DimensionCoordinate = DimensionCoordinate,
    DomainAncillary     = DomainAncillary,
    DomainAxis          = DomainAxis,
    Field               = Field,
    FieldAncillary      = FieldAncillary,
    
    Bounds = Bounds,
    
    CoordinateConversion = CoordinateConversion,
    Datum                = Datum,

    List          = List,
    Index         = Index,
    Count         = Count,
    
    Data                         = Data,
    GatheredArray                = GatheredArray,
    NetCDFArray                  = NetCDFArray,
    RaggedContiguousArray        = RaggedContiguousArray,
    RaggedIndexedArray           = RaggedIndexedArray,
    RaggedIndexedContiguousArray = RaggedIndexedContiguousArray,
)

def implementation():
    '''Return a container for the CF data model implementation.

.. versionadded:: 3.0.0

.. seealso:: `cf.read`, `cf.write`

:Returns:

    `CFImplementation`
        A container for the CF data model implementation.

**Examples:**

>>> i = cf.implementation()
>>> i
<CFDMImplementation: >
>>> i.classes()
{'AuxiliaryCoordinate': cf.auxiliarycoordinate.AuxiliaryCoordinate,
 'Bounds': cf.bounds.Bounds,
 'CellMeasure': cf.cellmeasure.CellMeasure,
 'CellMethod': cf.cellmethod.CellMethod,
 'CoordinateConversion': cf.coordinateconversion.CoordinateConversion,
 'CoordinateReference': cf.coordinatereference.CoordinateReference,
 'Count': cf.count.Count,
 'Data': cf.data.data.Data,
 'Datum': cf.datum.Datum,
 'DimensionCoordinate': cf.dimensioncoordinate.DimensionCoordinate,
 'DomainAncillary': cf.domainancillary.DomainAncillary,
 'DomainAxis': cf.domainaxis.DomainAxis,
 'Field': cf.field.Field,
 'FieldAncillary': cf.fieldancillary.FieldAncillary,
 'GatheredArray': cf.data.gatheredarray.GatheredArray,
 'Index': cf.index.Index,
 'List': cf.list.List,
 'NetCDFArray': cf.data.netcdfarray.NetCDFArray,
 'RaggedContiguousArray': cf.data.raggedcontiguousarray.RaggedContiguousArray,
 'RaggedIndexedArray': cf.data.raggedindexedarray.RaggedIndexedArray,
 'RaggedIndexedContiguousArray': cf.data.raggedindexedcontiguousarray.RaggedIndexedContiguousArray}

    '''
    return _implementation.copy()
#--- End: def
