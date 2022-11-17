import cfdm

from . import (
    AuxiliaryCoordinate,
    Bounds,
    CellMeasure,
    CellMethod,
    CoordinateConversion,
    CoordinateReference,
    Count,
    Datum,
    DimensionCoordinate,
    Domain,
    DomainAncillary,
    DomainAxis,
    Field,
    FieldAncillary,
    Index,
    InteriorRing,
    InterpolationParameter,
    List,
    NodeCountProperties,
    PartNodeCountProperties,
    TiePointIndex
)
from .data import Data
from .data.array import (
    CFANetCDFArray,
    GatheredArray,
    NetCDFArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
    SubsampledArray,
)
from .functions import CF


class CFImplementation(cfdm.CFDMImplementation):
    """A container for the CF data model implementation for `cf`.

    .. versionadded:: 3.0.0

    """

    def initialise_CFANetCDFArray(
        self,
        filename=None,
        ncvar=None,
        group=None,
        dtype=None,
        mask=True,
        units=False,
        calendar=False,
        instructions=None,
    ):
        """Return a `CFANetCDFArray` instance.

        :Parameters:

            filename: `str`

            ncvar: `str`

            group: `None` or sequence of str`

            dytpe: `numpy.dtype`

            mask: `bool`, optional

            units: `str` or `None`, optional

            calendar: `str` or `None`, optional

            instructions: `str`, optional

        :Returns:

            `CFANetCDFArray`

        """
        cls = self.get_class("CFANetCDFArray")
        return cls(
            filename=filename,
            ncvar=ncvar,
            group=group,
            dtype=dtype,
            mask=mask,
            units=units,
            calendar=calendar,
            instructions=instructions,
        )


# TODODASK: add missing classes, such as TiePointIndex
_implementation = CFImplementation(
    cf_version=CF(),
    AuxiliaryCoordinate=AuxiliaryCoordinate,
    CellMeasure=CellMeasure,
    CellMethod=CellMethod,
    CFANetCDFArray=CFANetCDFArray,
    CoordinateReference=CoordinateReference,
    DimensionCoordinate=DimensionCoordinate,
    Domain=Domain,
    DomainAncillary=DomainAncillary,
    DomainAxis=DomainAxis,
    Field=Field,
    FieldAncillary=FieldAncillary,
    Bounds=Bounds,
    InteriorRing=InteriorRing,
    InterpolationParameter=InterpolationParameter,
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
    SubsampledArray=SubsampledArray,
    TiePointIndex=TiePointIndex,
)


def implementation():
    """Return a container for the CF data model implementation.

    .. versionadded:: 3.1.0

    .. seealso:: `cf.example_field`, `cf.read`, `cf.write`

    :Returns:

        `CFImplementation`
            A container for the CF data model implementation.

    **Examples**

    >>> i = cf.implementation()
    >>> i
    <CFDMImplementation: >
    >>> i.classes()
    {'AuxiliaryCoordinate': cf.auxiliarycoordinate.AuxiliaryCoordinate,
     'CellMeasure': cf.cellmeasure.CellMeasure,
     'CellMethod': cf.cellmethod.CellMethod,
     'CoordinateReference': cf.coordinatereference.CoordinateReference,
     'DimensionCoordinate': cf.dimensioncoordinate.DimensionCoordinate,
     'Domain': cf.domain.Domain,
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

    """
    return _implementation.copy()
