import cfdm

from . import (
    AuxiliaryCoordinate,
    Bounds,
    CellConnectivity,
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
    DomainTopology,
    Field,
    FieldAncillary,
    Index,
    InteriorRing,
    InterpolationParameter,
    List,
    NodeCountProperties,
    PartNodeCountProperties,
    Quantization,
    TiePointIndex,
)
from .data import Data
from .data.array import (
    AggregatedArray,
    BoundsFromNodesArray,
    CellConnectivityArray,
    GatheredArray,
    H5netcdfArray,
    NetCDF4Array,
    PointTopologyArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
    SubsampledArray,
    ZarrArray,
)
from .functions import CF


class CFImplementation(cfdm.CFDMImplementation):
    """A container for the CF data model implementation for `cf`.

    .. versionadded:: 3.0.0

    """

    def nc_set_dataset_chunksizes(self, data, sizes, override=False):
        """Set the data dataset chunksizes.

        .. versionadded:: 3.16.2

        :Parameters:

            data: `Data`
                The data.

            sizes: sequence of `int`
                The new dataset chunk sizes.

            override: `bool`, optional
                If True then set the dataset chunks sizes even if some
                have already been specified. If False, the default,
                then only set the dataset chunks sizes if some none
                have already been specified.

        :Returns:

            `None`

        """
        if override or not data.nc_dataset_chunksizes():
            data.nc_set_dataset_chunksizes(sizes)

    def set_construct(self, parent, construct, axes=None, copy=True, **kwargs):
        """Insert a construct into a field or domain.

        Does not attempt to conform cell method nor coordinate
        reference constructs, as that has been handled by `cfdm`.

        .. versionadded:: 3.14.0

        :Parameters:

            parent: `Field` or `Domain`
               On what to set the construct

            construct:
                The construct to set.

            axes: `tuple` or `None`, optional
                The construct domain axes, if applicable.

            copy: `bool`, optional
                Whether or not to set a copy of *construct*.

            kwargs: optional
                Additional parameters to the `set_construct` method of
                *parent*.

        :Returns:

            `str`
                The construct identifier.

        """
        kwargs.setdefault("conform", False)
        return super().set_construct(
            parent, construct, axes=axes, copy=copy, **kwargs
        )


_implementation = CFImplementation(
    cf_version=CF(),
    AggregatedArray=AggregatedArray,
    AuxiliaryCoordinate=AuxiliaryCoordinate,
    CellConnectivity=CellConnectivity,
    CellMeasure=CellMeasure,
    CellMethod=CellMethod,
    CoordinateReference=CoordinateReference,
    DimensionCoordinate=DimensionCoordinate,
    Domain=Domain,
    DomainAncillary=DomainAncillary,
    DomainAxis=DomainAxis,
    DomainTopology=DomainTopology,
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
    BoundsFromNodesArray=BoundsFromNodesArray,
    CellConnectivityArray=CellConnectivityArray,
    GatheredArray=GatheredArray,
    H5netcdfArray=H5netcdfArray,
    NetCDF4Array=NetCDF4Array,
    PointTopologyArray=PointTopologyArray,
    Quantization=Quantization,
    RaggedContiguousArray=RaggedContiguousArray,
    RaggedIndexedArray=RaggedIndexedArray,
    RaggedIndexedContiguousArray=RaggedIndexedContiguousArray,
    SubsampledArray=SubsampledArray,
    TiePointIndex=TiePointIndex,
    ZarrArray=ZarrArray,
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
    <CFImplementation: >
    >>> i.classes()
    {'AuxiliaryCoordinate': cf.auxiliarycoordinate.AuxiliaryCoordinate,
     'BoundsFromNodesArray': cf.data.array.boundsfromnodesarray.BoundsFromNodesArray,
     'CellConnectivity': cf.cellconnectivity.CellConnectivity,
     'CellConnectivityArray': cf.data.array.cellconnectivityarray.CellConnectivityArray,
     'CellMeasure': cf.cellmeasure.CellMeasure,
     'CellMethod': cf.cellmethod.CellMethod,
     'CoordinateReference': cf.coordinatereference.CoordinateReference,
     'DimensionCoordinate': cf.dimensioncoordinate.DimensionCoordinate,
     'Domain': cf.domain.Domain,
     'DomainAncillary': cf.domainancillary.DomainAncillary,
     'DomainAxis': cf.domainaxis.DomainAxis,
     'DomainTopology': cf.domaintopology.DomainTopology,
     'Field': cf.field.Field,
     'FieldAncillary': cf.fieldancillary.FieldAncillary,
     'Bounds': cf.bounds.Bounds,
     'InteriorRing': cf.interiorring.InteriorRing,
     'InterpolationParameter': cf.interpolationparameter.InterpolationParameter,
     'CoordinateConversion': cf.coordinateconversion.CoordinateConversion,
     'Datum': cf.datum.Datum,
     'List': cf.list.List,
     'Index': cf.index.Index,
     'Count': cf.count.Count,
     'NodeCountProperties': cf.nodecountproperties.NodeCountProperties,
     'PartNodeCountProperties': cf.partnodecountproperties.PartNodeCountProperties,
     'Data': cf.data.data.Data,
     'GatheredArray': cf.data.array.gatheredarray.GatheredArray,
     'H5netcdfArray': cf.data.array.h5netcdfarray.H5netcdfArray,
     'NetCDF4Array': cf.data.array.netcdf4array.NetCDF4Array,
     'PointTopologyArray': <class 'cf.data.array.pointtopologyarray.PointTopologyArray'>,
     'Quantization': cf.quantization.Quantization,
     'RaggedContiguousArray': cf.data.array.raggedcontiguousarray.RaggedContiguousArray,
     'RaggedIndexedArray': cf.data.array.raggedindexedarray.RaggedIndexedArray,
     'RaggedIndexedContiguousArray': cf.data.array.raggedindexedcontiguousarray.RaggedIndexedContiguousArray,
     'SubsampledArray': cf.data.array.subsampledarray.SubsampledArray,
     'TiePointIndex': cf.tiepointindex.TiePointIndex,
     'ZarrArray': cf.data.array.zarrarray.ZarrArray,
    }

    """
    return _implementation.copy()
