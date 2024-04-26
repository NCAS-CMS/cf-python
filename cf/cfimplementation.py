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
    TiePointIndex,
)
from .data import Data
from .data.array import (
    BoundsFromNodesArray,
    CellConnectivityArray,
    CFANetCDFArray,
    GatheredArray,
    NetCDFArray,
    PointTopologyArray,
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

    def nc_set_hdf5_chunksizes(self, data, sizes, override=False):
        """Set the data HDF5 chunksizes.

        .. versionadded:: 3.16.2

        :Parameters:

            data: `Data`
                The data.

            sizes: sequence of `int`
                The new HDF5 chunk sizes.

            override: `bool`, optional
                If True then set the HDF5 chunks sizes even if some
                have already been specified. If False, the default,
                then only set the HDF5 chunks sizes if some none have
                already been specified.

        :Returns:

            `None`

        """
        if override or not data.nc_hdf5_chunksizes():
            data.nc_set_hdf5_chunksizes(sizes)

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

    def initialise_CFANetCDFArray(
        self,
        filename=None,
        address=None,
        dtype=None,
        mask=True,
        units=False,
        calendar=False,
        instructions=None,
        substitutions=None,
        term=None,
        x=None,
        **kwargs,
    ):
        """Return a `CFANetCDFArray` instance.

        :Parameters:

            filename: `str`

            address: (sequence of) `str` or `int`

            dytpe: `numpy.dtype`

            mask: `bool`, optional

            units: `str` or `None`, optional

            calendar: `str` or `None`, optional

            instructions: `str`, optional

            substitutions: `dict`, optional

            term: `str`, optional

            x: `dict`, optional

            kwargs: optional
                Ignored.

        :Returns:

            `CFANetCDFArray`

        """
        cls = self.get_class("CFANetCDFArray")
        return cls(
            filename=filename,
            address=address,
            dtype=dtype,
            mask=mask,
            units=units,
            calendar=calendar,
            instructions=instructions,
            substitutions=substitutions,
            term=term,
            x=x,
        )


_implementation = CFImplementation(
    cf_version=CF(),
    AuxiliaryCoordinate=AuxiliaryCoordinate,
    CellConnectivity=CellConnectivity,
    CellMeasure=CellMeasure,
    CellMethod=CellMethod,
    CFANetCDFArray=CFANetCDFArray,
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
    NetCDFArray=NetCDFArray,
    PointTopologyArray=PointTopologyArray,
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
    <CFImplementation: >
    >>> i.classes()
    {'AuxiliaryCoordinate': cf.auxiliarycoordinate.AuxiliaryCoordinate,
     'BoundsFromNodesArray': cf.data.array.boundsfromnodesarray.BoundsFromNodesArray,
     'CellConnectivity': cf.cellconnectivity.CellConnectivity,
     'CellConnectivityArray': cf.data.array.cellconnectivityarray.CellConnectivityArray,
     'CellMeasure': cf.cellmeasure.CellMeasure,
     'CellMethod': cf.cellmethod.CellMethod,
     'CFANetCDFArray': cf.data.array.cfanetcdfarray.CFANetCDFArray,
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
     'NetCDFArray': cf.data.array.netcdfarray.NetCDFArray,
     'PointTopologyArray': <class 'cf.data.array.pointtopologyarray.PointTopologyArray'>,
     'RaggedContiguousArray': cf.data.array.raggedcontiguousarray.RaggedContiguousArray,
     'RaggedIndexedArray': cf.data.array.raggedindexedarray.RaggedIndexedArray,
     'RaggedIndexedContiguousArray': cf.data.array.raggedindexedcontiguousarray.RaggedIndexedContiguousArray,
     'SubsampledArray': cf.data.array.subsampledarray.SubsampledArray,
     'TiePointIndex': cf.tiepointindex.TiePointIndex}

    """
    return _implementation.copy()
