import cfdm

from . import Quantization, mixin


class FieldAncillary(mixin.PropertiesData, cfdm.FieldAncillary):
    """A field ancillary construct of the CF data model.

    The field ancillary construct provides metadata which are
    distributed over the same sampling domain as the field itself. For
    example, if a data variable holds a variable retrieved from a
    satellite instrument, a related ancillary data variable might
    provide the uncertainty estimates for those retrievals (varying
    over the same spatiotemporal domain).

    The field ancillary construct consists of an array of the
    ancillary data, which is zero-dimensional or which depends on one
    or more of the domain axes, and properties to describe the
    data. It is assumed that the data do not depend on axes of the
    domain which are not spanned by the array, along which the values
    are implicitly propagated. CF-netCDF ancillary data variables
    correspond to field ancillary constructs. Note that a field
    ancillary construct is constrained by the domain definition of the
    parent field construct but does not contribute to the domain's
    definition, unlike, for instance, an auxiliary coordinate
    construct or domain ancillary construct.

    **NetCDF interface**

    {{netCDF variable}}

    {{netCDF dataset chunks}}

    .. versionadded:: 2.0

    """

    def __new__(cls, *args, **kwargs):
        """Store component classes."""
        instance = super().__new__(cls)
        instance._Quantization = Quantization
        return instance
