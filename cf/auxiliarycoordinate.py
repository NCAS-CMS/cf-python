import cfdm

from . import Bounds, mixin


class AuxiliaryCoordinate(
    mixin.Coordinate, mixin.PropertiesDataBounds, cfdm.AuxiliaryCoordinate
):
    """An auxiliary coordinate construct of the CF data model.

    An auxiliary coordinate construct provides information which
    locate the cells of the domain and which depend on a subset of the
    domain axis constructs. Auxiliary coordinate constructs have to be
    used, instead of dimension coordinate constructs, when a single
    domain axis requires more then one set of coordinate values, when
    coordinate values are not numeric, strictly monotonic, or contain
    missing values, or when they vary along more than one domain axis
    construct simultaneously. CF-netCDF auxiliary coordinate variables
    and non-numeric scalar coordinate variables correspond to
    auxiliary coordinate constructs.

    The auxiliary coordinate construct consists of a data array of the
    coordinate values which spans a subset of the domain axis
    constructs, an optional array of cell bounds recording the extents
    of each cell (stored in a `Bounds` object), and properties to
    describe the coordinates. An array of cell bounds spans the same
    domain axes as its coordinate array, with the addition of an extra
    dimension whose size is that of the number of vertices of each
    cell. This extra dimension does not correspond to a domain axis
    construct since it does not relate to an independent axis of the
    domain. Note that, for climatological time axes, the bounds are
    interpreted in a special way indicated by the cell method
    constructs.

    **NetCDF interface**

    {{netCDF variable}}

    {{netCDF UGRID node coordinate}}

    """

    def __new__(cls, *args, **kwargs):
        """Store component classes."""
        instance = super().__new__(cls)
        instance._Bounds = Bounds
        return instance
