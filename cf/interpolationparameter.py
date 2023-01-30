import cfdm

from . import mixin


class InterpolationParameter(
    mixin.PropertiesData, cfdm.InterpolationParameter
):
    """An interpolation parameter variable.

    Space may be saved by storing a subsample of the coordinates. The
    uncompressed coordinates can be reconstituted by interpolation
    from the subsampled coordinate values, also called either "tie
    points" or "bounds tie points".

    An interpolation parameter variable provides values for
    coefficient terms in the interpolation equation, or for any other
    terms that configure the interpolation process.

    **NetCDF interface**

    {{netCDF variable}}

    The netCDF subsampled dimension name and the netCDF interpolation
    subarea dimension name, if required, are set on the
    corresponding tie point index variable.

    .. versionadded:: 3.14.0

    .. seealso:: `TiePointIndex`

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)
