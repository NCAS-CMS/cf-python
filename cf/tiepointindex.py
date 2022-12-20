import cfdm

from . import mixin


class TiePointIndex(mixin.PropertiesData, cfdm.TiePointIndex):
    """A tie point index variable with properties.

    Space may be saved by storing a subsample of the coordinates. The
    uncompressed coordinates can be reconstituted by interpolation
    from the subsampled coordinate values, also called either "tie
    points" or "bounds tie points".

    For each interpolated dimension, the locations of the (bounds) tie
    points are defined by a corresponding tie point index variable,
    which also indicates the extent of each continuous area.

    **NetCDF interface**

    {{netCDF variable}}

    The netCDF subsampled dimension name may be accessed with the
    `nc_set_subsampled_dimension`, `nc_get_subsampled_dimension`,
    `nc_del_subsampled_dimension` and `nc_has_subsampled_dimension`
    methods.

    The netCDF subsampled dimension group structure may be accessed
    with the `nc_set_subsampled_dimension`,
    `nc_get_subsampled_dimension`, `nc_subsampled_dimension_groups`,
    `nc_clear_subsampled_dimension_groups`, and
    `nc_set_subsampled_dimension_groups` methods.

    The netCDF interpolation subarea dimension name may be accessed
    with the `nc_set_interpolation_subarea_dimension`,
    `nc_get_interpolation_subarea_dimension`,
    `nc_del_interpolation_subarea_dimension`, and
    `nc_has_interpolation_subarea_dimension` methods.

    The netCDF interpolation subarea dimension group structure may be
    accessed with the `nc_set_interpolation_subarea_dimension`,
    `nc_get_interpolation_subarea_dimension`,
    `nc_interpolation_subarea_dimension_groups`,
    `nc_clear_interpolation_subarea_dimension_groups` and
    `nc_set_interpolation_subarea_dimension_groups` methods.

    .. versionadded:: 3.14

    .. seealso:: `InterpolationParameter`

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)
