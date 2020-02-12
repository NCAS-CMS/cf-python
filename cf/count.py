import cfdm

from . import mixin


class Count(mixin.PropertiesData,
            cfdm.Count):
    '''A count variable required to uncompress a ragged array.

    A collection of features stored using a contiguous ragged array
    combines all features along a single dimension (the sample
    dimension) such that each feature in the collection occupies a
    contiguous block.

    The information needed to uncompress the data is stored in a count
    variable that gives the size of each block.

    **NetCDF interface**

    The netCDF variable name of the count variable may be accessed
    with the `nc_set_variable`, `nc_get_variable`, `nc_del_variable`
    and `nc_has_variable` methods.

    The name of the netCDF dimension spanned by the count variable's
    data may be accessed with the `nc_set_dimension`,
    `nc_get_dimension`, `nc_del_dimension` and `nc_has_dimension`
    methods.

    The name of the netCDF sample dimension spanned by the compressed
    data (that is stored in the "sample_dimension" netCDF attribute
    and which does not correspond to a domain axis construct) may be
    accessed with the `nc_set_sample_dimension`,
    `nc_get_sample_dimension`, `nc_del_sample_dimension` and
    `nc_has_sample_dimension` methods.

    .. versionadded:: 3.0.0

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)


# --- End: class
