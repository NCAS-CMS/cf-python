import cfdm

from . import mixin


class Index(mixin.PropertiesData,
            cfdm.Index):
    '''An index variable required to uncompress a ragged array.

    A collection of features stored using an indexed ragged array
    combines all features along a single dimension (the sample
    dimension) such that the values of each feature in the collection
    are interleaved.

    The information needed to uncompress the data is stored in an
    index variable that specifies the feature that each element of the
    sample dimension belongs to.

    **NetCDF interface**

    The netCDF variable name of the index variable may be accessed
    with the `nc_set_variable`, `nc_get_variable`, `nc_del_variable`
    and `nc_has_variable` methods.

    The name of the netCDF dimension spanned by the index variable's
    data (which does not correspond to a domain axis construct) may be
    accessed with the `nc_set_dimension`, `nc_get_dimension`,
    `nc_del_dimension` and `nc_has_dimension` methods.

    The name of the netCDF sample dimension spanned by the compressed
    data (which does not correspond to a domain axis contract) may be
    accessed with the `nc_set_sample_dimension`,
    `nc_get_sample_dimension`, `nc_del_sample_dimension` and
    `nc_has_sample_dimension` methods.

    The name of the netCDF instance dimension (that is stored in the
    "instance_dimension" netCDF attribute) is accessed via the
    corresponding domain axis construct.

    .. note:: The netCDF sample dimension and the netCDF dimension
              spanned by the index variable's data are should be the
              same, unless the compressed data is an indexed
              contiguous ragged array, in which case they must be
              different.

    .. versionadded:: 3.0.0

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)


# --- End: class
