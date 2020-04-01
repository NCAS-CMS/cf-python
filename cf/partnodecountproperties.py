import cfdm

from . import mixin


class PartNodeCountProperties(mixin.Properties,
                              cfdm.PartNodeCountProperties):
    '''Properties for a netCDF part node count variable.

    **NetCDF interface**

    The netCDF part node count variable name may be accessed with the
    `nc_set_variable`, `nc_get_variable`, `nc_del_variable` and
    `nc_has_variable` methods.

    The name of the netCDF dimension spanned by the netCDF part node
    count variable's data may be accessed with the `nc_set_dimension`,
    `nc_get_dimension`, `nc_del_dimension` and `nc_has_dimension`
    methods.

    .. versionadded:: 3.2.0

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

# --- End: class
