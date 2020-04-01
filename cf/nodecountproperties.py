import cfdm

from . import mixin


class NodeCountProperties(mixin.Properties,
                          cfdm.NodeCountProperties):
    '''Properties for a netCDF node count variable.

    **NetCDF interface**

    The netCDF node count variable name may be accessed with the
    `nc_set_variable`, `nc_get_variable`, `nc_del_variable` and
    `nc_has_variable` methods.

    .. versionadded:: 3.2.0

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

# --- End: class
