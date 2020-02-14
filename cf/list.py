import cfdm

from . import mixin


class List(mixin.PropertiesData,
           cfdm.List):
    '''A list variable required to uncompress a gathered array.

    Compression by gathering combines axes of a multidimensional array
    into a new, discrete axis whilst omitting the missing values and
    thus reducing the number of values that need to be stored.

    The information needed to uncompress the data is stored in a list
    variable that gives the indices of the required points.

    **NetCDF interface**

    The netCDF variable name of the list variable may be accessed with
    the `nc_set_variable`, `nc_get_variable`, `nc_del_variable` and
    `nc_has_variable` methods.

    .. versionadded:: 3.0.0

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)


# --- End: class
