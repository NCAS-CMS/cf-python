import cfdm

from . import mixin


class InteriorRing(mixin.PropertiesData,
                   cfdm.InteriorRing):
    '''An interior ring array with properties.

    If a cell is composed of multiple polygon parts, an individual
    polygon may define an "interior ring", i.e. a region that is to be
    omitted from, as opposed to included in, the cell extent. In this
    case an interior ring array is required that records whether each
    polygon is to be included or excluded from the cell, and is
    supplied by an interior ring variable in CF-netCDF. The interior
    ring array spans the same domain axis constructs as its coordinate
    array, with the addition of an extra dimension that indexes the
    parts for each cell. For example, a cell describing the land area
    surrounding a lake would require two polygon parts: one defines
    the outer boundary of the land area; the other, recorded as an
    interior ring, is the lake boundary, defining the inner boundary
    of the land area.

    **NetCDF interface**

    The netCDF variable name of the interior ring variable may be
    accessed with the `nc_set_variable`, `nc_get_variable`,
    `nc_del_variable` and `nc_has_variable` methods.

    The name of the netCDF dimension spanned by the interior ring
    variable's data (which does not correspond to a domain axis
    construct) may be accessed with the `nc_set_dimension`,
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
