import cfdm

from . import mixin


class FieldAncillary(mixin.PropertiesData,
                     cfdm.FieldAncillary):
    '''A field ancillary construct of the CF data model.

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

    The netCDF variable name of the construct may be accessed with the
    `nc_set_variable`, `nc_get_variable`, `nc_del_variable` and
    `nc_has_variable` methods.

    '''

    def __repr__(self):
        '''Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

    @property
    def isfieldancillary(self):
        '''True, denoting that the variable is a field ancillary object.

    .. versionadded:: 2.0

    **Examples:**

    >>> f.isfieldancillary
    True

        '''
        return True


# --- End: class
