import cfdm

from .data import Data
from .units import Units

from . import mixin

from .decorators import _deprecated_kwarg_check


class Bounds(mixin.Coordinate,
             mixin.PropertiesData,
             cfdm.Bounds):
    '''A cell bounds component of a coordinate or domain ancillary
    construct of the CF data model.

    An array of cell bounds spans the same domain axes as its
    coordinate array, with the addition of an extra dimension whose
    size is that of the number of vertices of each cell. This extra
    dimension does not correspond to a domain axis construct since it
    does not relate to an independent axis of the domain. Note that,
    for climatological time axes, the bounds are interpreted in a
    special way indicated by the cell method constructs.

    In the CF data model, a bounds component does not have its own
    properties because they can not logically be different to those of
    the coordinate construct itself. However, it is sometimes desired
    to store attributes on a CF-netCDF bounds variable, so it is also
    allowed to provide properties to a bounds component.

    **NetCDF interface**

    The netCDF variable name of the bounds may be accessed with the
    `nc_set_variable`, `nc_get_variable`, `nc_del_variable` and
    `nc_has_variable` methods.

    The name of the trailing netCDF dimension spanned by bounds (which
    does not correspond to a domain axis construct) may be accessed
    with the `nc_set_dimension`, `nc_get_dimension`,
    `nc_del_dimension` and `nc_has_dimension` methods.

    '''
    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return super().__repr__().replace('<', '<CF ', 1)

    def contiguous(self, overlap=True, direction=None, period=None,
                   verbose=1):
        '''Return True if the bounds are contiguous.

    Bounds are contiguous if the cell boundaries match up, or overlap,
    with the boundaries of adjacent cells.

    In general, it is only possible for 1 or 0 variable dimensional
    variables with bounds to be contiguous, but size 1 variables with
    any number of dimensions are always contiguous.

    An exception is raised if the variable is multdimensional and has
    more than one element.

    .. versionadded:: 2.0

   :Parameters:

        overlap: `bool`, optional
            If False then 1-d cells with two bounds vertices are not
            considered contiguous if any adjacent cells overlap each
            other. By default such cells are considered contiguous.

        direction:
            Specify the direction of 1-d coordinates with two bounds
            vertices. Either True for increasing coordinates, or False
            for descreasing coordinates. By default the direction is
            inferred from whether the first bound of the first cell is
            less than its second bound (direction is True), or not
            (direction is False).

        period: optional
            Define the period of cyclic values so that the test for
            contiguousness can be carried out with modulo
            arithmetic. By default the data are assumed to be
            non-cyclic, unless the bounds have units of longitude (or
            have units of ``'degrees'``), in which case a period of
            360 degrees is assumed.

        verbose: `int`, optional
            TODO

    :Returns:

        `bool`
            Whether or not the cells are contiguous.

    **Examples:**

    TODO

        '''
        data = self.get_data(None)
        if data is None:
            return False

        ndim = data.ndim - 1
        nbounds = data.shape[-1]

        if data.size == nbounds:
            return True

        if period is None:
            if self.Units.islongitude:
                period = Data(360.0, 'degrees_east')
            elif self.Units.equals(Units('degrees')):
                period = Data(360.0, 'degrees')
        # --- End: if
        if verbose >= 2:
            print("Period = {!r}".format(period))

        if ndim == 2:
            if nbounds != 4:
                raise ValueError("Can't tell if {}-d cells with {} vertices "
                                 "are contiguous".format(ndim, nbounds))

            # --------------------------------------------------------
            # 2-d coordinates with 4 vertices per cell
            # --------------------------------------------------------
#            if overlap:
#                raise ValueError(
#                    "overlap=True and can't tell if 2-d bounds are contiguous")

            # Check cells (j, i) and cells (j, i+1) are contiguous
            diff = data[:, :-1, 1] - data[:, 1:, 0]
            if period is not None:
                diff = diff % period

            if diff.any():
                return False

            diff = data[:, :-1, 2] - data[:, 1:, 3]
            if period is not None:
                diff = diff % period

            if diff.any():
                return False

            # Check cells (j, i) and (j+1, i) are contiguous
            diff = data[:-1, :, 3] - data[1:, :, 0]
            if period is not None:
                diff = diff % period

            if diff.any():
                return False

            diff = data[:-1, :, 2] - data[1:, :, 1]
            if period is not None:
                diff = diff % period

            if diff.any():
                return False


#            bnd = data.array
#            for j in range(data.shape[0] - 1):
#                for i in range(data.shape[1] - 1):
#
#                    if (bnd[j, i, 1] != bnd[j, i+1, 0] or
#                        bnd[j, i, 2] != bnd[j, i+1, 3]):
#                        return False
#
#                    # Check cells (j, i) and (j+1, i) are contiguous
#                    if (bnd[j, i, 3] != bnd[j+1, i, 0] or
#                            bnd[j, i, 2] != bnd[j+1, i, 1]):
#                        return False
#            # --- End: for

            return True

        if ndim > 2:
            raise ValueError("Can't tell if {}-d cells "
                             "are contiguous".format(ndim))

        if nbounds != 2:
            raise ValueError("Can't tell if {}-d cells with {} vertices "
                             "are contiguous".format(ndim, nbounds))

        if not overlap:
            diff = data[1:, 0] - data[:-1, 1]
            if period is not None:
                diff = diff % period

            return not diff.any()
#            return data[1:, 0].equals(data[:-1, 1])
        else:
            if direction is None:
                b = data[(0,) * ndim].array
                direction = b.item(0,) < b.item(1,)

            if direction:
                return (data[1:, 0] <= data[:-1, 1]).all()
            else:
                return (data[1:, 0] >= data[:-1, 1]).all()

    @_deprecated_kwarg_check('relaxed_identity')
    def identity(self, default='', strict=False, relaxed=False,
                 nc_only=False, relaxed_identity=None):
        '''Return the canonical identity.

    By default the identity is the first found of the following:

    1. The "standard_name" property.
    2. The "cf_role" property, preceeded by ``'cf_role='``.
    3. The "long_name" property, preceeded by ``'long_name='``.
    4. The netCDF variable name, preceeded by ``'ncvar%'``.
    5. The value of the *default* parameter.

    Properties include any inherited properties.

    .. versionadded:: 3.0.6

    .. seealso:: `identities`

    :Parameters:

        default: optional
            If no identity can be found then return the value of the
            default parameter.

        strict: `bool`, optional
            If True then the identity is the first found of only the
            "standard_name" property or the "id" attribute.

        relaxed: `bool`, optional
            If True then the identity is the first found of only the
            "standard_name" property, the "id" attribute, the
            "long_name" property or the netCDF variable name.

        nc_only: `bool`, optional
            If True then only take the identity from the netCDF
            variable name.

    :Returns:

            The identity.

    **Examples:**

    >>> b.inherited_properties()
    {'foo': 'bar',
     'long_name': 'Longitude'}
    >>> b.properties()
    {'long_name': 'A different long name'}
    >>> b.identity()
    'long_name=A different long name'
    >>> b.del_property('long_name')
    'A different long name'
    >>> b.identity()
    'long_name=Longitude'

        '''
        inherited_properties = self.inherited_properties()
        if inherited_properties:
            bounds = self.copy()
            properties = bounds.properties()
            bounds.set_properties(inherited_properties)
            bounds.set_properties(properties)
            self = bounds

        return super().identity(default=default, strict=strict,
                                relaxed=relaxed, nc_only=nc_only)


# --- End: class
