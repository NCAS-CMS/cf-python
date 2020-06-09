from numpy import empty as numpy_empty
from numpy import result_type as numpy_result_type

import cfdm

from . import Bounds

from .timeduration import TimeDuration
from .units import Units

from .data.data import Data

from . import mixin

from .functions import (_DEPRECATION_ERROR_KWARGS,
                        _DEPRECATION_ERROR_ATTRIBUTE)

from .decorators import (_inplace_enabled,
                         _inplace_enabled_define_and_cleanup,
                         _deprecated_kwarg_check)


class DimensionCoordinate(mixin.Coordinate,
                          mixin.PropertiesDataBounds,
                          cfdm.DimensionCoordinate):
    '''A dimension coordinate construct of the CF data model.

    A dimension coordinate construct provides information which locate
    the cells of the domain and which depend on a subset of the domain
    axis constructs. The dimension coordinate construct is able to
    unambiguously describe cell locations because a domain axis can be
    associated with at most one dimension coordinate construct, whose
    data array values must all be non-missing and strictly
    monotonically increasing or decreasing. They must also all be of
    the same numeric data type. If cell bounds are provided, then each
    cell must have exactly two vertices. CF-netCDF coordinate
    variables and numeric scalar coordinate variables correspond to
    dimension coordinate constructs.

    The dimension coordinate construct consists of a data array of the
    coordinate values which spans a subset of the domain axis
    constructs, an optional array of cell bounds recording the extents
    of each cell (stored in a `cf.Bounds` object), and properties to
    describe the coordinates. An array of cell bounds spans the same
    domain axes as its coordinate array, with the addition of an extra
    dimension whose size is that of the number of vertices of each
    cell. This extra dimension does not correspond to a domain axis
    construct since it does not relate to an independent axis of the
    domain. Note that, for climatological time axes, the bounds are
    interpreted in a special way indicated by the cell method
    constructs.

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

    def _centre(self, period):
        '''It assumed, but not checked, that the period has been set.

    .. seealso:: `roll`

        '''
        if self.direction():
            mx = self.data[-1]
        else:
            mx = self.data[0]

        return ((mx // period) * period).squeeze()

    def _infer_direction(self):
        '''Return True if a coordinate is increasing, otherwise return False.

    A dimension coordinate construct is considered to be increasing if
    its data array values are increasing in index space, or if it has
    no data nor bounds.

    If the direction can not be inferred from the data not bounds then
    the coordinate's units are used.

    The direction is inferred from the coordinate's data array values
    or its from coordinates. It is not taken directly from its
    `cf.Data` object.

    :Returns:

        `bool`
            Whether or not the coordinate is increasing.

    **Examples:**

    >>> c.array
    array([  0  30  60])
    >>> c._get_direction()
    True
    >>> c.array
    array([15])
    >>> c.bounds.array
    array([  30  0])
    >>> c._get_direction()
    False

        '''
        data = self.get_data(None)
        if data is not None:
            # Infer the direction from the data
            if data._size > 1:
                data = data[0:2].array
                return bool(data.item(0,) < data.item(1,))
        # --- End: if

        # Still here?
        data = self.get_bounds_data(None)
        if data is not None:
            # Infer the direction from the bounds
            b = data[(0,) * (data.ndim - 1)].array
            return bool(b.item(0,) < b.item(1,))

        # Still here? Then infer the direction from the units.
        return not self.Units.ispressure

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
#    @property
#    def cellsize(self):
#        '''The cell sizes.
#
#    :Returns:
#
#        `Data`
#            The size for each cell.
#
#    **Examples:**
#
#    >>> print(c.bounds)
#    <CF Bounds: latitude(3, 2) degrees_north>
#    >>> print(c.bounds.array)
#    [[-90. -87.]
#     [-87. -80.]
#     [-80. -67.]]
#    >>> print(d.cellsize)
#    <CF Data(3): [3.0, 7.0, 13.0] degrees_north>
#    >>> print(d.cellsize.array)
#    [  3.   7.  13.]
#    >>> print(c.sin().cellsize.array)
#    [ 0.00137047  0.01382178  0.0643029 ]
#
#    >>> del(c.bounds)
#    >>> c.cellsize
#    AttributeError: Can't get cell sizes when coordinates have no bounds
#
#        '''
#        cells = self.get_bounds_data(None)
#        if cells is None:
#            raise AttributeError(
#                "Can't get cell sizes when coordinates have no bounds")
#
#        if self.direction():
#            cells = cells[:, 1] - cells[:, 0]
#        else:
#            cells = cells[:, 0] - cells[:, 1]
#
#        cells.squeeze(1, inplace=True)
#
#        return cells

    @property
    def decreasing(self):
        '''True if the dimension coordinate is decreasing, otherwise False.

    A dimension coordinate is increasing if its coordinate values are
    increasing in index space.

    The direction is inferred from one of, in order of precedence:

    * The data array
    * The bounds data array
    * The `units` CF property

    :Returns:

        `bool`
            Whether or not the coordinate is decreasing.

    **Examples:**

    >>> c.decreasing
    False
    >>> c.flip().increasing
    True

        '''
        return not self.direction()

    @property
    def increasing(self):
        '''`True` for dimension coordinate constructs, `False` otherwise.

    A dimension coordinate is increasing if its coordinate values are
    increasing in index space.

    The direction is inferred from one of, in order of precedence:

    * The data array
    * The bounds data array
    * The `units` CF property

    :Returns:

        `bool`
            Whether or not the coordinate is increasing.

    **Examples:**

    >>> c.decreasing
    False
    >>> c.flip().increasing
    True

        '''
        return self.direction()

    @property
    def isdimension(self):
        '''True, denoting that the variable is a dimension coordinate object.

    .. seealso::`isauxiliary`, `isdomainancillary`, `isfieldancillary`,
                `ismeasure`

    **Examples:**

    >>> c.isdimension
    True

        '''
        return True

#    @property
#    def lower_bounds(self):
#        '''The lower dimension coordinate bounds in a `cf.Data` object.
#
#    .. seealso:: `bounds`, `upper_bounds`
#
#    **Examples:**
#
#    >>> print(c.bounds.array)
#    [[ 5  3]
#     [ 3  1]
#     [ 1 -1]]
#    >>> c.lower_bounds
#    <CF Data(3): [3, 1, -1]>
#    >>> print(c.lower_bounds.array)
#    [ 3  1 -1]
#
#        '''
#        data = self.get_bounds_data(None)
#        if data is None:
#            raise ValueError(
#                "Can't get lower bounds when there are no bounds")
#
#        if self.direction():
#            i = 0
#        else:
#            i = 1
#
#        out = data[..., i]
#        out.squeeze(1, inplace=True)
#        return out

#
#    @property
#    def upper_bounds(self):
#        '''The upper dimension coordinate bounds in a `cf.Data` object.
#
#    .. seealso:: `bounds`, `lower_bounds`
#
#    **Examples:**
#
#    >>> print(c.bounds.array)
#    [[ 5  3]
#     [ 3  1]
#     [ 1 -1]]
#    >>> c.upper_bounds
#    <CF Data(3): [5, 3, 1]>
#    >>> print(c.upper_bounds.array)
#    [5  3  1]
#
#        '''
#        data = self.get_bounds_data(None)
#        if data is None:
#            raise ValueError(
#                "Can't get upper bounds when there are no bounds")
#
#        if self.direction():
#            i = 1
#        else:
#            i = 0
#
#        return data[..., i].squeeze(1)

    def direction(self):
        '''Return True if the dimension coordinate values are increasing,
    otherwise return False.

    Dimension coordinates values are increasing if its coordinate
    values are increasing in index space.

    The direction is inferred from one of, in order of precedence:

    * The data array
    * The bounds data array
    * The `units` CF property

    :Returns:

        `bool`
            Whether or not the coordinate is increasing.

    **Examples:**

    >>> c.array
    array([  0  30  60])
    >>> c.direction()
    True

    >>> c.bounds.array
    array([  30  0])
    >>> c.direction()
    False

        '''
        _direction = self._custom.get('direction')
        if _direction is not None:
            return _direction

        _direction = self._infer_direction()
        self._custom['direction'] = _direction

        return _direction

    def create_bounds(self, bound=None, cellsize=None, flt=0.5,
                      max=None, min=None):
        '''Create cell bounds.

    :Parameters:

        bound: optional
            If set to a value larger (smaller) than the largest
            (smallest) coordinate value then bounds are created which
            include this value and for which each coordinate is in the
            centre of its bounds. Ignored if *create* is False.

        cellsize: optional
            Define the exact size of each cell that is
            created. Created cells are allowed to overlap do not have
            to be contigious.  Ignored if *create* is False. The
            *cellsize* parameter may be one of:

              * A data-like scalar (see below) that defines the cell size,
                either in the same units as the coordinates or in the
                units provided. Note that in this case, the position of
                each coordinate within the its cell is controlled by the
                *flt* parameter.

                *Parameter example:*
                    To specify cellsizes of 10, in the same units as the
                    coordinates: ``cellsize=10``.

                *Parameter example:*
                    To specify cellsizes of 1 day: ``cellsize=cf.Data(1,
                    'day')`` (see `cf.Data` for details).

                *Parameter example:*
                     For coordinates ``1, 2, 10``, setting ``cellsize=1``
                     will result in bounds of ``(0.5, 1.5), (1.5, 2.5),
                     (9.5, 10.5)``.

                *Parameter example:*
                     For coordinates ``1, 2, 10`` kilometres, setting
                     ``cellsize=cf.Data(5000, 'm')`` will result in bounds
                     of ``(-1.5, 3.5), (-0.5, 4.5), (7.5, 12.5)`` (see
                     `cf.Data` for details).

                *Parameter example:*
                  For decreasing coordinates ``2, 0, -12`` setting,
                  ``cellsize=2`` will result in bounds of ``(3, 1),
                  (1, -1), (-11, -13)``.

            ..

              * A `cf.TimeDuration` defining the cell size. Only
                applicable to reference time coordinates. It is possible
                to "anchor" the cell bounds via the `cf.TimeDuration`
                parameters. For example, to specify cell size of one
                calendar month, starting and ending on the 15th day:
                ``cellsize=cf.M(day=15)`` (see `cf.M` for details). Note
                that the *flt* parameter is ignored in this case.

                *Parameter example:*
                  For coordinates ``1984-12-01 12:00, 1984-12-02
                  12:00, 2000-04-15 12:00`` setting,
                  ``cellsize=cf.D()`` will result in bounds of
                  ``(1984-12-01, 1984-12-02), (1984-12-02,
                  1984-12-03), (2000-05-15, 2000-04-16)`` (see `cf.D`
                  for details).

                *Parameter example:*
                  For coordinates ``1984-12-01, 1984-12-02,
                  2000-04-15`` setting, ``cellsize=cf.D()`` will
                  result in bounds of ``(1984-12-01, 1984-12-02),
                  (1984-12-02, 1984-12-03), (2000-05-15, 2000-04-16)``
                  (see `cf.D` for details).

                *Parameter example:*
                  For coordinates ``1984-12-01, 1984-12-02,
                  2000-04-15`` setting, ``cellsize=cf.D(hour=12)``
                  will result in bounds of ``(1984-11:30 12:00,
                  1984-12-01 12:00), (1984-12-01 12:00, 1984-12-02
                  12:00), (2000-05-14 12:00, 2000-04-15 12:00)`` (see
                  `cf.D` for details).

                *Parameter example:*
                  For coordinates ``1984-12-16 12:00, 1985-01-16
                  12:00`` setting, ``cellsize=cf.M()`` will result in
                  bounds of ``(1984-12-01, 1985-01-01), (1985-01-01,
                  1985-02-01)`` (see `cf.M` for details).

                *Parameter example:*
                  For coordinates ``1984-12-01 12:00, 1985-01-01
                  12:00`` setting, ``cellsize=cf.M()`` will result in
                  bounds of ``(1984-12-01, 1985-01-01), (1985-01-01,
                  1985-02-01)`` (see `cf.M` for details).

                *Parameter example:*
                  For coordinates ``1984-12-01 12:00, 1985-01-01
                  12:00`` setting, ``cellsize=cf.M(day=20)`` will
                  result in bounds of ``(1984-11-20, 1984-12-20),
                  (1984-12-20, 1985-01-20)`` (see `cf.M` for details).

                *Parameter example:*
                  For coordinates ``1984-03-01, 1984-06-01`` setting,
                  ``cellsize=cf.Y()`` will result in bounds of
                  ``(1984-01-01, 1985-01-01), (1984-01-01,
                  1985-01-01)`` (see `cf.Y` for details). Note that in
                  this case each cell has the same bounds. This
                  because ``cf.Y()`` is equivalent to ``cf.Y(month=1,
                  day=1)`` and the closest 1st January to both
                  coordinates is 1st January 1984.

            {+data-like-scalar} TODO

        flt: `float`, optional
            When creating cells with sizes specified by the *cellsize*
            parameter, define the fraction of the each cell which is
            less its coordinate value. By default *flt* is 0.5, so that
            each cell has its coordinate at it's centre. Ignored if
            *cellsize* is not set.

            *Parameter example:*
               For coordinates ``1, 2, 10``, setting ``cellsize=1,
               flt=0.5`` will result in bounds of ``(0.5, 1.5), (1.5,
               2.5), (9.5, 10.5)``.

            *Parameter example:*
               For coordinates ``1, 2, 10``, setting ``cellsize=1,
               flt=0.25`` will result in bounds of ``(0.75, 1.75),
               (1.75, 2.75), (9.75, 10.75)``.

            *Parameter example:*
               For decreasing coordinates ``2, 0, -12``, setting
               ``cellsize=6, flt=0.9`` will result in bounds of
               ``(2.6, -3.4), (0.6, -5.4), (-11.4, -17.4)``.

        min: optional
            Limit the created bounds to be no less than this number.

            *Parameter example:*
               To ensure that all latitude bounds are at least -90:
               ``min=-90``.

        max: optional
            Limit the created bounds to be no more than this number.

            *Parameter example:*
               To ensure that all latitude bounds are at most 90:
               ``max=90``.

        copy: `bool`, optional
            If `False` then the returned bounds are not independent of
            the existing bounds, if any, or those inserted, if
            *create* and *insert* are both True. By default the
            returned bounds are independent.

    :Returns:

        `Bounds` or `None`
            TODO

    **Examples:**

    >>> c.create_bounds()
    >>> c.create_bounds(bound=-9000.0)

        '''
        array = self.array
        size = array.size

        if cellsize is not None:
            if bound:
                raise ValueError(
                    "bound parameter can't be True when setting the "
                    "cellsize parameter"
                )

            if not isinstance(cellsize, TimeDuration):
                # ----------------------------------------------------
                # Create bounds based on cell sizes defined by a
                # data-like object
                #
                # E.g. cellsize=10
                #      cellsize=cf.Data(1, 'day')
                # ----------------------------------------------------
                cellsize = Data.asdata(abs(cellsize))
                if cellsize.Units:
                    if self.Units.isreftime:
                        if not cellsize.Units.istime:
                            raise ValueError("q123423423jhgsjhbd jh ")
                        cellsize.Units = Units(self.Units._utime.units)
                    else:
                        if not cellsize.Units.equivalent(self.Units):
                            raise ValueError("jhgsjhbd jh ")
                        cellsize.Units = self.Units
                cellsize = cellsize.datum()

                cellsize0 = cellsize * flt
                cellsize1 = cellsize * (1 - flt)
                if not self.direction():
                    cellsize0, cellsize1 = -cellsize1, -cellsize0

                bounds = numpy_empty((size, 2), dtype=array.dtype)
                bounds[:, 0] = array - cellsize0
                bounds[:, 1] = array + cellsize1
            else:
                # ----------------------------------------------------
                # Create bounds based on cell sizes defined by a
                # TimeDuration object
                #
                # E.g. cellsize=cf.s()
                #      cellsize=cf.m()
                #      cellsize=cf.h()
                #      cellsize=cf.D()
                #      cellsize=cf.M()
                #      cellsize=cf.Y()
                #      cellsize=cf.D(hour=12)
                #      cellsize=cf.M(day=16)
                #      cellsize=cf.M(2)
                #      cellsize=cf.M(2, day=15, hour=12)
                # ----------------------------------------------------
                if not self.Units.isreftime:
                    raise ValueError(
                        "Can't create reference time bounds for "
                        "non-reference time coordinates: {0!r}".format(
                            self.Units)
                    )

                bounds = numpy_empty((size, 2), dtype=object)

                cellsize_bounds = cellsize.bounds
                calendar = getattr(self, 'calendar', None)
                direction = bool(self.direction())

                for c, b in zip(self.datetime_array, bounds):
                    b[...] = cellsize_bounds(c, direction=direction)
        else:
            if bound is None:
                # ----------------------------------------------------
                # Creat Voronoi bounds
                # ----------------------------------------------------
                if size < 2:
                    raise ValueError(
                        "Can't create bounds for Voronoi cells from one value")

                bounds_1d = [array.item(0,)*1.5 - array.item(1,)*0.5]
                bounds_1d.extend((array[0:-1] + array[1:])*0.5)
                bounds_1d.append(array.item(-1,)*1.5 - array.item(-2,)*0.5)

                dtype = type(bounds_1d[0])

                if max is not None:
                    if self.direction():
                        bounds_1d[-1] = max
                    else:
                        bounds_1d[0] = max
                if min is not None:
                    if self.direction():
                        bounds_1d[0] = min
                    else:
                        bounds_1d[-1] = min

            else:
                # ----------------------------------------------------
                # Create
                # ----------------------------------------------------
                direction = self.direction()
                if not direction and size > 1:
                    array = array[::-1]

                bounds_1d = [bound]
                if bound <= array.item(0,):
                    for i in range(size):
                        bound = 2.0*array.item(i,) - bound
                        bounds_1d.append(bound)
                elif bound >= array.item(-1,):
                    for i in range(size-1, -1, -1):
                        bound = 2.0*array.item(i,) - bound
                        bounds_1d.append(bound)

                    bounds_1d = bounds_1d[::-1]
                else:
                    raise ValueError("bad bound value")

                dtype = type(bounds_1d[-1])

                if not direction:
                    bounds_1d = bounds_1d[::-1]
            # --- End: if

            bounds = numpy_empty((size, 2), dtype=dtype)
            bounds[:, 0] = bounds_1d[:-1]
            bounds[:, 1] = bounds_1d[1:]
        # --- End: if

        # Create coordinate bounds object
        bounds = Bounds(data=Data(bounds, units=self.Units),
                        copy=False)

        return bounds

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def flip(self, axes=None, inplace=False, i=False):
        '''TODO
        '''
        d = _inplace_enabled_define_and_cleanup(self)
        super(DimensionCoordinate, d).flip(axes=axes, inplace=True)

        direction = d._custom.get('direction')
        if direction is not None:
            d._custom['direction'] = not direction

        return d

    def get_bounds(self, default=ValueError(), **kwargs):
        '''Return the bounds.

    .. versionadded:: 3.0.0

    .. seealso:: `bounds`, `create_bounds', `get_data`, `del_bounds`,
                 `has_bounds`, `set_bounds`

    :Parameters:

        default: optional
            Return the value of the default parameter if bounds have
            not been set. If set to an `Exception` instance then it
            will be raised instead.

    :Returns:

        `Bounds`
            The bounds.

    **Examples:**

    >>> b = Bounds(data=cfdm.Data(range(10).reshape(5, 2)))
    >>> c.set_bounds(b)
    >>> c.has_bounds()
    True
    >>> c.get_bounds()
    <Bounds: (5, 2) >
    >>> b = c.del_bounds()
    >>> b
    <Bounds: (5, 2) >
    >>> c.has_bounds()
    False
    >>> print(c.get_bounds(None))
    None
    >>> print(c.del_bounds(None))
    None

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'get_bounds', kwargs,
                "Bounds creation now uses the 'create_bounds' and "
                "'set_bounds' methods."
            )  # pragma: no cover

        return super().get_bounds(default=default)

#    def autoperiod(self, verbose=False):
#        '''TODO Set dimensions to be cyclic.
#
#    TODO A dimension is set to be cyclic if it has a unique longitude (or
#    grid longitude) dimension coordinate construct with bounds and the
#    first and last bounds values differ by 360 degrees (or an
#    equivalent amount in other units).
#
#    .. versionadded:: 3.0.0
#
#    .. seealso:: `isperiodic`, `period`
#
#    :Parameters:
#
#        TODO
#
#    :Returns:
#
#       `bool`
#
#    **Examples:**
#
#    >>> f.autocyclic()
#
#        '''
#        if not self.Units.islongitude:
#            if verbose:
#                print(0)
#            if (self.get_property('standard_name', None) not in
#                    ('longitude', 'grid_longitude')):
#                if verbose:
#                    print(1)
#                return False
#        # --- End: if
#
#        bounds = self.get_bounds(None)
#        if bounds is None:
#            if verbose:
#                print(2)
#            return False
#
#        bounds_data = bounds.get_data(None)
#        if bounds_data is None:
#            if verbose:
#                print(3)
#            return False
#
#        bounds = bounds_data.array
#
#        period = Data(360.0, units='degrees')
#
#        period.Units = bounds_data.Units
#
#        if abs(bounds[-1, -1] - bounds[0, 0]) != period.array:
#            if verbose:
#                print(4)
#            return False
#
#        self.period(period)
#
#        return True

    @_deprecated_kwarg_check('i')
    @_inplace_enabled
    def roll(self, axis, shift, inplace=False, i=False):
        '''TODO

        '''
        if self.size <= 1:
            if inplace:
                return
            else:
                return self.copy()
        # --- End: if

        shift %= self.size

#        period = self._custom.get('period')
        period = self.period()

        if not shift:
            # Null roll
            if inplace:
                return
            else:
                return self.copy()
        elif period is None:
            raise ValueError(
                "Can't roll {} when no period has been set".format(
                    self.__class__.__name__))

        direction = self.direction()

        centre = self._centre(period)

        if axis not in [0, -1]:
            raise ValueError(
                "Can't roll axis {} when there is only one axis".format(axis))

        c = _inplace_enabled_define_and_cleanup(self)
        super(DimensionCoordinate, c).roll(axis, shift, inplace=True)

        c.dtype = numpy_result_type(c.dtype, period.dtype)

        b = c.get_bounds(None)
        bounds_data = c.get_bounds_data(None)

        if bounds_data is not None:
            b.dtype = numpy_result_type(bounds_data.dtype, period.dtype)
            bounds_data = b.get_data(None)

        if direction:
            # Increasing
            c[:shift] -= period
            if bounds_data is not None:
                b[:shift] -= period

            if c.data[0] <= centre - period:
                c += period
                if bounds_data is not None:
                    b += period
        else:
            # Decreasing
            c[:shift] += period
            if bounds_data is not None:
                b[:shift] += period

            if c.data[0] >= centre + period:
                c -= period
                if bounds_data is not None:
                    b -= period
        # --- End: if

        c._custom['direction'] = direction

        return c

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def role(self):
        '''Deprecated at version 3.0.0. Use attribute 'construct_type'
    instead.

        '''
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, 'role',
            "Use attribute 'construct_type' instead"
        )  # pragma: no cover


# --- End: class
