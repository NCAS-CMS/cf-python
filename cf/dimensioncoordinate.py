import cfdm
import numpy as np

from . import Bounds, mixin
from .data.data import Data
from .decorators import (
    _deprecated_kwarg_check,
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
)
from .functions import _DEPRECATION_ERROR_ATTRIBUTE, _DEPRECATION_ERROR_KWARGS
from .timeduration import TimeDuration
from .units import Units


class DimensionCoordinate(
    mixin.Coordinate, mixin.PropertiesDataBounds, cfdm.DimensionCoordinate
):
    """A dimension coordinate construct of the CF data model.

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
    of each cell (stored in a `Bounds` object), and properties to
    describe the coordinates. An array of cell bounds spans the same
    domain axes as its coordinate array, with the addition of an extra
    dimension whose size is that of the number of vertices of each
    cell. This extra dimension does not correspond to a domain axis
    construct since it does not relate to an independent axis of the
    domain. Note that, for climatological time axes, the bounds are
    interpreted in a special way indicated by the cell method
    constructs.

    **NetCDF interface**

    {{netcdf variable}}

    """

    def __new__(cls, *args, **kwargs):
        """Store component classes."""
        instance = super().__new__(cls)
        instance._Bounds = Bounds
        return instance

    def __init__(
        self,
        properties=None,
        data=None,
        bounds=None,
        geometry=None,
        interior_ring=None,
        source=None,
        copy=True,
        _use_data=True,
    ):
        """**Initialisation**

        :Parameters:

            {{init properties: `dict`, optional}}

               *Parameter example:*
                  ``properties={'standard_name': 'time'}``

            {{init data: data_like, optional}}

            {{init bounds: `Bounds`, optional}}

            {{init geometry: `str`, optional}}

            {{init interior_ring: `InteriorRing`, optional}}

            {{init source: optional}}

            {{init copy: `bool`, optional}}

        """
        super().__init__(
            properties=properties,
            data=data,
            bounds=bounds,
            geometry=geometry,
            interior_ring=interior_ring,
            source=source,
            copy=copy,
            _use_data=_use_data,
        )

        if source:
            # Reset cell characteristics after set_data/set_bounds has
            # removed them
            try:
                chars = source._get_component("cell_characteristics", None)
            except AttributeError:
                chars = None

            if chars is not None:
                self._set_component("cell_characteristics", chars, copy=False)

    def _centre(self, period):
        """It assumed, but not checked, that the period has been set.

        .. seealso:: `roll`

        """
        if self.direction():
            mx = self.data[-1]
        else:
            mx = self.data[0]

        return ((mx // period) * period).squeeze()

    def _infer_direction(self):
        """Return True if a coordinate is increasing, otherwise return
        False.

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

        **Examples**

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

        """
        data = self.get_data(None, _fill_value=False)
        if data is not None:
            # Infer the direction from the data
            if data.size > 1:
                c = data._get_cached_elements()
                if c:
                    try:
                        return bool(c.get(0) < c.get(1))
                    except TypeError:
                        pass

                data = data[:2].compute()
                return bool(data.item(0) < data.item(1))

        # Still here?
        data = self.get_bounds_data(None, _fill_value=False)
        if data is not None:
            # Infer the direction from the bounds
            c = data._get_cached_elements()
            if c:
                try:
                    return bool(c.get(0) < c.get(1))
                except TypeError:
                    pass

            b = data[0].compute()
            return bool(b.item(0) < b.item(1))

        # Still here? Then infer the direction from the units.
        return not self.Units.ispressure

    @property
    def decreasing(self):
        """True if the dimension coordinate is decreasing, otherwise
        False.

        A dimension coordinate is increasing if its coordinate values are
        increasing in index space.

        The direction is inferred from one of, in order of precedence:

        * The data array
        * The bounds data array
        * The `units` CF property

        :Returns:

            `bool`
                Whether or not the coordinate is decreasing.

        **Examples**

        >>> c.decreasing
        False
        >>> c.flip().increasing
        True

        """
        return not self.direction()

    @property
    def increasing(self):
        """`True` for dimension coordinate constructs, `False`
        otherwise.

        A dimension coordinate is increasing if its coordinate values are
        increasing in index space.

        The direction is inferred from one of, in order of precedence:

        * The data array
        * The bounds data array
        * The `units` CF property

        :Returns:

            `bool`
                Whether or not the coordinate is increasing.

        **Examples**

        >>> c.decreasing
        False
        >>> c.flip().increasing
        True

        """
        return self.direction()

    def direction(self):
        """Return True if the dimension coordinate values are
        increasing, otherwise return False.

        Dimension coordinates values are increasing if its coordinate
        values are increasing in index space.

        The direction is inferred from one of, in order of precedence:

        * The data array
        * The bounds data array
        * The `units` CF property

        :Returns:

            `bool`
                Whether or not the coordinate is increasing.

        **Examples**

        >>> c.array
        array([  0  30  60])
        >>> c.direction()
        True

        >>> c.bounds.array
        array([  30  0])
        >>> c.direction()
        False

        """
        _direction = self._custom.get("direction")
        if _direction is not None:
            return _direction

        _direction = self._infer_direction()
        self._custom["direction"] = _direction

        return _direction

    @classmethod
    def create_regular(cls, args, units=None, standard_name=None, bounds=True):
        """
        Create a new `DimensionCoordinate` with the given range and cellsize.

        .. versionadded:: 3.15.0

        :Note: This method does not set the cyclicity of the
               `DimensionCoordinate`.

        :Parameters:

            args: sequence of numbers
                {{regular args}}

            bounds: `bool`, optional
                If True (the default) then the given range represents
                the bounds, and the coordinate points will be the midpoints of
                the bounds. If False, the range represents the coordinate points
                directly.

            units: str or `Units`, optional
                The units of the new `DimensionCoordinate` object.

            standard_name: str, optional
                The standard_name of the `DimensionCoordinate` object.

        :Returns:

            `DimensionCoordinate`
                The newly created DimensionCoordinate object.

        **Examples**

        >>> longitude = cf.DimensionCoordinate.create_regular(
                (-180, 180, 1), units='degrees_east', standard_name='longitude'
            )
        >>> longitude.dump()
        Dimension coordinate: longitude
            standard_name = 'longitude'
            units = 'degrees_east'
            Data(360) = [-179.5, ..., 179.5] degrees_east
            Bounds:units = 'degrees_east'
            Bounds:Data(360, 2) = [[-180.0, ..., 180.0]] degrees_east

        """
        args = np.array(args)

        if args.shape != (3,) or args.dtype.kind not in "fi":
            raise ValueError(
                "The args argument was incorrectly formatted. "
                f"Expected a sequence of three numbers, got {args}."
            )

        range = (args[0], args[1])
        cellsize = args[2]

        range_diff = range[1] - range[0]
        if cellsize > 0 and range_diff <= 0:
            raise ValueError(
                f"Range ({range[0], range[1]}) must be increasing for a "
                f"positive cellsize ({cellsize})"
            )
        elif cellsize < 0 and range_diff >= 0:
            raise ValueError(
                f"Range ({range[0], range[1]}) must be decreasing for a "
                f"negative cellsize ({cellsize})"
            )

        if standard_name is not None and not isinstance(standard_name, str):
            raise ValueError("standard_name must be either None or a string.")

        if bounds:
            cellsize2 = cellsize / 2
            start = range[0] + cellsize2
            end = range[1] - cellsize2
        else:
            start = range[0]
            end = range[1]

        points = np.arange(start, end + cellsize, cellsize)

        coordinate = cls(
            data=Data(points, units=units),
            properties={"standard_name": standard_name},
            copy=False,
        )

        if bounds:
            b = coordinate.create_bounds()
            coordinate.set_bounds(b, copy=False)

        return coordinate

    def create_bounds(
        self, bound=None, cellsize=None, flt=0.5, max=None, min=None
    ):
        """Create cell bounds.

        Creates new cell bounds, irrespective of whether the cells
        already have cell bounds. The new bounds are not set on the
        dimension coordinate construct, but if that is desired they
        may always be added with the `set_bounds` method, for
        instance:

        >>> b = d.create_bounds()
        >>> d.set_bounds(b)

        By default, Voronoi cells are created by defining cell bounds
        that are half way between adjacent coordinate values. For
        non-cyclic coordinates, extrapolation is used so that the
        first and last cells have their coordinates in their centres.

        For instance, the Voronoi cells created for non-cyclic
        coordinates ``0, 40, 90, 180`` have bounds ``(-20, 20), (20,
        65), (65, 135), (135, 225)``. If those coordinates were
        instead cyclic with a period of ``360``, then the bounds would
        be ``(-90, 20), (20, 65), (65, 135), (135, 270)``.

        .. seealso:: `del_bounds`, `get_bounds`, `set_bounds`

        :Parameters:

            bound: data_like scalar, optional
                If set to a value larger (smaller) than the largest
                (smallest) coordinate value, then bounds are created
                which include this value and for which each coordinate
                is in the centre of its bounds. This can be useful for
                recreating the bounds of vertical coordinates from
                Earth system models.

                *Parameter example:*
                  For coordinates ``1, 2, 10``, setting ``bound=0.5``
                  will result in bounds of ``(0.5, 1.5), (1.5, 2.5),
                  (2.5, 17.5)``.

                *Parameter example:*
                  Beware that if *bound* is too large or small, then
                  some returned bounds values will by physically
                  incorrect. For coordinates ``1, 2, 10``, the setting
                  of ``bound=-2`` will result in bounds of ``(-2, 4),
                  (4, 0), (0, 20)``.

            cellsize: optional
                Define the size of each cell that is created. Created
                cells are allowed to overlap and do not have to be
                contigious. The *cellsize* parameter may be one of:

                * data_like scalar

                  Defines the cell size, either in the same units as
                  the coordinates or in the units provided. Note that
                  in this case, the position of each coordinate within
                  the its cell is determined by the *flt* parameter.

            ..

                * `cf.TimeDuration`

                  A time duration defining the cell size. Only
                  applicable to reference time coordinates. It is
                  possible to "anchor" the cell bounds via the
                  `cf.TimeDuration` instance parameters. For example,
                  to specify cell size of one calendar month, starting
                  and ending on the 15th day:
                  ``cellsize=cf.M(day=15)`` (see `cf.M` for
                  details). Note that the *flt* parameter is ignored
                  in this case.

            flt: array_like scalar, optional
                When creating cells with sizes specified by the
                *cellsize* parameter, define the fraction of each
                cell which is less than its coordinate value. By default
                *flt* is 0.5, so that each cell has its coordinate at
                its centre. Ignored if *cellsize* is not set.

                *Parameter example:*
                  For coordinates ``1, 2, 10`` setting ``flt=0.25``
                  will result in bounds of ``(0.75, 1.75), (1.75,
                  2.75), (9.75, 10.75)``.

                *Parameter example:*
                  For decreasing coordinates ``2, 0, -12``, setting
                  ``cellsize=6, flt=0.9`` will result in bounds of
                  ``(2.6, -3.4), (0.6, -5.4), (-11.4, -17.4)``.

            max: data_like scalar, optional
                Limit the created bounds to be no more than this number.

                *Parameter example:*
                  To ensure that all latitude bounds are at most
                  ``90``: ``max=90``, or ``max=cf.Data(90,
                  'degrees_north')``.

            min: data_like scalar, optional
                Limit the created bounds to be no less than this number.

                *Parameter example:*
                  To ensure that all latitude bounds are at least
                  ``-90``: ``min=-90``, or ``min=cf.Data(-90,
                  'degrees_north')``.

        :Returns:

            `Bounds`
                The new coordinate cell bounds.

        **Examples**

        Create bounds for Voronoi cells:

        >>> d = cf.DimensionCoordinate(data=[0.0, 40, 90, 180])
        >>> b = d.create_bounds()
        >>> print(b.array)
        [[-20.  20.]
         [ 20.  65.]
         [ 65. 135.]
         [135. 225.]]
        >>> d = d[::-1]
        >>> b = d.create_bounds()
        >>> print(b.array)
        [[225. 135.]
         [135.  65.]
         [ 65.  20.]
         [ 20. -20.]]

        Cyclic coordinates have a co-located first and last bound:

        >>> d.period(360)
        >>> d.cyclic(0)
        set()
        >>> b = d.create_bounds()
        >>> print(b.array)
        [[270. 135.]
         [135.  65.]
         [ 65.  20.]
         [ 20. -90.]]

        Cellsize units must be equivalent to the coordinate units, or
        if the cell has no units then they are assumed to be the
        same as the coordinates:

        >>> d = cf.DimensionCoordinate(data=cf.Data([0, 2, 4], 'km'))
        >>> b = d.create_bounds(cellsize=cf.Data(2000, 'm'))
        >>> print(b.array)
        [[-1  1]
         [ 1  3]
         [ 3  5]]
        >>> b = d.create_bounds(cellsize=2)
        >>> print(b.array)
        [[-1  1]
         [ 1  3]
         [ 3  5]]

        Cells may be non-contiguous and may overlap:

        >>> d = cf.DimensionCoordinate(data=[1.0, 2, 10])
        >>> b = d.create_bounds(cellsize=1)
        >>> print(b.array)
        [[ 0.5  1.5]
         [ 1.5  2.5]
         [ 9.5 10.5]]
        >>> b = d.create_bounds(cellsize=5)
        >>> print(b.array)
        [[-1.5  3.5]
         [-0.5  4.5]
         [ 7.5 12.5]]

        Reference date-time coordinates can use `cf.TimeDuration` cell
        sizes:

        >>> d = cf.DimensionCoordinate(
        ...   data=cf.Data(['1984-12-01', '1984-12-02'], dt=True)
        ... )
        >>> b = d.create_bounds(cellsize=cf.D())
        >>> print(b.datetime_array)
        [[cftime.DatetimeGregorian(1984, 12, 1, 0, 0, 0, 0)
          cftime.DatetimeGregorian(1984, 12, 2, 0, 0, 0, 0)]
         [cftime.DatetimeGregorian(1984, 12, 2, 0, 0, 0, 0)
          cftime.DatetimeGregorian(1984, 12, 3, 0, 0, 0, 0)]]
        >>> b = d.create_bounds(cellsize=cf.D(hour=12))
        >>> print(b.datetime_array)
        [[cftime.DatetimeGregorian(1984, 11, 30, 12, 0, 0, 0)
          cftime.DatetimeGregorian(1984, 12, 1, 12, 0, 0, 0)]
         [cftime.DatetimeGregorian(1984, 12, 1, 12, 0, 0, 0)
          cftime.DatetimeGregorian(1984, 12, 2, 12, 0, 0, 0)]]

        Cell bounds defined by calendar months:

        >>> d = cf.DimensionCoordinate(
        ...   data=cf.Data(['1985-01-16: 12:00', '1985-02-15'], dt=True)
        ... )
        >>> b = d.create_bounds(cellsize=cf.M())
        >>> print(b.datetime_array)
        [[cftime.DatetimeGregorian(1985, 1, 1, 0, 0, 0, 0)
          cftime.DatetimeGregorian(1985, 2, 1, 0, 0, 0, 0)]
         [cftime.DatetimeGregorian(1985, 2, 1, 0, 0, 0, 0)
          cftime.DatetimeGregorian(1985, 3, 1, 0, 0, 0, 0)]]
        >>> b = d.create_bounds(cellsize=cf.M(day=20))
        >>> print(b.datetime_array)
        [[cftime.DatetimeGregorian(1984, 12, 20, 0, 0, 0, 0
          cftime.DatetimeGregorian(1985, 1, 20, 0, 0, 0, 0]
         [cftime.DatetimeGregorian(1985, 1, 20, 0, 0, 0, 0
          cftime.DatetimeGregorian(1985, 2, 20, 0, 0, 0, 0]]

        Cell bounds defined by calendar years:

        >>> d = cf.DimensionCoordinate(
        ...   data=cf.Data(['1984-06-01', '1985-06-01'], dt=True)
        ... )
        >>> b = d.create_bounds(cellsize=cf.Y(month=12))
        >>> print(b.datetime_array)
        [[cftime.DatetimeGregorian(1983, 12, 1, 0, 0, 0, 0)
          cftime.DatetimeGregorian(1984, 12, 1, 0, 0, 0, 0)]
         [cftime.DatetimeGregorian(1984, 12, 1, 0, 0, 0, 0)
          cftime.DatetimeGregorian(1985, 12, 1, 0, 0, 0, 0)]]

        """
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
                    err_msg = (
                        "Can't create bounds because the bound units "
                        f"({cellsize.Units}) are not compatible with "
                        f"the coordinate units ({self.Units})."
                    )
                    if self.Units.isreftime:
                        if not cellsize.Units.istime:
                            raise ValueError(err_msg)
                        cellsize.Units = Units(self.Units._utime.units)
                    else:
                        if not cellsize.Units.equivalent(self.Units):
                            raise ValueError(err_msg)
                        cellsize.Units = self.Units
                cellsize = cellsize.datum()

                cellsize0 = cellsize * flt
                cellsize1 = cellsize * (1 - flt)
                if not self.direction():
                    cellsize0, cellsize1 = -cellsize1, -cellsize0

                bounds = np.empty((size, 2), dtype=array.dtype)
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
                        f"non-reference time coordinates: {self.Units!r}"
                    )

                bounds = np.empty((size, 2), dtype=object)

                cellsize_bounds = cellsize.bounds
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
                        "Can't create Voronoi bounds when there is "
                        "only one cell"
                    )

                bounds_1d = [array.item(0) * 1.5 - array.item(1) * 0.5]
                bounds_1d.extend((array[0:-1] + array[1:]) * 0.5)
                bounds_1d.append(array.item(-1) * 1.5 - array.item(-2) * 0.5)

                dtype = type(bounds_1d[0])

                if self.cyclic():
                    period = self.period()
                    if period is None:
                        raise ValueError(
                            "Can't create Voronoi bounds for cyclic "
                            "coordinates with no period"
                        )

                    if self.direction():
                        # Increasing
                        b = 0.5 * (array[0] + period - array[-1])
                    else:
                        # Decreasing
                        b = 0.5 * (array[-1] - period + array[0])

                    bounds_1d[0] = array.item(0) - b
                    bounds_1d[-1] = array.item(-1) + b
                else:
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
                if bound <= array.item(0):
                    for i in range(size):
                        bound = 2.0 * array.item(i) - bound
                        bounds_1d.append(bound)
                elif bound >= array.item(-1):
                    for i in range(size - 1, -1, -1):
                        bound = 2.0 * array.item(i) - bound
                        bounds_1d.append(bound)

                    bounds_1d = bounds_1d[::-1]
                else:
                    raise ValueError("bad bound value")

                dtype = type(bounds_1d[-1])

                if not direction:
                    bounds_1d = bounds_1d[::-1]

            bounds = np.empty((size, 2), dtype=dtype)
            bounds[:, 0] = bounds_1d[:-1]
            bounds[:, 1] = bounds_1d[1:]

        # Create coordinate bounds object
        bounds = Bounds(data=Data(bounds, units=self.Units), copy=False)

        return bounds

    def del_cell_characteristics(self, default=ValueError()):
        """Remove the cell characteristics.

        A cell characteristic is assumed to be valid for each cell. Cell
        characteristics are not inferred from the coordinate or bounds
        data, but may be defined with the `set_cell_characteristics`
        method. Cell characteristics are automatically removed
        whenever the new data or bounds are set with `set_data` or
        `set_bounds` respectively.

        .. versionadded:: 3.15.4

        .. seealso:: `get_cell_characteristics`,
                     `has_cell_characteristics`,
                     `set_cell_characteristics`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if cell
                characteristics have not been set.

                {{default Exception}}

        :Returns:

            `dict`
                The removed cell size characteristics, as would have
                been returned by `get_cell_characteristics`.

        """
        out = self.get_cell_characteristics(default=default)
        self._del_component("cell_characteristics", default=None)
        return out

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def flip(self, axes=None, inplace=False, i=False):
        """Flips the dimension coordinate, that is reverses its
        direction."""
        d = _inplace_enabled_define_and_cleanup(self)

        super(DimensionCoordinate, d).flip(axes=axes, inplace=True)

        direction = d._custom.get("direction")
        if direction is not None:
            d._custom["direction"] = not direction

        return d

    def get_bounds(self, default=ValueError(), **kwargs):
        """Return the bounds.

        .. versionadded:: 3.0.0

        .. seealso:: `bounds`, `create_bounds', `get_data`,
                     `del_bounds`, `has_bounds`, `set_bounds`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if bounds
                have not been set.

                {{default Exception}}

        :Returns:

            `Bounds`
                The bounds.

        **Examples**

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

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "get_bounds",
                kwargs,
                "Bounds creation now uses the 'create_bounds' and "
                "'set_bounds' methods.",
            )  # pragma: no cover

        return super().get_bounds(default=default)

    def get_cell_characteristics(self, default=ValueError()):
        """Return cell characteristics.

        A cell characteristic is assumed to be valid for each cell. Cell
        characteristics are not inferred from the coordinate or bounds
        data, but may be defined with the `set_cell_characteristics`
        method. Cell characteristics are automatically removed
        whenever the new data or bounds are set with `set_data` or
        `set_bounds` respectively.

        .. versionadded:: 3.15.4

        .. seealso:: `del_cell_characteristics`,
                     `has_cell_characteristics`,
                     `set_cell_characteristics`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if cell
                characteristics have not been set.

                {{default Exception}}

        :Returns:

            `dict`
                The cell size characteristic (i.e. the absolute
                difference between the cell bounds) and cell spacing
                characteristic (i.e. the absolute difference between
                two neighbouring coordinate values), with keys
                ``'cellsize'`` and ``'spacing'`` respectively. If
                either has a value of `None` then no characteristic
                has been stored for that type.

        """
        out = self._get_component("cell_characteristics", default=None)
        if out is None:
            if default is None:
                return

            return self._default(
                default,
                f"{self.__class__.__name__} has no 'cell_characteristics' "
                "component",
            )

        from copy import deepcopy

        return deepcopy(out)

    def has_cell_characteristics(self):
        """Whether or not there are any cell characteristics.

        A cell characteristic is assumed to be valid for each cell. Cell
        characteristics are not inferred from the coordinate or bounds
        data, but may be defined with the `set_cell_characteristics`
        method. Cell characteristics are automatically removed
        whenever the new data or bounds are set with `set_data` or
        `set_bounds` respectively.

        .. versionadded:: 3.15.4

        .. seealso:: `del_cell_characteristics`,
                     `get_cell_characteristics`,
                     `set_cell_characteristics`

        :Returns:

            `bool`
                Whether or not there are any cell characteristics.

        """
        return self._has_component("cell_characteristics")

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def roll(self, axis, shift, inplace=False, i=False):
        """Rolls the dimension coordinate along a cyclic axis."""
        c = _inplace_enabled_define_and_cleanup(self)

        size = c.size
        if size <= 1:
            return c

        shift %= size
        if not shift:
            # Null roll
            return c

        period = c.period()

        if period is None:
            raise ValueError(
                f"Can't roll {self.__class__.__name__} when no period has "
                "been set"
            )

        direction = c.direction()

        centre = c._centre(period)

        if axis not in [0, -1]:
            raise ValueError(
                f"Can't roll axis {axis} when there is only one axis"
            )

        super(DimensionCoordinate, c).roll(axis, shift, inplace=True)

        c.dtype = np.result_type(c.dtype, period.dtype)

        data = c.data

        b = c.get_bounds(None)
        bounds_data = c.get_bounds_data(None, _fill_value=False)

        if bounds_data is not None:
            b.dtype = np.result_type(bounds_data.dtype, period.dtype)
            bounds_data = b.get_data(None, _fill_value=False)

        if direction:
            # Increasing
            data[:shift] -= period
            if bounds_data is not None:
                bounds_data[:shift] -= period

            if data[0] <= centre - period:
                data += period
                if bounds_data is not None:
                    bounds_data += period
        else:
            # Decreasing
            data[:shift] += period
            if bounds_data is not None:
                bounds_data[:shift] += period

            if data[0] >= centre + period:
                data -= period
                if bounds_data is not None:
                    bounds_data -= period

        c._custom["direction"] = direction

        return c

    def set_bounds(self, bounds, copy=True):
        """Set the bounds.

        .. versionadded:: 3.15.4

        .. seealso: `del_bounds`, `get_bounds`, `has_bounds`, `set_data`

        :Parameters:

            bounds: `Bounds`
                The bounds to be inserted.

            copy: `bool`, optional
                If True then copy the bounds prior to
                insertion. By default the bounds are copied.

        :Returns:

            `None`

        **Examples**

        >>> import numpy
        >>> b = {{package}}.Bounds(data=numpy.arange(10).reshape(5, 2))
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

        """
        self._del_component("cell_characteristics", default=None)
        super().set_bounds(bounds, copy=copy)

    def set_cell_characteristics(self, cellsize, spacing):
        """Set cell characteristics.

        A cell characteristic is assumed to be valid for each cell. Cell
        characteristics are not inferred from the coordinate or bounds
        data, but may be set with this method. Cell characteristics
        are automatically removed whenever the new data or bounds are
        set with `set_data` or `set_bounds` respectively.

        .. versionadded:: 3.15.4

        .. seealso:: `del_cell_characteristics`,
                     `get_cell_characteristics`,
                     `has_cell_characteristics`

        :Parameters:

            cellsize:
                The cell size (i.e. the absolute difference between
                the cell bounds) characteristic. May be a `Query`,
                `TimeDuration`, scalar `Data`, scalar data_like
                object, or `None`. A value of `None` means no
                characteristic has been set.

            spacing:
                The cell spacing (i.e. the absolute difference between
                two neighbouring coordinate values) characteristic.
                May be a `Query`, `TimeDuration`, scalar `Data`,
                scalar data_like object, or `None`. A value of `None`
                means no characteristic has been set.

        :Returns:

            `None`

        **Examples**

        >>> d.set_cell_characteristics(cellsize=cf.D(5), spacing=cf.D(1))

        >>> d.set_cell_characteristics(cf.Data(10, 'degree_E'), None)

        >>> d.set_cell_characteristics(cf.wi(100, 200), 150)

        """
        chars = {}
        if cellsize is not None:
            chars["cellsize"] = cellsize

        if spacing is not None:
            chars["spacing"] = spacing

        if not chars:
            self.del_cell_characteristics(None)
            return

        self._set_component("cell_characteristics", chars, copy=False)

    def set_data(self, data, copy=True, inplace=True):
        """Set the data.

        The units, calendar and fill value of the incoming `Data`
        instance are removed prior to insertion.

        .. versionadded:: 3.15.4

        .. seealso:: `data`, `del_data`, `get_data`, `has_data`

        :Parameters:

            data: `Data`
                The data to be inserted.

                {{data_like}}

            copy: `bool`, optional
                If True then copy the data prior to
                insertion. By default the data are copied.

            {{inplace: `bool`, optional (default True)}}

                .. versionadded:: 3.7.0

        :Returns:

            `None` or `{{class}}`
                If the operation was in-place then `None` is returned,
                otherwise return a new `{{class}}` instance containing
                the new data.

        **Examples**

        >>> f = cf.{{class}}()
        >>> f.set_data([1, 2, 3])
        >>> f.has_data()
        True
        >>> f.get_data()
        <CF Data(3): [1, 2, 3]>
        >>> f.data
        <CF Data(3): [1, 2, 3]>
        >>> f.del_data()
        <CF Data(3): [1, 2, 3]>
        >>> g = f.set_data([4, 5, 6], inplace=False)
        >>> g.data
        <CF Data(3): [4, 5, 6]>
        >>> f.has_data()
        False
        >>> print(f.get_data(None))
        None
        >>> print(f.del_data(None))
        None

        """
        self._del_component("cell_characteristics", default=None)
        return super().set_data(data, copy=copy, inplace=inplace)

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def role(self):
        """Deprecated at version 3.0.0, use `construct_type` attribute
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "role",
            "Use attribute 'construct_type' instead",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover
