"""General functions useful for `Data` functionality."""
from functools import lru_cache, partial
from itertools import product

import numpy as np

from ..cfdatetime import (
    canonical_calendar,
    default_calendar,
    dt,
    dt2rt,
    rt2dt,
    st2rt,
)
from ..units import Units
from .dask_utils import cf_YMDhms


def _is_numeric_dtype(array):
    """True if the given array is of a numeric or boolean data type.

    .. versionadded:: 4.0.0

        :Parameters:

            array: numpy-like array

        :Returns:

            `bool`
                Whether or not the array holds numeric elements.

    **Examples:**

    >>> a = np.array([0, 1, 2])
    >>> _is_numeric_dtype(a)
    True
    >>> a = np.array([False, True, True])
    >>> _is_numeric_dtype(a)
    True
    >>> a = np.array(["a", "b", "c"], dtype="S1")
    >>> _is_numeric_dtype(a)
    False
    >>> a = np.ma.array([10.0, 2.0, 3.0], mask=[1, 0, 0])
    >>> _is_numeric_dtype(a)
    True
    >>> a = np.array(10)
    >>> _is_numeric_dtype(a)
    True
    >>> a = np.empty(1, dtype=object)
    >>> _is_numeric_dtype(a)
    False

    """
    # TODODASK: do we need to make any specific checks relating to ways of
    # encoding datetimes, which could be encoded as strings, e.g. as in
    # "2000-12-3 12:00", yet could be considered, or encoded as, numeric?
    dtype = array.dtype
    # This checks if the dtype is either a standard "numeric" type (i.e.
    # int types, floating point types or complex floating point types)
    # or Boolean, which are effectively a restricted int type (0 or 1).
    # We determine the former by seeing if it sits under the 'np.number'
    # top-level dtype in the NumPy dtype hierarchy; see the
    # 'Hierarchy of type objects' figure diagram under:
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def convert_to_datetime(array, units):
    """Convert a dask array of numbers to one of date-time objects.

    .. versionadded:: 4.0.0

    .. seealso `convert_to_reftime`

    :Parameters:

        array: `dask.array.Array`
            The input reference time values.

        units: `Units`
            The reference time units that define the output
            date-time objects.

    :Returns:

        `dask.array.Array`
            A new dask array containing date-time objects.

    **Examples**

    >>> import dask.array as da
    >>> d = da.from_array(2.5)
    >>> e = convert_to_datetime(d, cf.Units("days since 2000-12-01"))
    >>> print(e.compute())
    2000-12-03 12:00:00

    """
    return array.map_blocks(
        partial(rt2dt, units_in=units),
        dtype=object,
        meta=np.array((), dtype=object),
    )


def convert_to_reftime(array, units=None, first_value=None):
    """Convert a dask array of string or object date-times to floating
    point reference times.

    .. versionadded:: 4.0.0

    .. seealso `convert_to_datetime`

    :Parameters:

        array: `dask.array.Array`

        units: `Units`, optional
             Specify the units for the output reference time
             values. By default the the units are inferred from first
             non-missing value in the array, or set to ``<Units: days
             since 1970-01-01 gregorian>`` if all values are missing.

        first_value: optional
            If set, then assumed to be equal to the first non-missing
            value of the array, thereby removing the need to find it
            by inspection of *array*, which may be expensive. By
            default the first non-missing value is found from *array*.

    :Returns:

        `dask.array.Array`, `Units`
            The reference times, and their units.

    >>> import dask.array as da
    >>> d = da.from_array(2.5)
    >>> e = convert_to_datetime(d, cf.Units("days since 2000-12-01"))

    >>> f, u = convert_to_reftime(e)
    >>> f.compute()
    0.5
    >>> u
    <Units: days since 2000-12-3 standard>

    >>> f, u = convert_to_reftime(e, cf.Units("days since 1999-12-01"))
    >>> f.compute()
    368.5
    >>> u
    <Units: days since 1999-12-01 standard>

    """
    kind = array.dtype.kind
    if kind in "US":
        # Convert date-time strings to reference time floats
        if not units:
            first_value = first_non_missing_value(array, cached=first_value)
            if first_value is not None:
                YMD = str(first_value).partition("T")[0]
            else:
                YMD = "1970-01-01"

            units = Units("days since " + YMD, default_calendar)

        array = array.map_blocks(
            partial(st2rt, units_in=units, units_out=units), dtype=float
        )

    elif kind == "O":
        # Convert date-time objects to reference time floats
        first_value = first_non_missing_value(array, cached=first_value)
        if first_value is not None:
            x = first_value
        else:
            x = dt(1970, 1, 1, calendar=default_calendar)

        x_since = "days since " + "-".join(map(str, (x.year, x.month, x.day)))
        x_calendar = getattr(x, "calendar", default_calendar)

        d_calendar = getattr(units, "calendar", None)
        d_units = getattr(units, "units", None)

        if x_calendar != "":
            if units is None:
                d_calendar = x_calendar
            elif not units.equivalent(Units(x_since, x_calendar)):
                raise ValueError(
                    "Incompatible units: "
                    f"{units!r}, {Units(x_since, x_calendar)!r}"
                )

        if not units:
            # Set the units to something that is (hopefully) close to
            # all of the datetimes, in an attempt to reduce errors
            # arising from the conversion to reference times
            units = Units(x_since, calendar=d_calendar)
        else:
            units = Units(d_units, calendar=d_calendar)

        # Check that all date-time objects have correct and equivalent
        # calendars
        #        calendars = unique_calendars(array)
        #        if len(calendars) > 1:
        #            raise ValueError(
        #                "Not all date-time objects have equivalent "
        #                f"calendars: {tuple(calendars)}"
        #            )

        # If the date-times are calendar-agnostic, assign the given
        # calendar, defaulting to Gregorian.
        #        if calendars.pop() == "":
        #        if first_value.calendar == "":
        #
        #            def set_calendar(x, calendar=default_calendar):
        #                np.vectorize(
        #                    partial(dt, calendar=calendar), otypes=[np.dtype(object)]
        #                )
        #
        #            array = array.map_blocks(
        #                set_calendar,
        #                calendar=getattr(units, "calendar", default_calendar),
        #                dtype=object,
        #            )

        # Convert the date-time objects to reference times
        array = array.map_blocks(
            dt2rt, units_in=None, units_out=units, dtype=float
        )

    if not units.isreftime:
        raise ValueError(
            f"Can't create a reference time array with units {units!r}"
        )

    return array, units


def first_non_missing_value(array, cached=None):
    """Return the first non-missing value of an array.

    .. versionadded:: 4.0.0

    :Parameters:

        array: `dask.array.Array`
            The array to be inspected.

        cached: scalar, optional
            If set to a value other than `None`, then return this
            value instead of inspecting the array.

    :Returns:

            If set, then *cached* is returned. Otherwise returns the
            first non-missing value of *array*, or `None` if there
            isn't one.

    """
    if cached is not None:
        return cached

    # This does not look particularly efficient, but the expectation
    # is that the first element in the array will not be missing data.

    shape = array.shape
    for i in range(array.size):
        index = np.unravel_index(i, shape)
        x = array[index].compute()
        if x is not np.ma.masked:
            try:
                return x.item()
            except AttributeError:
                return x

    return None


def unique_calendars(array):
    """Find the unique calendars from a dask array of date-time objects.

    .. versionadded:: 4.0.0

    :Parameters:

        array: `dask.array.Array`
            A dask array of data-time objects.

    :Returns:

        `set`
            The unique calendars.

    """

    def _get_calendar(x):
        return getattr(x, "calendar", default_calendar)

    _calendars = np.vectorize(_get_calendar, otypes=[np.dtype(str)])

    # TODO
    #
    # da.unique doesn't work well with masked data (2022-02-07), so do
    # move to numpy-space for now. When da.unique is better we can
    # replace the next two lines of code with:
    #
    #   array = array.map_blocks(_calendars, dtype=str)
    #   calendars = da.unique(array).compute()
    array = _calendars(array.compute())
    calendars = np.unique(array)

    if np.ma.isMA(calendars):
        calendars = calendars.compressed()

    # Replace each calendar with its canonical name
    out = [canonical_calendar[cal] for cal in calendars.tolist()]

    return set(out)


@lru_cache(maxsize=32)
def new_axis_identifier(existing_axes=(), basename="dim"):
    """Return a new, unique axis identifier.

    The name is arbitrary and has no semantic meaning.

    .. versionadded:: TODODASK

    :Parameters:

        existing_axes: sequence of `str`, optional
            Any existing axis names that are not to be duplicated.

        basename: `str`, optional
            The root of the new axis identifier. The new axis
            identifier will be this root followed by an integer.

    :Returns:

        `str`
            The new axis idenfifier.

    **Examples:**

    >>> new_axis_identifier()
    'dim0'
    >>> new_axis_identifier(['dim0'])
    'dim1'
    >>> new_axis_identifier(['dim3'])
    'dim1'
     >>> new_axis_identifier(['dim1'])
    'dim2'
    >>> new_axis_identifier(['dim1', 'dim0'])
    'dim2'
    >>> new_axis_identifier(['dim3', 'dim4'])
    'dim2'
    >>> new_axis_identifier(['dim2', 'dim0'])
    'dim3'
    >>> new_axis_identifier(['dim3', 'dim4', 'dim0'])
    'dim5'
    >>> d._new_axis_identifier(basename='axis')
    'axis0'
    >>> d._new_axis_identifier(basename='axis')
    'axis0'
    >>> d._new_axis_identifier(['dim0'], basename='axis')
    'axis1'
    >>> d._new_axis_identifier(['dim0', 'dim1'], basename='axis')
    'axis2'

    """
    n = len(existing_axes)
    axis = f"{basename}{n}"
    while axis in existing_axes:
        n += 1
        axis = f"{basename}{n}"

    return axis


def chunk_positions(chunks):
    """Find the position of each chunk.

    .. versionadded:: TODODASK

    .. seealso:: `chunk_shapes`

    :Parameters:

        chunks: `tuple`
            The chunk sizes along each dimension, as output by
            `dask.array.Array.chunks`.

    **Examples**

    >>> chunks = ((1, 2), (9,), (44, 55, 66))
    >>> for position in chunk_positions(chunks):
    ...     print(position)
    ...
    (0, 0, 0)
    (0, 0, 1)
    (0, 0, 2)
    (1, 0, 0)
    (1, 0, 1)
    (1, 0, 2)

    """
    return product(*(range(len(bds)) for bds in chunks))


def chunk_shapes(chunks):
    """Find the shape of each chunk.

    .. versionadded:: TODODASK

    .. seealso:: `chunk_positions`

    :Parameters:

        chunks: `tuple`
            The chunk sizes along each dimension, as output by
            `dask.array.Array.chunks`.

    **Examples**

    >>> chunks = ((1, 2), (9,), (4, 5, 6))
    >>> for shape in chunk_shapes(chunks):
    ...     print(shape)
    ...
    (1, 9, 4)
    (1, 9, 5)
    (1, 9, 6)
    (2, 9, 4)
    (2, 9, 5)
    (2, 9, 6)

    """
    return product(*chunks)


def is_small(array, threshold=None):
    """TODODASK - need to define what 'small' is, and consider the API
     in general

    We adjust the size of the data here for the potiential of a mask

    Returns False if size is unknown

    .. versionadded:: 4.0.0
    """
    if threshold is None:
        threshold = 2 ** 90  # TODODASK - True for now!

    return array.size * (array.dtype.itemsize + 1) < threshold


def is_very_small(array, threshold=None):
    """
    TODODASK - need to define what 'very small' is, and consider the API
    in general

    .. versionadded:: 4.0.0

    """
    if threshold is None:
        threshold = 0.125 * 2 ** 90  # TODODASK - True for now!

    return is_small(array, threshold)


def dask_compatible(a):
    """Convert an object to one which is dask compatible.

    The object is returned unchanged unless it is a cf object
    containing data, in which case the dask array of the data is
    returned instead.

    .. versionadded:: 4.0.0

    """
    try:
        return a.data._get_dask()
    except AttributeError:
        return a


def scalar_masked_array(dtype=float):
    """Return a scalar masked array.

     .. versionadded:: TODODASK

     :Parmaeters:

         dtype: data-type, optional
             Desired output data-type for the array, e.g,
             `numpy.int8`. Default is `numpy.float64`.

     :Returns:

         `np.ma.core.MaskedArray`
             The scalar masked array.

     **Examples**

     >>> scalar_masked_array()
     masked_array(data=--,
                  mask=True,
            fill_value=1e+20,
                 dtype=float64)
     >>> scalar_masked_array(dtype('int32'))
     masked_array(data=--,
                  mask=True,
            fill_value=999999,
                 dtype=int32)
     >>> scalar_masked_array('U45')
     masked_array(data=--,
                  mask=True,
            fill_value='N/A',
                dtype='<U45')
    >>> scalar_masked_array(bool)
    masked_array(data=--,
                 mask=True,
            fill_value=True,
                dtype=bool)

    """
    a = np.ma.empty((), dtype=dtype)
    a.mask = True
    return a


def conform_units(value, units):
    """Conform units.

    If *value* has units defined by its `Units` attribute then

    * if the value units are equal to *units* then *value* is returned
      unchanged;

    * if the value units are equivalent to *units* then a copy of
      *value* converted to *units* is returned;

    * if the value units are not equivalent to *units* then an
      exception is raised.

    In all other cases *value* is returned unchanged.

    .. versionadded:: TODODASK

    :Parameters:

        value:
            The value whose units are to be conformed to *units*.

        units: `Units`
            The units to conform to.

    **Examples**

    >>> conform_units(1, cf.Units('metres'))
    1
    >>> conform_units([1, 2, 3], cf.Units('metres'))
    [1, 2, 3]
    >>> import numpy
    >>> conform_units(numpy.array([1, 2, 3]), cf.Units('metres'))
    array([1, 2, 3])
    >>> conform_units('string', cf.Units('metres'))
    'string'
    >>> d = cf.Data([1, 2] , 'm')
    >>> conform_units(d, cf.Units('metres'))
    <CF Data(2): [1, 2] m>
    >>> d = cf.Data([1, 2] , 'km')
    >>> conform_units(d, cf.Units('metres'))
    <CF Data(2): [1000.0, 2000.0] metres>
    >>> conform_units(d, cf.Units('s'))
        ...
    ValueError: Units <Units: km> are incompatible with units <Units: s>

    """
    try:
        value_units = value.Units
    except AttributeError:
        pass
    else:
        if value_units.equivalent(units):
            if value_units != units:
                value = value.copy()
                value.Units = units
        elif value_units and units:
            raise ValueError(
                f"Units {value_units!r} are incompatible with units {units!r}"
            )

    return value


def YMDhms(d, attr):
    """Return a date-time component of the data.

    Only applicable for data with reference time units. The returned
    `Data` will have the same mask hardness as the original array.

    .. versionadded:: TODODASK

    .. seealso:: `~cf.Data.year`, ~cf.Data.month`, `~cf.Data.day`,
                 `~cf.Data.hour`, `~cf.Data.minute`, `~cf.Data.second`

    :Parameters:

        d: `Data`
            The data from which to extract date-time component.

        attr: `str`
            The name of the date-time component, one of ``'year'``,
            ``'month'``, ``'day'``, ``'hour'``, ``'minute'``,
            ``'second'``.

    :Returns:

        `Data`
            The date-time component

    **Examples**

    >>> d = cf.Data([0, 1, 2], 'days since 1999-12-31')
    >>> YMDhms(d, 'year').array
    >>> array([1999, 2000, 2000])

    """
    units = d.Units
    if not units.isreftime:
        raise ValueError(f"Can't get {attr}s from data with {units!r}")

    d = d._asdatetime()
    d._map_blocks(partial(cf_YMDhms, attr=attr), dtype=int)
    d.override_units(Units(None), inplace=True)
    return d
