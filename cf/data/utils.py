"""General functions useful for `Data` functionality."""
from functools import lru_cache, partial
from itertools import product

import dask.array as da
import numpy as np

from ..cfdatetime import dt as cf_dt
from ..cfdatetime import dt2rt, rt2dt, st2rt
from ..units import Units


def _da_ma_allclose(x, y, masked_equal=True, rtol=1e-05, atol=1e-08):
    """An effective dask.array.ma.allclose method.

    True if two dask arrays are element-wise equal within
    a tolerance.

    Equivalent to allclose except that masked values are treated
    as equal (default) or unequal, depending on the masked_equal
    argument.

    Define an effective da.ma.allclose method here because one is
    currently missing in the Dask codebase.

    Note that all default arguments are the same as those provided to
    the corresponding NumPy method (see the `numpy.ma.allclose` API
    reference).

    TODODASK: put in a PR to Dask to request to add as genuine method.

    .. versionadded:: 4.0.0

        :Parameters:

            x: a dask array to compare with y

            y: a dask array to compare with x

            masked_equal:
                Whether masked values in a and b are considered
                equal (True) or not (False). They are considered equal
                by default.

            rtol:
                Relative tolerance. Default is 1e-05.

            atol:
                Absolute tolerance. Default is 1e-08.

        :Returns:

            Boolean
                A Boolean value indicating whether or not the
                two dask arrays are element-wise equal to
                the given rtol and atol tolerance.

    """

    def allclose(a_blocks, b_blocks):
        """Run `ma.allclose` across multiple blocks over two arrays."""
        result = True
        for a, b in zip(a_blocks, b_blocks):
            result &= np.ma.allclose(
                a, b, rtol=rtol, atol=atol, masked_equal=masked_equal
            )

        return result

    # Handle scalars, which are not valid inputs to da.blockwise, though
    # test for scalars by checking the shape to avoid computation, etc.
    if not x.shape and not y.shape:  # i.e. both are scalars
        return np.ma.allclose(x, y)
    elif not x.shape or not y.shape:
        return False  # one is a scalar and the other isn't => not all close

    axes = tuple(range(x.ndim))
    return da.blockwise(
        allclose,
        "",
        x,
        axes,
        y,
        axes,
        dtype=bool,
    )


def convert_to_datetime(array, units):
    """Convert a daskarray to.

    .. versionadded:: 4.0.0

        :Parameters:

            array: dask array

            units : `Units`

        :Returns:

            dask array
                A new dask array containing datetime objects.

    """
    dx = array.map_blocks(partial(rt2dt, units_in=units), dtype=object)
    return dx


def convert_to_reftime(array, units, first_value=None):
    """Convert a dask array of string or object date-times to floating
    point reference times.

    .. versionadded:: 4.0.0

        :Parameters:

            array: dask array

            units : `Units`

            first_value : scalar, optional

        :Returns:

            dask array, `Units`
                A new dask array containing reference times, and its
                units.

    """
    kind = array.dtype.kind
    if kind in "US":
        # Convert date-time strings to reference time floats
        if not units:
            value = first_value(array, first_value)
            if value is not None:
                YMD = str(value).partition("T")[0]
            else:
                YMD = "1970-01-01"

            units = Units("days since " + YMD, units._calendar)

        array = array.map_blocks(
            partial(st2rt, units_in=units, units_out=units), dtype=float
        )

    elif kind == "O":
        # Convert date-time objects to reference time floats
        value = first_value(array, first_value)
        if value is not None:
            x = value
        else:
            x = cf_dt(1970, 1, 1, calendar="gregorian")

        x_since = "days since " + "-".join(map(str, (x.year, x.month, x.day)))
        x_calendar = getattr(x, "calendar", "gregorian")

        d_calendar = getattr(units, "calendar", None)
        d_units = getattr(units, "units", None)

        if x_calendar != "":
            if d_calendar is not None:
                if not units.equivalent(Units(x_since, x_calendar)):
                    raise ValueError(
                        f"Incompatible units: "
                        f"{units!r}, {Units(x_since, x_calendar)!r}"
                    )
            else:
                d_calendar = x_calendar

        if not units:
            # Set the units to something that is (hopefully)
            # close to all of the datetimes, in an attempt to
            # reduce errors arising from the conversion to
            # reference times
            units = Units(x_since, calendar=d_calendar)
        else:
            units = Units(d_units, calendar=d_calendar)

        # Check that all date-time objects have correct and
        # equivalent calendars
        calendars = unique_calendars(array)
        if len(calendars) > 1:
            raise ValueError(
                "Not all date-time objects have equivalent "
                f"calendars: {tuple(calendars)}"
            )

        # If the date-times are calendar-agnostic, assign the
        # given calendar, defaulting to Gregorian.
        if calendars.pop() == "":
            calendar = getattr(units, "calendar", "gregorian")

            # TODODASK: can map_blocks this, I think
            new_array = da.empty_like(array, dtype=object)
            for i in np.ndindex(new_array.shape):
                new_array[i] = cf_dt(array[i], calendar=calendar)

            array = new_array

        # Convert the date-time objects to reference times
        array = array.map_blocks(dt2rt, units_out=units, dtype=float)

    if not units.isreftime:
        raise ValueError(
            f"Can't create a reference time array with units {units!r}"
        )

    return array, units


def first_non_missing_value(array, cached=None):
    """Return the first non-missing value of an array.

    If the array contains only missing data then `None` is returned.

    If a cached value is provided then that is returned without
    looking for the actual first non-missing value.

    .. versionadded:: 4.0.0

    :Parameters:

    array: dask array
        The array to be inspected.

    cached: scalar, optional
        If set to a value other than `Ǹone`, then return this value
        instead of inspecting the array.

    :Returns:

            If the *cached* parameter is set then its value is
            returned. Otherwise return the first non-missing value, or
            `None` if there isn't one.

    """
    if cached is not None:
        return cached

    # This does not look particularly efficient, but the expectation
    # is that the first element in the array will not be missing data.

    shape = array.shape
    for i in range(array.size):
        index = np.unravel_index(i, shape)
        x = array[index].compute()
        if x is np.ma.masked:
            continue

        return x.item()

    return None


def unique_calendars(array):
    """Find the unique calendars from a dask array of date-time objects.

    .. versionadded:: 4.0.0

    :Returns:

        `set`
            The unique calendars.

    """

    def _get_calendar(x):
        getattr(x, "calendar", "gregorian")

    _calendars = np.vectorize(_get_calendar, otypes=[np.dtype(str)])

    array = array.map_blocks(_calendars, dtype=str)

    cals = da.unique(array).compute()
    if np.ma.isMA(cals):
        cals = cals.compressed()

    # TODODASK - need to allow differetn bu equivalent calendars, such
    # as "gregorian" and 'standard'. Or perhaps this should by the
    # caller?

    return set(cals.tolist())


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

    **Examples:**

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
