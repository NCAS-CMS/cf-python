"""General functions useful for `Data` functionality."""
from functools import lru_cache, partial
from itertools import product

import dask.array as da
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

_units_None = Units(None)


def is_numeric_dtype(array):
    """True if the given array is of a numeric or boolean data type.

    .. versionadded:: 3.14.0

        :Parameters:

            array: numpy-like array

        :Returns:

            `bool`
                Whether or not the array holds numeric elements.

    **Examples**

    >>> a = np.array([0, 1, 2])
    >>> cf.data.utils.is_numeric_dtype(a)
    True
    >>> a = np.array([False, True, True])
    >>> cf.data.utils.is_numeric_dtype(a)
    True
    >>> a = np.array(["a", "b", "c"], dtype="S1")
    >>> cf.data.utils.is_numeric_dtype(a)
    False
    >>> a = np.ma.array([10.0, 2.0, 3.0], mask=[1, 0, 0])
    >>> cf.data.utils.is_numeric_dtype(a)
    True
    >>> a = np.array(10)
    >>> cf.data.utils.is_numeric_dtype(a)
    True
    >>> a = np.empty(1, dtype=object)
    >>> cf.data.utils.is_numeric_dtype(a)
    False

    """
    dtype = array.dtype

    # This checks if the dtype is either a standard "numeric" type (i.e.
    # int types, floating point types or complex floating point types)
    # or Boolean, which are effectively a restricted int type (0 or 1).
    # We determine the former by seeing if it sits under the 'np.number'
    # top-level dtype in the NumPy dtype hierarchy; see the
    # 'Hierarchy of type objects' figure diagram under:
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#scalars
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def convert_to_datetime(a, units):
    """Convert a dask array of numbers to one of date-time objects.

    .. versionadded:: 3.14.0

    .. seealso `convert_to_reftime`

    :Parameters:

        a: `dask.array.Array`
            The input numeric reference time values.

        units: `Units`
            The reference time units that define the output
            date-time objects.

    :Returns:

        `dask.array.Array`
            A new dask array containing date-time objects.

    **Examples**

    >>> import dask.array as da
    >>> d = da.from_array(2.5)
    >>> e = cf.data.utils.convert_to_datetime(d, cf.Units("days since 2000-12-01"))
    >>> print(e.compute())
    2000-12-03 12:00:00

    """
    return a.map_blocks(
        partial(rt2dt, units_in=units),
        dtype=object,
        meta=np.array((), dtype=object),
    )


def convert_to_reftime(a, units=None, first_value=None):
    """Convert a dask array of string or object date-times to floating
    point reference times.

    .. versionadded:: 3.14.0

    .. seealso `convert_to_datetime`

    :Parameters:

        a: `dask.array.Array`

        units: `Units`, optional
             Specify the units for the output reference time
             values. By default the units are inferred from the first
             non-missing value in the array, or set to ``<Units: days
             since 1970-01-01 gregorian>`` if all values are missing.

        first_value: optional
            If set, then assumed to be equal to the first non-missing
            value of the array, thereby removing the need to find it
            by inspection of *a*, which may be expensive. By default
            the first non-missing value is found from *a*.

    :Returns:

        (`dask.array.Array`, `Units`)
            The reference times, and their units.

    >>> import dask.array as da
    >>> d = da.from_array(2.5)
    >>> e = cf.data.utils.convert_to_datetime(d, cf.Units("days since 2000-12-01"))

    >>> f, u = cf.data.utils.convert_to_reftime(e)
    >>> f.compute()
    0.5
    >>> u
    <Units: days since 2000-12-3 standard>

    >>> f, u = cf.data.utils.convert_to_reftime(e, cf.Units("days since 1999-12-01"))
    >>> f.compute()
    368.5
    >>> u
    <Units: days since 1999-12-01 standard>

    """
    kind = a.dtype.kind
    if kind in "US":
        # Convert date-time strings to reference time floats
        if not units:
            first_value = first_non_missing_value(a, cached=first_value)
            if first_value is not None:
                YMD = str(first_value).partition("T")[0]
            else:
                YMD = "1970-01-01"

            units = Units(
                "days since " + YMD,
                getattr(units, "calendar", default_calendar),
            )

        a = a.map_blocks(
            partial(st2rt, units_in=units, units_out=units), dtype=float
        )

    elif kind == "O":
        # Convert date-time objects to reference time floats
        first_value = first_non_missing_value(a, cached=first_value)
        if first_value is not None:
            x = first_value
        else:
            x = dt(1970, 1, 1, calendar=default_calendar)

        x_since = "days since " + "-".join(map(str, (x.year, x.month, x.day)))
        x_calendar = getattr(x, "calendar", default_calendar)

        d_calendar = getattr(units, "calendar", None)
        d_units = getattr(units, "units", None)

        if x_calendar != "":
            if not units:
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

        # Convert the date-time objects to reference times
        a = a.map_blocks(dt2rt, units_in=None, units_out=units, dtype=float)

    if not units.isreftime:
        raise ValueError(
            f"Can't create a reference time array with units {units!r}"
        )

    return a, units


def first_non_missing_value(a, cached=None, method="index"):
    """Return the first non-missing value of a dask array.

    .. versionadded:: 3.14.0

    :Parameters:

        a: `dask.array.Array`
            The array to be inspected.

        cached: scalar, optional
            If set to a value other than `None`, then return without
            inspecting the array. This allows a previously found first
            value to be used instead of a potentially costly array
            access.

        method: `str`, optional
            Select the method used to find the first non-missing
            value.

            The default ``'index'`` method evaulates sequentially the
            elements of the flattened array and returns when the first
            non-missing value is found.

            The ``'mask'`` method finds the first non-missing value of
            the flattened array as that which has the same location as
            the first False element of the flattened array mask.

            It is considered likely that the ``'index'`` method is
            fastest for data for which the first element is not
            missing, but this may not always be the case.

    :Returns:

            If set, then *cached* is returned. Otherwise returns the
            first non-missing value of *a*, or `None` if there isn't
            one.

    **Examples**

    >>> import dask.array as da
    >>> d = da.arange(8).reshape(2, 4)
    >>> print(d.compute())
    [[0 1 2 3]
     [4 5 6 7]]
    >>> cf.data.utils.first_non_missing_value(d)
    0
    >>> cf.data.utils.first_non_missing_value(d, cached=99)
    99
    >>> d[0, 0] = cf.masked
    >>> cf.data.utils.first_non_missing_value(d)
    1
    >>> d[0, :] = cf.masked
    >>> cf.data.utils.first_non_missing_value(d)
    4
    >>> cf.data.utils.first_non_missing_value(d, cached=99)
    99
    >>> d[...] = cf.masked
    >>> print(cf.data.utils.first_non_missing_value(d))
    None
    >>> print(cf.data.utils.first_non_missing_value(d, cached=99))
    99

    """
    if cached is not None:
        return cached

    if method == "index":
        shape = a.shape
        for i in range(a.size):
            index = np.unravel_index(i, shape)
            x = a[index].compute()
            if not (x is np.ma.masked or np.ma.getmask(x)):
                try:
                    return x.item()
                except AttributeError:
                    return x

        return

    if method == "mask":
        mask = da.ma.getmaskarray(a)
        if not a.ndim:
            # Scalar data
            if mask:
                return

            a = a.compute()
            try:
                return a.item()
            except AttributeError:
                return a

        x = a[da.unravel_index(mask.argmin(), a.shape)].compute()
        if x is np.ma.masked:
            return

        try:
            return x.item()
        except AttributeError:
            return x

    raise ValueError(f"Unknown value of 'method': {method!r}")


def unique_calendars(a):
    """Find the unique calendars from a dask array of date-time objects.

    .. versionadded:: 3.14.0

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

    # TODODASK
    #
    # da.unique doesn't work well with masked data (2022-02-07), so do
    # move to numpy-space for now. When da.unique is better we can
    # replace the next two lines of code with:
    #
    #   a = a.map_blocks(_calendars, dtype=str)
    #   calendars = da.unique(array).compute()
    a = _calendars(a.compute())
    calendars = np.unique(a)

    if np.ma.isMA(calendars):
        calendars = calendars.compressed()

    # Replace each calendar with its canonical name
    out = [canonical_calendar[cal] for cal in calendars.tolist()]

    return set(out)


@lru_cache(maxsize=32)
def new_axis_identifier(existing_axes=(), basename="dim"):
    """Return a new, unique axis identifier.

    The name is arbitrary and has no semantic meaning.

    .. versionadded:: 3.14.0

    :Parameters:

        existing_axes: sequence of `str`, optional
            Any existing axis names that are not to be duplicated.

        basename: `str`, optional
            The root of the new axis identifier. The new axis
            identifier will be this root followed by an integer.

    :Returns:

        `str`
            The new axis idenfifier.

    **Examples**

    >>> cf.data.utils.new_axis_identifier()
    'dim0'
    >>> cf.data.utils.new_axis_identifier(['dim0'])
    'dim1'
    >>> cf.data.utils.new_axis_identifier(['dim3'])
    'dim1'
     >>> cf.data.utils.new_axis_identifier(['dim1'])
    'dim2'
    >>> cf.data.utils.new_axis_identifier(['dim1', 'dim0'])
    'dim2'
    >>> cf.data.utils.new_axis_identifier(['dim3', 'dim4'])
    'dim2'
    >>> cf.data.utils.new_axis_identifier(['dim2', 'dim0'])
    'dim3'
    >>> cf.data.utils.new_axis_identifier(['dim3', 'dim4', 'dim0'])
    'dim5'
    >>> cf.data.utils.new_axis_identifier(basename='axis')
    'axis0'
    >>> cf.data.utils.new_axis_identifier(basename='axis')
    'axis0'
    >>> cf.data.utils.new_axis_identifier(['dim0'], basename='axis')
    'axis1'
    >>> cf.data.utils.new_axis_identifier(['dim0', 'dim1'], basename='axis')
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

    .. versionadded:: 3.14.0

    .. seealso:: `chunk_shapes`

    :Parameters:

        chunks: `tuple`
            The chunk sizes along each dimension, as output by
            `dask.array.Array.chunks`.

    **Examples**

    >>> chunks = ((1, 2), (9,), (44, 55, 66))
    >>> for position in cf.data.utils.chunk_positions(chunks):
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

    .. versionadded:: 3.14.0

    .. seealso:: `chunk_positions`

    :Parameters:

        chunks: `tuple`
            The chunk sizes along each dimension, as output by
            `dask.array.Array.chunks`.

    **Examples**

    >>> chunks = ((1, 2), (9,), (4, 5, 6))
    >>> for shape in cf.data.utils.chunk_shapes(chunks):
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


def scalar_masked_array(dtype=float):
    """Return a scalar masked array.

     .. versionadded:: 3.14.0

     :Parmaeters:

         dtype: data-type, optional
             Desired output data-type for the array, e.g,
             `numpy.int8`. Default is `numpy.float64`.

     :Returns:

         `np.ma.core.MaskedArray`
             The scalar masked array.

     **Examples**

     >>> cf.data.utils.scalar_masked_array()
     masked_array(data=--,
                  mask=True,
            fill_value=1e+20,
                 dtype=float64)
     >>> cf.data.utils.scalar_masked_array(dtype('int32'))
     masked_array(data=--,
                  mask=True,
            fill_value=999999,
                 dtype=int32)
     >>> cf.data.utils.scalar_masked_array('U45')
     masked_array(data=--,
                  mask=True,
            fill_value='N/A',
                dtype='<U45')
    >>> cf.data.utils.scalar_masked_array(bool)
    masked_array(data=--,
                 mask=True,
            fill_value=True,
                dtype=bool)

    """
    a = np.ma.empty((), dtype=dtype)
    a.mask = True
    return a


def conform_units(value, units, message=None):
    """Conform units.

    If *value* has units defined by its `Units` attribute then

    * If the value units are equal to *units* then *value* is returned
      unchanged;

    * If the value units are equivalent to *units* then a copy of
      *value* converted to *units* is returned;

    * If the value units are not equivalent to *units* then an
      exception is raised.

    In all other cases *value* is returned unchanged.

    .. versionadded:: 3.14.0

    :Parameters:

        value:
            The value whose units are to be conformed to *units*.

        units: `Units`
            The units to conform to.

        message: `str`, optional
            If the value units are not equivalent to *units* then use
            this message when the exception is raised. By default a
            message that is independent of the calling context is
            used.

    **Examples**

    >>> cf.data.utils.conform_units(1, cf.Units('m'))
    1
    >>> cf.data.utils.conform_units([1, 2, 3], cf.Units('m'))
    [1, 2, 3]
    >>> import numpy as np
    >>> cf.data.utils.conform_units(np.array([1, 2, 3]), cf.Units('m'))
    array([1, 2, 3])
    >>> cf.data.utils.conform_units('string', cf.Units('m'))
    'string'
    >>> d = cf.Data([1, 2] , 'm')
    >>> cf.data.utils.conform_units(d, cf.Units('m'))
    <CF Data(2): [1, 2] m>
    >>> d = cf.Data([1, 2] , 'km')
    >>> cf.data.utils.conform_units(d, cf.Units('m'))
    <CF Data(2): [1000.0, 2000.0] m>
    >>> cf.data.utils.conform_units(d, cf.Units('s'))
    Traceback (most recent call last):
        ...
    ValueError: Units <Units: km> are incompatible with units <Units: s>
    >>> cf.data.utils.conform_units(d, cf.Units('s'), message='My message')
    Traceback (most recent call last):
        ...
    ValueError: My message

    """
    value_units = getattr(value, "Units", None)
    if value_units is None:
        return value

    if value_units.equivalent(units):
        if value_units != units:
            value = value.copy()
            value.Units = units
    elif value_units and units:
        if message is None:
            message = (
                f"Units {value_units!r} are incompatible with units {units!r}"
            )

        raise ValueError(message)

    return value


def YMDhms(d, attr):
    """Return a date-time component of the data.

    Only applicable for data with reference time units. The returned
    `Data` will have the same mask hardness as the original array.

    .. versionadded:: 3.14.0

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
    >>> cf.data.utils.YMDhms(d, 'year').array
    >>> array([1999, 2000, 2000])

    """
    units = d.Units
    if not units.isreftime:
        raise ValueError(f"Can't get {attr}s from data with {units!r}")

    d = d._asdatetime()
    dx = d.to_dask_array()
    dx = dx.map_blocks(partial(cf_YMDhms, attr=attr), dtype=int)
    d._set_dask(dx)
    d.override_units(Units(None), inplace=True)
    return d


def where_broadcastable(data, x, name):
    """Check broadcastability for `cf.Data.where` assignments.

    Raises an exception unless the *data* and *x* parameters are
    broadcastable across each other, such that the size of the result
    is identical to the size of *data*. Leading size 1 dimensions of
    *x* are ignored, thereby also ensuring that the shape of the
    result is identical to the shape of *data*.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.where`

    :Parameters:

        data, x: `Data`
            The arrays to compare.

        name: `str`
            A name for *x* that is used in exception error messages.

    :Returns:

        `Data`
             The input parameter *x*, or a modified copy without
             leading size 1 dimensions. If *x* can not be acceptably
             broadcast to *data* then a `ValueError` is raised.

    """
    ndim_x = x.ndim
    if not ndim_x:
        return x

    error = 0

    shape_x = x.shape
    shape_data = data.shape

    shape_x0 = shape_x
    ndim_difference = ndim_x - data.ndim

    if ndim_difference > 0:
        if shape_x[:ndim_difference] == (1,) * ndim_difference:
            # Remove leading ize 1 dimensions
            x = x.reshape(shape_x[ndim_difference:])
            shape_x = x.shape
        else:
            error += 1

    for n, m in zip(shape_x[::-1], shape_data[::-1]):
        if n != m and m > 1 and n > 1:
            raise ValueError(
                f"where: {name!r} parameter with shape {shape_x0} can not "
                f"be broadcast across data with shape {shape_data}"
            )

        if m == 1 and n > 1:
            error += 1

    if error:
        raise ValueError(
            f"where: {name!r} parameter with shape {shape_x0} can not "
            f"be broadcast across data with shape {shape_data} when the "
            "result will have a different shape to the data"
        )

    return x
