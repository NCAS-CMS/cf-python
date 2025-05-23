"""General functions useful for `Data` functionality."""

from functools import partial, reduce
from operator import mul

import numpy as np

from ..cfdatetime import canonical_calendar, default_calendar
from ..units import Units
from .dask_utils import cf_YMDhms

_units_None = Units(None)


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

    :Returns:

            The *value* with conformed units.

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
    if value_units is None or value_units == units:
        return value

    if value_units.equivalent(units):
        value = value.copy()
        value.Units = units
        return value

    if value_units and units:
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


def collapse(
    func,
    d,
    axis=None,
    weights=None,
    keepdims=True,
    mtol=1,
    ddof=None,
    split_every=None,
):
    """Collapse data in-place using a given funcion.

     .. versionadded:: 3.14.0

     .. seealso:: `parse_weights`

    :Parameters:

        func: callable
            The function that collapses the underlying `dask` array of
            *d*. Must have the minimum signature (parameters and
            default values) ``func(dx, axis=None, keepdims=False,
            mtol=1, split_every=None)`` (optionally including
            ``weights=None`` or ``ddof=None``), where ``dx`` is a the
            dask array contained in *d*.

        d: `Data`
            The data to be collapsed.

        axis: (sequence of) int, optional
            The axes to be collapsed. By default all axes are
            collapsed, resulting in output with size 1. Each axis is
            identified by its integer position. If *axes* is an empty
            sequence then the collapse is applied to each scalar
            element and the reuslt has the same shape as the input
            data.

        weights: data_like, `dict`, or `None`, optional
            Weights associated with values of the data. By default
            *weights* is `None`, meaning that all non-missing elements
            of the data have a weight of 1 and all missing elements
            have a weight of 0.

            If *weights* is a data_like object then it must be
            broadcastable to the array.

            If *weights* is a dictionary then each key specifies axes
            of the data (an `int` or `tuple` of `int`), with a
            corresponding value of data_like weights for those
            axes. The dimensions of a weights value must correspond to
            its key axes in the same order. Not all of the axes need
            weights assigned to them. The weights that will be used
            will be an outer product of the dictionary's values.

            However they are specified, the weights are internally
            broadcast to the shape of the data, and those weights that
            are missing data, or that correspond to the missing
            elements of the data, are assigned a weight of 0.

            For collapse functions that do not have a ``weights``
            parameter, *weights* must be `None`.

        keepdims: `bool`, optional
            By default, the axes which are collapsed are left in the
            result as dimensions with size one, so that the result
            will broadcast correctly against the input array. If set
            to False then collapsed axes are removed from the data.

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. It is defined as a fraction (between
            0 and 1 inclusive) of the contributing input data
            values. A missing value in the output array occurs
            whenever more than ``100*mtol%`` of its contributing input
            array elements are missing data.

            The default of *mtol* is 1, meaning that a missing value
            in the output array occurs whenever all of its
            contributing input array elements are missing data.

            Note that for non-zero values of *mtol*, different
            collapsed elements may have different sample sizes,
            depending on the distribution of missing data in the input
            data.

        ddof: number, optional
            The delta degrees of freedom, a non-negative number. The
            number of degrees of freedom used in the calculation is
            ``N-ddof`` where ``N`` is the number of non-missing
            elements. A value of 1 applies Bessel's correction. If the
            calculation is weighted then *ddof* can only be 0 or 1.

            For collapse functions for which delta degrees of freedom
            is not applicable (such as `max`), *ddof* must be `None`.

        split_every: `int` or `dict`, optional
            Determines the depth of the recursive aggregation. See
            `dask.array.reduction` for details.

    :Returns:

        (`Data`, formatted weights)
            The collapsed data and the output of ``parse_weights(d,
            weights, axis)``.

    """
    original_size = d.size
    if axis is None:
        axis = range(d.ndim)
    else:
        axis = d._parse_axes(axis)

    kwargs = {
        "axis": tuple(axis),
        "keepdims": keepdims,
        "split_every": split_every,
        "mtol": mtol,
    }

    weights = parse_weights(d, weights, axis)
    if weights is not None:
        kwargs["weights"] = weights

    if ddof is not None:
        kwargs["ddof"] = ddof

    # The applicable chunk function will have its own call to
    # 'cfdm_to_memory', so we can set '_force_to_memory=False'. Also,
    # setting _force_to_memory=False will ensure that any active
    # storage operations are not compromised.
    dx = d.to_dask_array(_force_to_memory=False)
    dx = func(dx, **kwargs)
    d._set_dask(dx)

    if not keepdims:
        # Remove collapsed axis names
        d._axes = [a for i, a in enumerate(d._axes) if i not in axis]

    if d.size != original_size:
        # Remove the out-dated dataset chunking strategy
        d.nc_clear_dataset_chunksizes()

    return d, weights


def parse_weights(d, weights, axis=None):
    """Parse the weights input to `collapse`.

     .. versionadded:: 3.14.0

     .. seealso:: `collapse`

    :Parameters:

        d: `Data`
            The data to be collapsed.

        weights: data_like or `dict`
            See `collapse` for details.

        axis: (sequence of) `int`, optional
            See `collapse` for details.

    :Returns:

        `Data` or `None`
            * If *weights* is a data_like object then they are
              returned unchanged as a `Data` object. It is up to the
              downstream functions to check if the weights can be
              broadcast to the data.

            * If *weights* is a dictionary then the dictionary
              values', i.e. the weights components, outer product is
              returned in `Data` object that is broadcastable to the
              data.

              If the dictionary is empty, or none of the axes defined
              by the keys correspond to collapse axes defined by
              *axis*, then then the collapse is unweighted and `None`
              is returned.

            Note that, in all cases, the returned weights are *not*
            modified to account for missing values in the data.

    **Examples**

    >>> d = cf.Data(np.arange(12)).reshape(4, 3)

    >>> cf.data.utils.parse_weights(d, [1, 2, 1], (0, 1))
    <CF Data(3): [1, 2, 1]>

    >>> cf.data.utils.parse_weights(d, [[1, 2, 1]], (0, 1))
    <CF Data(1, 3): [[1, 2, 1]]>

    >>> cf.data.utils.parse_weights(d, {1: [1, 2, 1]}, (0, 1))
    <CF Data(1, 3): [[1, 2, 1]]>

    >>> print(
    ...     cf.data.utils.parse_weights(
    ...         d, {0: [1, 2, 3, 4], 1: [1, 2, 1]}, (0, 1)
    ...     )
    ... )
    [[1 2 1]
     [2 4 2]
     [3 6 3]
     [4 8 4]]

    >>> print(cf.data.utils.parse_weights(d, {}, (0, 1)))
    None

    >>> print(cf.data.utils.parse_weights(d, {1: [1, 2, 1]}, 0))
    None

    """
    if weights is None:
        # No weights
        return

    if not isinstance(weights, dict):
        # Weights is data_like. Don't check broadcastability to d,
        # leave that to whatever uses the weights.
        return type(d).asdata(weights)

    if not weights:
        # No weights (empty dictionary)
        return

    if axis is None:
        axis = tuple(range(d.ndim))
    else:
        axis = d._parse_axes(axis)

    weights = weights.copy()
    weights_axes = set()
    for key, value in tuple(weights.items()):
        del weights[key]
        key = d._parse_axes(key)
        if weights_axes.intersection(key):
            raise ValueError("Duplicate weights axis")

        weights[tuple(key)] = value
        weights_axes.update(key)

    if not weights_axes.intersection(axis):
        # No weights span collapse axes
        return

    # For each component, add missing dimensions as size 1.
    w = []
    shape = d.shape
    axes = d._axes
    Data = type(d)
    for key, value in weights.items():
        value = Data.asdata(value)

        # Make sure axes are in ascending order
        if key != tuple(sorted(key)):
            key1 = [axes[i] for i in key]
            new_order = [key1.index(axis) for axis in axes if axis in key1]
            value = value.transpose(new_order)

        new_shape = [n if i in key else 1 for i, n in enumerate(shape)]
        w.append(value.reshape(new_shape))

    # Return the product of the weights components, which will be
    # broadcastable to d
    return reduce(mul, w)
