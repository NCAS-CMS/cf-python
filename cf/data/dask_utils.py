"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

from functools import partial

import numpy as np
from cfdm.data.dask_utils import cfdm_to_memory
from scipy.ndimage import convolve1d

from ..cfdatetime import dt, dt2rt, rt2dt
from ..units import Units


def cf_contains(a, value):
    """Whether or not an array contains a value.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.__contains__`

    :Parameters:

        a: array_like
            The array.

        value: array_like
            The value.

    :Returns:

        `numpy.ndarray`
            A size 1 Boolean array, with the same number of dimensions
            as *a*, that indicates whether or not *a* contains the
            value.

    """
    a = cfdm_to_memory(a)
    value = cfdm_to_memory(value)
    return np.array(value in a).reshape((1,) * a.ndim)


def cf_convolve1d(a, window=None, axis=-1, origin=0):
    """Calculate a 1-d convolution along the given axis.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.convolution_filter`

    :Parameters:

        a: `numpy.ndarray`
            The float array to be filtered.

        window: 1-d sequence of numbers
            The window of weights to use for the filter.

        axis: `int`, optional
            The axis of input along which to calculate. Default is -1.

        origin: `int`, optional
            Controls the placement of the filter on the input array’s
            pixels. A value of 0 (the default) centers the filter over
            the pixel, with positive values shifting the filter to the
            left, and negative ones to the right.

    :Returns:

        `numpy.ndarray`
            Convolved float array with same shape as input.

    """
    a = cfdm_to_memory(a)

    # Cast to float to ensure that NaNs can be stored
    if a.dtype != float:
        a = a.astype(float, copy=False)

    masked = np.ma.is_masked(a)
    if masked:
        # convolve1d does not deal with masked arrays, so uses NaNs
        # instead.
        a = a.filled(np.nan)

    c = convolve1d(
        a, window, axis=axis, mode="constant", cval=0.0, origin=origin
    )

    if masked or np.isnan(c).any():
        with np.errstate(invalid="ignore"):
            c = np.ma.masked_invalid(c)

    return c


def cf_percentile(a, q, axis, method, keepdims=False, mtol=1):
    """Compute percentiles of the data along the specified axes.

    See `cf.Data.percentile` for further details.

    .. note:: This function correctly sets the mask hardness of the
              output array.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.percentile`

    :Parameters:

        a: array_like
            Input array.

        q: `numpy.ndarray`
            Percentile or sequence of percentiles to compute, which
            must be between 0 and 100 inclusive.

        axis: `tuple` of `int`
            Axes along which the percentiles are computed.

        method: `str`
            Specifies the interpolation method to use when the desired
            percentile lies between two data points ``i < j``.

        keepdims: `bool`, optional
            If this is set to True, the axes which are reduced are
            left in the result as dimensions with size one. With this
            option, the result will broadcast correctly against the
            original array *a*.

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. It is defined as a fraction (between
            0 and 1 inclusive) of the contributing input data values.

            The default of *mtol* is 1, meaning that a missing datum
            in the output array occurs whenever all of its
            contributing input array elements are missing data.

            For other values, a missing datum in the output array
            occurs whenever more than ``100*mtol%`` of its
            contributing input array elements are missing data.

            Note that for non-zero values of *mtol*, different
            collapsed elements may have different sample sizes,
            depending on the distribution of missing data in the input
            data.

    :Returns:

        `numpy.ndarray`

    """
    from math import prod

    a = cfdm_to_memory(a)

    if np.ma.isMA(a) and not np.ma.is_masked(a):
        # Masked array with no masked elements
        a = a.data

    if np.ma.isMA(a):
        # ------------------------------------------------------------
        # Input array is masked: Replace missing values with NaNs and
        # remask later.
        # ------------------------------------------------------------
        if a.dtype != float:
            # Can't assign NaNs to integer arrays
            a = a.astype(float, copy=True)

        mask = None
        if mtol < 1:
            # Count the number of missing values that contribute to
            # each output percentile value and make a corresponding
            # mask
            full_size = prod(
                [size for i, size in enumerate(a.shape) if i in axis]
            )
            n_missing = full_size - np.ma.count(
                a, axis=axis, keepdims=keepdims
            )
            if n_missing.any():
                mask = np.where(n_missing > mtol * full_size, True, False)
                if q.ndim:
                    mask = np.expand_dims(mask, 0)

        a = np.ma.filled(a, np.nan)

        with np.testing.suppress_warnings() as sup:
            sup.filter(
                category=RuntimeWarning,
                message=".*All-NaN slice encountered.*",
            )
            p = np.nanpercentile(
                a,
                q,
                axis=axis,
                method=method,
                keepdims=keepdims,
                overwrite_input=True,
            )

        # Update the mask for NaN points
        nan_mask = np.isnan(p)
        if nan_mask.any():
            if mask is None:
                mask = nan_mask
            else:
                mask = np.ma.where(nan_mask, True, mask)

        # Mask any NaNs and elements below the mtol threshold
        if mask is not None:
            p = np.ma.where(mask, np.ma.masked, p)

    else:
        # ------------------------------------------------------------
        # Input array is not masked
        # ------------------------------------------------------------
        p = np.percentile(
            a,
            q,
            axis=axis,
            method=method,
            keepdims=keepdims,
            overwrite_input=False,
        )

    return p


def _getattr(x, attr):
    return getattr(x, attr, False)


_array_getattr = np.vectorize(_getattr, excluded="attr")


def cf_YMDhms(a, attr):
    """Return a date-time component from an array of date-time objects.

    Only applicable for data with reference time units. The returned
    array will have the same mask hardness as the original array.

    .. versionadded:: 3.14.0

    .. seealso:: `~cf.Data.year`, ~cf.Data.month`, `~cf.Data.day`,
                 `~cf.Data.hour`, `~cf.Data.minute`, `~cf.Data.second`

    :Parameters:

        a: `numpy.ndarray`
            The array from which to extract date-time component.

        attr: `str`
            The name of the date-time component, one of ``'year'``,
            ``'month'``, ``'day'``, ``'hour'``, ``'minute'``,
            ``'second'``.

    :Returns:

        `numpy.ndarray`
            The date-time component.

    **Examples**

    >>> import numpy as np
    >>> a = np.array([
    ...  cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False)
    ...  cftime.DatetimeGregorian(2000, 1, 2, 0, 0, 0, 0, has_year_zero=False)
    ... ])
    >>> cf_YMDmhs(a, 'day')
    array([1, 2])

    """
    a = cfdm_to_memory(a)
    return _array_getattr(a, attr=attr)


def cf_rt2dt(a, units):
    """Convert an array of reference times to date-time objects.

    .. versionadded:: 3.14.0

    .. seealso:: `cf._dt2rt`, `cf.Data._asdatetime`

    :Parameters:

        a: `numpy.ndarray`
            An array of numeric reference times.

        units: `Units`
            The units for the reference times


    :Returns:

        `numpy.ndarray`
            A array containing date-time objects.

    **Examples**

    >>> import numpy as np
    >>> print(cf_rt2dt(np.array([0, 1]), cf.Units('days since 2000-01-01')))
    [cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False)
     cftime.DatetimeGregorian(2000, 1, 2, 0, 0, 0, 0, has_year_zero=False)]

    """
    a = cfdm_to_memory(a)

    if not units.iscalendartime:
        return rt2dt(a, units_in=units)

    # Calendar month/year units
    from ..timeduration import TimeDuration

    def _convert(x, units, reftime):
        t = TimeDuration(x, units=units)
        if x > 0:
            return t.interval(reftime, end=False)[1]
        else:
            return t.interval(reftime, end=True)[0]

    return np.vectorize(
        partial(
            _convert,
            units=units._units_since_reftime,
            reftime=dt(units.reftime, calendar=units._calendar),
        ),
        otypes=[object],
    )(a)


def cf_dt2rt(a, units):
    """Convert an array of date-time objects to reference times.

    .. versionadded:: 3.14.0

    .. seealso:: `cf._rt2dt`, `cf.Data._asreftime`

    :Parameters:

        a: `numpy.ndarray`
            An array of date-time objects.

        units: `Units`
            The units for the reference times

    :Returns:

        `numpy.ndarray`
            An array containing numeric reference times

    **Examples**

    >>> import numpy as np
    >>> a = np.array([
    ...  cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False)
    ...  cftime.DatetimeGregorian(2000, 1, 2, 0, 0, 0, 0, has_year_zero=False)
    ... ])
    >>> print(cf_dt2rt(a, cf.Units('days since 1999-01-01')))
    [365 366]

    """
    a = cfdm_to_memory(a)
    return dt2rt(a, units_out=units, units_in=None)


def cf_units(a, from_units, to_units):
    """Convert array values to have different equivalent units.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.Units`

    :Parameters:

        a: `numpy.ndarray`
            The array.

        from_units: `Units`
            The existing units of the array.

        to_units: `Units`
            The units that the array should be converted to. Must be
            equivalent to *from_units*.

    :Returns:

        `numpy.ndarray`
            An array containing values in the new units. In order to
            represent the new units, the returned data type may be
            different from that of the input array. For instance, if
            *a* has an integer data type, *from_units* are kilometres,
            and *to_units* are ``'miles'`` then the returned array
            will have a float data type.

    **Examples**

    >>> import numpy as np
    >>> a = np.array([1, 2])
    >>> print(cf.data.dask_utils.cf_units(a, cf.Units('km'), cf.Units('m')))
    [1000. 2000.]

    """
    a = cfdm_to_memory(a)
    return Units.conform(
        a, from_units=from_units, to_units=to_units, inplace=False
    )


def cf_is_masked(a):
    """Determine whether an array has masked values.

    .. versionadded:: 3.16.3

    :Parameters:

        a: array_like
            The array.

    :Returns:

        `numpy.ndarray`
            A size 1 Boolean array with the same number of dimensions
            as *a*, for which `True` indicates that there are masked
            values.

    """
    a = cfdm_to_memory(a)
    out = np.ma.is_masked(a)
    return np.array(out).reshape((1,) * a.ndim)


def cf_filled(a, fill_value=None):
    """Replace masked elements with a fill value.

    .. versionadded:: 3.16.3

    :Parameters:

        a: array_like
            The array.

        fill_value: scalar
            The fill value.

    :Returns:

        `numpy.ndarray`
            The filled array.

    **Examples**

    >>> a = np.array([[1, 2, 3]])
    >>> print(cf.data.dask_utils.cf_filled(a, -999))
    [[1 2 3]]
    >>> a = np.ma.array([[1, 2, 3]], mask=[[True, False, False]])
    >>> print(cf.data.dask_utils.cf_filled(a, -999))
    [[-999    2    3]]

    """
    a = cfdm_to_memory(a)
    return np.ma.filled(a, fill_value=fill_value)
