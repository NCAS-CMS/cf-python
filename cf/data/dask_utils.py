"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

from functools import partial

import dask.array as da
import numpy as np
from dask.core import flatten
from scipy.ndimage import convolve1d

from ..cfdatetime import dt, dt2rt, rt2dt
from ..functions import atol as cf_atol
from ..functions import rtol as cf_rtol
from ..units import Units


def _da_ma_allclose(x, y, masked_equal=True, rtol=None, atol=None):
    """An effective dask.array.ma.allclose method.

    True if two dask arrays are element-wise equal within a tolerance.

    Equivalent to allclose except that masked values are treated as
    equal (default) or unequal, depending on the masked_equal
    argument.

    Define an effective da.ma.allclose method here because one is
    currently missing in the Dask codebase.

    Note that all default arguments are the same as those provided to
    the corresponding NumPy method (see the `numpy.ma.allclose` API
    reference).

    .. versionadded:: 3.14.0

    :Parameters:

        x: a dask array to compare with y

        y: a dask array to compare with x

        masked_equal: `bool`, optional
            Whether masked values in a and b are considered equal
            (True) or not (False). They are considered equal by
            default.

        {{rtol: number, optional}}

        {{atol: number, optional}}

    :Returns:

        `bool`
            A Boolean value indicating whether or not the two dask
            arrays are element-wise equal to the given *rtol* and
            *atol* tolerance.

    """
    # TODODASK: put in a PR to Dask to request to add as genuine method.

    if rtol is None:
        rtol = cf_rtol()
    if atol is None:
        atol = cf_atol()

    # Must pass rtol=rtol, atol=atol in as kwargs to allclose, rather than it
    # using those in local scope from the outer function arguments, because
    # Dask's internal algorithms require these to be set as parameters.
    def allclose(a_blocks, b_blocks, rtol=rtol, atol=atol):
        """Run `ma.allclose` across multiple blocks over two arrays."""
        result = True
        # Handle scalars, including 0-d arrays, for which a_blocks and
        # b_blocks will have the corresponding type and hence not be iterable.
        # With this approach, we avoid inspecting sizes or lengths, and for
        # the 0-d array blocks the following iteration can be used unchanged
        # and will only execute once with block sizes as desired of:
        # (np.array(<int size>),)[0] = array(<int size>). Note
        # can't check against more general case of collections.abc.Iterable
        # because a 0-d array is also iterable, but in practice always a list.
        if not isinstance(a_blocks, list):
            a_blocks = (a_blocks,)
        if not isinstance(b_blocks, list):
            b_blocks = (b_blocks,)

        # Note: If a_blocks or b_blocks has more than one chunk in
        #       more than one dimension they will comprise a nested
        #       sequence of sequences, that needs to be flattened so
        #       that we can safely iterate through the actual numpy
        #       array elements.

        for a, b in zip(flatten(a_blocks), flatten(b_blocks)):
            result &= np.ma.allclose(
                a, b, masked_equal=masked_equal, rtol=rtol, atol=atol
            )

        return result

    axes = tuple(range(x.ndim))
    return da.blockwise(
        allclose, "", x, axes, y, axes, dtype=bool, rtol=rtol, atol=atol
    )


def cf_contains(a, value):
    """Whether or not an array contains a value.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.__contains__`

    :Parameters:

        a: `numpy.ndarray`
            The array.

        value: array_like
            The value.

    :Returns:

        `numpy.ndarray`
            A size 1 Boolean array, with the same number of dimensions
            as *a*, that indicates whether or not *a* contains the
            value.

    """
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


def cf_harden_mask(a):
    """Harden the mask of a masked `numpy` array.

    Has no effect if the array is not a masked array.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.harden_mask`

    :Parameters:

        a: `numpy.ndarray`
            The array to have a hardened mask.

    :Returns:

        `numpy.ndarray`
            The array with hardened mask.

    """
    if np.ma.isMA(a):
        try:
            a.harden_mask()
        except AttributeError:
            # Trap cases when the input array is not a numpy array
            # (e.g. it might be numpy.ma.masked).
            pass

    return a


def cf_percentile(a, q, axis, method, keepdims=False, mtol=1):
    """Compute percentiles of the data along the specified axes.

    See `cf.Data.percentile` for further details.

    .. note:: This function correctly sets the mask hardness of the
              output array.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.percentile`

    :Parameters:

        a: `numpy.ndarray`
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


def cf_soften_mask(a):
    """Soften the mask of a masked `numpy` array.

    Has no effect if the array is not a masked array.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.soften_mask`

    :Parameters:

        a: `numpy.ndarray`
            The array to have a softened mask.

    :Returns:

        `numpy.ndarray`
            The array with softened mask.

    """
    if np.ma.isMA(a):
        try:
            a.soften_mask()
        except AttributeError:
            # Trap cases when the input array is not a numpy array
            # (e.g. it might be numpy.ma.masked).
            pass

    return a


def cf_where(array, condition, x, y, hardmask):
    """Set elements of *array* from *x* or *y* depending on *condition*.

    The input *array* is not changed in-place.

    See `where` for details on the expected functionality.

    .. note:: This function correctly sets the mask hardness of the
              output array.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.where`

    :Parameters:

        array: numpy.ndarray
            The array to be assigned to.

        condition: numpy.ndarray
            Where False or masked, assign from *y*, otherwise assign
            from *x*.

        x: numpy.ndarray or `None`
            *x* and *y* must not both be `None`.

        y: numpy.ndarray or `None`
            *x* and *y* must not both be `None`.

        hardmask: `bool`
           Set the mask hardness for a returned masked array. If True
           then a returned masked array will have a hardened mask, and
           the mask of the input *array* (if there is one) will be
           applied to the returned array, in addition to any masked
           elements arising from assignments from *x* or *y*.

    :Returns:

        `numpy.ndarray`
            A copy of the input *array* with elements from *y* where
            *condition* is False or masked, and elements from *x*
            elsewhere.

    """
    mask = None

    if np.ma.isMA(array):
        # Do a masked where
        where = np.ma.where
        if hardmask:
            mask = array.mask
    elif np.ma.isMA(x) or np.ma.isMA(y):
        # Do a masked where
        where = np.ma.where
    else:
        # Do a non-masked where
        where = np.where
        hardmask = False

    condition_is_masked = np.ma.isMA(condition)
    if condition_is_masked:
        condition = condition.astype(bool)

    if x is not None:
        # Assign values from x
        if condition_is_masked:
            # Replace masked elements of condition with False, so that
            # masked locations are assigned from array
            c = condition.filled(False)
        else:
            c = condition

        array = where(c, x, array)

    if y is not None:
        # Assign values from y
        if condition_is_masked:
            # Replace masked elements of condition with True, so that
            # masked locations are assigned from array
            c = condition.filled(True)
        else:
            c = condition

        array = where(c, array, y)

    if hardmask:
        if mask is not None and mask.any():
            # Apply the mask from the input array to the result
            array.mask |= mask

        array.harden_mask()

    return array


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
    return Units.conform(
        a, from_units=from_units, to_units=to_units, inplace=False
    )
