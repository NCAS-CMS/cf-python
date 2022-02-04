"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""
from functools import reduce
from operator import mul

import dask.array as da
import numpy as np

from ..functions import atol as cf_atol
from ..functions import rtol as cf_rtol


def _da_ma_allclose(x, y, masked_equal=True, rtol=None, atol=None):
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

            masked_equal: `bool`, optional
                Whether masked values in a and b are considered
                equal (True) or not (False). They are considered equal
                by default.

            {{rtol: number, optional}}

            {{atol: number, optional}}

        :Returns:

            Boolean
                A Boolean value indicating whether or not the
                two dask arrays are element-wise equal to
                the given *rtol* and *atol* tolerance.

    """
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

        for a, b in zip(a_blocks, b_blocks):
            result &= np.ma.allclose(
                a,
                b,
                masked_equal=masked_equal,
                rtol=rtol,
                atol=atol,
            )

        return result

    axes = tuple(range(x.ndim))
    return da.blockwise(
        allclose,
        "",
        x,
        axes,
        y,
        axes,
        dtype=bool,
        rtol=rtol,
        atol=atol,
    )


try:
    from scipy.ndimage.filters import convolve1d
except ImportError:
    pass


def cf_convolve1d(a, window=None, axis=-1, origin=0):
    """Calculate a 1-d convolution along the given axis.

    .. versionadded:: TODODASK

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

    .. versionadded:: TODODASK

    .. seealso:: `cf.Data.harden_mask`

    :Parameters:

        a: `numpy.ndarray`
            The array to have a hardened mask.

    :Returns:

        `numpy.ndarray`
            The array with hardened mask.

    """
    if np.ma.isMA(a):
        a.harden_mask()

    return a


def cf_percentile(a, q, axis, interpolation, keepdims=False, mtol=1):
    """Compute the q-th percentile of the data along the specified axis.

    See `cf.Data.percentile` for further details.

    .. note:: This function correctly sets the mask hardness of the
              output array.

    .. versionadded:: TODODASK

    .. seealso:: `cf.Data.percentile`

    :Parameters:

        a: `numpy.ndarray`
            Input array,

        q: `numpy.ndarray`
            Percentile or sequence of percentiles to compute, which
            must be between 0 and 100 inclusive.

        axis: `tuple` of `int`
            Axes along which the percentiles are computed.

        interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Specifies the interpolation method to use when the desired
            percentile lies between two data points ``i < j``:

        keepdims: `bool`, optional
            If this is set to True, the axes which are reduced are
            left in the result as dimensions with size one. With this
            option, the result will broadcast correctly against the
            original array *a*.

        mtol: number, optional
            Set the fraction (between 0 and 1 inclusive) of input data
            elements which is allowed to contain missing data when
            contributing to an individual output data element. The
            default is 1, meaning that a missing datum in the output
            array occurs when all of its contributing input array
            elements are missing data. A value of 0 means that a
            missing datum in the output array occurs whenever any of
            its contributing input array elements are missing data.

    :Returns:

        `numpy.ndarray`

    """
    if not len(a):
        return None

    if np.ma.is_masked(a):
        # ------------------------------------------------------------
        # Input array is masked
        # ------------------------------------------------------------
        if a.dtype != float:
            # Can't assign NaNs to integer arrays
            a = a.astype(float, copy=True)

        if mtol < 1:
            # Count the number of missing values that contribute to
            # each output percentile value
            n_missing = np.ma.count(a, axis=axis)
            if q.ndim:
                n_missing = np.expand_dims(n_missing, 0)

        a = np.ma.filled(a, np.nan)

        with np.testing.suppress_warnings() as sup:
            sup.filter(
                category=RuntimeWarning, message=".*All-NaN slice encountered"
            )
            p = np.nanpercentile(
                a,
                q,
                axis=axis,
                interpolation=interpolation,
                keepdims=keepdims,
                overwrite_input=True,
            )

        # Replace NaNs with missing data
        mask = np.isnan(p)
        if not mask.any():
            mask = None

        # Mask any percentiles which are the result of too few input
        # values
        if mtol < 1:
            full_size = reduce(
                mul, [size for i, size in enumerate(a.shape) if i in axis], 1
            )
            mtol_mask = np.where(n_missing < mtol * full_size, 1, 0)

            if mask is None:
                mask = mtol_mask
            else:
                mask |= mtol_mask

        if mask is not None:
            print(9999)
            p = np.ma.masked_where(mask, p, copy=False)

    else:
        # ------------------------------------------------------------
        # Input array is not masked
        # ------------------------------------------------------------
        p = np.percentile(
            a,
            q,
            axis=axis,
            interpolation=interpolation,
            keepdims=keepdims,
            overwrite_input=False,
        )

    return p


def cf_soften_mask(a):
    """Soften the mask of a masked `numpy` array.

    Has no effect if the array is not a masked array.

    .. versionadded:: TODODASK

    .. seealso:: `cf.Data.soften_mask`

    :Parameters:

        a: `numpy.ndarray`
            The array to have a softened mask.

    :Returns:

        `numpy.ndarray`
            The array with softened mask.

    """
    if np.ma.isMA(a):
        a.soften_mask()

    return a


def cf_where(array, condition, x, y, hardmask):
    """Set elements of *array* from *x* or *y* depending on *condition*.

    The input *array* is not changed in-place.

    See `where` for details on the expected functionality.

    .. note:: This function correctly sets the mask hardness of the
              output array.

    .. versionadded:: TODODASK

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
