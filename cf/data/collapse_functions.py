from functools import partial as functools_partial

from numpy import abs         as numpy_abs
from numpy import allclose    as numpy_allclose
from numpy import amax        as numpy_amax
from numpy import amin        as numpy_amin
from numpy import any         as numpy_any
from numpy import array       as numpy_array
from numpy import asanyarray  as numpy_asanyarray
from numpy import average     as numpy_average
from numpy import bool_       as numpy_bool_
from numpy import copy        as numpy_copy
from numpy import empty       as numpy_empty
from numpy import expand_dims as numpy_expand_dims
from numpy import integer     as numpy_integer
from numpy import maximum     as numpy_maximum
from numpy import minimum     as numpy_minimum
from numpy import ndim        as numpy_ndim
from numpy import sum         as numpy_sum
from numpy import where       as numpy_where
from numpy import zeros       as numpy_zeros
import numpy

from numpy.ma import array        as numpy_ma_array
from numpy.ma import average      as numpy_ma_average
from numpy.ma import expand_dims  as numpy_ma_expand_dims
from numpy.ma import isMA         as numpy_ma_isMA
from numpy.ma import masked       as numpy_ma_masked
from numpy.ma import masked_less  as numpy_ma_masked_less
from numpy.ma import masked_where as numpy_ma_masked_where
from numpy.ma import nomask       as numpy_ma_nomask
from numpy.ma import where        as numpy_ma_where

from ..functions import broadcast_array


def asanyarray(*args):
    '''TODO

    :Parameters:

        args: sequence of `numpy.ndarray`

    :Returns:

        out: `tuple`

    '''
    out = []
    for x in args:
        if x is not None and not numpy_ndim(x):
            # Make sure that we have a numpy array (as opposed to, e.g. a
            # numpy.float64)
            out.append(numpy_asanyarray(x))
        else:
            out.append(x)
    # --- End: for

    return out


def psum(x, y):
    '''Add two arrays element-wise.

    If either or both of the arrays are masked then the output array is
    masked only where both input arrays are masked.

    :Parameters:

        x: numpy array-like
           *Might be updated in place*.

        y: numpy array-like
           Will not be updated in place.

    :Returns:

        out: `numpy.ndarray`

    **Examples:**

    >>> c = psum(a, b)

    '''
    if numpy_ma_isMA(x):
        if numpy_ma_isMA(y):
            # x and y are both masked
            x_mask = x.mask
            x = x.filled(0)
            x += y.filled(0)
            x = numpy_ma_array(x, mask=x_mask & y.mask, copy=False)
        else:
            # Only x is masked
            x = x.filled(0)
            x += y
    elif numpy_ma_isMA(y):
        # Only y is masked
        x += y.filled(0)
    else:
        # x and y are both unmasked
        x += y

    return x


def pmax(x, y):
    '''TODO

    :Parameters:

        x: array-like
           May be updated in place and should not be used again.

        y: array-like
           Will not be updated in place.

    :Returns:

        out: `numpy.ndarray`

    '''
    if numpy_ma_isMA(x):
        if numpy_ma_isMA(y):
            # x and y are both masked
            z = numpy_maximum(x, y)
            z = numpy_ma_where(x.mask & ~y.mask, y, z)
            x = numpy_ma_where(y.mask & ~x.mask, x, z)
            if x.mask is numpy_ma_nomask:  # not numpy_any(x.mask):
                x = numpy_array(x)
        else:
            # Only x is masked
            z = numpy_maximum(x, y)
            x = numpy_ma_where(x.mask, y, z)
            if x.mask is numpy_ma_nomask:  # not numpy_any(x.mask):
                x = numpy_array(x)
    elif numpy_ma_isMA(y):
        # Only y is masked
        z = numpy_maximum(x, y)
        x = numpy_ma_where(y.mask, x, z)
        if x.mask is numpy_ma_nomask:  # not numpy_any(x.mask):
            x = numpy_array(x)
    else:
        # x and y are both unmasked
        if not numpy_ndim(x):
            # Make sure that we have a numpy array (as opposed to,
            # e.g. a numpy.float64)
            x = numpy_asanyarray(x)

        numpy_maximum(x, y, out=x)

    return x


def pmin(x, y):
    '''TODO

    :Parameters:

        x: `numpy.ndarray`
           May be updated in place and should not be used again.

        y: `numpy.ndarray`
           Will not be updated in place.

    :Returns:

        out: `numpy.ndarray`

    '''
    if numpy_ma_isMA(x):
        if numpy_ma_isMA(y):
            # x and y are both masked
            z = numpy_minimum(x, y)
            z = numpy_ma_where(x.mask & ~y.mask, y, z)
            x = numpy_ma_where(y.mask & ~x.mask, x, z)
            if x.mask is numpy_ma_nomask:
                x = numpy_array(x)
        else:
            # Only x is masked
            z = numpy_minimum(x, y)
            x = numpy_ma_where(x.mask, y, z)
            if x.mask is numpy_ma_nomask:
                x = numpy_array(x)
    elif numpy_ma_isMA(y):
        # Only y is masked
        z = numpy_minimum(x, y)
        x = numpy_ma_where(y.mask, x, z)
        if x.mask is numpy_ma_nomask:
            x = numpy_array(x)
    else:
        # x and y are both unmasked
        if not numpy_ndim(x):
            # Make sure that we have a numpy array (as opposed to,
            # e.g. a numpy.float64)
            x = numpy_asanyarray(x)

        numpy_minimum(x, y, out=x)

    return x


def mask_where_too_few_values(Nmin, N, x):
    '''Mask elements of N and x where N is strictly less than Nmin.

    :Parameters:

        Nmin: `int`

        N: `numpy.ndarray`

        x: `numpy.ndarray`

    :Returns:

        (`numpy.ndarray`, `numpy.ndarray`)
            A tuple containing *N* and *x*, both masked where *N* is
            strictly less than *Nmin*.

    '''
    if N.min() < Nmin:
        mask = N < Nmin
        N = numpy_ma_array(N, mask=mask, copy=False, shrink=False)
        x = numpy_ma_array(x, mask=mask, copy=False, shrink=True)

    return asanyarray(N, x)


def double_precision(a):
    '''Convert the input array to double precision.

    :Parameters:

        a: `numpy.ndarray`

    :Returns:

        out: `numpy.ndarray`

    '''
    char = a.dtype.char
    if char == 'f':
        newtype = float
    elif char == 'i':
        newtype = int
    else:
        return a

    if numpy_ma_isMA(a):
        return a.astype(newtype)
    else:
        return a.astype(newtype, copy=False)


# --------------------------------------------------------------------
# Maximum
# --------------------------------------------------------------------
def max_f(a, axis=None, masked=False):
    '''Return the maximum of an array, or the maxima of an array along an
    axis.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        masked: `bool`

    :Returns:

        out: 2-tuple of numpy arrays
            The sample sizes and the maxima.

    '''
    N,   = sample_size_f(a, axis=axis, masked=masked)
    amax = numpy_amax(a, axis=axis)

    if not numpy_ndim(amax):
        # Make sure that we have a numpy array (as opposed to, e.g. a
        # numpy.float64)
        amax = numpy_asanyarray(amax)

    return asanyarray(N, amax)


def max_fpartial(out, out1=None, group=False):
    '''TODO
    '''
    N, amax = out

    if out1 is not None:
        N1, amax1 = out1
        N = psum(N, N1)
        amax = pmax(amax, amax1)

    return asanyarray(N, amax)


def max_ffinalise(out, sub_samples=None):
    '''TODO

    :Parameters:

        sub_samples: *optional*
            Ignored.

    '''
    return mask_where_too_few_values(1, *out)


# --------------------------------------------------------------------
# Minimum
# --------------------------------------------------------------------
def min_f(a, axis=None, masked=False):
    '''Return the minimum of an array, or the minima of an array along an
    axis.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        masked: `bool`

    :Returns:

        out: 2-tuple of `numpy.ndarray`
            The sample sizes and the minima.

    '''
    N,   = sample_size_f(a, axis=axis, masked=masked)
    amin = numpy_amin(a, axis=axis)

    return asanyarray(N, amin)


def min_fpartial(out, out1=None, group=False):
    '''TODO
    '''
    N, amin = out

    if out1 is not None:
        N1, amin1 = out1
        N = psum(N, N1)
        amin = pmin(amin, amin1)

    return asanyarray(N, amin)


def min_ffinalise(out, sub_samples=None):
    '''TODO

    :Parameters:

        sub_samples: optional
            Ignored.

    '''
    return mask_where_too_few_values(1, *out)


# --------------------------------------------------------------------
# maximum_absolute_value
# --------------------------------------------------------------------
def max_abs_f(a, axis=None, masked=False):
    '''Return the maximum of the absolute array, or the maxima of the
    absolute array along an axis.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        masked: bool

    :Returns:

        out: 2-tuple of numpy arrays
            The sample sizes and the maxima of the absolute values.

    '''
    return max_f(numpy_abs(a), axis=axis, masked=masked)


max_abs_fpartial = max_fpartial
max_abs_ffinalise = max_ffinalise


# --------------------------------------------------------------------
# minimum_absolute_value
# --------------------------------------------------------------------
def min_abs_f(a, axis=None, masked=False):
    '''Return the minimum of the absolute array, or the minima of the
    absolute array along an axis.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input is
            used.

        masked: `bool`

    :Returns:

        out: 2-tuple of `numpy.ndarray`
            The sample sizes and the minima of the absolute values.

    '''
    return min_f(numpy_abs(a), axis=axis, masked=masked)


min_abs_fpartial = min_fpartial
min_abs_ffinalise = min_ffinalise


# --------------------------------------------------------------------
# mean
# --------------------------------------------------------------------
def mean_f(a, axis=None, weights=None, masked=False):
    '''The weighted average along the specified axes.

    :Parameters:

        a: array-like
            Input array. Not all missing data

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        weights: array-like, optional

        masked: `bool`, optional

    :Returns:

        out: `tuple`
            3-tuple.

    '''
    a = double_precision(a)

    if masked:
        average = numpy_ma_average
    else:
        average = numpy_average

    avg, sw = average(a, axis=axis, weights=weights, returned=True)

    if not numpy_ndim(avg):
        avg = numpy_asanyarray(avg)
        sw = numpy_asanyarray(sw)

    if weights is None:
        N = sw.copy()
    else:
        N, = sample_size_f(a, axis=axis, masked=masked)

    return asanyarray(N, avg, sw)


def mean_fpartial(out, out1=None, group=False):
    '''Return the partial sample size,the partial sum and partial sum of
    the weights.

    :Parameters:

        out: 3-`tuple` of `numpy.ndarray`
            Either an output from a previous call to `mean_fpartial`;
            or, if *out1* is `None`, an output from `mean_f`.

        out1: 3-`tuple` of `numpy.ndarray`, optional
            An output from `mean_f`.

    :Returns:

        out: 3-`tuple` of `numpy.ndarray`

    '''
    N, avg, sw = out

    if out1 is None and not group:
        # This is the first partition to be processed

        # Convert the partition average to a partition sum
        avg *= sw
    else:
        # Combine this partition with existing parital combination
        N1, avg1, sw1 = out1

        # Convert the partition average to a partition sum
        if not group:
            avg1 *= sw1

        N = psum(N, N1)
        avg = psum(avg, avg1)  # Now a partial sum
        sw = psum(sw, sw1)

    return asanyarray(N, avg, sw)


def mean_ffinalise(out, sub_samples=None):
    '''Divide the weighted sum by the sum of weights.

    Also mask out any values derived from a too-small sample size.

    :Parameters:

        out: 3-`tuple` of `numpy.ndarray`
            An output from `mean_fpartial`.

        sub_samples: optional

    :Returns:

        out: 2-`tuple` of `numpy.ndarray`
            The sample size and the mean.

    '''
    N, avg, sw = out

    if sub_samples:
        avg /= sw

    return mask_where_too_few_values(1, N, avg)


# --------------------------------------------------------------------
# mean_absolute_value
# --------------------------------------------------------------------
def mean_abs_f(a, axis=None, weights=None, masked=False):
    '''Return the mean of the absolute array, or the means of the absolute
    array along an axis.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input is
            used.

        masked: `bool`

    :Returns:

        out: 2-tuple of `numpy.ndarray`
            The sample sizes and the means of the absolute values.

    '''
    return mean_f(numpy_abs(a), axis=axis, weights=weights, masked=masked)


mean_abs_fpartial = mean_fpartial
mean_abs_ffinalise = mean_ffinalise


# --------------------------------------------------------------------
# root_mean_square
# --------------------------------------------------------------------
def root_mean_square_f(a, axis=None, weights=None, masked=False):
    '''The RMS along the specified axes.

    :Parameters:

        a: array-like
            Input array. Not all missing data

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        weights: array-like, optional

        masked: `bool`, optional

    :Returns:

        out: `tuple`
            3-tuple.

    '''
    a = double_precision(a)

    return mean_f(a**2, axis=axis, weights=weights, masked=masked)


root_mean_square_fpartial = mean_fpartial


def root_mean_square_ffinalise(out, sub_samples=None):
    '''Divide the weighted sum by the sum of weights and take the square
    root.

    Also mask out any values derived from a too-small sample size.

    :Parameters:

        out: 3-`tuple` of `numpy.ndarray`
            An output from `root_mean_square_fpartial`.

        sub_samples: optional

    :Returns:

        out: 2-`tuple` of `numpy.ndarray`
            The sample size and the RMS.

    '''
    N, avg = mean_ffinalise(out, sub_samples=sub_samples)

    avg **= 0.5

    return asanyarray(N, avg)


# --------------------------------------------------------------------
# Mid range: Average of maximum and minimum
# --------------------------------------------------------------------
def mid_range_f(a, axis=None, masked=False):
    '''Return the minimum and maximum of an array or the minima and maxima
    along an axis.

    ``mid_range_f(a, axis=axis)`` is equivalent to ``(numpy.amin(a,
    axis=axis), numpy.amax(a, axis=axis))``

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        kwargs: ignored

    :Returns:

        out: `tuple`
            The minimum and maximum inside a 2-tuple.

    '''
    N, = sample_size_f(a, axis=axis, masked=masked)
    amin = numpy_amin(a, axis=axis)
    amax = numpy_amax(a, axis=axis)

    if not numpy_ndim(amin):
        # Make sure that we have a numpy array (as opposed to, e.g. a
        # numpy.float64)
        amin = numpy_asanyarray(amin)
        amax = numpy_asanyarray(amax)

    return asanyarray(N, amin, amax)


def mid_range_fpartial(out, out1=None, group=False):
    '''TODO

    '''
    N, amin, amax = out

    if out1 is not None:
        N1, amin1, amax1 = out1

        N = psum(N, N1)
        amin = pmin(amin, amin1)
        amax = pmax(amax, amax1)

    return asanyarray(N, amin, amax)


def mid_range_ffinalise(out, sub_samples=None):
    '''TODO

    :Parameters:

        out: ordered sequence

        amin: `numpy.ndarray`

        amax: `numpy.ndarray`

        sub_samples: *optional*
            Ignored.

    '''
    N, amin, amax = out

    amax = double_precision(amax)

    # Cast bool, unsigned int, and int to float64
    if issubclass(amax.dtype.type, (numpy_integer, numpy_bool_)):
        amax = amax.astype(float)

    amax += amin
    amax *= 0.5

    return mask_where_too_few_values(1, N, amax)


# ---------------------------------------------------------------------
# Range: Absolute difference between maximum and minimum
# ---------------------------------------------------------------------
range_f = mid_range_f
range_fpartial = mid_range_fpartial


def range_ffinalise(out, sub_samples=None):
    '''Absolute difference between maximum and minimum

    :Parameters:

        out: ordered sequence

        sub_samples: optional
            Ignored.

    '''
    N, amin, amax = out

    amax = double_precision(amax)
    amax -= amin

    return mask_where_too_few_values(1, N, amax)


# ---------------------------------------------------------------------
# Sample size
# ---------------------------------------------------------------------
def sample_size_f(a, axis=None, masked=False):
    '''TODO

    :Parameters:

        axis: `int`, optional
            non-negative


    '''
    if masked:
        N = numpy_sum(~a.mask, axis=axis, dtype=float)
        if not numpy_ndim(N):
            N = numpy_asanyarray(N)
    else:
        if axis is None:
            N = numpy_array(a.size, dtype=float)
        else:
            shape = a.shape
            N = numpy_empty(shape[:axis]+shape[axis+1:], dtype=float)
            N[...] = shape[axis]
    # --- End: if

    return asanyarray(N)


def sample_size_fpartial(out, out1=None, group=False):
    '''TODO

    :Parameters:

        out: ordered sequence of one numpy array

    :Returns:

        out: `numpy.ndarray`

    '''
    N, = out
    if out1 is not None:
        N1, = out1
        N = psum(N, N1)

    return asanyarray(N)


def sample_size_ffinalise(out, sub_samples=None):
    '''TODO

    :Parameters:

        out: ordered sequence of one numpy array

        sub_samples: *optional*
            Ignored.

    :Returns:

        out: `tuple`
            A 2-tuple containing *N* twice.

    '''
    N, = out
    return asanyarray(N, N)


# ---------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------
def sum_f(a, axis=None, weights=None, masked=False):
    '''Return the sum of an array or the sum along an axis.

    ``sum_f(a, axis=axis)`` is equivalent to ``(numpy.sum(a,
    axis=axis),)``

    :Parameters:

        a: numpy array-like
            Input array

        weights: numpy array-like, optional
            TODO

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

    :Returns:

        `tuple`
            2-tuple

    '''
    a = double_precision(a)

    N,   = sample_size_f(a, axis=axis, masked=masked)

    if weights is not None:
        # A weights array has been provided
        weights = double_precision(weights)

        if weights.ndim < a.ndim:
            weights = broadcast_array(weights, a.shape)

        a = a * weights

    asum = a.sum(axis=axis)

    if not numpy_ndim(asum):
        asum = numpy_asanyarray(asum)

    return asanyarray(N, asum)


def sum_fpartial(out, out1=None, group=False):
    '''TODO
    '''
    N, asum = out

    if out1 is not None:
        N1, asum1 = out1
        N = psum(N, N1)
        asum = psum(asum, asum1)

    return asanyarray(N, asum)


def sum_ffinalise(out, sub_samples=None):
    '''TODO

    :Parameters:

        sub_samples: *optional*
            Ignored.
    '''
    return mask_where_too_few_values(1, *out)


# ---------------------------------------------------------------------
# sum_of_squares
# ---------------------------------------------------------------------
def sum_of_squares_f(a, axis=None, weights=None, masked=False):
    '''Return the sum of the square of an array or the sum of squares
    along an axis.

    :Parameters:

        a: numpy array-like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

    :Returns:

        `tuple`
            2-tuple

    '''
    a = double_precision(a)
    return sum_f(a**2, axis=axis, weights=weights, masked=masked)


sum_of_squares_fpartial = sum_fpartial
sum_of_squares_ffinalise = sum_ffinalise


# ---------------------------------------------------------------------
# Sum of weights
# ---------------------------------------------------------------------
def sw_f(a, axis=None, masked=False, weights=None, N=None,
         sum_of_squares=False):
    '''TODO

    '''
    if N is None:
        N, = sample_size_f(a, axis=axis, masked=masked)

    if weights is not None:
        # A weights array has been provided
        weights = double_precision(weights)

        if weights.ndim < a.ndim:
            weights = broadcast_array(weights, a.shape)

        if masked:
            weights = numpy_ma_array(weights, mask=a.mask, copy=False)

        if sum_of_squares:
            weights = weights * weights

        sw = weights.sum(axis=axis)

        if not numpy_ndim(sw):
            sw = numpy_asanyarray(sw)
    else:
        # The sum of weights is equal to the sample size (i.e. an
        # unweighted sample)
        sw = N.copy()

    return asanyarray(N, sw)


sw_fpartial = sum_fpartial
sw_ffinalise = sum_ffinalise

# ---------------------------------------------------------------------
# Sum of squares of weights
# ---------------------------------------------------------------------
sw2_f = functools_partial(sw_f, sum_of_squares=True)
sw2_fpartial = sum_fpartial
sw2_ffinalise = sum_ffinalise


# ---------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------
def var_f(a, axis=None, weights=None, masked=False, ddof=0):
    '''TODO

    ========  ============================================================
    Variable  Description
    ========  ============================================================
    N         Sample size

    var       Sample variance (ddof=0)

    avg       Weighted mean

    V1        Sum of weights

    V2        Sum of squares of weights

    ddof      Delta degrees of freedom

    weighted  Whether or not the sample is weighted
    ========  ============================================================

    :Returns:

        out: `tuple`

    '''
    # Make sure that a is double precision
    a = double_precision(a)

    weighted = weights is not None

    # ----------------------------------------------------------------
    # Methods:
    #
    # http://en.wikipedia.org/wiki/Standard_deviation#Population-based_statistics
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    # ----------------------------------------------------------------

    # Calculate N   = number of data points
    # Calculate avg = mean of data
    # Calculate V1  = sum of weights
    N, avg, V1 = mean_f(a, weights=weights, axis=axis, masked=masked)

    # Calculate V2 = sum of squares of weights
    if weighted and ddof == 1:
        N, V2 = sw2_f(a, axis=axis, masked=masked, weights=weights, N=N)
    else:
        V2 = None

    if axis is not None and avg.size > 1:
        # We collapsed over a single axis and the array has 2 or more
        # axes, so add an extra size 1 axis to the mean so that
        # broadcasting works when we calculate the variance.
        reshape_avg = True
        if masked:
            expand_dims = numpy_ma_expand_dims
        else:
            expand_dims = numpy_expand_dims

        avg = expand_dims(avg, axis)
    else:
        reshape_avg = False

    var = a - avg
    var *= var

    if masked:
        average = numpy_ma_average
    else:
        average = numpy_average

    var = average(var, axis=axis, weights=weights)

    if reshape_avg:
        shape = avg.shape
        avg = avg.reshape(shape[:axis] + shape[axis+1:])

    (N, var, avg, V1, V2) = asanyarray(N, var, avg, V1, V2)

    return (N, var, avg, V1, V2, ddof, weighted)


def var_fpartial(out, out1=None, group=False):
    '''Return:

* The partial sum sample sizes

* Partial sum of V1*(variance + mean^2)

* Unweighted partial sum

========  ============================================================
Variable  Description
========  ============================================================
N         Partial sample size

var       Partial sum of V1*(variance + mean^2)

avg       Unweighted partial sum

V1        Partial sum of weights

V2        Partial sum of squares of weights

ddof      Delta degrees of freedom

weighted  Whether or not the population is weighted
========  ============================================================


https://en.wikipedia.org/wiki/Pooled_variance#Population-based_statistics

:Parameters:

    out: 7-`tuple`

    out1: 7-`tuple`, optional

:Returns:

    '''
    (N, var, avg, V1, V2, ddof, weighted) = out

    if out1 is None and not group:
        # ------------------------------------------------------------
        # var = V1(var+avg**2)
        # avg = V1*avg = unweighted partial sum
        # ------------------------------------------------------------
        var += avg*avg
        var *= V1
        avg *= V1
    else:
        # ------------------------------------------------------------
        # var = var + V1b(varb+avgb**2)
        # avg = avg + V1b*avgb
        # V1  = V1 + V1b
        # V2  = V2 + V2b
        # ------------------------------------------------------------
        (Nb, varb, avgb, V1b, V2b, ddof, weighted) = out1

        N = psum(N, Nb)

        if not group:
            varb += avgb*avgb
            varb *= V1b
            avgb *= V1b

        var = psum(var, varb)
        avg = psum(avg, avgb)
        V1 = psum(V1, V1b)

        if weighted and ddof == 1:
            V2 = psum(V2, V2b)
    # --- End: if

    (N, var, avg, V1, V2) = asanyarray(N, var, avg, V1, V2)

    return (N, var, avg, V1, V2, ddof, weighted)


def var_ffinalise(out, sub_samples=None):
    '''https://en.wikipedia.org/wiki/Pooled_variance#Population-based_statistics

    '''
    (N, var, avg, V1, V2, ddof, weighted) = out

    N, var = mask_where_too_few_values(max(2, ddof+1), N, var)
    N, V1 = mask_where_too_few_values(max(2, ddof+1), N, V1)
    if V2 is not None:
        N, V2 = mask_where_too_few_values(max(2, ddof+1), N, V2)

    if sub_samples:
        # ----------------------------------------------------------------
        # The global biased variance = {[SUM(pV1(pv+pm**2)]/V1} - m**2
        #
        #   where pV1 = partial sum of weights
        #         pv  = partial biased variance
        #         pm  = partial mean
        #         V1  = global sum of weights
        #         m   = global mean
        #
        # Currently: var = SUM(pV1(pv+pm**2)
        #            avg = V1*m
        #
        # http://en.wikipedia.org/wiki/Standard_deviation#Population-based_statistics
        # ----------------------------------------------------------------
        avg /= V1
        avg *= avg
        var /= V1
        var -= avg

    # ----------------------------------------------------------------
    # var is now the biased global variance
    # ----------------------------------------------------------------
    if not weighted:
        if ddof:
            # The unweighted variance with N-ddof degrees of freedom is
            # [V1/(V1-ddof)]*var. In this case V1 equals the sample size,
            # N. ddof=1 provides an unbiased estimator of the variance of
            # a hypothetical infinite population.
            V1 /= V1 - ddof
            var *= V1
    elif ddof == 1:
        # Calculate the weighted unbiased variance. The unbiased
        # variance weighted with _reliability_ weights is
        # [V1**2/(V1**2-V2)]*var.
        #
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
        V1 **= 2
        var *= V1
        V1 -= V2
        var /= V1
    elif ddof:
        raise ValueError(
            "Can only calculate a weighted variance with a delta degrees "
            "of freedom (ddof) of 0 or 1: Got {}".format(ddof)
        )

    return asanyarray(N, var)


# ---------------------------------------------------------------------
# standard_deviation
# ---------------------------------------------------------------------
sd_f = var_f
sd_fpartial = var_fpartial


def sd_ffinalise(out, sub_samples=None):
    '''TODO

    :Parameters:

        out: `tuple`
            A 2-tuple

        sub_samples: *optional*
            Ignored.


    '''
    N, sd = var_ffinalise(out, sub_samples)

    sd **= 0.5

    return asanyarray(N, sd)
