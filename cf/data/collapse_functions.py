from functools import partial
from functools import partial as functools_partial
from functools import reduce
from operator import mul

import numpy as np
from dask.array.reductions import reduction
from numpy import abs as numpy_abs
from numpy import amax as numpy_amax
from numpy import amin as numpy_amin
from numpy import array as numpy_array
from numpy import asanyarray as numpy_asanyarray
from numpy import average as numpy_average
from numpy import bool_ as numpy_bool_
from numpy import empty as numpy_empty
from numpy import expand_dims as numpy_expand_dims
from numpy import integer as numpy_integer
from numpy import maximum as numpy_maximum
from numpy import minimum as numpy_minimum
from numpy import ndim as numpy_ndim
from numpy import sum as numpy_sum
from numpy.ma import array as numpy_ma_array
from numpy.ma import average as numpy_ma_average
from numpy.ma import expand_dims as numpy_ma_expand_dims
from numpy.ma import isMA as numpy_ma_isMA
from numpy.ma import nomask as numpy_ma_nomask
from numpy.ma import where as numpy_ma_where

from ..functions import broadcast_array


def asanyarray(*args):
    """Return every input array as an numpy ndarray, or a subclass of.

    :Parameters:

        args: sequence of array-like input objects

    :Returns:

        `list`
            The input objects left as, else converted to, `numpy.ndarray`

    """
    out = []
    for x in args:
        if x is not None and not np.ndim(x):
            # Make sure that we have a numpy array (as opposed to, e.g. a
            # numpy.float64)
            out.append(np.asanyarray(x))
        else:
            out.append(x)

    return out


def psum(x, y):
    """Add two arrays element-wise.

    If either or both of the arrays are masked then the output array is
    masked only where both input arrays are masked.

    :Parameters:

        x: numpy array-like
           *Might be updated in place*.

        y: numpy array-like
           Will not be updated in place.

    :Returns:

        `numpy.ndarray`

    **Examples:**

    >>> c = psum(a, b)

    """
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
    """The element-wise maximum of two arrays.

    :Parameters:

        x: array-like
           May be updated in place and should not be used again.

        y: array-like
           Will not be updated in place.

    :Returns:

        `numpy.ndarray`

    """
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
    """The element-wise minimum of two arrays.

    :Parameters:

        x: `numpy.ndarray`
           May be updated in place and should not be used again.

        y: `numpy.ndarray`
           Will not be updated in place.

    :Returns:

        `numpy.ndarray`

    """
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
    """Mask elements of N and x where N is strictly less than Nmin.

    :Parameters:

        Nmin: `int`

        N: `numpy.ndarray`

        x: `numpy.ndarray`

    :Returns:

        (`numpy.ndarray`, `numpy.ndarray`)
            A tuple containing *N* and *x*, both masked where *N* is
            strictly less than *Nmin*.

    """
    print(" N.min() =", N.min(), Nmin)
    if N.min() < Nmin:
        mask = N < Nmin
        N = numpy_ma_array(N, mask=mask, copy=False, shrink=False)
        x = numpy_ma_array(x, mask=mask, copy=False, shrink=True)

    return asanyarray(N, x)


def mask_small_sample_size(x, N, axis, mtol, original_shape):
    """Mask elements of N and x where N is strictly less than Nmin.

    :Parameters:

        Nmin: `int`

        N: `numpy.ndarray`

        x: `numpy.ndarray`

    :Returns:

        (`numpy.ndarray`, `numpy.ndarray`)
            A tuple containing *N* and *x*, both masked where *N* is
            strictly less than *Nmin*.

    """
    if mtol < 1:
        Nmax = reduce(mul, [original_shape[i] for i in axis], 1)
        x = np.ma.masked_where(N < (1 - mtol) * Nmax, x, copy=False)

    return asanyarray(x)[0]


def double_precision(a):
    """Convert the input array to double precision.

    :Parameters:

        a: `numpy.ndarray`

    :Returns:

        `numpy.ndarray`

    """
    char = a.dtype.char
    if char == "f":
        newtype = float
    elif char == "i":
        newtype = int
    else:
        return a

    if numpy_ma_isMA(a):
        return a.astype(newtype)
    else:
        return a.astype(newtype, copy=False)




# --------------------------------------------------------------------
# mean
# --------------------------------------------------------------------
def mean_f(a, axis=None, weights=None, masked=False):
    """The weighted average along the specified axes.

    :Parameters:

        a: array-like
            Input array. Not all missing data

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        weights: numpy array-like, optional
            Weights associated with values of the array. By default the
            statistics are unweighted.

        masked: `bool`, optional

    :Returns:

        3-`tuple` of `numpy.ndarray`
            The sample size, average and sum of weights inside a 3-tuple.

    """
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
        (N,) = sample_size_f(a, axis=axis, masked=masked)

    return asanyarray(N, avg, sw)


def mean_fpartial(out, out1=None, group=False):
    """Return the partial sample size, the partial sum and partial sum
    of the weights.

    :Parameters:

        out: 3-`tuple` of `numpy.ndarray`
            Either an output from a previous call to `mean_fpartial`;
            or, if *out1* is `None`, an output from `mean_f`.

        out1: 3-`tuple` of `numpy.ndarray`, optional
            An output from `mean_f`.

    :Returns:

        3-`tuple` of `numpy.ndarray`
            The sample size, average and sum of weights inside a 3-tuple.

    """
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
    """Divide the weighted sum by the sum of weights.

    Also mask out any values derived from a too-small sample size.

    :Parameters:

        out: 3-`tuple` of `numpy.ndarray`
            An output from `mean_fpartial`.

        sub_samples: optional

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the mean.

    """
    N, avg, sw = out

    if sub_samples:
        avg /= sw

    return mask_where_too_few_values(1, N, avg)


# --------------------------------------------------------------------
# mean_absolute_value
# --------------------------------------------------------------------
def mean_abs_f(a, axis=None, weights=None, masked=False):
    """Return the mean of the absolute array, or the means of the
    absolute array along an axis.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input is
            used.

        masked: `bool`

    :Returns:

        2-tuple of `numpy.ndarray`
            The sample sizes and the means of the absolute values.

    """
    return mean_f(numpy_abs(a), axis=axis, weights=weights, masked=masked)


mean_abs_fpartial = mean_fpartial
mean_abs_ffinalise = mean_ffinalise


# --------------------------------------------------------------------
# root_mean_square
# --------------------------------------------------------------------
def root_mean_square_f(a, axis=None, weights=None, masked=False):
    """The RMS along the specified axes.

    :Parameters:

        a: array-like
            Input array. Not all missing data

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        weights: numpy array-like, optional
            Weights associated with values of the array. By default the
            statistics are unweighted.

        masked: `bool`, optional

    :Returns:

        `tuple`
            3-tuple.

    """
    a = double_precision(a)

    return mean_f(a ** 2, axis=axis, weights=weights, masked=masked)


root_mean_square_fpartial = mean_fpartial


def root_mean_square_ffinalise(out, sub_samples=None):
    """Divide the weighted sum by the sum of weights and take the square
    root.

    Also mask out any values derived from a too-small sample size.

    :Parameters:

        out: 3-`tuple` of `numpy.ndarray`
            An output from `root_mean_square_fpartial`.

        sub_samples: optional

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the RMS.

    """
    N, avg = mean_ffinalise(out, sub_samples=sub_samples)

    avg **= 0.5

    return asanyarray(N, avg)



def sum_f(a, axis=None, weights=None, masked=False):
    """Return the sum of an array or the sum along an axis.

    ``sum_f(a, axis=axis)`` is equivalent to ``(numpy.sum(a,
    axis=axis),)``

    :Parameters:

        a: numpy array-like
            Input array

        weights: numpy array-like, optional
            Weights associated with values of the array. By default the
            statistics are unweighted.

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the sum.

    """
    a = double_precision(a)

    (N,) = sample_size_f(a, axis=axis, masked=masked)

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
    """Return the partial sum of an array.

    :Parameters:

        out: 2-`tuple` of `numpy.ndarray`

        out1: 2-`tuple` of `numpy.ndarray`, optional

        group: *optional*
            Ignored.

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the sum.

    """
    N, asum = out

    if out1 is not None:
        N1, asum1 = out1
        N = psum(N, N1)
        asum = psum(asum, asum1)

    return asanyarray(N, asum)


def sum_ffinalise(out, sub_samples=None):
    """Apply any logic to finalise the collapse to the sum of an array.

    Here mask out any values derived from a too-small sample size.

    :Parameters:

        sub_samples: optional
            Ignored.

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the sum.

    """
    return mask_where_too_few_values(1, *out)


# ---------------------------------------------------------------------
# sum_of_squares
# ---------------------------------------------------------------------
def sum_of_squares_f(a, axis=None, weights=None, masked=False):
    """Return the sum of the square of an array or the sum of squares
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

    """
    a = double_precision(a)
    return sum_f(a ** 2, axis=axis, weights=weights, masked=masked)


sum_of_squares_fpartial = sum_fpartial
sum_of_squares_ffinalise = sum_ffinalise


# ---------------------------------------------------------------------
# Sum of weights
# ---------------------------------------------------------------------
def sw_f(
    a, axis=None, masked=False, weights=None, N=None, sum_of_squares=False
):
    """Return the sum of weights for an array.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        masked: `bool`, optional

        weights: numpy array-like, optional
            Weights associated with values of the array. By default the
            statistics are unweighted.

        N:  `numpy.ndarray`
            Sample size

        sum_of_squares: delta degrees of freedom

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the sum of weights.

    """
    if N is None:
        (N,) = sample_size_f(a, axis=axis, masked=masked)

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
    """Return a tuple containing metrics relating to the array variance.

    The tuple is a 7-tuple that contains, in the order given, the
    following variables:

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

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        weights: numpy array-like, optional
            Weights associated with values of the array. By default the
            statistics are unweighted.

        masked: `bool`, optional

        ddof: delta degrees of freedom, optional

    :Returns:

        7-`tuple`
            Tuple containing the value of the statistical metrics described
            in the above table, in the given order.

    """
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
        avg = avg.reshape(shape[:axis] + shape[axis + 1 :])

    (N, var, avg, V1, V2) = asanyarray(N, var, avg, V1, V2)

    return (N, var, avg, V1, V2, ddof, weighted)


def var_fpartial(out, out1=None, group=False):
    """Return a tuple of partial metrics relating to the array variance.

    The tuple is a 7-tuple that contains, in the order given, the
    following variables:

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

    For further information, see:
    https://en.wikipedia.org/wiki/Pooled_variance#Population-based_statistics

    :Parameters:

        out: 7-`tuple`

        out1: 7-`tuple`, optional

    :Returns:

        7-`tuple`
            Tuple containing the value of the statistical metrics described
            in the above table, in the given order.

    """
    (N, var, avg, V1, V2, ddof, weighted) = out

    if out1 is None and not group:
        # ------------------------------------------------------------
        # var = V1(var+avg**2)
        # avg = V1*avg = unweighted partial sum
        # ------------------------------------------------------------
        var += avg * avg
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
            varb += avgb * avgb
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
    """Calculate the variance of the array and return it with the sample
    size.

    Also mask out any values derived from a too-small sample size.

    :Parameters:

        out: 7-`tuple`

        sub_samples: optional

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the variance.

    """
    (N, var, avg, V1, V2, ddof, weighted) = out

    N, var = mask_where_too_few_values(max(2, ddof + 1), N, var)
    N, V1 = mask_where_too_few_values(max(2, ddof + 1), N, V1)
    if V2 is not None:
        N, V2 = mask_where_too_few_values(max(2, ddof + 1), N, V2)

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
        # https://en.wikipedia.org/wiki/Pooled_variance#Population-based_statistics
        #
        # For the general case of M non-overlapping data sets, X_{1}
        # through X_{M}, and the aggregate data set X=\bigcup_{i}X_{i}
        # we have the unweighted mean and variance is:
        #
        # \mu_{X}={\frac{1}{\sum_{i}{N_{X_{i}}}}}\left(\sum_{i}{N_{X_{i}}\mu_{X_{i}}}\right)
        #
        # var_{X}={{\frac{1}{\sum_{i}{N_{X_{i}}-ddof}}}\left(\sum_{i}{\left[(N_{X_{i}}-1)\sigma_{X_{i}}^{2}+N_{X_{i}}\mu_{X_{i}}^{2}\right]}-\left[\sum_{i}{N_{X_{i}}}\right]\mu_{X}^{2}\right)}
        #
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
    """Apply any logic to finalise the collapse to the standard
    deviation.

    :Parameters:

        out: `tuple`
            A 2-tuple

        sub_samples: *optional*
            Ignored.

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the standard deviation.

    """
    N, sd = var_ffinalise(out, sub_samples)

    sd **= 0.5

    return asanyarray(N, sd)


from dask.array import chunk
from dask.array.core import _concatenate2, broadcast_to
from dask.array.reductions import divide
from dask.array.ufunc import multiply
from dask.utils import deepmap  # Apply function inside nested lists

def sum_of_weights(x, weights=None, N=None, squared=False, **kwargs):
    """TODO"""
    if weights is None:
        if N is None:
            N = cf_sample_size_chunk(x, **kwargs)["N"]
            
        sw = N
    else:
        if squared:
            weights = multiply(weights, weights, dtype=float)

        if np.ma.is_masked(x):
            weights = np.ma.masked_where(x.mask, weights)

        sw = weights.sum(**kwargs)

    return sw

def combine_sample_sizes(pairs, axis, **kwargs):
    # Create a nested list of N and recursively concatenate it
    # along the specified axes
    return combine_arrays(pairs, "N", chunk.sum, axis, int, False, **kwargs)


def combine_arrays(
    pairs, key, func, axis, dtype, computing_meta=False, **kwargs
):
    # Create a nested list of N and recursively concatenate it
    # along the specified axes
    x = deepmap(lambda pair: pair[key], pairs) if not computing_meta else pairs
    if dtype:
        kwargs["dtype"] = dtype

    x = func(_concatenate2(x, axes=axis), axis=axis, **kwargs)
    return x


def sum_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    # Create a nested list of N and recursively concatenate it
    # along the specified axes    
    return combine_arrays(pairs, key, chunk.sum, axis, dtype,
                          computing_meta, **kwargs)


# --------------------------------------------------------------------
# mean
# --------------------------------------------------------------------
def cf_mean_chunk(x, weights=None, dtype=float, computing_meta=False, **kwargs):
    """Find the max of an array."""
    if computing_meta:
        return x

    d = cf_sum_chunk(x, weights=weights, **kwargs)

    if weights is None:
        sw = d["N"]
    else:
        sw = chunk.sum(weights, **kwargs)
        
    d["sw"] = sw
    return d

def cf_mean_combine(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    **kwargs,
):
    """Apply the function to the data in a nested list of arrays."""
    if not isinstance(pairs, list):
        pairs = [pairs]
        
    d = {}
    for key in ("sum", "sw"):        
        d[key] = sum_arrays(pairs, key, axis, dtype, computing_meta, **kwargs)
        if computing_meta:
            return x
    
    d["N"] = combine_sample_sizes(pairs, axis, **kwargs)
    return d

def cf_mean_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Apply the function to the data in a nested list of arrays and
    mask where the sample size is below the threshold."""
    d = cf_mean_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d
    
    x = divide(d["sum"], d["sw"]) # dtype?
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


def cf_mean(a, axis=None, weights=None, keepdims=False, mtol=1,
            split_every=None):
    """TODODASK."""
    dtype = float
    return reduction(
        a,
        cf_mean_chunk,
        partial(cf_mean_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_mean_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
        weights=weights
    )


# --------------------------------------------------------------------
# max
# --------------------------------------------------------------------
def cf_max_chunk(x, computing_meta=False, **kwargs):
    """Find the max of an array."""
    if computing_meta:
        return x

    return {
        "max": chunk.max(x, **kwargs),
        "N": cf_sample_size_chunk(x, **kwargs)["N"],
    }


def cf_max_combine(
    pairs,
    axis=None,
    computing_meta=False,
    **kwargs,
):
    """Find the max and min of a nested list of arrays."""
    if not isinstance(pairs, list):
        pairs = [pairs]

    # Create a nested list of maxima and recursively concatenate it
    # along the specified axes
    m = combine_arrays(
        pairs, "max", chunk.max, axis, None, computing_meta, **kwargs
    )
    if computing_meta:
        return m

    return {
        "max": m,
        "N": combine_sample_sizes(pairs, axis, **kwargs),
    }


def cf_max_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Find the range of a nested list of arrays."""
    d = cf_max_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d
    

    x = d["max"]
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


def cf_max(a, axis=None, keepdims=False, mtol=1, split_every=None):
    """TODODASK."""
    dtype = a.dtype
    return reduction(
        a,
        cf_max_chunk,
        partial(cf_max_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_max_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
    )



# --------------------------------------------------------------------
# maximum_absolute_value
# --------------------------------------------------------------------
def cf_max_abs_chunk(x, computing_meta=False, **kwargs):
    """Return the maximum of the absolute array, or the maximum of the
    absolute array along an axis.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        masked: bool

    :Returns:

        2-tuple of numpy arrays
            The sample sizes and the maxima of the absolute values.

    """
    if computing_meta:
        return x
    
    return cf_max_chunk(np.abs(x), **kwargs)


cf_max_abs_combine = cf_max_combine
cf_max_abs_agg = cf_max_agg

# --------------------------------------------------------------------
# min
# --------------------------------------------------------------------
def cf_min_chunk(x, computing_meta=False, **kwargs):
    """Find the max of an array."""
    if computing_meta:
        return x

    return  {
        "min": chunk.min(x, **kwargs),
        "N": cf_sample_size_chunk(x, **kwargs)["N"],
    }


def cf_min_combine(
    pairs,
    axis=None,
    computing_meta=False,
    **kwargs,
):
    """Find the max and min of a nested list of arrays."""
    if not isinstance(pairs, list):
        pairs = [pairs]

    # Create a nested list of maxima and recursively concatenate it
    # along the specified axes
    x = combine_arrays(
        pairs, "min", chunk.min, axis, None, computing_meta, **kwargs
    )
    if computing_meta:
        return m

    return {
        "min": x,
        "N": combine_sample_sizes(pairs, axis, **kwargs),
    }


def cf_min_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Find the range of a nested list of arrays."""
    d = cf_min_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d
    
    x = d["min"]
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x



def cf_min(a, axis=None, keepdims=False, mtol=1, split_every=None):
    """TODODASK."""
    dtype = a.dtype
    return reduction(
        a,
        cf_min_chunk,
        partial(cf_min_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_min_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
    )

# --------------------------------------------------------------------
# minimum absolute value
# --------------------------------------------------------------------
def cf_min_abs_chunk(x, computing_meta=False, **kwargs):
    """Return the maximum of the absolute array, or the maximum of the
    absolute array along an axis.

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        masked: bool

    :Returns:

        2-tuple of numpy arrays
            The sample sizes and the maxima of the absolute values.

    """
    if computing_meta:
        return x

    return cf_min_chunk(np.abs(x), **kwargs)


cf_min_abs_combine = cf_min_combine
cf_min_abs_agg = cf_min_agg

# --------------------------------------------------------------------
# range
# --------------------------------------------------------------------
def cf_range_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Find the max and min of an array."""
    if computing_meta:
        return x

    d = cf_max_chunk(x, **kwargs)
    d["min"] = chunk.min(x, **kwargs)
    return  d


def cf_range_combine(
    pairs,
    axis=None,
    dtype=None,
    computing_meta=False,
    **kwargs,
):
    """Find the max and min of a nested list of arrays."""
    if not isinstance(pairs, list):
        pairs = [pairs]

    # Create a nested list of maxima and recursively concatenate it
    # along the specified axes
    mx = combine_arrays(
        pairs, "max", chunk.max, axis, None, computing_meta, **kwargs
    )
    if computing_meta:
        return mx

    mn = combine_arrays(
        pairs, "min", chunk.min, axis, None, computing_meta, **kwargs
    )

    return {
        "max": mx,
        "min": mn,
        "N": combine_sample_sizes(pairs, axis, **kwargs),
    }


def cf_range_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Find the range of a nested list of arrays."""
    d = cf_range_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d
    
    # Calculate the range
    x = d["max"] - d["min"]
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


def cf_range(a, axis=None, keepdims=False, mtol=1, split_every=None):
    """TODODASK."""
    dtype = a.dtype
    return reduction(
        a,
        cf_range_chunk,
        partial(cf_range_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_range_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
    )
# --------------------------------------------------------------------
# mid-range
# --------------------------------------------------------------------
cf_mid_range_chunk = cf_range_chunk
cf_mid_range_combine = cf_range_combine


def cf_mid_range_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Find the mid-range of a nested list of arrays."""
    d = cf_range_combine(pairs, axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d
    

    # Calculate the mid-range
    x = divide(d["max"] + d["min"], 2.0, dtype=dtype)
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


def cf_mid_range(
    a, axis=None, dtype=float, keepdims=False, mtol=1, split_every=None
):
    """TODODASK."""
    dtype = float
    return reduction(
        a,
        cf_mid_range_chunk,
        partial(cf_mid_range_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_mid_range_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
    )


# --------------------------------------------------------------------
# sample_size
# --------------------------------------------------------------------
def cf_sample_size_chunk(x, dtype=int, computing_meta=False, **kwargs):
    if computing_meta:
        return x

    if np.ma.isMA(x):
        N = chunk.sum(~np.ma.getmaskarray(x), dtype=dtype, **kwargs)
        if not np.ndim(N):
            N = np.asanyarray(N)
    else:
        axis = kwargs["axis"]
        shape = [1 if i in axis else n for i, n in enumerate(x.shape)]
        size = reduce(mul, [n for i, n in enumerate(x.shape) if i in axis], 1)
        N = np.full(shape, size, dtype=dtype)

    return {"N": N}


def cf_sample_size_combine(
    pairs,
    axis=None,
    computing_meta=False,
    **kwargs,
):
    if not isinstance(pairs, list):
        pairs = [pairs]

    x = combine_arrays(pairs, "N", chunk.sum, axis, None,
                       computing_meta, **kwargs)
    if computing_meta:
        return x
    
    return {"N": x}


def cf_sample_size_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    d = cf_sample_size_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d
    
    x = d["N"]
    x = mask_small_sample_size(x, x, axis, mtol, original_shape)
    return x


def cf_sample_size(a, axis=None, keepdims=False, mtol=1, split_every=None):
    """TODODASK."""
    dtype = int
    return reduction(
        a,
        cf_sample_size_chunk,
        partial(cf_sample_size_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_sample_size_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
    )



# --------------------------------------------------------------------
# sum
# --------------------------------------------------------------------
def cf_sum_chunk(x, weights=None, dtype=float, computing_meta=False, **kwargs):
    """Find the max of an array."""
    if computing_meta:
        return x

    if weights is not None:
        x = multiply(x, weights) # sort out  dtype=result_dtype)

    d = cf_sample_size_chunk(x, **kwargs)
    d["sum"] = chunk.sum(x, dtype=dtype, **kwargs)
    return d


def cf_sum_combine(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    **kwargs,
):
    """Apply the function to the data in a nested list of arrays."""
    if not isinstance(pairs, list):
        pairs = [pairs]

    # Create a nested list of maxima and recursively concatenate it
    # along the specified axes
    x = sum_arrays(pairs, "sum", axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return x

    return {
        "sum": x,
        "N": combine_sample_sizes(pairs, axis, **kwargs),
    }


def cf_sum_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Apply the function to the data in a nested list of arrays and
    mask where the sample size is below the threshold."""
    d = cf_sum_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = d["sum"]
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


def cf_sum(a, axis=None, weights=None, keepdims=False, mtol=1, split_every=None):
    """TODODASK."""
    dtype = float
    return reduction(
        a,
        cf_sum_chunk,
        partial(cf_sum_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_sum_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
        weights=weights
    )
# --------------------------------------------------------------------
# variance
# --------------------------------------------------------------------
def cf_var_chunk(x, weights=None, dtype=float, computing_meta=False,
                 ddof=0, **kwargs):
    """Return a tuple containing metrics relating to the array variance.

    The tuple is a 7-tuple that contains, in the order given, the
    following variables:

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

    :Parameters:

        a: numpy array_like
            Input array

        axis: `int`, optional
            Axis along which to operate. By default, flattened input
            is used.

        weights: numpy array-like, optional
            Weights associated with values of the array. By default the
            statistics are unweighted.

        masked: `bool`, optional

        ddof: delta degrees of freedom, optional

    :Returns:

        7-`tuple`
            Tuple containing the value of the statistical metrics described
            in the above table, in the given order.

    """
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
    d = cf_mean_chunk(x, weights, dtype=dtype, **kwargs)

    N = d["N"]
    V1 = d["sw"]
    avg = d["avg"] #divide(d"sum"], V1) # dtype
    
#    N, avg, V1 = mean_f(a, weights=weights, axis=axis, masked=masked)

    # Calculate V2 = sum of squares of weights
    if weights is not None and ddof == 1:
        V2 = sum_of_weights(x, weights, N=N, squared=True, **kwargs)
    else:
        V2 = None

#    if axis is not None and avg.size > 1:
#        # We collapsed over a single axis and the array has 2 or more
#        # axes, so add an extra size 1 axis to the mean so that
#        # broadcasting works when we calculate the variance.
#        reshape_avg = True
#        if masked:
#            expand_dims = numpy_ma_expand_dims
#        else:
#            expand_dims = numpy_expand_dims
#
#        avg = expand_dims(avg, axis)
#    else:
#        reshape_avg = False

    var = x - avg
    var *= var

    if np.ma.isMA(var):
        average = np.ma.average
    else:
        average = np.average

    var = average(var, weights=weights, **kwargs)

#    if reshape_avg:
#        shape = avg.shape
#        avg = avg.reshape(shape[:axis] + shape[axis + 1 :])

#    (N, var, avg, V1, V2) = asanyarray(N, var, avg, V1, V2)

    

    return {"var": var,
            "avg": avg,
            "N": N,
            "V1": V1,
            "V2": V2,
#            "ddof": ddof,
#            "weighted": weighted,
    }



def cf_var_combine(out, out1=None, group=False):
    """Return a tuple of partial metrics relating to the array variance.

    The tuple is a 7-tuple that contains, in the order given, the
    following variables:

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

    For further information, see:
    https://en.wikipedia.org/wiki/Pooled_variance#Population-based_statistics

    :Parameters:

        out: 7-`tuple`

        out1: 7-`tuple`, optional

    :Returns:

        7-`tuple`
            Tuple containing the value of the statistical metrics described
            in the above table, in the given order.

    """
    (N, var, avg, V1, V2, ddof, weighted) = out

    if out1 is None and not group:
        # ------------------------------------------------------------
        # var = V1(var+avg**2)
        # avg = V1*avg = unweighted partial sum
        # ------------------------------------------------------------
        var += avg * avg
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
            varb += avgb * avgb
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
    """Calculate the variance of the array and return it with the sample
    size.

    Also mask out any values derived from a too-small sample size.

    :Parameters:

        out: 7-`tuple`

        sub_samples: optional

    :Returns:

        2-`tuple` of `numpy.ndarray`
            The sample size and the variance.

    """
    (N, var, avg, V1, V2, ddof, weighted) = out

    N, var = mask_where_too_few_values(max(2, ddof + 1), N, var)
    N, V1 = mask_where_too_few_values(max(2, ddof + 1), N, V1)
    if V2 is not None:
        N, V2 = mask_where_too_few_values(max(2, ddof + 1), N, V2)

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
        # https://en.wikipedia.org/wiki/Pooled_variance#Population-based_statistics
        #
        # For the general case of M non-overlapping data sets, X_{1}
        # through X_{M}, and the aggregate data set X=\bigcup_{i}X_{i}
        # we have the unweighted mean and variance is:
        #
        # \mu_{X}={\frac{1}{\sum_{i}{N_{X_{i}}}}}\left(\sum_{i}{N_{X_{i}}\mu_{X_{i}}}\right)
        #
        # var_{X}={{\frac{1}{\sum_{i}{N_{X_{i}}-ddof}}}\left(\sum_{i}{\left[(N_{X_{i}}-1)\sigma_{X_{i}}^{2}+N_{X_{i}}\mu_{X_{i}}^{2}\right]}-\left[\sum_{i}{N_{X_{i}}}\right]\mu_{X}^{2}\right)}
        #
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


