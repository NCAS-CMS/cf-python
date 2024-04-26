"""Reduction functions intended to be passed to be dask.

Most of these functions are expected to be set as *chunk*, *combine* and
*aggregate* parameters of `dask.array.reduction`

"""

from functools import reduce
from operator import mul

import numpy as np
from dask.array import chunk
from dask.array.core import _concatenate2
from dask.array.reductions import divide, numel
from dask.core import flatten
from dask.utils import deepmap

from .collapse_utils import double_precision_dtype


def mask_small_sample_size(x, N, axis, mtol, original_shape):
    """Mask elements where the sample size is below a threshold.

    .. versionadded:: 3.14.0

    :Parameters:

        x: `numpy.ndarray`
            The collapsed data.

        N: `numpy.ndarray`
            The sample sizes of the collapsed values.

        axis: sequence of `int`
            The axes being collapsed.

        mtol: number
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

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

    :Returns:

        `numpy.ndarray`
            Array *x* masked where *N* is sufficiently small. Note
            that the input *x* might be modified in-place with the
            contents of the output.

    """
    if not x.ndim:
        # Make sure that we have a numpy array (e.g. as opposed to a
        # numpy.float64)
        x = np.asanyarray(x)

    if mtol < 1:
        # Nmax = total number of elements, including missing values
        Nmax = reduce(mul, [original_shape[i] for i in axis], 1)
        x = np.ma.masked_where(N < (1 - mtol) * Nmax, x, copy=False)

    return x


def sum_weights_chunk(
    x, weights=None, square=False, N=None, check_weights=True, **kwargs
):
    """Sum the weights.

    .. versionadded:: 3.14.0

    :Parameters:

        x: `numpy.ndarray`
            The data.

        weights: `numpy.ndarray`, optional
            The weights associated with values of the data. Must have
            the same shape as *x*. By default *weights* is `None`,
            meaning that all non-missing elements of the data have a
            weight of 1 and all missing elements have a weight of
            0. If given as an array then those weights that are
            missing data, or that correspond to the missing elements
            of the data, are assigned a weight of 0.

        square: `bool`, optional
            If True calculate the sum of the squares of the weights.

        N: `numpy.ndarray`, optional
            The sample size. If provided as an array and there are no
            weights, then the *N* is returned instead of calculating
            the sum (of the squares) of weights. Ignored of *weights*
            is not `None`.

        check_weights: `bool`, optional
            If True, the default, then check that all weights are
            positive.

            .. versionadded:: 3.16.0

    :Returns:

        `numpy.ndarray`
            The sum of the weights, with data type "i8" or "f8".

    """
    if weights is None:
        # All weights are 1, so the sum of the weights and the sum of
        # the squares of the weights are both equal to the sample
        # size.
        if N is None:
            N = cf_sample_size_chunk(x, **kwargs)["N"]

        return N
    elif check_weights:
        w_min = weights.min()
        if w_min <= 0:
            raise ValueError(
                "All collapse weights must be positive. "
                f"Got a weight of {w_min!r}. Consider replacing "
                "non-positive values with missing data."
            )

    dtype = double_precision_dtype(weights)
    if square:
        weights = np.multiply(weights, weights, dtype=dtype)

    if np.ma.is_masked(x):
        weights = np.ma.masked_where(x.mask, weights)

    return chunk.sum(weights, dtype=dtype, **kwargs)


def combine_arrays(
    pairs, key, func, axis, dtype=None, computing_meta=False, **kwargs
):
    """Worker function for Combine callables.

    Select arrays by dictionary key from a nested list of
    dictionaries, concatenate the resulting nested list of arrays
    along the specified axes, and then apply a function to the result
    along those same axes.

    See `dask.array.reductions.mean_combine` for an example.

    .. versionadded:: 3.14.0

    :Returns:

        `numpy.ndarray`

    """
    x = deepmap(lambda pair: pair[key], pairs) if not computing_meta else pairs

    if dtype:
        kwargs["dtype"] = dtype

    x = _concatenate2(x, axes=axis)
    return func(x, axis=axis, **kwargs)


def sum_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``func=chunk.sum``.

    .. versionadded:: 3.14.0

    """
    return combine_arrays(
        pairs, key, chunk.sum, axis, dtype, computing_meta, **kwargs
    )


def max_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``func=chunk.max``.

    .. versionadded:: 3.14.0

    """
    return combine_arrays(
        pairs, key, chunk.max, axis, dtype, computing_meta, **kwargs
    )


def min_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``func=chunk.min``.

    .. versionadded:: 3.14.0

    """
    return combine_arrays(
        pairs, key, chunk.min, axis, dtype, computing_meta, **kwargs
    )


def sum_sample_sizes(pairs, axis, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``key="N", func=chunk.sum,
    dtype="i8"``.

    .. versionadded:: 3.14.0

    """
    return combine_arrays(
        pairs,
        "N",
        chunk.sum,
        axis,
        dtype="i8",
        computing_meta=computing_meta,
        **kwargs,
    )


# --------------------------------------------------------------------
# mean
# --------------------------------------------------------------------
def cf_mean_chunk(
    x,
    weights=None,
    dtype="f8",
    computing_meta=False,
    check_weights=True,
    **kwargs,
):
    """Chunk calculations for the mean.

     This function is passed to `dask.array.reduction` as its *chunk*
     parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        check_weights: `bool`, optional
            If True then check that all weights are positive.

            .. versionadded:: 3.16.0

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * V1: The sum of ``weights`` (equal to ``N`` if weights
                  are not set).
            * sum: The weighted sum of ``x``.
            * weighted: True if weights have been set.

    """
    if computing_meta:
        return x

    # N, sum
    d = cf_sum_chunk(x, weights, dtype=dtype, **kwargs)

    d["V1"] = sum_weights_chunk(
        x, weights, N=d["N"], check_weights=False, **kwargs
    )
    d["weighted"] = weights is not None

    return d


def cf_mean_combine(
    pairs, axis=None, dtype="f8", computing_meta=False, **kwargs
):
    """Combination calculations for the mean.

    This function is passed to `dask.array.reduction` as its *combine*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        As for `cf_mean_chunk`.

    """
    if not isinstance(pairs, list):
        pairs = [pairs]

    weighted = next(flatten(pairs))["weighted"]
    d = {"weighted": weighted}

    d["sum"] = sum_arrays(pairs, "sum", axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d["sum"]

    d["N"] = sum_sample_sizes(pairs, axis, **kwargs)
    if weighted:
        d["V1"] = sum_arrays(pairs, "V1", axis, dtype, **kwargs)
    else:
        d["V1"] = d["N"]

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
    """Aggregation calculations for the mean.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_mean_combine(pairs, axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = divide(d["sum"], d["V1"], dtype=dtype)
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


# --------------------------------------------------------------------
# maximum
# --------------------------------------------------------------------
def cf_max_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Chunk calculations for the maximum.

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * max: The maximum of `x``.

    """
    if computing_meta:
        return x

    return {
        "max": chunk.max(x, **kwargs),
        "N": cf_sample_size_chunk(x, **kwargs)["N"],
    }


def cf_max_combine(pairs, axis=None, computing_meta=False, **kwargs):
    """Combination calculations for the maximum.

    This function is passed to `dask.array.reduction` as its *combine*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        As for `cf_max_chunk`.

    """
    if not isinstance(pairs, list):
        pairs = [pairs]

    mx = max_arrays(pairs, "max", axis, None, computing_meta, **kwargs)
    if computing_meta:
        return mx

    return {"max": mx, "N": sum_sample_sizes(pairs, axis, **kwargs)}


def cf_max_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Aggregation calculations for the maximum.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_max_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = d["max"]
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


# --------------------------------------------------------------------
# mid-range
# --------------------------------------------------------------------
def cf_mid_range_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Aggregation calculations for the mid-range.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_range_combine(pairs, axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d

    # Calculate the mid-range
    x = divide(d["max"] + d["min"], 2.0, dtype=dtype)
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


# --------------------------------------------------------------------
# minimum
# --------------------------------------------------------------------
def cf_min_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Chunk calculations for the minimum.

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * min: The minimum of ``x``.

    """
    if computing_meta:
        return x

    return {
        "min": chunk.min(x, **kwargs),
        "N": cf_sample_size_chunk(x, **kwargs)["N"],
    }


def cf_min_combine(pairs, axis=None, computing_meta=False, **kwargs):
    """Combination calculations for the minimum.

    This function is passed to `dask.array.reduction` as its *combine*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        As for `cf_min_chunk`.

    """
    if not isinstance(pairs, list):
        pairs = [pairs]

    mn = min_arrays(pairs, "min", axis, None, computing_meta, **kwargs)
    if computing_meta:
        return mn

    return {"min": mn, "N": sum_sample_sizes(pairs, axis, **kwargs)}


def cf_min_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Aggregation calculations for the minimum.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_min_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = d["min"]
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


# --------------------------------------------------------------------
# range
# --------------------------------------------------------------------
def cf_range_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Chunk calculations for the range.

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * min: The minimum of ``x``.
            * max: The maximum of ``x`.

    """
    if computing_meta:
        return x

    # N, max
    d = cf_max_chunk(x, **kwargs)

    d["min"] = chunk.min(x, **kwargs)
    return d


def cf_range_combine(
    pairs, axis=None, dtype=None, computing_meta=False, **kwargs
):
    """Combination calculations for the range.

    This function is passed to `dask.array.reduction` as its *combine*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        As for `cf_range_chunk`.

    """
    if not isinstance(pairs, list):
        pairs = [pairs]

    mx = max_arrays(pairs, "max", axis, None, computing_meta, **kwargs)
    if computing_meta:
        return mx

    mn = min_arrays(pairs, "min", axis, None, **kwargs)

    return {"max": mx, "min": mn, "N": sum_sample_sizes(pairs, axis, **kwargs)}


def cf_range_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Aggregation calculations for the range.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_range_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d

    # Calculate the range
    x = d["max"] - d["min"]
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


# --------------------------------------------------------------------
# root mean square
# --------------------------------------------------------------------
def cf_rms_chunk(x, weights=None, dtype="f8", computing_meta=False, **kwargs):
    """Chunk calculations for the root mean square (RMS).

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * sum: The weighted sum of ``x**2``.

    """
    if computing_meta:
        return x

    return cf_mean_chunk(
        np.multiply(x, x, dtype=dtype), weights=weights, dtype=dtype, **kwargs
    )


def cf_rms_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Aggregation calculations for the root mean square (RMS).

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_mean_combine(pairs, axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = np.sqrt(d["sum"] / d["V1"], dtype=dtype)
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


# --------------------------------------------------------------------
# sample size
# --------------------------------------------------------------------
def cf_sample_size_chunk(x, dtype="i8", computing_meta=False, **kwargs):
    """Chunk calculations for the sample size.

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

    """
    if computing_meta:
        return x

    if np.ma.isMA(x):
        N = chunk.sum(np.ones_like(x, dtype=dtype), **kwargs)
    else:
        if dtype:
            kwargs["dtype"] = dtype

        N = numel(x, **kwargs)

    return {"N": N}


def cf_sample_size_combine(
    pairs, axis=None, dtype="i8", computing_meta=False, **kwargs
):
    """Combination calculations for the sample size.

    This function is passed to `dask.array.reduction` as its *combine*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        As for `cf_sample_size_chunk`.

    """
    if not isinstance(pairs, list):
        pairs = [pairs]

    x = sum_arrays(pairs, "N", axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return x

    return {"N": x}


def cf_sample_size_agg(
    pairs,
    axis=None,
    computing_meta=False,
    dtype="i8",
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Aggregation calculations for the sample size.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_sample_size_combine(pairs, axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = d["N"]
    x = mask_small_sample_size(x, x, axis, mtol, original_shape)
    return x


# --------------------------------------------------------------------
# sum
# --------------------------------------------------------------------
def cf_sum_chunk(
    x,
    weights=None,
    dtype="f8",
    computing_meta=False,
    check_weights=True,
    **kwargs,
):
    """Chunk calculations for the sum.

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        check_weights: `bool`, optional
            If True, the default, then check that all weights are
            positive.

            .. versionadded:: 3.16.0

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * sum: The weighted sum of ``x``

    """
    if computing_meta:
        return x

    if weights is not None:
        if check_weights:
            w_min = weights.min()
            if w_min <= 0:
                raise ValueError(
                    "All collapse weights must be positive. "
                    f"Got a weight of {w_min!r}. Consider replacing "
                    "non-positive values with missing data."
                )

        x = np.multiply(x, weights, dtype=dtype)

    d = cf_sample_size_chunk(x, **kwargs)
    d["sum"] = chunk.sum(x, dtype=dtype, **kwargs)
    return d


def cf_sum_combine(
    pairs, axis=None, dtype="f8", computing_meta=False, **kwargs
):
    """Combination calculations for the sum.

    This function is passed to `dask.array.reduction` as its *combine*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        As for `cf_sum_chunk`.

    """
    if not isinstance(pairs, list):
        pairs = [pairs]

    x = sum_arrays(pairs, "sum", axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return x

    return {"sum": x, "N": sum_sample_sizes(pairs, axis, **kwargs)}


def cf_sum_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Aggregation calculations for the sum.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_sum_combine(pairs, axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = d["sum"]
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


# --------------------------------------------------------------------
# sum of weights
# --------------------------------------------------------------------
def cf_sum_of_weights_chunk(
    x, weights=None, dtype="f8", computing_meta=False, square=False, **kwargs
):
    """Chunk calculations for the sum of the weights.

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    :Parameters:

        square: `bool`, optional
            If True then calculate the sum of the squares of the
            weights.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * sum: The sum of ``weights``, or the sum of
                   ``weights**2`` if *square* is True.

    """
    if computing_meta:
        return x

    # N
    d = cf_sample_size_chunk(x, **kwargs)

    d["sum"] = sum_weights_chunk(
        x, weights=weights, square=square, N=d["N"], **kwargs
    )

    return d


# --------------------------------------------------------------------
# unique
# --------------------------------------------------------------------
def cf_unique_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Chunk calculations for the unique values.

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

    See `dask.array.reductions` for details of the parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * unique: The unique values.

    """
    if computing_meta:
        return x

    return {"unique": np.unique(x)}


def cf_unique_agg(pairs, axis=None, computing_meta=False, **kwargs):
    """Aggregation calculations for the unique values.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    It is assumed that the arrays are one-dimensional.

    .. versionadded:: 3.14.0

    :Parameters:

    See `dask.array.reductions` for details of the parameters.

    :Returns:

        `dask.array.Array`
            The unique values.

    """
    x = (
        deepmap(lambda pair: pair["unique"], pairs)
        if not computing_meta
        else pairs
    )
    if computing_meta:
        return x

    x = _concatenate2(x, axes=[0])
    return np.unique(x)


# --------------------------------------------------------------------
# variance
# --------------------------------------------------------------------
def cf_var_chunk(
    x, weights=None, dtype="f8", computing_meta=False, ddof=None, **kwargs
):
    """Chunk calculations for the variance.

    This function is passed to `dask.array.reduction` as its *chunk*
    parameter.

    See
    https://en.wikipedia.org/wiki/Pooled_variance#Sample-based_statistics
    for details.

    .. versionadded:: 3.14.0

    :Parameters:

        ddof: number
            The delta degrees of freedom. The number of degrees of
            freedom used in the calculation is (N-*ddof*) where N
            represents the number of non-missing elements. A value of
            1 applies Bessel's correction.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * V1: The sum of ``weights`` (equal to ``N`` if weights
                  are not set).
            * V2: The sum of ``weights**2``, or `None` of not
                  required.
            * sum: The weighted sum of ``x``.
            * part: ``V1 * (sigma**2 + mu**2)``, where ``sigma**2`` is
                    the weighted biased (i.e. ``ddof=0``) variance of
                    ``x``, and ``mu`` is the weighted mean of ``x``.
            * weighted: True if weights have been set.
            * ddof: The delta degrees of freedom.

    """
    if computing_meta:
        return x

    weighted = weights is not None

    # N, V1, sum
    d = cf_mean_chunk(x, weights, dtype=dtype, **kwargs)

    wsum = d["sum"]
    V1 = d["V1"]

    avg = divide(wsum, V1, dtype=dtype)
    part = x - avg
    part *= part
    if weighted:
        part = part * weights

    part = chunk.sum(part, dtype=dtype, **kwargs)
    part = part + avg * wsum

    d["part"] = part

    if weighted and ddof == 1:
        d["V2"] = sum_weights_chunk(
            x, weights, square=True, check_weights=False, **kwargs
        )
    else:
        d["V2"] = None

    d["weighted"] = weighted
    d["ddof"] = ddof

    return d


def cf_var_combine(
    pairs, axis=None, dtype="f8", computing_meta=False, **kwargs
):
    """Combination calculations for the variance.

    This function is passed to `dask.array.reduction` as its *combine*
    parameter.

    .. versionadded:: 3.14.0

    :Parameters:

        See `dask.array.reductions` for details of the parameters.

    :Returns:

        As for `cf_var_chunk`.

    """
    if not isinstance(pairs, list):
        pairs = [pairs]

    d = next(flatten(pairs))
    weighted = d["weighted"]
    ddof = d["ddof"]
    d = {"weighted": weighted, "ddof": ddof}

    d["part"] = sum_arrays(
        pairs, "part", axis, dtype, computing_meta, **kwargs
    )
    if computing_meta:
        return d["part"]

    d["sum"] = sum_arrays(pairs, "sum", axis, dtype, **kwargs)

    d["N"] = sum_sample_sizes(pairs, axis, **kwargs)
    d["V1"] = d["N"]
    d["V2"] = None
    if weighted:
        d["V1"] = sum_arrays(pairs, "V1", axis, dtype, **kwargs)
        if ddof == 1:
            d["V2"] = sum_arrays(pairs, "V2", axis, dtype, **kwargs)

    return d


def cf_var_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=1,
    original_shape=None,
    **kwargs,
):
    """Aggregation calculations for the variance.

    This function is passed to `dask.array.reduction` as its
    *aggregate* parameter.

    .. note:: Weights are interpreted as reliability weights, as
              opposed to frequency weights.

    See
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    for details.

    .. versionadded:: 3.14.0

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for details of the other
        parameters.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_var_combine(pairs, axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d

    ddof = d["ddof"]
    V1 = d["V1"]
    wsum = d["sum"]
    var = d["part"] - wsum * wsum / V1

    # Note: var is now the global value of V1 * sigma**2, where sigma
    #       is the global weighted biased (i.e. ddof=0) variance.

    if ddof is None:
        raise ValueError(f"Must set ddof to a numeric value. Got: {ddof!r}")

    if not ddof:
        # Weighted or unweighted variance with ddof=0
        f = 1 / V1
    elif not d["weighted"]:
        # Unweighted variance with any non-zero value of ddof
        f = 1 / (V1 - ddof)
    elif ddof == 1:
        # Weighted variance with ddof=1
        f = V1 / (V1 * V1 - d["V2"])
    else:
        raise ValueError(
            "Can only calculate a weighted variance with ddof=0 or ddof=1. "
            f"Got: {ddof!r}"
        )

    # Now get the required global variance with the requested ddof
    var = f * var

    var = mask_small_sample_size(var, d["N"], axis, mtol, original_shape)
    return var
