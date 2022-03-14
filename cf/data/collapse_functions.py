from functools import partial, reduce
from operator import mul

import numpy as np
from dask.array import chunk
from dask.array.core import _concatenate2
from dask.array.reductions import divide, numel, reduction
from dask.core import flatten
from dask.utils import deepmap  # Apply function inside nested lists


def mask_small_sample_size(x, N, axis, mtol, original_shape):
    """Mask elements where the sample size of the collapsed data is
    below a threshold.

    :Parameters:

        x: `numpy.ndarray`
            The collapsed data.

        N: `numpy.ndarray`
            The sample sizes of the collapsed values.

        axis: sequence of `int`
            The axes being collapsed.

        mtol: number

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

    :Returns:

        `numpy.ndarray`
            Array *x* masked where *N* is sufficiently small. Note
            that input *x* may be modified in-place with the output.

    """
    if not x.ndim:
        # Make sure that we have a numpy array (e.g. as opposed to a
        # numpy.float64)
        x = np.asanyarray(x)

    if mtol < 1:
        # Nmax = total number of element, including an missing values
        Nmax = reduce(mul, [original_shape[i] for i in axis], 1)
        x = np.ma.masked_where(N < (1 - mtol) * Nmax, x, copy=False)

    return x


def sum_of_weights(
    x, weights=None, squared=False, dtype="f8", N=None, **kwargs
):
    """TODO."""
    if weights is None:
        if N is None:
            N = cf_sample_size_chunk(x, **kwargs)["N"]

        return N

    if squared:
        weights = np.multiply(weights, weights, dtype=dtype)

    if np.ma.is_masked(x):
        weights = np.ma.masked_where(x.mask, weights)

    return chunk.sum(weights, dtype=dtype, **kwargs)


def combine_arrays(
    pairs, key, func, axis, dtype=None, computing_meta=False, **kwargs
):
    # Create a nested list of N and recursively concatenate it
    # along the specified
    x = deepmap(lambda pair: pair[key], pairs) if not computing_meta else pairs

    if dtype:
        kwargs["dtype"] = dtype

    return func(_concatenate2(x, axes=axis), axis=axis, **kwargs)


def sum_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``func=chunk.sum``."""
    return combine_arrays(
        pairs, key, chunk.sum, axis, dtype, computing_meta, **kwargs
    )


def max_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``func=chunk.max``."""
    return combine_arrays(
        pairs, key, chunk.max, axis, dtype, computing_meta, **kwargs
    )


def min_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``func=chunk.min``."""
    return combine_arrays(
        pairs, key, chunk.min, axis, dtype, computing_meta, **kwargs
    )


def sum_sample_sizes(pairs, axis, **kwargs):
    """Alias of `combine_arrays` with ``key="N", func=chunk.sum,
    dtype="i8", computing_meta=False``."""
    return combine_arrays(
        pairs, "N", chunk.sum, axis, dtype="i8", computing_meta=False, **kwargs
    )


# --------------------------------------------------------------------
# mean
# --------------------------------------------------------------------
def cf_mean_chunk(x, weights=None, dtype="f8", computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global mean.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * V1: The sum of ``weights`` (set to ``N`` if weights are
                  not present).

            * sum: The weighted sum of ``x``.

    """
    if computing_meta:
        return x

    # N, sum
    d = cf_sum_chunk(x, weights, dtype=dtype, **kwargs)

    if weights is None:
        d["V1"] = d["N"]
    else:
        d["V1"] = chunk.sum(weights, dtype=dtype, **kwargs)

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
    for key in ("sum", "V1"):
        d[key] = sum_arrays(pairs, key, axis, dtype, computing_meta, **kwargs)
        if computing_meta:
            return d[key]

    d["N"] = sum_sample_sizes(pairs, axis, **kwargs)
    return d


def cf_mean_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=None,
    original_shape=None,
    **kwargs,
):
    """Apply the function to the data in a nested list of arrays and
    mask where the sample size is below the threshold."""
    d = cf_mean_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = divide(d["sum"], d["V1"], dtype=dtype)
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


def cf_mean(
    a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
):
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
        weights=weights,
    )


# --------------------------------------------------------------------
# mean_absolute_value
# --------------------------------------------------------------------
def cf_mean_abs_chunk(
    x, weights=None, dtype=None, computing_meta=False, **kwargs
):
    """Return chunk-based values for calculating the global absolute
    mean.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * V1: The sum of the weights (set to N if weights are not
                  present).

            * sum: The weighted sum of ``abs(x)``.

    """
    if computing_meta:
        return x

    return cf_mean_chunk(np.abs(x), weights, dtype=dtype, **kwargs)


def cf_mean_abs(
    a, weights=None, axis=None, keepdims=False, mtol=None, split_every=None
):
    """TODODASK."""
    dtype = a.dtype
    return reduction(
        a,
        cf_mean_abs_chunk,
        partial(cf_mean_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_mean_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
        weights=weights,
    )


# --------------------------------------------------------------------
# maximum
# --------------------------------------------------------------------
def cf_max_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global maximum.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction. Ignored.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

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
    m = max_arrays(pairs, "max", axis, None, computing_meta, **kwargs)
    if computing_meta:
        return m

    return {
        "max": m,
        "N": sum_sample_sizes(pairs, axis, **kwargs),
    }


def cf_max_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=None,
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


def cf_max(a, axis=None, keepdims=False, mtol=None, split_every=None):
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
def cf_max_abs_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global absolute
    max.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * max: The maximum of ``abs(x)``.

    """
    if computing_meta:
        return x

    return cf_max_chunk(np.abs(x), **kwargs)


def cf_max_abs(a, axis=None, keepdims=False, mtol=None, split_every=None):
    """TODODASK."""
    dtype = a.dtype
    return reduction(
        a,
        cf_max_abs_chunk,
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
# minimum
# --------------------------------------------------------------------
def cf_min_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global minimum.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

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
    x = min_arrays(pairs, "min", axis, None, computing_meta, **kwargs)
    if computing_meta:
        return x

    return {
        "min": x,
        "N": sum_sample_sizes(pairs, axis, **kwargs),
    }


def cf_min_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=None,
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


def cf_min(a, axis=None, keepdims=False, mtol=None, split_every=None):
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
def cf_min_abs_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global absolute
    min.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * min: The minimum of ``abs(x)``.

    """
    if computing_meta:
        return x

    return cf_min_chunk(np.abs(x), **kwargs)


def cf_min_abs(a, axis=None, keepdims=False, mtol=None, split_every=None):
    """TODODASK."""
    dtype = a.dtype
    return reduction(
        a,
        cf_min_abs_chunk,
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
# range
# --------------------------------------------------------------------
def cf_range_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global range.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * min: The minimum of ``x``.

            * max: The maximum of ``x`.

    """
    if computing_meta:
        return x

    d = cf_max_chunk(x, **kwargs)
    d["min"] = chunk.min(x, **kwargs)
    return d


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
    mx = max_arrays(pairs, "max", axis, None, computing_meta, **kwargs)
    if computing_meta:
        return mx

    mn = min_arrays(pairs, "min", axis, None, computing_meta, **kwargs)

    return {
        "max": mx,
        "min": mn,
        "N": sum_sample_sizes(pairs, axis, **kwargs),
    }


def cf_range_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=None,
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


def cf_range(a, axis=None, keepdims=False, mtol=None, split_every=None):
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
    mtol=None,
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
    a, axis=None, dtype=float, keepdims=False, mtol=None, split_every=None
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
# root mean square
# --------------------------------------------------------------------
def cf_rms_chunk(x, weights=None, dtype="f8", computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global RMS.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

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
    mtol=None,
    original_shape=None,
    **kwargs,
):
    """Apply the function to the data in a nested list of arrays and
    mask where the sample size is below the threshold."""
    d = cf_sum_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = np.sqrt(d["sum"], dtype=dtype)
    x = mask_small_sample_size(x, d["N"], axis, mtol, original_shape)
    return x


def cf_rms(
    a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
):
    """TODODASK."""
    dtype = float
    return reduction(
        a,
        cf_rms_chunk,
        partial(cf_rms_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_sum_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
        weights=weights,
    )


# --------------------------------------------------------------------
# sample_size
# --------------------------------------------------------------------
def cf_sample_size_chunk(x, dtype="i8", computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global sample size.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

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
    pairs,
    axis=None,
    computing_meta=False,
    **kwargs,
):
    if not isinstance(pairs, list):
        pairs = [pairs]

    x = sum_arrays(pairs, "N", axis, None, computing_meta, **kwargs)
    if computing_meta:
        return x

    return {"N": x}


def cf_sample_size_agg(
    pairs,
    axis=None,
    computing_meta=False,
    mtol=None,
    original_shape=None,
    **kwargs,
):
    d = cf_sample_size_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d

    x = d["N"]
    x = mask_small_sample_size(x, x, axis, mtol, original_shape)
    return x


def cf_sample_size(a, axis=None, keepdims=False, mtol=None, split_every=None):
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
def cf_sum_chunk(x, weights=None, dtype="f8", computing_meta=False, **kwargs):
    """Return chunk-based values for calculating the global sum.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * sum: The weighted sum of ``x``.

    """
    if computing_meta:
        return x

    if weights is not None:
        x = np.multiply(x, weights, dtype=dtype)

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
        "N": sum_sample_sizes(pairs, axis, **kwargs),
    }


def cf_sum_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=None,
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


def cf_sum(
    a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
):
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
        weights=weights,
    )


# --------------------------------------------------------------------
# sum of sqaures
# --------------------------------------------------------------------
def cf_sum_of_squares_chunk(
    x, weights=None, dtype="f8", computing_meta=False, **kwargs
):
    """Return chunk-based values for calculating the global sum.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * sum: The weighted sum of ``x**2``

    """
    if computing_meta:
        return x

    return cf_sum_chunk(
        np.multiply(x, x, dtype=dtype), weights, dtype=dtype, **kwargs
    )


def cf_sum_of_squares(
    a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
):
    """TODODASK."""
    dtype = float
    return reduction(
        a,
        cf_sum_of_squares_chunk,
        partial(cf_sum_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_sum_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
        weights=weights,
    )


# --------------------------------------------------------------------
# sum of weights
# --------------------------------------------------------------------
def cf_sum_of_weights_chunk(
    x, weights=None, dtype="f8", computing_meta=False, squared=False, **kwargs
):
    """Return chunk-based values for calculating the global sum.

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * sum: The sum of ``weights``, or the sum of
                   ``weights**2`` if *squared* is True.

    """
    if computing_meta:
        return x

    # N
    d = cf_sample_size_chunk(x, **kwargs)

    # sum
    d["sum"] = sum_of_weights(
        x, weights=weights, dtype=dtype, N=d["N"], squared=squared, **kwargs
    )
    return d


def cf_sum_of_weights(
    a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
):
    """TODODASK."""
    dtype = float
    return reduction(
        a,
        cf_sum_of_weights_chunk,
        partial(cf_sum_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_sum_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
        weights=weights,
    )


def cf_sum_of_weights2(
    a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
):
    """TODODASK."""
    dtype = float
    return reduction(
        a,
        partial(cf_sum_of_weights_chunk, squared=True),
        partial(cf_sum_agg, mtol=mtol, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_sum_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
        weights=weights,
    )


# --------------------------------------------------------------------
# variance
# --------------------------------------------------------------------
def cf_var_chunk(
    x, weights=None, dtype="f8", computing_meta=False, ddof=None, **kwargs
):
    """Return chunk-based values for calculating the global variance.

    .. note:: If weights are provided then they are interpreted as
              reliability weights, as opposed to frequency weights
              (where a weight equals the number of occurrences).

    :Parameters:

        x: numpy.ndarray
            Chunks data being reduced along one or more axes.

        weights: numpy array-like, optional
            Weights to be used in the reduction of *x*, with the same
            shape as *x*. By default the reduction is unweighted.

        dtype: data_type, optional
            Data type of global reduction.

        ddof: number, optional
            The delta degrees of freedom. The number of degrees of
            freedom used in the calculation is (N-*ddof*) where N
            represents the number of non-missing elements. By default
            *ddof* is 0, for the biased variance. Setting ddof to
            ``1`` applies Bessel's correction
            (https://en.wikipedia.org/wiki/Bessel's_correction)

        computing_meta: `bool` optional
            See `dask.array.reductions` for details.

        kwargs: `dict`, optional
            See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.

            * V1: The sum of ``weights`` (set to ``N`` if weights are
                  not set).

            * sum: The weighted sum of ``x``.

            * part: ``V1 * (sigma**2 + mu**2)``, where ``sigma**2`` is
                    the weighted biased (i.e. ``ddof=0``) variance of
                    ``x``, and ``mu`` is the weighted mean of
                    ``x``. See
                    https://en.wikipedia.org/wiki/Pooled_variance#Sample-based_statistics
                    for details.

            * V2: The sum of ``weights**2``. Only present if *weights*
                  are set and ``ddof=1``.

    """
    if computing_meta:
        return x

    # N, V1, sum
    d = cf_mean_chunk(x, weights, dtype=dtype, **kwargs)

    wsum = d["sum"]
    V1 = d["V1"]

    # part
    avg = divide(wsum, V1, dtype=dtype)
    part = x - avg
    part *= part
    if weights is not None:
        part = part * weights

    part = chunk.sum(part, dtype=dtype, **kwargs)
    part = part + avg * wsum
    d["part"] = part

    if weights is not None and ddof == 1:
        d["V2"] = sum_of_weights(x, weights, squared=True, **kwargs)

    return d


def cf_var_combine(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    **kwargs,
):
    """TODO."""
    d = {}

    weighted = "V2" in flatten(pairs)

    keys = ("part", "sum")
    if weighted:
        keys += ("V1", "V2")

    for key in keys:
        d[key] = sum_arrays(pairs, key, axis, dtype, computing_meta, **kwargs)
        if computing_meta:
            return d[key]

    d["N"] = sum_sample_sizes(pairs, axis, **kwargs)

    if not weighted:
        d["V1"] = d["N"]

    return d


def cf_var_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=None,
    original_shape=None,
    ddof=None,
    **kwargs,
):
    """TODO."""
    d = cf_var_combine(pairs, axis, computing_meta, **kwargs)
    if computing_meta:
        return d

    V1 = d["V1"]
    V2 = d.get("V2")
    weighted = V2 is not None

    wsum = d["sum"]
    var = d["part"] - wsum * wsum / V1

    # Note: var is currently the global value of V1 * sigma**2, where
    # sigma is the global weighted biased (i.e. ddof=0) variance.

    if ddof == 0:  # intended equality with zero
        # Weighted or unweighted variance with ddof=0
        f = 1 / V1
    elif not weighted:
        # Unweighted variance with any non-zero value of ddof.
        f = 1 / (V1 - ddof)
    elif ddof == 1:
        # Weighted variance with ddof=1. For details see
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
        f = V1 / (V1 * V1 - V2)
    else:
        raise ValueError(
            "Can only calculate a weighted variance with ddof=0 or ddof=1: "
            f"Got {ddof!r}"
        )

    # Calculate the global variance, with the specified weighting and
    # ddof.
    var = f * var

    # Note: var is now the global value of sigma**2

    var = mask_small_sample_size(var, d["N"], axis, mtol, original_shape)
    return var


def cf_var(
    a,
    axis=None,
    weights=None,
    keepdims=False,
    mtol=None,
    ddof=None,
    split_every=None,
):
    """TODODASK."""
    dtype = float
    return reduction(
        a,
        partial(cf_var_chunk, ddof=ddof),
        partial(cf_var_agg, mtol=mtol, ddof=ddof, original_shape=a.shape),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        combine=cf_var_combine,
        out=None,
        concatenate=False,
        meta=np.array((), dtype=dtype),
        weights=weights,
    )
