from functools import partial, reduce
from operator import mul

import numpy as np
from cfdm.core import DocstringRewriteMeta
from dask.array import chunk
from dask.array.core import _concatenate2
from dask.array.reductions import divide, numel, reduction
from dask.core import flatten
from dask.utils import deepmap  # Apply function inside nested lists


class Collapse(metaclass=DocstringRewriteMeta):
    """Container for functions that collapse `dask` arrays.

    .. versionadded:: TODODASK

    """

    @staticmethod
    def max(a, axis=None, keepdims=False, mtol=None, split_every=None):
        """Return maximum values of an array.

        Calculates the maximum value of an array or the maximum values
        along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
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
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @staticmethod
    def max_abs(a, axis=None, keepdims=False, mtol=None, split_every=None):
        """Return maximum absolute values of an array.

        Calculates the maximum absolute value of an array or the
        maximum absolute values along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
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
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @staticmethod
    def mean(
        a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
    ):
        """Return mean values of an array.

        Calculates the mean value of an array or the mean values along
        axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        dtype = "f8"
        return reduction(
            a,
            cf_mean_chunk,
            partial(cf_mean_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_mean_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )

    @staticmethod
    def mean_abs(
        a, weights=None, axis=None, keepdims=False, mtol=None, split_every=None
    ):
        """"Return mean absolute values of an array.

        Calculates the mean absolute value of an array or the mean
        absolute values along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        dtype = "f8"
        return reduction(
            a,
            cf_mean_abs_chunk,
            partial(cf_mean_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_mean_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )

    @staticmethod
    def mid_range(
        a, axis=None, dtype=None, keepdims=False, mtol=None, split_every=None
    ):
        """Return mid-range values of an array.

        Calculates the mid-range value of an array or the mid-range
        values along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        dtype = "f8"
        return reduction(
            a,
            cf_range_chunk,
            partial(cf_mid_range_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_range_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @staticmethod
    def min(a, axis=None, keepdims=False, mtol=None, split_every=None):
        """Return minimum values of an array.

        Calculates the minimum value of an array or the minimum values
        along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
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
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @staticmethod
    def min_abs(a, axis=None, keepdims=False, mtol=None, split_every=None):
        """Return minimum absolute values of an array.

        Calculates the minimum absolute value of an array or the
        minimum absolute values along axes.

        .. versionadded:: TODODASK


        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
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
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @staticmethod
    def range(a, axis=None, keepdims=False, mtol=None, split_every=None):
        """Return range values of an array.

        Calculates the range value of an array or the range values
        along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
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
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @staticmethod
    def rms(
        a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
    ):
        """Return root mean square (RMS) values of an array.

        Calculates the RMS value of an array or the RMS values along
        axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        dtype = "f8"
        return reduction(
            a,
            cf_rms_chunk,
            partial(cf_rms_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_mean_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )

    @staticmethod
    def sample_size(a, axis=None, keepdims=False, mtol=None, split_every=None):
        """Return sample size values of an array.

        Calculates the sample size value of an array or the sample
        size values along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        dtype = "i8"
        return reduction(
            a,
            cf_sample_size_chunk,
            partial(cf_sample_size_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_sample_size_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @staticmethod
    def sum(
        a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
    ):
        """Return sum values of an array.

        Calculates the sum value of an array or the sum values along
        axes.

        .. versionadded:: TODODASK


        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        if weights is None:
            dtype = double_precision_dtype(a)
        else:
            dtype = "f8"

        return reduction(
            a,
            cf_sum_chunk,
            partial(cf_sum_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_sum_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )

    @staticmethod
    def sum_of_squares(
        a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
    ):
        """Return sum of square values of an array.

        Calculates the sum of square value of an array or the sum of
        square values along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        if weights is None:
            dtype = double_precision_dtype(a)
        else:
            dtype = "f8"

        return reduction(
            a,
            partial(cf_sum_chunk, squared=True),
            partial(cf_sum_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_sum_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )

    @staticmethod
    def sum_of_weights(
        a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
    ):
        """Return sum of weights values for an array.

        Calculates the sum of weights value for an array or the sum of
        weights values along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        dtype = "f8"
        return reduction(
            a,
            cf_sum_of_weights_chunk,
            partial(cf_sum_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_sum_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )

    @staticmethod
    def sum_of_weights2(
        a, axis=None, weights=None, keepdims=False, mtol=None, split_every=None
    ):
        """Return sum of squares of weights values for an array.

        Calculates the sum of squares of weights value for an array or
        the sum of squares of weights values along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        dtype = "f8"
        return reduction(
            a,
            partial(cf_sum_of_weights_chunk, squared=True),
            partial(cf_sum_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_sum_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )

    @staticmethod
    def var(
        a,
        axis=None,
        weights=None,
        keepdims=False,
        mtol=None,
        ddof=None,
        split_every=None,
    ):
        """Return variances of an array.

        Calculates the variance value of an array or the variance
        values along axes.

        .. versionadded:: TODODASK

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}

            {{ddof: number}}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        dtype = "f8"
        return reduction(
            a,
            partial(cf_var_chunk, ddof=ddof),
            partial(cf_var_agg, mtol=mtol, ddof=ddof, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_var_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )


def double_precision_dtype(a):
    """Return the double precision data type of an aray.

    :Parameters:

        a: array_like

    :Returns:

        `str`
            The double precision type.

    **Examples**

    >>> for d in (float, 'float32', int, 'int32'):
    ...     print(double_precision_dtype(np.array(9, dtype=d)))
    ...
    f8
    f8
    i8
    i8

    """
    return a.dtype.kind + "8"


def mask_small_sample_size(x, N, axis, mtol, original_shape):
    """Mask elements where the sample size of the collapsed data is
    below a threshold.

    .. versionadded:: TODODASK

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
            A missing datum in the output array occurs whenever at
            least ``100*mtol%`` of its contributing input array
            elements are non-missing data. The default of *mtol* is 1,
            meaning that a missing datum in the output array only
            occurs when all of its contributing input array elements
            are missing data. A value of 0 means that a missing datum
            in the output array occurs whenever any of its
            contributing input array elements are missing. Any
            intermediate value is allowed. Note that for non-zero
            values of *mtol*, different collapsed elements may have
            different sample sizes, depending on the distribution of
            missing data in the input data.

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


def sum_weights(x, weights=None, squared=False, dtype="f8", N=None, **kwargs):
    """TODO.

    .. versionadded:: TODODASK

    :Parameters:

        squared: `bool`, optional
            If True calculate the sum of the squares of the weights.

    """
    if weights is None:
        # All weights are 1, so the sum of the weights and the sum of
        # the squares of the weights are both equal to the sample
        # size.
        if N is None:
            return cf_sample_size_chunk(x, dtype=dtype, **kwargs)["N"]

        return N.astype(dtype)

    if squared:
        weights = np.multiply(weights, weights, dtype=dtype)

    if np.ma.is_masked(x):
        weights = np.ma.masked_where(x.mask, weights)

    return chunk.sum(weights, dtype=dtype, **kwargs)


def combine_arrays(
    pairs, key, func, axis, dtype=None, computing_meta=False, **kwargs
):
    """Worker function for Combine functions.

    Select arrays by dictionary key from a nested list of
    dictionaries, concatenate the resulting a nested list of arrays
    along the axes, and apply a function to the result along the same
    axes.

    :Returns:

        `numpy.ndarray`

    """
    x = deepmap(lambda pair: pair[key], pairs) if not computing_meta else pairs

    if dtype:
        kwargs["dtype"] = dtype

    x = _concatenate2(x, axes=axis)
    return func(x, axis=axis, **kwargs)


def sum_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``func=chunk.sum``."""
    return combine_arrays(
        pairs, key, chunk.sum, axis, dtype, computing_meta, **kwargs
    )


def max_arrays(pairs, key, axis, dtype, computing_meta=False, **kwargs):
    """Alias of `combine_arrays` with ``func=chunk.max``.

    .. versionadded:: TODODASK

    """
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
    dtype="i8", computing_meta=False``.

    .. versionadded:: TODODASK

    """
    return combine_arrays(
        pairs, "N", chunk.sum, axis, dtype="i8", computing_meta=False, **kwargs
    )


# --------------------------------------------------------------------
# mean
# --------------------------------------------------------------------
def cf_mean_chunk(x, weights=None, dtype="f8", computing_meta=False, **kwargs):
    """Chunk calculations for the mean.

     This function is passed to `dask.array.reduction` as callable
     *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:
            * N: The sample size.
            * V1: The sum of ``weights`` (equal to ``N`` if weights
                  are not set).
            * sum: The weighted sum of ``x``.

    """
    if computing_meta:
        return x

    # N, sum
    d = cf_sum_chunk(x, weights, dtype=dtype, **kwargs)

    d["V1"] = sum_weights(x, weights, N=d["N"], **kwargs)

    return d


def cf_mean_combine(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    **kwargs,
):
    """Combine calculations for the mean.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

    :Returns:

        As for `cf_mean_chunk`.

    """
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
    """"Aggregate calculations for the mean.

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

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
# mean_absolute_value
# --------------------------------------------------------------------
def cf_mean_abs_chunk(
    x, weights=None, dtype=None, computing_meta=False, **kwargs
):
    """Chunk calculations for the mean of the absolute values.

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:
            * N: The sample size.
            * V1: The sum of ``weights`` (equal to ``N`` if weights
                  are not set).
            * sum: The weighted sum of ``abs(x)``.

    """
    if computing_meta:
        return x

    return cf_mean_chunk(np.abs(x), weights, dtype=dtype, **kwargs)


# --------------------------------------------------------------------
# maximum
# --------------------------------------------------------------------
def cf_max_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Chunk calculations for the maximum.

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

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
    """Combine  calculations for the maximum.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

    :Returns:

        As for `cf_max_chunk`.

    """
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
    """Aggregate calculations for the maximum.

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

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
# maximum_absolute_value
# --------------------------------------------------------------------
def cf_max_abs_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Chunk calculations for the maximum of absolute values.

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

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


# --------------------------------------------------------------------
# mid-range
# --------------------------------------------------------------------
def cf_mid_range_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=None,
    original_shape=None,
    **kwargs,
):
    """Aggregate calculations for the mid-range.

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

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

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

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
    """Combine calculations for the minimum.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

    :Returns:

        As for `cf_min_chunk`.

    """
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
    """Aggregate calculations for the minimum.

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

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
# minimum absolute value
# --------------------------------------------------------------------
def cf_min_abs_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Chunk calculations for the minimum of absolute values.

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

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


# --------------------------------------------------------------------
# range
# --------------------------------------------------------------------
def cf_range_chunk(x, dtype=None, computing_meta=False, **kwargs):
    """Chunk calculations for the range.

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

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
    """Combine calculations for the range.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

    :Returns:

        As for `cf_range_chunk`.

    """
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
    """Aggregate calculations for the range.

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

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
    """Chunk calculations for the root mean square (RMS)..

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

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
    """Aggregate calculations for the root mean square (RMS).

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

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

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

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
    dtype="i8",
    computing_meta=False,
    **kwargs,
):
    """Combine calculations for the sample size.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

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
    mtol=None,
    original_shape=None,
    **kwargs,
):
    """Aggregate calculations for the sample size.

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

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
    x, weights=None, dtype="f8", computing_meta=False, squared=False, **kwargs
):
    """Chunk calculations for the sum.

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        squared: `bool`, optional
            If True then calculate the weighted sum of the squares.

        See `dask.array.reductions` for details.

    :Returns:

        `dict`
            Dictionary with the keys:
            * N: The sample size.
            * sum: The weighted sum of ``x``, or the weighted sum of
                   ``x**2`` if *squared* is True.

    """
    if computing_meta:
        return x

    if squared:
        x = np.multiply(x, x, dtype=dtype)

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
    """Combine calculations for the sum.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

    :Returns:

        As for `cf_sum_chunk`.

    """
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
    """Aggregate calculations for the sum.

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

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
    x, weights=None, dtype="f8", computing_meta=False, squared=False, **kwargs
):
    """Chunk calculations for the sum of the weights.

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    :Parameters:

        squared: `bool`, optional
            If True then calculate the sum of the squares of the
            weights.

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
    d["sum"] = sum_weights(x, weights=weights, squared=squared,
                           N=d["N"], **kwargs )
    return d


# --------------------------------------------------------------------
# variance
# --------------------------------------------------------------------
def cf_var_chunk(
    x, weights=None, dtype="f8", computing_meta=False, ddof=None, **kwargs
):
    """Chunk calculations for the variance.

    This function is passed to `dask.array.reduction` as callable
    *chunk* parameter.

    See
    https://en.wikipedia.org/wiki/Pooled_variance#Sample-based_statistics
    for details.

    .. versionadded:: TODODASK

    :Parameters:

        ddof: number
            The delta degrees of freedom. The number of degrees of
            freedom used in the calculation is (N-*ddof*) where N
            represents the number of non-missing elements. A value of
            1 applies Bessel's correction.

        See `dask.array.reductions` for further details.

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * V1: The sum of ``weights`` (equal to ``N`` if weights
                  are not set).
            * sum: The weighted sum of ``x``.
            * part: ``V1 * (sigma**2 + mu**2)``, where ``sigma**2`` is
                    the weighted biased (i.e. ``ddof=0``) variance of
                    ``x``, and ``mu`` is the weighted mean of ``x``.
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
        d["V2"] = sum_weights(x, weights, squared=True, **kwargs)

    return d


def cf_var_combine(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    **kwargs,
):
    """Combine calculations for the variance.

    .. versionadded:: TODODASK

    :Parameters:

        See `dask.array.reductions` for details.

    :Returns:

        As for `cf_var_chunk`.

    """
    if not isinstance(pairs, list):
        pairs = [pairs]

    weighted = "V2" in flatten(pairs)

    keys = ("part", "sum")
    if weighted:
        keys += ("V1", "V2")

    d = {}
    for key in keys:
        d[key] = sum_arrays(pairs, key, axis, dtype, computing_meta, **kwargs)
        if computing_meta:
            return d[key]

    d["N"] = sum_sample_sizes(pairs, axis, **kwargs)

    if not weighted:
        d["V1"] = d["N"].astype("f8")

    return d


def cf_var_agg(
    pairs,
    axis=None,
    dtype="f8",
    computing_meta=False,
    mtol=None,
    ddof=None,
    original_shape=None,
    **kwargs,
):
    """Aggregate calculations for the variance.

    This function is passed to `dask.array.reduction` as callable
    *aggregate* parameter.

    .. note:: If weights are provided then they are interpreted as
              reliability weights, as opposed to frequency weights.

    See
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    for details

    .. versionadded:: TODODASK

    :Parameters:

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. See `mask_small_sample_size` for
            details.

        ddof: number
            The delta degrees of freedom. The number of degrees of
            freedom used in the calculation is (N-*ddof*) where N
            represents the number of non-missing elements. A value of
            1 applies Bessel's correction.

        original_shape: `tuple`
            The shape of the original, uncollapsed data.

        See `dask.array.reductions` for further details.

    :Returns:

        `dask.array.Array`
            The collapsed array.

    """
    d = cf_var_combine(pairs, axis, dtype, computing_meta, **kwargs)
    if computing_meta:
        return d

    V1 = d["V1"]
    wsum = d["sum"]
    var = d["part"] - wsum * wsum / V1

    # Note: var is now the global value of V1 * sigma**2, where sigma
    #       is the global weighted biased (i.e. ddof=0) variance.

    V2 = d.get("V2")
    weighted = V2 is not None
    
    if ddof is None:
        raise ValueError(f"Must set ddof to a numeric value. Got: {ddof!r}")

    if not ddof:
        # Weighted or unweighted variance with ddof=0
        f = 1 / V1
    elif not weighted:
        # Unweighted variance with any non-zero value of ddof
        f = 1 / (V1 - ddof)
    elif ddof == 1:
        # Weighted variance with ddof=1
        f = V1 / (V1 * V1 - V2)
    else:
        raise ValueError(
            "Can only calculate a weighted variance with ddof=0 or ddof=1. "
            f"Got: {ddof!r}"
        )

    # Now get the required global variance
    var = f * var

    var = mask_small_sample_size(var, d["N"], axis, mtol, original_shape)
    return var
