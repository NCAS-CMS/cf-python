"""Functions used for `Data` object collapses."""
from functools import partial

import numpy as np
from cfdm.core import DocstringRewriteMeta
from dask.array.reductions import reduction

from ..docstring import _docstring_substitution_definitions
from .collapse_utils import actify, check_input_dtype, double_precision_dtype


class Collapse(metaclass=DocstringRewriteMeta):
    """Container for functions that collapse dask arrays.

    .. versionadded:: TODODASKVER

    """

    def __docstring_substitutions__(self):
        """Define docstring substitutions that apply to this class and
        all of its subclasses.

        These are in addtion to, and take precendence over, docstring
        substitutions defined by the base classes of this class.

        See `_docstring_substitutions` for details.

        .. versionadded:: TODODASKVER

        .. seealso:: `_docstring_substitutions`

        :Returns:

            `dict`
                The docstring substitutions that have been applied.

        """
        return _docstring_substitution_definitions

    def __docstring_package_depth__(self):
        """Return the package depth for "package" docstring
        substitutions.

        See `_docstring_package_depth` for details.

        .. versionadded:: TODODASKVER

        """
        return 0

    @classmethod
    def max(
        cls,
        a,
        axis=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return maximum values of an array.

        Calculates the maximum value of an array or the maximum values
        along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import cf_max_agg, cf_max_chunk, cf_max_combine

        check_input_dtype(a)
        dtype = a.dtype

        # Rewrite data and chunk function if active storage operations
        # are available.
        a, chunk_function = actify(
            a,
            method="max",
            axis=axis,
            chunk_function=cf_max_chunk,
            active_storage=active_storage,
        )

        return reduction(
            a,
            chunk_function,
            partial(cf_max_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_max_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @classmethod
    def max_abs(
        cls,
        a,
        axis=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return maximum absolute values of an array.

        Calculates the maximum absolute value of an array or the
        maximum absolute values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        return cls.max(
            abs(a),
            axis=axis,
            keepdims=keepdims,
            mtol=mtol,
            split_every=split_every,
        )

    @classmethod
    def mean(
        cls,
        a,
        axis=None,
        weights=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return mean values of an array.

        Calculates the mean value of an array or the mean values along
        axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import cf_mean_agg, cf_mean_chunk, cf_mean_combine

        check_input_dtype(a)
        dtype = "f8"

        # Rewrite data and chunk function if active storage operations
        # are available.
        a, chunk_function = actify(
            a,
            method="mean",
            axis=axis,
            chunk_function=cf_mean_chunk,
            active_storage=active_storage,
        )

        return reduction(
            a,
            chunk_function,
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

    @classmethod
    def mean_abs(
        cls,
        a,
        weights=None,
        axis=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return mean absolute values of an array.

        Calculates the mean absolute value of an array or the mean
        absolute values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        return cls.mean(
            abs(a),
            weights=weights,
            axis=axis,
            keepdims=keepdims,
            mtol=mtol,
            split_every=split_every,
        )

    @classmethod
    def mid_range(
        cls,
        a,
        axis=None,
        dtype=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return mid-range values of an array.

        Calculates the mid-range value of an array or the mid-range
        values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import (
            cf_mid_range_agg,
            cf_range_chunk,
            cf_range_combine,
        )

        check_input_dtype(a, allowed="fi")
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

    @classmethod
    def min(
        cls,
        a,
        axis=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return minimum values of an array.

        Calculates the minimum value of an array or the minimum values
        along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import cf_min_agg, cf_min_chunk, cf_min_combine

        check_input_dtype(a)
        dtype = a.dtype

        # Rewrite data and chunk function if active storage operations
        # are available.
        a, chunk_function = actify(
            a,
            method="min",
            axis=axis,
            chunk_function=cf_min_chunk,
            active_storage=active_storage,
        )

        return reduction(
            a,
            chunk_function,
            partial(cf_min_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_min_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )

    @classmethod
    def min_abs(
        cls,
        a,
        axis=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return minimum absolute values of an array.

        Calculates the minimum absolute value of an array or the
        minimum absolute values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        return cls.min(
            abs(a),
            axis=axis,
            keepdims=keepdims,
            mtol=mtol,
            split_every=split_every,
        )

    @classmethod
    def range(
        cls,
        a,
        axis=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return range values of an array.

        Calculates the range value of an array or the range values
        along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import (
            cf_range_agg,
            cf_range_chunk,
            cf_range_combine,
        )

        check_input_dtype(a, allowed="fi")
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

    @classmethod
    def rms(
        cls,
        a,
        axis=None,
        weights=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return root mean square (RMS) values of an array.

        Calculates the RMS value of an array or the RMS values along
        axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import cf_mean_combine, cf_rms_agg, cf_rms_chunk

        check_input_dtype(a)
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

    @classmethod
    def sample_size(
        cls,
        a,
        axis=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return sample size values of an array.

        Calculates the sample size value of an array or the sample
        size values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import (
            cf_sample_size_agg,
            cf_sample_size_chunk,
            cf_sample_size_combine,
        )

        check_input_dtype(a)
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

    @classmethod
    def sum(
        cls,
        a,
        axis=None,
        weights=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return sum values of an array.

        Calculates the sum value of an array or the sum values along
        axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import cf_sum_agg, cf_sum_chunk, cf_sum_combine

        check_input_dtype(a)
        dtype = double_precision_dtype(a)
        if weights is not None:
            dtype = np.result_type(double_precision_dtype(weights), dtype)

        # Rewrite data and chunk function if active storage operations
        # are available.
        a, chunk_function = actify(
            a,
            method="sum",
            axis=axis,
            chunk_function=cf_sum_chunk,
            active_storage=active_storage,
        )

        return reduction(
            a,
            chunk_function,
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

    @classmethod
    def sum_of_weights(
        cls,
        a,
        axis=None,
        weights=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return sum of weights values for an array.

        Calculates the sum of weights value for an array or the sum of
        weights values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import (
            cf_sum_agg,
            cf_sum_combine,
            cf_sum_of_weights_chunk,
        )

        check_input_dtype(a)
        dtype = double_precision_dtype(weights, default="i8")
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

    @classmethod
    def sum_of_weights2(
        cls,
        a,
        axis=None,
        weights=None,
        keepdims=False,
        mtol=1,
        split_every=None,
        active_storage=False,
    ):
        """Return sum of squares of weights values for an array.

        Calculates the sum of squares of weights value for an array or
        the sum of squares of weights values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import (
            cf_sum_agg,
            cf_sum_combine,
            cf_sum_of_weights_chunk,
        )

        check_input_dtype(a)
        dtype = double_precision_dtype(weights, default="i8")
        return reduction(
            a,
            partial(cf_sum_of_weights_chunk, square=True),
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

    @classmethod
    def var(
        cls,
        a,
        axis=None,
        weights=None,
        keepdims=False,
        mtol=1,
        ddof=None,
        split_every=None,
        active_storage=False,
    ):
        """Return variances of an array.

        Calculates the variance value of an array or the variance
        values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{Collapse weights: data_like or `None`, optional}}

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse keepdims: `bool`, optional}}

            {{mtol: number, optional}}

            {{ddof: number}}

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The collapsed array.

        """
        from .dask_collapse import cf_var_agg, cf_var_chunk, cf_var_combine

        check_input_dtype(a)
        dtype = "f8"
        return reduction(
            a,
            partial(cf_var_chunk, ddof=ddof),
            partial(cf_var_agg, mtol=mtol, original_shape=a.shape),
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            split_every=split_every,
            combine=cf_var_combine,
            concatenate=False,
            meta=np.array((), dtype=dtype),
            weights=weights,
        )

    @classmethod
    def unique(cls, a, split_every=None, active_storage=False):
        """Return unique elements of the data.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dask.array.Array`
                The array to be collapsed.

            {{split_every: `int` or `dict`, optional}}

            {{active_storage: `bool`, optional}}

        :Returns:

            `dask.array.Array`
                The unique values in a 1-d array.

        """
        from .dask_collapse import cf_unique_agg, cf_unique_chunk

        check_input_dtype(a, "fibUS")

        # Flatten the array so that it has the same number of
        # dimensions as the result (i.e. 1). This ensures that the
        # combination of `keepdims=True, output_size=np.nan` will
        # result in a correct output chunk size `np.nan`. See
        # `dask.array.reduction` for details.
        a = a.flatten()

        dtype = a.dtype
        return reduction(
            a,
            cf_unique_chunk,
            cf_unique_agg,
            keepdims=True,
            output_size=np.nan,
            dtype=dtype,
            split_every=split_every,
            concatenate=False,
            meta=np.array((), dtype=dtype),
        )
