import logging
import math
from functools import partial, reduce
from itertools import product
from operator import mul

import cfdm
import cftime
import dask.array as da
import numpy as np
from cfdm.data.dask_utils import cfdm_where
from cfdm.data.utils import new_axis_identifier
from dask import compute, delayed  # noqa: F401
from dask.array.core import normalize_chunks
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph

from ..cfdatetime import dt as cf_dt
from ..constants import masked
from ..decorators import (
    _deprecated_kwarg_check,
    _display_or_return,
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _manage_log_level_via_verbosity,
)
from ..functions import (
    _DEPRECATION_ERROR_KWARGS,
    _section,
    free_memory,
    parse_indices,
)
from ..mixin2 import Container
from ..units import Units
from .collapse import Collapse
from .dask_utils import (
    cf_contains,
    cf_dt2rt,
    cf_is_masked,
    cf_percentile,
    cf_rt2dt,
    cf_units,
)
from .mixin import DataClassDeprecationsMixin
from .utils import YMDhms, collapse, conform_units, scalar_masked_array

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
_year_length = 365.242198781
_month_length = _year_length / 12

_empty_set = set()

_units_None = Units()
_units_1 = Units("1")
_units_radians = Units("radians")

_month_units = ("month", "months")
_year_units = ("year", "years", "yr")

_dtype_float32 = np.dtype("float32")
_dtype_float = np.dtype(float)
_dtype_bool = np.dtype(bool)


class Data(DataClassDeprecationsMixin, Container, cfdm.Data):
    """An N-dimensional data array with units and masked values.

    * Contains an N-dimensional, indexable and broadcastable array with
      many similarities to a `numpy` array.

    * Contains the units of the array elements.

    * Supports masked arrays, regardless of whether or not it was
      initialised with a masked array.

    * Stores and operates on data arrays which are larger than the
      available memory.

    **Indexing**

    A data array is indexable in a similar way to numpy array:

    >>> d.shape
    (12, 19, 73, 96)
    >>> d[...].shape
    (12, 19, 73, 96)
    >>> d[slice(0, 9), 10:0:-2, :, :].shape
    (9, 5, 73, 96)

    There are three extensions to the numpy indexing functionality:

    * Size 1 dimensions are never removed by indexing.

      An integer index i takes the i-th element but does not reduce the
      rank of the output array by one:

      >>> d.shape
      (12, 19, 73, 96)
      >>> d[0, ...].shape
      (1, 19, 73, 96)
      >>> d[:, 3, slice(10, 0, -2), 95].shape
      (12, 1, 5, 1)

      Size 1 dimensions may be removed with the `squeeze` method.

    * The indices for each axis work independently.

      When more than one dimension's slice is a 1-d boolean sequence or
      1-d sequence of integers, then these indices work independently
      along each dimension (similar to the way vector subscripts work in
      Fortran), rather than by their elements:

      >>> d.shape
      (12, 19, 73, 96)
      >>> d[0, :, [0, 1], [0, 13, 27]].shape
      (1, 19, 2, 3)

    * Boolean indices may be any object which exposes the numpy array
      interface.

      >>> d.shape
      (12, 19, 73, 96)
      >>> d[..., d[0, 0, 0]>d[0, 0, 0].min()]

    **Cyclic axes**

    """

    def __new__(cls, *args, **kwargs):
        """Store component classes."""
        instance = super().__new__(cls)
        instance._Units_class = Units
        return instance

    def __contains__(self, value):
        """Membership test operator ``in``

        x.__contains__(y) <==> y in x

        Returns True if the scalar *value* is contained anywhere in
        the data. If *value* is not scalar then an exception is
        raised.

        **Performance**

        `__contains__` causes all delayed operations to be computed
        unless *value* is a `Data` object with incompatible units, in
        which case `False` is always returned.

        **Examples**

        >>> d = cf.Data([[0, 1, 2], [3, 4, 5]], 'm')
        >>> 4 in d
        True
        >>> 4.0 in d
        True
        >>> cf.Data(5) in d
        True
        >>> cf.Data(5, 'm') in d
        True
        >>> cf.Data(0.005, 'km') in d
        True

        >>> 99 in d
        False
        >>> cf.Data(2, 'seconds') in d
        False

        >>> [1] in d
        Traceback (most recent call last):
            ...
        TypeError: elementwise comparison failed; must test against a scalar, not [1]
        >>> [1, 2] in d
        Traceback (most recent call last):
            ...
        TypeError: elementwise comparison failed; must test against a scalar, not [1, 2]

        >>> d = cf.Data(["foo", "bar"])
        >>> 'foo' in d
        True
        >>> 'xyz' in d
        False

        """
        # Check that value is scalar by seeing if its shape is ()
        shape = getattr(value, "shape", None)
        if shape is None:
            if isinstance(value, str):
                # Strings are scalars, even though they have a len().
                shape = ()
            else:
                try:
                    len(value)
                except TypeError:
                    # value has no len() so assume that it is a scalar
                    shape = ()
                else:
                    # value has a len() so assume that it is not a scalar
                    shape = True
        elif is_dask_collection(value) and math.isnan(value.size):
            # value is a dask array with unknown size, so calculate
            # the size. This is acceptable, as we're going to compute
            # it anyway at the end of this method.
            value.compute_chunk_sizes()
            shape = value.shape

        if shape:
            raise TypeError(
                "elementwise comparison failed; must test against a scalar, "
                f"not {value!r}"
            )

        # If value is a scalar Data object then conform its units
        if isinstance(value, self.__class__):
            self_units = self.Units
            value_units = value.Units
            if value_units.equivalent(self_units):
                if not value_units.equals(self_units):
                    value = value.copy()
                    value.Units = self_units
            elif value_units:
                # No need to check the dask array if the value units
                # are incompatible
                return False

            # 'cf_contains' has its own calls to 'cfdm_to_memory', so
            # we can set '_force_to_memory=False'.
            value = value.to_dask_array(_force_to_memory=False)

        # 'cf_contains' has its own calls to 'cfdm_to_memory', so we
        # can set '_force_to_memory=False'.
        dx = self.to_dask_array(_force_to_memory=False)

        out_ind = tuple(range(dx.ndim))
        dx_ind = out_ind

        dx = da.blockwise(
            cf_contains,
            out_ind,
            dx,
            dx_ind,
            value,
            (),
            adjust_chunks={i: 1 for i in out_ind},
            dtype=bool,
        )

        return bool(dx.any())

    def __getitem__(self, indices):
        """Return a subspace of the data defined by indices.

        d.__getitem__(indices) <==> d[indices]

        Indexing follows rules that are very similar to the numpy indexing
        rules, the only differences being:

        * An integer index i takes the i-th element but does not reduce
          the rank by one.

        * When two or more dimensions' indices are sequences of integers
          then these indices work independently along each dimension
          (similar to the way vector subscripts work in Fortran). This is
          the same behaviour as indexing on a `netCDF4.Variable` object.

        **Performance**

        If the shape of the data is unknown then it is calculated
        immediately by executing all delayed operations.

        . seealso:: `__keepdims_indexing__`,
                    `__orthogonal_indexing__`, `__setitem__`

        :Returns:

            `Data`
                The subspace of the data.

        **Examples**

        >>> import numpy
        >>> d = Data(numpy.arange(100, 190).reshape(1, 10, 9))
        >>> d.shape
        (1, 10, 9)
        >>> d[:, :, 1].shape
        (1, 10, 1)
        >>> d[:, 0].shape
        (1, 1, 9)
        >>> d[..., 6:3:-1, 3:6].shape
        (1, 3, 3)
        >>> d[0, [2, 9], [4, 8]].shape
        (1, 2, 2)
        >>> d[0, :, -2].shape
        (1, 10, 1)

        """
        if indices is Ellipsis:
            return self.copy()

        ancillary_mask = ()
        try:
            arg = indices[0]
        except (IndexError, TypeError):
            pass
        else:
            if isinstance(arg, str) and arg == "mask":
                ancillary_mask = indices[1]
                indices = indices[2:]

        shape = self.shape
        axes = self._axes
        cyclic_axes = self._cyclic
        keepdims = self.__keepdims_indexing__

        indices, roll = parse_indices(
            shape, indices, cyclic=True, keepdims=keepdims
        )
        indices = tuple(indices)
        if roll:
            #  Roll axes with cyclic slices.
            #
            # For example, if slice(-2, 3) has been requested on a
            # cyclic axis, then we roll that axis by two points and
            # apply the slice(0, 5) instead.
            if not cyclic_axes.issuperset([axes[i] for i in roll]):
                raise IndexError(
                    "Can't take a cyclic slice of a non-cyclic axis"
                )

            d = self.roll(axis=tuple(roll.keys()), shift=tuple(roll.values()))
        else:
            d = self

        new = super(Data, d).__getitem__(indices)

        if cyclic_axes:
            # Cyclic axes that have been reduced in size are no longer
            # considered to be cyclics
            shape0 = [
                n for n, axis in zip(shape, self._axes) if axis in new._axes
            ]
            x = [
                axis
                for axis, n0, n1 in zip(new._axes, shape0, new.shape)
                if axis in cyclic_axes and n0 != n1
            ]
            if x:
                # Never change the value of the _cyclic attribute
                # in-place
                new._cyclic = cyclic_axes.difference(x)

        if ancillary_mask:
            # Apply ancillary masks
            for mask in ancillary_mask:
                new.where(mask, masked, None, inplace=True)

        return new

    def __setitem__(self, indices, value):
        """Implement indexed assignment.

        x.__setitem__(indices, y) <==> x[indices]=y

        Assignment to data array elements defined by indices.

        Elements of a data array may be changed by assigning values to
        a subspace. See `__getitem__` for details on how to define
        subspace of the data array.

        .. note:: Currently at most one dimension's assignment index
                  may be a 1-d array of integers or booleans. This is
                  is different to `__getitem__`, which by default
                  applies 'orthogonal indexing' when multiple indices
                  of 1-d array of integers or booleans are present.

        **Missing data**

        The treatment of missing data elements during assignment to a
        subspace depends on the value of the `hardmask` attribute. If
        it is True then masked elements will not be unmasked,
        otherwise masked elements may be set to any value.

        In either case, unmasked elements may be set, (including
        missing data).

        Unmasked elements may be set to missing data by assignment to
        the `cf.masked` constant or by assignment to a value which
        contains masked elements.

        **Performance**

        If the shape of the data is unknown then it is calculated
        immediately by executing all delayed operations.

        If indices for two or more dimensions are lists or 1-d arrays
        of Booleans or integers, and any of these are dask
        collections, then these dask collections will be
        computed immediately.

        .. seealso:: `__getitem__`, `__keedims_indexing__`,
                     `__orthogonal_indexing__`, `cf.masked`,
                     `hardmask`, `where`

        """
        ancillary_mask = ()
        try:
            arg = indices[0]
        except (IndexError, TypeError):
            pass
        else:
            if isinstance(arg, str) and arg == "mask":
                # The indices include an ancillary mask that defines
                # elements which are protected from assignment
                original_self = self.copy()
                ancillary_mask = indices[1]
                indices = indices[2:]

        indices, roll = parse_indices(
            self.shape,
            indices,
            cyclic=True,
            keepdims=self.__keepdims_indexing__,
        )

        if roll:
            # Roll axes with cyclic slices
            #
            # For example, if assigning to slice(-2, 3) has been
            # requested on a cyclic axis (and we're not using numpy
            # indexing), then we roll that axis by two points and
            # assign to slice(0, 5) instead. The axis is then unrolled
            # by two points afer the assignment has been made.
            axes = self._axes
            if not self._cyclic.issuperset([axes[i] for i in roll]):
                raise IndexError(
                    "Can't do a cyclic assignment to a non-cyclic axis"
                )

            roll_axes = tuple(roll.keys())
            shifts = tuple(roll.values())
            self.roll(shift=shifts, axis=roll_axes, inplace=True)

        # Make sure that the units of value are the same as self
        value = conform_units(value, self.Units)

        # Do the assignment
        indices = tuple(indices)
        super().__setitem__(indices, value)

        if roll:
            # Unroll any axes that were rolled to enable a cyclic
            # assignment
            shifts = [-shift for shift in shifts]
            self.roll(shift=shifts, axis=roll_axes, inplace=True)

        if ancillary_mask:
            # Reset the original array values at locations that are
            # excluded from the assignment by True values in any
            # ancillary masks
            original_self = original_self[indices]
            reset = self[indices]
            for mask in ancillary_mask:
                reset.where(mask, original_self, inplace=True)

            self[indices] = reset

        return

    @_inplace_enabled(default=False)
    def diff(self, axis=-1, n=1, inplace=False):
        """Calculate the n-th discrete difference along the given axis.

        The first difference is given by ``x[i+1] - x[i]`` along the
        given axis, higher differences are calculated by using `diff`
        recursively.

        The shape of the output is the same as the input except along
        the given axis, where the dimension is smaller by *n*. The
        data type of the output is the same as the type of the
        difference between any two elements of the input.

        .. versionadded:: 3.2.0

        .. seealso:: `cumsum`, `sum`

        :Parameters:

            axis: int, optional
                The axis along which the difference is taken. By
                default the last axis is used. The *axis* argument is
                an integer that selects the axis corresponding to the
                given position in the list of axes of the data array.

            n: int, optional
                The number of times values are differenced. If zero,
                the input is returned as-is. By default *n* is ``1``.

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The n-th differences, or `None` if the operation was
                in-place.

        **Examples**

        >>> d = cf.Data(numpy.arange(12.).reshape(3, 4))
        >>> d[1, 1] = 4.5
        >>> d[2, 2] = 10.5
        >>> print(d.array)
        [[ 0.   1.   2.   3. ]
         [ 4.   4.5  6.   7. ]
         [ 8.   9.  10.5 11. ]]
        >>> print(d.diff().array)
        [[1.  1.  1. ]
         [0.5 1.5 1. ]
         [1.  1.5 0.5]]
        >>> print(d.diff(n=2).array)
        [[ 0.   0. ]
         [ 1.  -0.5]
         [ 0.5 -1. ]]
        >>> print(d.diff(axis=0).array)
        [[4.  3.5 4.  4. ]
         [4.  4.5 4.5 4. ]]
        >>> print(d.diff(axis=0, n=2).array)
        [[0.  1.  0.5 0. ]]
        >>> d[1, 2] = cf.masked
        >>> print(d.array)
        [[0.0 1.0  2.0  3.0]
         [4.0 4.5   --  7.0]
         [8.0 9.0 10.5 11.0]]
        >>> print(d.diff().array)
        [[1.0 1.0 1.0]
         [0.5  --  --]
         [1.0 1.5 0.5]]
        >>> print(d.diff(n=2).array)
        [[0.0  0.0]
         [ --   --]
         [0.5 -1.0]]
        >>> print(d.diff(axis=0).array)
        [[4.0 3.5 -- 4.0]
         [4.0 4.5 -- 4.0]]
        >>> print(d.diff(axis=0, n=2).array)
        [[0.0 1.0 -- 0.0]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        dx = self.to_dask_array()
        dx = da.diff(dx, axis=axis, n=n)
        d._set_dask(dx)

        # Convert to "difference" units
        #
        # TODO: Think about temperature units in relation to
        #       https://github.com/cf-convention/discuss/issues/101,
        #       whenever that issue is resolved.
        units = self.Units
        if units.isreftime:
            units = d._Units_class(units._units_since_reftime)
            d.override_units(units, inplace=True)

        return d

    @_inplace_enabled(default=False)
    def digitize(
        self,
        bins,
        upper=False,
        open_ends=False,
        closed_ends=None,
        return_bins=False,
        inplace=False,
    ):
        """Return the indices of the bins to which each value belongs.

        Values (including masked values) that do not belong to any bin
        result in masked values in the output data.

        Bins defined by percentiles are easily created with the
        `percentiles` method

        *Example*:
          Find the indices for bins defined by the 10th, 50th and 90th
          percentiles:

          >>> bins = d.percentile([0, 10, 50, 90, 100], squeeze=True)
          >>> i = f.digitize(bins, closed_ends=True)

        .. versionadded:: 3.0.2

        .. seealso:: `percentile`

        :Parameters:

            bins: array_like
                The bin boundaries. One of:

                * An integer.

                  Create this many equally sized, contiguous bins spanning
                  the range of the data. I.e. the smallest bin boundary is
                  the minimum of the data and the largest bin boundary is
                  the maximum of the data. In order to guarantee that each
                  data value lies inside a bin, the *closed_ends*
                  parameter is assumed to be True.

                * A 1-d array of numbers.

                  When sorted into a monotonically increasing sequence,
                  each boundary, with the exception of the two end
                  boundaries, counts as the upper boundary of one bin and
                  the lower boundary of next. If the *open_ends* parameter
                  is True then the lowest lower bin boundary also defines
                  a left-open (i.e. not bounded below) bin, and the
                  largest upper bin boundary also defines a right-open
                  (i.e. not bounded above) bin.

                * A 2-d array of numbers.

                  The second dimension, that must have size 2, contains
                  the lower and upper bin boundaries. Different bins may
                  share a boundary, but may not overlap. If the
                  *open_ends* parameter is True then the lowest lower bin
                  boundary also defines a left-open (i.e. not bounded
                  below) bin, and the largest upper bin boundary also
                  defines a right-open (i.e. not bounded above) bin.

            upper: `bool`, optional
                If True then each bin includes its upper bound but not its
                lower bound. By default the opposite is applied, i.e. each
                bin includes its lower bound but not its upper bound.

            open_ends: `bool`, optional
                If True then create left-open (i.e. not bounded below) and
                right-open (i.e. not bounded above) bins from the lowest
                lower bin boundary and largest upper bin boundary
                respectively. By default these bins are not created

            closed_ends: `bool`, optional
                If True then extend the most extreme open boundary by a
                small amount so that its bin includes values that are
                equal to the unadjusted boundary value. This is done by
                multiplying it by ``1.0 - epsilon`` or ``1.0 + epsilon``,
                whichever extends the boundary in the appropriate
                direction, where ``epsilon`` is the smallest positive
                64-bit float such that ``1.0 + epsilson != 1.0``. I.e. if
                *upper* is False then the largest upper bin boundary is
                made slightly larger and if *upper* is True then the
                lowest lower bin boundary is made slightly lower.

                By default *closed_ends* is assumed to be True if *bins*
                is a scalar and False otherwise.

            return_bins: `bool`, optional
                If True then also return the bins in their 2-d form.

            {{inplace: `bool`, optional}}

        :Returns:

            `Data`, [`Data`]
                The indices of the bins to which each value belongs.

                If *return_bins* is True then also return the bins in
                their 2-d form.

        **Examples**

        >>> d = cf.Data(numpy.arange(12).reshape(3, 4))
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]

        Equivalent ways to create indices for the four bins ``[-inf, 2),
        [2, 6), [6, 10), [10, inf)``

        >>> e = d.digitize([2, 6, 10])
        >>> e = d.digitize([[2, 6], [6, 10]])
        >>> print(e.array)
        [[0 0 1 1]
         [1 1 2 2]
         [2 2 3 3]]

        Equivalent ways to create indices for the two bins ``(2, 6], (6, 10]``

        >>> e = d.digitize([2, 6, 10], upper=True, open_ends=False)
        >>> e = d.digitize([[2, 6], [6, 10]], upper=True, open_ends=False)
        >>> print(e.array)
        [[-- -- --  0]
         [ 0  0  0  1]
         [ 1  1  1 --]]

        Create indices for the two bins ``[2, 6), [8, 10)``, which are
        non-contiguous

        >>> e = d.digitize([[2, 6], [8, 10]])
        >>> print(e.array)
        [[ 0 0  1  1]
         [ 1 1 -- --]
         [ 2 2  3  3]]

        Masked values result in masked indices in the output array.

        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4 --  6  7]
         [ 8  9 10 11]]
        >>> print(d.digitize([2, 6, 10], open_ends=True).array)
        [[ 0  0  1  1]
         [ 1 --  2  2]
         [ 2  2  3  3]]
        >>> print(d.digitize([2, 6, 10]).array)
        [[-- --  0  0]
         [ 0 --  1  1]
         [ 1  1 -- --]]
        >>> print(d.digitize([2, 6, 10], closed_ends=True).array)
        [[-- --  0  0]
         [ 0 --  1  1]
         [ 1  1  1 --]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        org_units = d.Units

        bin_units = getattr(bins, "Units", None)

        if bin_units:
            if not bin_units.equivalent(org_units):
                raise ValueError(
                    "Can't put data into bins that have units that are "
                    "not equivalent to the units of the data."
                )

            if not bin_units.equals(org_units):
                bins = bins.copy()
                bins.Units = org_units
        else:
            bin_units = org_units

        # Get bins as a numpy array
        if isinstance(bins, np.ndarray):
            bins = bins.copy()
        else:
            bins = np.asanyarray(bins)

        if bins.ndim > 2:
            raise ValueError(
                "The 'bins' parameter must be scalar, 1-d or 2-d. "
                f"Got: {bins!r}"
            )

        two_d_bins = None

        if bins.ndim == 2:
            # --------------------------------------------------------
            # 2-d bins: Make sure that each bin is increasing and sort
            #           the bins by lower bounds
            # --------------------------------------------------------
            if bins.shape[1] != 2:
                raise ValueError(
                    "The second dimension of the 'bins' parameter must "
                    f"have size 2. Got: {bins!r}"
                )

            bins.sort(axis=1)
            bins.sort(axis=0)

            # Check for overlaps
            for i, (u, l) in enumerate(zip(bins[:-1, 1], bins[1:, 0])):
                if u > l:
                    raise ValueError(
                        f"Overlapping bins: "
                        f"{tuple(bins[i])}, {tuple(bins[i + i])}"
                    )

            two_d_bins = bins
            bins = np.unique(bins)

            # Find the bins that were omitted from the original 2-d
            # bins array. Note that this includes the left-open and
            # right-open bins at the ends.
            delete_bins = [
                n + 1
                for n, (a, b) in enumerate(zip(bins[:-1], bins[1:]))
                if (a, b) not in two_d_bins
            ]
        elif bins.ndim == 1:
            # --------------------------------------------------------
            # 1-d bins:
            # --------------------------------------------------------
            bins.sort()
            delete_bins = []
        else:
            # --------------------------------------------------------
            # 0-d bins:
            # --------------------------------------------------------
            if closed_ends is None:
                closed_ends = True

            if not closed_ends:
                raise ValueError(
                    "Can't set closed_ends=False when specifying bins as "
                    "a scalar."
                )

            if open_ends:
                raise ValueError(
                    "Can't set open_ends=True when specifying bins as a "
                    "scalar."
                )

            mx = d.max().datum()
            mn = d.min().datum()
            bins = np.linspace(mn, mx, int(bins) + 1, dtype=float)

            delete_bins = []

        if closed_ends:
            # Adjust the lowest/largest bin boundary to be inclusive
            if open_ends:
                raise ValueError(
                    "Can't set open_ends=True when closed_ends is True."
                )

            if bins.dtype.kind != "f":
                bins = bins.astype(float, copy=False)

            epsilon = np.finfo(float).eps
            ndim = bins.ndim
            if upper:
                mn = bins[(0,) * ndim]
                bins[(0,) * ndim] -= abs(mn) * epsilon
            else:
                mx = bins[(-1,) * ndim]
                bins[(-1,) * ndim] += abs(mx) * epsilon

        if not open_ends:
            delete_bins.insert(0, 0)
            delete_bins.append(bins.size)

        # Digitise the array
        dx = d.to_dask_array()
        dx = da.digitize(dx, bins, right=upper)
        d._set_dask(dx)
        d.override_units(_units_None, inplace=True)

        # More elegant to handle 'delete_bins' in cf- rather than Dask- space
        # i.e. using cf.where with d in-place rather than da.where with dx
        # just after the digitize operation above (cf.where already applies
        # equivalent logic element-wise).
        if delete_bins:
            for n, db in enumerate(delete_bins):
                db -= n
                d.where(d == db, np.ma.masked, None, inplace=True)
                # x = d - 1 rather than = d here since there is one fewer bin
                # therefore we need to adjust to the new corresponding indices
                d.where(d > db, d - 1, None, inplace=True)

        if return_bins:
            if two_d_bins is None:
                two_d_bins = np.empty((bins.size - 1, 2), dtype=bins.dtype)
                two_d_bins[:, 0] = bins[:-1]
                two_d_bins[:, 1] = bins[1:]

            two_d_bins = type(self)(two_d_bins, units=bin_units)
            return d, two_d_bins

        return d

    def median(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        inplace=False,
    ):
        """Calculate median values.

        Calculates the median value or the median values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `mean_of_upper_decile`, `percentile`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2])
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.median()
        <CF Data(1, 1): [[6.0]] K>

        """
        return self.percentile(
            50, axes=axes, squeeze=squeeze, mtol=mtol, inplace=inplace
        )

    @_inplace_enabled(default=False)
    def mean_of_upper_decile(
        self,
        axes=None,
        weights=None,
        method="linear",
        squeeze=False,
        mtol=1,
        include_decile=True,
        split_every=None,
        inplace=False,
    ):
        """Mean of values defined by the upper tenth of their
        distribution.

        For the values defined by the upper tenth of their
        distribution, calculates their mean, or their mean along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `mean`, `median`, `percentile`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

                .. note:: *weights* only applies to the calculation of
                          the mean defined by the upper tenth of their
                          distribution.

            {{percentile method: `str`, optional}}

                .. versionadded:: 3.14.0

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

                .. note:: *mtol* only applies to the calculation of
                          the location of the 90th percentile.

            include_decile: `bool`, optional
                If True then include in the mean any values that are
                equal to the 90th percentile. By default these are
                excluded.

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> d = cf.Data(np.arange(20).reshape(4, 5), 'm')
        >>> print(d.array)
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
        >>> e = d.mean_of_upper_decile()
        >>> e
        <CF Data(1, 1): [[18.5]] m>

        """
        d = _inplace_enabled_define_and_cleanup(self)

        # Find the 90th percentile
        p90 = d.percentile(
            90, axes=axes, squeeze=False, mtol=mtol, inplace=False
        )

        # Mask all elements that are less than (or equal to) the 90th
        # percentile
        if include_decile:
            less_than_p90 = d < p90
        else:
            less_than_p90 = d <= p90

        if mtol < 1:
            # Set missing values to True to ensure that 'd' gets
            # masked at those locations
            less_than_p90.filled(True, inplace=True)

        d.where(less_than_p90, masked, inplace=True)

        # Find the mean of elements greater than (or equal to) the
        # 90th percentile
        d.mean(
            axes=axes,
            weights=weights,
            squeeze=squeeze,
            mtol=1,
            split_every=split_every,
            inplace=True,
        )

        return d

    @_inplace_enabled(default=False)
    def percentile(
        self,
        ranks,
        axes=None,
        method="linear",
        squeeze=False,
        mtol=1,
        inplace=False,
        interpolation=None,
        interpolation2=None,
    ):
        """Compute percentiles of the data along the specified axes.

        The default is to compute the percentiles along a flattened
        version of the data.

        If the input data are integers, or floats smaller than
        float64, or the input data contains missing values, then
        output data-type is float64. Otherwise, the output data-type
        is the same as that of the input.

        If multiple percentile ranks are given then a new, leading
        data dimension is created so that percentiles can be stored
        for each percentile rank.

        **Accuracy**

        The `percentile` method returns results that are consistent
        with `numpy.percentile`, which may be different to those
        created by `dask.percentile`. The dask method uses an
        algorithm that calculates approximate percentiles which are
        likely to be different from the correct values when there are
        two or more dask chunks.

        >>> import numpy as np
        >>> import dask.array as da
        >>> import cf
        >>> a = np.arange(101)
        >>> dx = da.from_array(a, chunks=10)
        >>> da.percentile(dx, 40).compute()
        array([40.36])
        >>> np.percentile(a, 40)
        40.0
        >>> d = cf.Data(a, chunks=10)
        >>> d.percentile(40).array
        array([40.])

        .. versionadded:: 3.0.4

        .. seealso:: `digitize`, `median`, `mean_of_upper_decile`,
                     `where`

        :Parameters:

            ranks: (sequence of) number
                Percentile rank, or sequence of percentile ranks, to
                compute, which must be between 0 and 100 inclusive.

            axes: (sequence of) `int`, optional
                Select the axes. The *axes* argument may be one, or a
                sequence, of integers that select the axis
                corresponding to the given position in the list of
                axes of the data array.

                By default, of *axes* is `None`, all axes are selected.

            {{percentile method: `str`, optional}}

                .. versionadded:: 3.14.0

            squeeze: `bool`, optional
                If True then all axes over which percentiles are
                calculated are removed from the returned data. By
                default axes over which percentiles have been
                calculated are left in the result as axes with size 1,
                meaning that the result is guaranteed to broadcast
                correctly against the original data.

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            interpolation: deprecated at version 3.14.0
                Use the *method* parameter instead.

        :Returns:

            `Data` or `None`
                The percentiles of the original data, or `None` if the
                operation was in-place.

        **Examples**

        >>> d = cf.Data(numpy.arange(12).reshape(3, 4), 'm')
        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> p = d.percentile([20, 40, 50, 60, 80])
        >>> p
        <CF Data(5, 1, 1): [[[2.2, ..., 8.8]]] m>

        >>> p = d.percentile([20, 40, 50, 60, 80], squeeze=True)
        >>> print(p.array)
        [2.2 4.4 5.5 6.6 8.8]

        Find the standard deviation of the values above the 80th percentile:

        >>> p80 = d.percentile(80)
        <CF Data(1, 1): [[8.8]] m>
        >>> e = d.where(d<=p80, cf.masked)
        >>> print(e.array)
        [[-- -- -- --]
         [-- -- -- --]
         [-- 9 10 11]]
        >>> e.std()
        <CF Data(1, 1): [[0.816496580927726]] m>

        Find the mean of the values above the 45th percentile along the
        second axis:

        >>> p45 = d.percentile(45, axes=1)
        >>> print(p45.array)
        [[1.35],
         [5.35],
         [9.35]]
        >>> e = d.where(d<=p45, cf.masked)
        >>> print(e.array)
        [[-- -- 2 3]
         [-- -- 6 7]
         [-- -- 10 11]]
        >>> f = e.mean(axes=1)
        >>> f
        <CF Data(3, 1): [[2.5, ..., 10.5]] m>
        >>> print(f.array)
        [[ 2.5]
         [ 6.5]
         [10.5]]

        Find the histogram bin boundaries associated with given
        percentiles, and digitize the data based on these bins:

        >>> bins = d.percentile([0, 10, 50, 90, 100], squeeze=True)
        >>> print(bins.array)
        [ 0.   1.1  5.5  9.9 11. ]
        >>> e = d.digitize(bins, closed_ends=True)
        >>> print(e.array)
        [[0 0 1 1]
         [1 1 2 2]
         [2 2 3 3]]

        """
        from dask.core import flatten

        # TODODASKAPI: interpolation -> method
        if interpolation is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "percentile",
                {"interpolation": None},
                message="Use the 'method' parameter instead.",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        d = _inplace_enabled_define_and_cleanup(self)

        # Parse percentile ranks
        q = ranks
        if not (isinstance(q, np.ndarray) or is_dask_collection(q)):
            q = np.array(ranks)

        if q.ndim > 1:
            q = q.flatten()

        if not np.issubdtype(d.dtype, np.number):
            method = "nearest"

        if axes is None:
            axes = tuple(range(d.ndim))
        else:
            axes = tuple(sorted(d._parse_axes(axes)))

        # 'cf_percentile' has its own call to 'cfdm_to_memory', so we
        # can set '_force_to_memory=False'.
        dx = d.to_dask_array(_force_to_memory=False)
        dtype = dx.dtype
        shape = dx.shape

        # Rechunk the data so that the dimensions over which
        # percentiles are being calculated all have one chunk.
        #
        # Make sure that no new chunks are larger (in bytes) than any
        # original chunk.
        new_chunks = normalize_chunks(
            [-1 if i in axes else "auto" for i in range(dx.ndim)],
            shape=shape,
            dtype=dtype,
            limit=dtype.itemsize * reduce(mul, map(max, dx.chunks), 1),
        )
        dx = dx.rechunk(new_chunks)

        # Initialise the indices of each chunk of the result
        #
        # E.g. [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
        keys = [key[1:] for key in flatten(dx.__dask_keys__())]

        keepdims = not squeeze
        if not keepdims:
            # Remove axes that will be dropped in the result
            indices = [i for i in range(len(keys[0])) if i not in axes]
            keys = [tuple([k[i] for i in indices]) for k in keys]

        if q.ndim:
            # Insert a leading rank dimension for non-scalar input
            # percentile ranks
            keys = [(0,) + k for k in keys]

        # Create a new dask dictionary for the result
        name = "cf-percentile-" + tokenize(dx, axes, q, method)
        name = (name,)
        dsk = {
            name
            + chunk_index: (
                cf_percentile,
                dask_key,
                q,
                axes,
                method,
                keepdims,
                mtol,
            )
            for chunk_index, dask_key in zip(keys, flatten(dx.__dask_keys__()))
        }

        # Define the chunks for the result
        if q.ndim:
            out_chunks = [(q.size,)]
        else:
            out_chunks = []

        for i, c in enumerate(dx.chunks):
            if i in axes:
                if keepdims:
                    out_chunks.append((1,))
            else:
                out_chunks.append(c)

        name = name[0]
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[dx])
        dx = da.Array(graph, name, chunks=out_chunks, dtype=float)

        d._set_dask(dx)

        # Add a new axis identifier for a leading rank axis
        if q.ndim:
            axes = d._axes
            d._axes = (new_axis_identifier(axes),) + axes

        d._update_deterministic(q)

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def ceil(self, inplace=False, i=False):
        """The ceiling of the data, element-wise.

        The ceiling of ``x`` is the smallest integer ``n``, such that
        ``n>=x``.

        .. versionadded:: 1.0

        .. seealso:: `floor`, `rint`, `trunc`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The ceiling of the data. If the operation was in-place
                then `None` is returned.

        **Examples**

        >>> d = cf.Data([-1.9, -1.5, -1.1, -1, 0, 1, 1.1, 1.5 , 1.9])
        >>> print(d.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(d.ceil().array)
        [-1. -1. -1. -1.  0.  1.  2.  2.  2.]

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = d.to_dask_array()
        dx = da.ceil(dx)
        d._set_dask(dx)
        return d

    @_inplace_enabled(default=False)
    def convolution_filter(
        self,
        window=None,
        axis=None,
        mode=None,
        cval=None,
        origin=0,
        inplace=False,
    ):
        """Return the data convolved along the given axis with the
        specified filter.

        The magnitude of the integral of the filter (i.e. the sum of the
        weights defined by the *weights* parameter) affects the convolved
        values. For example, filter weights of ``[0.2, 0.2 0.2, 0.2,
        0.2]`` will produce a non-weighted 5-point running mean; and
        weights of ``[1, 1, 1, 1, 1]`` will produce a 5-point running
        sum. Note that the weights returned by functions of the
        `scipy.signal.windows` package do not necessarily sum to 1 (see
        the examples for details).

        .. versionadded:: 3.3.0

        :Parameters:

            window: sequence of numbers
                Specify the window of weights to use for the filter.

                *Parameter example:*
                  An unweighted 5-point moving average can be computed
                  with ``weights=[0.2, 0.2, 0.2, 0.2, 0.2]``

                Note that the `scipy.signal.windows` package has suite of
                window functions for creating weights for filtering (see
                the examples for details).

            axis: `int`
                Select the axis over which the filter is to be applied.
                removed. The *axis* parameter is an integer that selects
                the axis corresponding to the given position in the list
                of axes of the data.

                *Parameter example:*
                  Convolve the second axis: ``axis=1``.

                *Parameter example:*
                  Convolve the last axis: ``axis=-1``.

            mode: `str`, optional
                The *mode* parameter determines how the input array is
                extended when the filter overlaps an array border. The
                default value is ``'constant'`` or, if the dimension being
                convolved is cyclic (as ascertained by the `iscyclic`
                method), ``'wrap'``. The valid values and their behaviours
                are as follows:

                ==============  ==========================  ============================
                *mode*          Description                 Behaviour
                ==============  ==========================  ============================
                ``'reflect'``   The input is extended by    ``(c b a | a b c | c b a)``
                                reflecting about the edge

                ``'constant'``  The input is extended by    ``(k k k | a b c | k k k)``
                                filling all values beyond
                                the edge with the same
                                constant value (``k``),
                                defined by the *cval*
                                parameter.

                ``'nearest'``   The input is extended by    ``(a a a | a b c | c c c )``
                                replicating the last point

                ``'mirror'``    The input is extended by    ``(c b | a b c | b a)``
                                reflecting about the
                                centre of the last point.

                ``'wrap'``      The input is extended by    ``(a b c | a b c | a b c)``
                                wrapping around to the
                                opposite edge.

                ``'periodic'``  This is a synonym for
                                ``'wrap'``.
                ==============  ==========================  ============================

                The position of the window relative to each value can be
                changed by using the *origin* parameter.

            cval: scalar, optional
                Value to fill past the edges of the array if *mode* is
                ``'constant'``. Defaults to `None`, in which case the
                edges of the array will be filled with missing data.

                *Parameter example:*
                   To extend the input by filling all values beyond the
                   edge with zero: ``cval=0``

            origin: `int`, optional
                Controls the placement of the filter. Defaults to 0, which
                is the centre of the window. If the window has an even
                number of weights then then a value of 0 defines the index
                defined by ``width/2 -1``.

                *Parameter example:*
                  For a weighted moving average computed with a weights
                  window of ``[0.1, 0.15, 0.5, 0.15, 0.1]``, if
                  ``origin=0`` then the average is centred on each
                  point. If ``origin=-2`` then the average is shifted to
                  include the previous four points. If ``origin=1`` then
                  the average is shifted to include the previous point and
                  the and the next three points.

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The convolved data, or `None` if the operation was
                in-place.

        """
        from .dask_utils import cf_convolve1d

        d = _inplace_enabled_define_and_cleanup(self)

        iaxis = d._parse_axes(axis)
        if len(iaxis) != 1:
            raise ValueError(
                "Must specify a unique domain axis with the 'axis' "
                f"parameter. {axis!r} specifies axes {iaxis!r}"
            )

        iaxis = iaxis[0]

        if mode is None:
            # Default mode is 'wrap' if the axis is cyclic, or else
            # 'constant'.
            if iaxis in d.cyclic():
                boundary = "periodic"
            else:
                boundary = cval
        elif mode == "wrap":
            boundary = "periodic"
        elif mode == "constant":
            boundary = cval
        elif mode == "mirror":
            raise ValueError(
                "'mirror' mode is no longer available. Please raise an "
                "issue at https://github.com/NCAS-CMS/cf-python/issues "
                "if you would like it to be re-implemented."
            )
            # This re-implementation would involve getting a 'mirror'
            # function added to dask.array.overlap, along similar
            # lines to the existing 'reflect' function in that module.
        else:
            boundary = mode

        # Set the overlap depth large enough to accommodate the
        # filter.
        #
        # For instance, for a 5-point window, the calculated value at
        # each point requires 2 points either side if the filter is
        # centred (i.e. origin is 0) and (up to) 3 points either side
        # if origin is 1 or -1.
        #
        # It is a restriction of dask.array.map_overlap that we can't
        # use asymmetric halos for general 'boundary' types.
        size = len(window)
        depth = int(size / 2)
        if not origin and not size % 2:
            depth += 1

        depth += abs(origin)

        dx = d.to_dask_array()

        # Cast to float to ensure that NaNs can be stored (so
        # map_overlap can correctly assign the halos)
        if dx.dtype != float:
            dx = dx.astype(float, copy=False)

        # Convolve each chunk
        convolve1d = partial(
            cf_convolve1d, window=window, axis=iaxis, origin=origin
        )

        dx = dx.map_overlap(
            convolve1d,
            depth={iaxis: depth},
            boundary=boundary,
            trim=True,
            meta=np.array((), dtype=float),
        )

        d._set_dask(dx)

        return d

    @_inplace_enabled(default=False)
    def cumsum(
        self,
        axis=None,
        masked_as_zero=False,
        method="sequential",
        inplace=False,
    ):
        """Return the data cumulatively summed along the given axis.

        .. versionadded:: 3.0.0

        .. seealso:: `diff`, `sum`

        :Parameters:

            axis: `int`, optional
                Select the axis over which the cumulative sums are to
                be calculated. By default the cumulative sum is
                computed over the flattened array.

            method: `str`, optional
                Choose which method to use to perform the cumulative
                sum. See `dask.array.cumsum` for details.

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

                .. versionadded:: 3.3.0

            masked_as_zero: deprecated at version 3.14.0
                See the examples for the new behaviour when there are
                masked values.

        :Returns:

             `Data` or `None`
                The data with the cumulatively summed axis, or `None`
                if the operation was in-place.

        **Examples**

        >>> d = cf.Data(numpy.arange(12).reshape(3, 4))
        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> print(d.cumsum().array)
        [ 0  1  3  6 10 15 21 28 36 45 55 66]
        >>> print(d.cumsum(axis=0).array)
        [[ 0  1  2  3]
         [ 4  6  8 10]
         [12 15 18 21]]
        >>> print(d.cumsum(axis=1).array)
        [[ 0  1  3  6]
         [ 4  9 15 22]
         [ 8 17 27 38]]

        >>> d[0, 0] = cf.masked
        >>> d[1, [1, 3]] = cf.masked
        >>> d[2, 0:2] = cf.masked
        >>> print(d.array)
        [[-- 1 2 3]
         [4 -- 6 --]
         [-- -- 10 11]]
        >>> print(d.cumsum(axis=0).array)
        [[-- 1 2 3]
         [4 -- 8 --]
         [-- -- 18 14]]
        >>> print(d.cumsum(axis=1).array)
        [[-- 1 3 6]
         [4 -- 10 --]
         [-- -- 10 21]]

        """
        if masked_as_zero:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "cumsum",
                {"masked_as_zero": None},
                message="",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        d = _inplace_enabled_define_and_cleanup(self)

        dx = d.to_dask_array()
        dx = dx.cumsum(axis=axis, method=method)
        d._set_dask(dx)

        return d

    @_inplace_enabled(default=False)
    def _asdatetime(self, inplace=False):
        """Change the internal representation of data array elements
        from numeric reference times to datetime-like objects.

        If the calendar has not been set then the default CF calendar will
        be used and the units' and the `calendar` attribute will be
        updated accordingly.

        If the internal representations are already datetime-like objects
        then no change occurs.

        .. versionadded:: 1.3

        .. seealso:: `_asreftime`, `_isdatetime`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d = cf.Data([[1.93, 5.17]], "days since 2000-12-29")
        >>> e = d._asdatetime()
        >>> print(e.array)
        [[cftime.DatetimeGregorian(2000, 12, 30, 22, 19, 12, 0, has_year_zero=False)
          cftime.DatetimeGregorian(2001, 1, 3, 4, 4, 48, 0, has_year_zero=False)]]
        >>> f = e._asreftime()
        >>> print(f.array)
        [[1.93 5.17]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        units = d.Units
        if not units.isreftime:
            raise ValueError(
                f"Can't convert {units!r} values to date-time objects"
            )

        if not d._isdatetime():
            # 'cf_rt2dt' has its own call to 'cfdm_to_memory', so we
            # can set '_force_to_memory=False'.
            dx = d.to_dask_array(_force_to_memory=False)
            dx = dx.map_blocks(cf_rt2dt, units=units, dtype=object)
            d._set_dask(dx)

        return d

    def _isdatetime(self):
        """True if the internal representation is a datetime object."""
        return self.dtype.kind == "O" and self.Units.isreftime

    @_inplace_enabled(default=False)
    def _asreftime(self, inplace=False):
        """Change the internal representation of data array elements
        from datetime-like objects to numeric reference times.

        If the calendar has not been set then the default CF calendar will
        be used and the units' and the `calendar` attribute will be
        updated accordingly.

        If the internal representations are already numeric reference
        times then no change occurs.

        .. versionadded:: 1.3

        .. seealso:: `_asdatetime`, `_isdatetime`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d = cf.Data([[1.93, 5.17]], "days since 2000-12-29")
        >>> e = d._asdatetime()
        >>> print(e.array)
        [[cftime.DatetimeGregorian(2000, 12, 30, 22, 19, 12, 0, has_year_zero=False)
          cftime.DatetimeGregorian(2001, 1, 3, 4, 4, 48, 0, has_year_zero=False)]]
        >>> f = e._asreftime()
        >>> print(f.array)
        [[1.93 5.17]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        units = d.Units
        if not units.isreftime:
            raise ValueError(
                f"Can't convert {units!r} values to numeric reference times"
            )

        if d._isdatetime():
            # 'cf_dt2rt' has its own call to 'cfdm_to_memory', so we
            # can set '_force_to_memory=False'.
            dx = d.to_dask_array(_force_to_memory=False)
            dx = dx.map_blocks(cf_dt2rt, units=units, dtype=float)
            d._set_dask(dx)

        return d

    def _combined_units(self, data1, method, inplace):
        """Combines by given method the data's units with other units.

        :Parameters:

            data1: `Data`

            method: `str`

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`, `Data` or `None`, `Units`

        **Examples**

        >>> d._combined_units(e, '__sub__')
        >>> d._combined_units(e, '__imul__')
        >>> d._combined_units(e, '__irdiv__')
        >>> d._combined_units(e, '__lt__')
        >>> d._combined_units(e, '__rlshift__')
        >>> d._combined_units(e, '__iand__')

        """
        method_type = method[-5:-2]

        data0 = self

        units0 = data0.Units
        units1 = data1.Units

        if not units0 and not units1:
            return data0, data1, units0
        if (
            units0.isreftime
            and units1.isreftime
            and not units0.equivalent(units1)
        ):
            # Both are reference_time, but have non-equivalent
            # calendars
            if units0._canonical_calendar and not units1._canonical_calendar:
                data1 = data1._asdatetime()
                data1.override_units(units0, inplace=True)
                data1._asreftime(inplace=True)
                units1 = units0
            elif units1._canonical_calendar and not units0._canonical_calendar:
                if not inplace:
                    inplace = True
                    data0 = data0.copy()
                data0._asdatetime(inplace=True)
                data0.override_units(units1, inplace=True)
                data0._asreftime(inplace=True)
                units0 = units1
        # --- End: if

        if method_type in ("_eq", "_ne", "_lt", "_le", "_gt", "_ge"):
            # ---------------------------------------------------------
            # Operator is one of ==, !=, >=, >, <=, <
            # ---------------------------------------------------------
            if units0.equivalent(units1):
                # Units are equivalent
                if not units0.equals(units1):
                    data1 = data1.copy()
                    data1.Units = units0
                return data0, data1, _units_None
            elif not units1 or not units0:
                # At least one of the units is undefined
                return data0, data1, _units_None
            else:
                raise ValueError(
                    "Can't compare {0!r} to {1!r}".format(units0, units1)
                )
        # --- End: if

        # still here?
        if method_type in ("and", "_or", "ior", "ror", "xor", "ift"):
            # ---------------------------------------------------------
            # Operation is one of &, |, ^, >>, <<
            # ---------------------------------------------------------
            if units0.equivalent(units1):
                # Units are equivalent
                if not units0.equals(units1):
                    data1 = data1.copy()
                    data1.Units = units0
                return data0, data1, units0
            elif not units1:
                # units1 is undefined
                return data0, data1, units0
            elif not units0:
                # units0 is undefined
                return data0, data1, units1
            else:
                # Both units are defined and not equivalent
                raise ValueError(
                    "Can't operate with {} on data with {!r} to {!r}".format(
                        method, units0, units1
                    )
                )
        # --- End: if

        # Still here?
        if units0.isreftime:
            # ---------------------------------------------------------
            # units0 is reference time
            # ---------------------------------------------------------
            if method_type == "sub":
                if units1.isreftime:
                    if units0.equivalent(units1):
                        # Equivalent reference_times: the output units
                        # are time
                        if not units0.equals(units1):
                            data1 = data1.copy()
                            data1.Units = units0
                        return (
                            data0,
                            data1,
                            self._Units_class(_ut_unit=units0._ut_unit),
                        )
                    else:
                        # Non-equivalent reference_times: raise an
                        # exception
                        getattr(units0, method)(units1)
                elif units1.istime:
                    # reference_time minus time: the output units are
                    # reference_time
                    time0 = self._Units_class(_ut_unit=units0._ut_unit)
                    if not units1.equals(time0):
                        data1 = data1.copy()
                        data1.Units = time0
                    return data0, data1, units0
                elif not units1:
                    # reference_time minus no_units: the output units
                    # are reference_time
                    return data0, data1, units0
                else:
                    # reference_time minus something not yet accounted
                    # for: raise an exception
                    getattr(units0, method)(units1)

            elif method_type in ("add", "mul", "div", "mod"):
                if units1.istime:
                    # reference_time plus regular_time: the output
                    # units are reference_time
                    time0 = Units(_ut_unit=units0._ut_unit)
                    if not units1.equals(time0):
                        data1 = data1.copy()
                        data1.Units = time0
                    return data0, data1, units0
                elif not units1:
                    # reference_time plus no_units: the output units
                    # are reference_time
                    return data0, data1, units0
                else:
                    # reference_time plus something not yet accounted
                    # for: raise an exception
                    getattr(units0, method)(units1)

            else:
                # Raise an exception
                getattr(units0, method)(units1)

        elif units1.isreftime:
            # ---------------------------------------------------------
            # units1 is reference time
            # ---------------------------------------------------------
            if method_type == "add":
                if units0.istime:
                    # Time plus reference_time: the output units are
                    # reference_time
                    time1 = self._Units_class(_ut_unit=units1._ut_unit)
                    if not units0.equals(time1):
                        if not inplace:
                            data0 = data0.copy()
                        data0.Units = time1
                    return data0, data1, units1
                elif not units0:
                    # No_units plus reference_time: the output units
                    # are reference_time
                    return data0, data1, units1
                else:
                    # Raise an exception
                    getattr(units0, method)(units1)
        # --- End: if

        # Still here?
        if method_type in ("mul", "div"):
            # ---------------------------------------------------------
            # Method is one of *, /, //
            # ---------------------------------------------------------
            if not units1:
                # units1 is undefined
                return data0, data1, getattr(units0, method)(_units_1)
            elif not units0:
                # units0 is undefined
                return data0, data1, getattr(_units_1, method)(units1)
                #  !!!!!!! units0*units0 YOWSER
            else:
                # Both units are defined (note: if the units are
                # noncombinable then this will raise an exception)
                return data0, data1, getattr(units0, method)(units1)
        # --- End: if

        # Still here?
        if method_type in ("sub", "add", "mod"):
            # ---------------------------------------------------------
            # Operator is one of +, -
            # ---------------------------------------------------------
            if units0.equivalent(units1):
                # Units are equivalent
                if not units0.equals(units1):
                    data1 = data1.copy()
                    data1.Units = units0
                return data0, data1, units0
            elif not units1:
                # units1 is undefined
                return data0, data1, units0
            elif not units0:
                # units0 is undefined
                return data0, data1, units1
            else:
                # Both units are defined and not equivalent (note: if
                # the units are noncombinable then this will raise an
                # exception)
                return data0, data1, getattr(units0, method)(units1)
        # --- End: if

        # Still here?
        if method_type == "pow":
            if method == "__rpow__":
                # -----------------------------------------------------
                # Operator is __rpow__
                # -----------------------------------------------------
                if not units1:
                    # units1 is undefined
                    if not units0:
                        # units0 is undefined
                        return data0, data1, _units_None
                    elif units0.isdimensionless:
                        # units0 is dimensionless
                        if not units0.equals(_units_1):
                            if not inplace:
                                data0 = data0.copy()
                            data0.Units = _units_1

                        return data0, data1, _units_None
                elif units1.isdimensionless:
                    # units1 is dimensionless
                    if not units1.equals(_units_1):
                        data1 = data1.copy()
                        data1.Units = _units_1

                    if not units0:
                        # units0 is undefined
                        return data0, data1, _units_1
                    elif units0.isdimensionless:
                        # units0 is dimensionless
                        if not units0.equals(_units_1):
                            if not inplace:
                                data0 = data0.copy()
                            data0.Units = _units_1

                        return data0, data1, _units_1
                else:
                    # units1 is defined and is not dimensionless
                    if data0.size > 1:
                        raise ValueError(
                            "Can only raise units to the power of a single "
                            "value at a time. Asking to raise to the power of "
                            "{}".format(data0)
                        )

                    if not units0:
                        # Check that the units are not shifted, as
                        # raising this to a power is a nonlinear
                        # operation
                        p = data0.datum(0)
                        if units0 != (units0**p) ** (1.0 / p):
                            raise ValueError(
                                "Can't raise shifted units {!r} to the "
                                "power {}".format(units0, p)
                            )

                        return data0, data1, units1**p
                    elif units0.isdimensionless:
                        # units0 is dimensionless
                        if not units0.equals(_units_1):
                            if not inplace:
                                data0 = data0.copy()
                            data0.Units = _units_1

                        # Check that the units are not shifted, as
                        # raising this to a power is a nonlinear
                        # operation
                        p = data0.datum(0)
                        if units0 != (units0**p) ** (1.0 / p):
                            raise ValueError(
                                "Can't raise shifted units {!r} to the "
                                "power {}".format(units0, p)
                            )

                        return data0, data1, units1**p
                # --- End: if

                # This will deliberately raise an exception
                units1**units0
            else:
                # -----------------------------------------------------
                # Operator is __pow__
                # -----------------------------------------------------
                if not units0:
                    # units0 is undefined
                    if not units1:
                        # units0 is undefined
                        return data0, data1, _units_None
                    elif units1.isdimensionless:
                        # units0 is dimensionless
                        if not units1.equals(_units_1):
                            data1 = data1.copy()
                            data1.Units = _units_1

                        return data0, data1, _units_None
                elif units0.isdimensionless:
                    # units0 is dimensionless
                    if not units0.equals(_units_1):
                        if not inplace:
                            data0 = data0.copy()
                        data0.Units = _units_1

                    if not units1:
                        # units1 is undefined
                        return data0, data1, _units_1
                    elif units1.isdimensionless:
                        # units1 is dimensionless
                        if not units1.equals(_units_1):
                            data1 = data1.copy()
                            data1.Units = _units_1

                        return data0, data1, _units_1
                else:
                    # units0 is defined and is not dimensionless
                    if data1.size > 1:
                        raise ValueError(
                            "Can only raise units to the power of a single "
                            "value at a time. Asking to raise to the power of "
                            "{}".format(data1)
                        )

                    if not units1:
                        # Check that the units are not shifted, as
                        # raising this to a power is a nonlinear
                        # operation
                        p = data1.datum(0)
                        if units0 != (units0**p) ** (1.0 / p):
                            raise ValueError(
                                "Can't raise shifted units {!r} to the "
                                "power {}".format(units0, p)
                            )

                        return data0, data1, units0**p
                    elif units1.isdimensionless:
                        # units1 is dimensionless
                        if not units1.equals(_units_1):
                            data1 = data1.copy()
                            data1.Units = _units_1

                        # Check that the units are not shifted, as
                        # raising this to a power is a nonlinear
                        # operation
                        p = data1.datum(0)
                        if units0 != (units0**p) ** (1.0 / p):
                            raise ValueError(
                                "Can't raise shifted units {!r} to the "
                                "power {}".format(units0, p)
                            )

                        return data0, data1, units0**p
                # --- End: if

                # This will deliberately raise an exception
                units0**units1
            # --- End: if
        # --- End: if

        # Still here?
        raise ValueError(
            "Can't operate with {} on data with {!r} to {!r}".format(
                method, units0, units1
            )
        )

    @classmethod
    def _binary_operation(cls, data, other, method):
        """Implement binary arithmetic and comparison operations with
        the numpy broadcasting rules.

        It is called by the binary arithmetic and comparison
        methods, such as `__sub__`, `__imul__`, `__rdiv__`, `__lt__`, etc.

        .. seealso:: `_unary_operation`

        :Parameters:

            other:
                The object on the right hand side of the operator.

            method: `str`
                The binary arithmetic or comparison method name (such as
                ``'__imul__'`` or ``'__ge__'``).

        :Returns:

            `Data`
                A new data object, or if the operation was in place, the
                same data object.

        **Examples**

        >>> d = cf.Data([0, 1, 2, 3])
        >>> e = cf.Data([1, 1, 3, 4])

        >>> f = d._binary_operation(e, '__add__')
        >>> print(f.array)
        [1 2 5 7]

        >>> e = d._binary_operation(e, '__lt__')
        >>> print(e.array)
        [ True False  True  True]

        >>> d._binary_operation(2, '__imul__')
        >>> print(d.array)
        [0 2 4 6]

        """
        if getattr(other, "_NotImplemented_RHS_Data_op", False):
            return NotImplemented

        inplace = method[2] == "i"

        # ------------------------------------------------------------
        # Ensure other is an independent Data object, for example
        # so that combination with cf.Query objects works.
        # ------------------------------------------------------------
        if not isinstance(other, cls):
            if (
                isinstance(other, cftime.datetime)
                and other.calendar == ""
                and data.Units.isreftime
            ):
                other = cf_dt(
                    other, calendar=getattr(data.Units, "calendar", "standard")
                )
            elif other is None:
                # Can't sensibly initialise a Data object from a bare
                # `None` (issue #281)
                other = np.array(None, dtype=object)

            other = cls.asdata(other)

        # ------------------------------------------------------------
        # Prepare data0 (i.e. self copied) and data1 (i.e. other)
        # ------------------------------------------------------------
        data0 = data.copy()

        # Parse units
        data0, other, new_Units = data0._combined_units(other, method, True)

        d = super()._binary_operation(data0, other, method)

        d.override_units(new_Units, inplace=True)

        if inplace:
            data.__dict__ = d.__dict__
        else:
            data = d

        return data

    def _parse_indices(self, *args, **kwargs):
        """'cf.Data._parse_indices' is not available.

        Use function `cf.parse_indices` instead.

        """
        raise NotImplementedError(
            "'cf.Data._parse_indices' is no longer available. "
            "Use function 'cf.parse_indices' instead."
        )

    def _regrid(
        self,
        method=None,
        operator=None,
        regrid_axes=None,
        regridded_sizes=None,
        min_weight=None,
    ):
        """Regrid the data.

        See `cf.regrid.regrid` for details.

        .. versionadded:: 3.14.0

        .. seealso:: `cf.Field.regridc`, `cf.Field.regrids`

        :Parameters:

            {{method: `str` or `None`, optional}}

            operator: `RegridOperator`
                The definition of the source and destination grids and
                the regridding weights.

            regrid_axes: sequence of `int`
                The positions of the regrid axes in the data, given in
                the relative order expected by the regrid
                operator. For spherical regridding this order is [Y,
                X] or [Z, Y, X].

                *Parameter example:*
                  ``[2, 3]``

            regridded_sizes: `dict`
                Mapping of the regrid axes, defined by the integer
                elements of *regrid_axes*, to their regridded sizes.

                *Parameter example:*
                  ``{3: 128, 2: 64}``

            {{min_weight: float, optional}}

        :Returns:

            `Data`
                The regridded data.

        """
        from .dask_regrid import regrid, regrid_weights

        shape = self.shape
        ndim = self.ndim
        src_shape = tuple(shape[i] for i in regrid_axes)
        if src_shape != operator.src_shape:
            raise ValueError(
                f"Regrid axes shape {src_shape} does not match "
                f"the shape of the regrid operator: {operator.src_shape}"
            )

        # 'regrid' has its own calls to 'cfdm_to_memory', so we can set
        # '_force_to_memory=False'.
        dx = self.to_dask_array(_force_to_memory=False)

        # Rechunk so that each chunk contains data in the form
        # expected by the regrid operator, i.e. the regrid axes all
        # have chunksize -1.
        numblocks = dx.numblocks
        if not all(numblocks[i] == 1 for i in regrid_axes):
            chunks = [
                -1 if i in regrid_axes else c for i, c in enumerate(dx.chunks)
            ]
            dx = dx.rechunk(chunks)

        # Define the regridded chunksizes (allowing for the regridded
        # data to have more/fewer axes).
        regridded_chunks = []  # The 'chunks' parameter to `map_blocks`
        drop_axis = []  # The 'drop_axis' parameter to `map_blocks`
        new_axis = []  # The 'new_axis' parameter to `map_blocks`
        n = 0
        for i, c in enumerate(dx.chunks):
            if i in regridded_sizes:
                sizes = regridded_sizes[i]
                n_sizes = len(sizes)
                if not n_sizes:
                    drop_axis.append(i)
                    continue

                regridded_chunks.extend(sizes)
                if n_sizes > 1:
                    new_axis.extend(range(n + 1, n + n_sizes))
                    n += n_sizes - 1
            else:
                regridded_chunks.append(c)

            n += 1

        # Update the axis identifiers.
        #
        # This is necessary when regridding changes the number of data
        # dimensions (e.g. as happens when regridding a mesh topology
        # axis to/from separate lat and lon axes).
        if new_axis:
            axes = list(self._axes)
            for i in new_axis:
                axes.insert(i, new_axis_identifier(tuple(axes)))

            self._axes = axes
        elif drop_axis:
            axes = self._axes
            axes = [axes[i] for i in range(ndim) if i not in drop_axis]
            self._axes = axes

        # Set the output data type
        if method in ("nearest_dtos", "nearest_stod"):
            dst_dtype = dx.dtype
        else:
            dst_dtype = float

        non_regrid_axes = [i for i in range(ndim) if i not in regrid_axes]

        src_mask = operator.src_mask
        if src_mask is not None:
            src_mask = da.asanyarray(src_mask)

        weights_dst_mask = delayed(regrid_weights, pure=True)(
            operator=operator, dst_dtype=dst_dtype
        )

        # Create a regridding function to apply to each chunk
        cf_regrid_func = partial(
            regrid,
            method=method,
            src_shape=src_shape,
            dst_shape=operator.dst_shape,
            axis_order=non_regrid_axes + list(regrid_axes),
            min_weight=min_weight,
        )

        # Performance note:
        #
        # The function 'regrid_func' is copied into every Dask
        # task. If we included the large 'weights_dst_mask' in the
        # 'partial' definition then it would also be copied to every
        # task, which "will start to be a pain in a few parts of the
        # pipeline" definition. Instead we can pass it in via a
        # keyword argument to 'map_blocks'.
        # github.com/pangeo-data/pangeo/issues/334#issuecomment-403787663

        dx = dx.map_blocks(
            cf_regrid_func,
            weights_dst_mask=weights_dst_mask,
            ref_src_mask=src_mask,
            chunks=regridded_chunks,
            drop_axis=drop_axis,
            new_axis=new_axis,
            meta=np.array((), dtype=dst_dtype),
        )

        d = self.copy()
        d._set_dask(dx)

        # Don't know (yet) if 'operator' has a deterministic name
        d._update_deterministic(False)

        return d

    def __add__(self, other):
        """The binary arithmetic operation ``+``

        x.__add__(y) <==> x+y

        """
        return self._binary_operation(self, other, "__add__")

    def __iadd__(self, other):
        """The augmented arithmetic assignment ``+=``

        x.__iadd__(y) <==> x+=y

        """
        return self._binary_operation(self, other, "__iadd__")

    def __radd__(self, other):
        """The binary arithmetic operation ``+`` with reflected
        operands.

        x.__radd__(y) <==> y+x

        """
        return self._binary_operation(self, other, "__radd__")

    def __sub__(self, other):
        """The binary arithmetic operation ``-``

        x.__sub__(y) <==> x-y

        """
        return self._binary_operation(self, other, "__sub__")

    def __isub__(self, other):
        """The augmented arithmetic assignment ``-=``

        x.__isub__(y) <==> x-=y

        """
        return self._binary_operation(self, other, "__isub__")

    def __rsub__(self, other):
        """The binary arithmetic operation ``-`` with reflected
        operands.

        x.__rsub__(y) <==> y-x

        """
        return self._binary_operation(self, other, "__rsub__")

    def __mul__(self, other):
        """The binary arithmetic operation ``*``

        x.__mul__(y) <==> x*y

        """
        return self._binary_operation(self, other, "__mul__")

    def __imul__(self, other):
        """The augmented arithmetic assignment ``*=``

        x.__imul__(y) <==> x*=y

        """
        return self._binary_operation(self, other, "__imul__")

    def __rmul__(self, other):
        """The binary arithmetic operation ``*`` with reflected
        operands.

        x.__rmul__(y) <==> y*x

        """
        return self._binary_operation(self, other, "__rmul__")

    def __div__(self, other):
        """The binary arithmetic operation ``/``

        x.__div__(y) <==> x/y

        """
        return self._binary_operation(self, other, "__div__")

    def __idiv__(self, other):
        """The augmented arithmetic assignment ``/=``

        x.__idiv__(y) <==> x/=y

        """
        return self._binary_operation(self, other, "__idiv__")

    def __rdiv__(self, other):
        """The binary arithmetic operation ``/`` with reflected
        operands.

        x.__rdiv__(y) <==> y/x

        """
        return self._binary_operation(self, other, "__rdiv__")

    def __floordiv__(self, other):
        """The binary arithmetic operation ``//``

        x.__floordiv__(y) <==> x//y

        """
        return self._binary_operation(self, other, "__floordiv__")

    def __ifloordiv__(self, other):
        """The augmented arithmetic assignment ``//=``

        x.__ifloordiv__(y) <==> x//=y

        """
        return self._binary_operation(self, other, "__ifloordiv__")

    def __rfloordiv__(self, other):
        """The binary arithmetic operation ``//`` with reflected
        operands.

        x.__rfloordiv__(y) <==> y//x

        """
        return self._binary_operation(self, other, "__rfloordiv__")

    def __truediv__(self, other):
        """The binary arithmetic operation ``/`` (true division)

        x.__truediv__(y) <==> x/y

        """
        return self._binary_operation(self, other, "__truediv__")

    def __itruediv__(self, other):
        """The augmented arithmetic assignment ``/=`` (true division)

        x.__itruediv__(y) <==> x/=y

        """
        return self._binary_operation(self, other, "__itruediv__")

    def __rtruediv__(self, other):
        """The binary arithmetic operation ``/`` (true division) with
        reflected operands.

        x.__rtruediv__(y) <==> y/x

        """
        return self._binary_operation(self, other, "__rtruediv__")

    def __pow__(self, other, modulo=None):
        """The binary arithmetic operations ``**`` and ``pow``

        x.__pow__(y) <==> x**y

        """
        if modulo is not None:
            raise NotImplementedError(
                "3-argument power not supported for {!r}".format(
                    self.__class__.__name__
                )
            )

        return self._binary_operation(self, other, "__pow__")

    def __ipow__(self, other, modulo=None):
        """The augmented arithmetic assignment ``**=``

        x.__ipow__(y) <==> x**=y

        """
        if modulo is not None:
            raise NotImplementedError(
                "3-argument power not supported for {!r}".format(
                    self.__class__.__name__
                )
            )

        return self._binary_operation(self, other, "__ipow__")

    def __rpow__(self, other, modulo=None):
        """The binary arithmetic operations ``**`` and ``pow`` with
        reflected operands.

        x.__rpow__(y) <==> y**x

        """
        if modulo is not None:
            raise NotImplementedError(
                "3-argument power not supported for {!r}".format(
                    self.__class__.__name__
                )
            )

        return self._binary_operation(self, other, "__rpow__")

    def __mod__(self, other):
        """The binary arithmetic operation ``%``

        x.__mod__(y) <==> x % y

        """
        return self._binary_operation(self, other, "__mod__")

    def __imod__(self, other):
        """The binary arithmetic operation ``%=``

        x.__imod__(y) <==> x %= y

        """
        return self._binary_operation(self, other, "__imod__")

    def __rmod__(self, other):
        """The binary arithmetic operation ``%`` with reflected
        operands.

        x.__rmod__(y) <==> y % x

        """
        return self._binary_operation(self, other, "__rmod__")

    def __query_isclose__(self, value, rtol, atol):
        """Query interface method for an "is close" condition.

        :Parameters:

            value:
                The object to test against.

            rtol: number
                The tolerance on relative numerical differences.

            atol: number
                The tolerance on absolute numerical differences.

        .. versionadded:: 3.15.2

        """
        return self.isclose(value, rtol=rtol, atol=atol)

    @property
    def _cyclic(self):
        """Storage for axis cyclicity.

        Contains a `set` that identifies which axes are cyclic (and
        therefore allows cyclic slicing). The set contains a subset of
        the axis identifiers defined by the `_axes` attribute.

        .. warning:: Never change the value of the `_cyclic` attribute
                     in-place.

        .. note:: When an axis identifier is removed from the `_axes`
                  attribute then it is automatically also removed from
                  the `_cyclic` attribute.

        """
        return self._custom.get("_cyclic", _empty_set)

    @_cyclic.setter
    def _cyclic(self, value):
        self._custom["_cyclic"] = value

    @_cyclic.deleter
    def _cyclic(self):
        self._custom["_cyclic"] = _empty_set

    @property
    def _axes(self):
        """Storage for the axis identifiers.

        Contains a `tuple` of identifiers, one for each array axis.

        """
        return super()._axes

    @_axes.setter
    def _axes(self, value):
        self._set_component("axes", tuple(value), copy=False)

        # Remove cyclic axes that are not in the new axes
        cyclic = self._cyclic
        if cyclic:
            # Never change the value of the _cyclic attribute in-place
            self._cyclic = cyclic.intersection(value)

    @property
    def Units(self):
        """The `Units` object containing the units of the data array.

        **Examples**

        >>> d = cf.Data([1, 2, 3], units='m')
        >>> d.Units
        <Units: m>
        >>> d.Units = cf.Units('kilometres')
        >>> d.Units
        <Units: kilometres>
        >>> d.Units = cf.Units('km')
        >>> d.Units
        <Units: km>

        """
        return self._Units

    @Units.setter
    def Units(self, value):
        try:
            old_units = self._Units
        except ValueError:
            pass
        else:
            if not old_units or self.Units.equals(value):
                self._Units = value
                return

            if old_units and not old_units.equivalent(value):
                raise ValueError(
                    f"Can't set Units to {value!r} that are not "
                    f"equivalent to the current units {old_units!r}. "
                    "Consider using the override_units method instead."
                )

        try:
            dtype = self.dtype
        except ValueError:
            dtype = None

        if dtype is not None:
            if dtype.kind in "iu":
                if dtype.char in "iI":
                    dtype = _dtype_float32
                else:
                    dtype = _dtype_float

            cf_func = partial(cf_units, from_units=old_units, to_units=value)

            # 'cf_units' has its own call to 'cfdm_to_memory', so we
            # can set '_force_to_memory=False'.
            dx = self.to_dask_array(_force_to_memory=False)
            dx = dx.map_blocks(cf_func, dtype=dtype)

            # Setting equivalent units doesn't affect the CFA write
            # status. Nor does it invalidate any cached values, but
            # only because we'll adjust those, too.
            self._set_dask(dx, clear=self._ALL ^ self._CACHE ^ self._CFA)

            # Adjust cached values for the new units
            cache = self._get_cached_elements()
            if cache:
                self._set_cached_elements(
                    {index: cf_func(value) for index, value in cache.items()}
                )

        self._Units = value

    @Units.deleter
    def Units(self):
        raise ValueError(
            "Can't delete the Units attribute. "
            "Consider using the override_units method instead."
        )

    @property
    def is_masked(self):
        """True if the data array has any masked values.

        **Performance**

        `is_masked` causes all delayed operations to be executed.

        **Examples**

        >>> d = cf.Data([[1, 2, 3], [4, 5, 6]])
        >>> print(d.is_masked)
        False
        >>> d[0, ...] = cf.masked
        >>> d.is_masked
        True

        """
        # 'cf_is_masked' has its own call to 'cfdm_to_memory', so we
        # can set '_force_to_memory=False'.
        dx = self.to_dask_array(_force_to_memory=False)

        out_ind = tuple(range(dx.ndim))
        dx_ind = out_ind

        dx = da.blockwise(
            cf_is_masked,
            out_ind,
            dx,
            dx_ind,
            adjust_chunks={i: 1 for i in out_ind},
            dtype=bool,
        )

        return bool(dx.any())

    @classmethod
    def _concatenate_conform_units(cls, data1, units0, relaxed_units, copy):
        """Check and conform the units of data prior to concatenation.

        This is a helper function for `concatenate` that may be easily
        overridden in subclasses, to allow for customisation of the
        concatenation process.

        .. versionadded:: 3.17.0

        .. seealso:: `concatenate`

        :Parameters:

            data1: `Data`
                Data with units.

            units0: `Units`
                The units to conform *data1* to.

            {{relaxed_units: `bool`, optional}}

            copy: `bool`
                If False then modify *data1* in-place. Otherwise a
                copy of it is modified.

        :Returns:

            `Data`
                Returns *data1*, possibly modified so that it conforms
                to *units0*. If *copy* is False and *data1* is
                modified, then it is done so in-place.

        """
        # Check and conform, if necessary, the units of all inputs
        units1 = data1.Units
        if (
            relaxed_units
            and not units0.isvalid
            and not units1.isvalid
            and units0.__dict__ == units1.__dict__
        ):
            # Allow identical invalid units to be equal
            pass
        elif units0.equals(units1):
            pass
        elif units0.equivalent(units1):
            if copy:
                data1 = data1.copy()

            data1.Units = units0
        else:
            raise ValueError(
                "Can't concatenate: All the input arrays must have "
                f"equivalent units. Got {units0!r} and {units1!r}"
            )

        return data1

    @classmethod
    def _concatenate_post_process(
        cls, concatenated_data, axis, conformed_data
    ):
        """Post-process concatenated data.

        This is a helper function for `concatenate` that may be easily
        overridden in subclasses, to allow for customisation of the
        concatenation process.

        .. versionadded:: 3.17.0

        .. seealso:: `concatenate`

        :Parameters:

            concatenated_data: `Data`
                The concatenated data array.

            axis: `int`
                The axis of concatenation.

            conformed_data: sequence of `Data`
                The ordered sequence of data arrays that were
                concatenated.

        :Returns:

            `Data`
                Returns *concatenated_data*, possibly modified
                in-place.

        """
        concatenated_data = super()._concatenate_post_process(
            concatenated_data, axis, conformed_data
        )

        # Manage cyclicity of axes: if join axis was cyclic, it is no
        # longer.
        axis = concatenated_data._parse_axes(axis)[0]
        if axis in concatenated_data.cyclic():
            logger.warning(
                f"Concatenating along a cyclic axis ({axis}) therefore the "
                "axis has been set as non-cyclic in the output."
            )
            concatenated_data.cyclic(axes=axis, iscyclic=False)

        return concatenated_data

    @_inplace_enabled(default=False)
    def arctan(self, inplace=False):
        """Take the trigonometric inverse tangent of the data element-
        wise.

        Units are ignored in the calculation. The result has units of radians.

        .. versionadded:: 3.0.7

        .. seealso:: `tan`, `arcsin`, `arccos`, `arctanh`, `arctan2`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> print(d.array)
        [[0.5 0.7]
         [0.9 1.1]]
        >>> e = d.arctan()
        >>> e.Units
        <Units: radians>
        >>> print(e.array)
        [[0.46364761 0.61072596]
         [0.7328151  0.83298127]]

        >>> print(d.array)
        [1.2 1.0 0.8 0.6 --]
        >>> d.arctan(inplace=True)
        >>> print(d.array)
        [0.8760580505981934 0.7853981633974483 0.6747409422235527
         0.5404195002705842 --]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        dx = d.to_dask_array()
        dx = da.arctan(dx)
        d._set_dask(dx)

        d.override_units(_units_radians, inplace=True)

        return d

    @_inplace_enabled(default=False)
    def arctanh(self, inplace=False):
        """Take the inverse hyperbolic tangent of the data element-wise.

        Units are ignored in the calculation. The result has units of radians.

        .. versionadded:: 3.2.0

        .. seealso::  `tanh`, `arcsinh`, `arccosh`, `arctan`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> print(d.array)
        [[0.5 0.7]
         [0.9 1.1]]
        >>> e = d.arctanh()
        >>> e.Units
        <Units: radians>
        >>> print(e.array)
        [[0.54930614 0.86730053]
         [1.47221949        nan]]

        >>> print(d.array)
        [1.2 1.0 0.8 0.6 --]
        >>> d.arctanh(inplace=True)
        >>> print(d.array)
        [nan inf 1.0986122886681098 0.6931471805599453 --]
        >>> d.masked_invalid(inplace=True)
        >>> print(d.array)
        [-- -- 1.0986122886681098 0.6931471805599453 --]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        # Data.func is used instead of the Dask built-in in this case because
        # arctanh has a restricted domain therefore it is necessary to use our
        # custom logic implemented via the `preserve_invalid` keyword to func.
        d.func(
            np.arctanh,
            units=_units_radians,
            inplace=True,
            preserve_invalid=True,
        )

        return d

    @_inplace_enabled(default=False)
    def arcsin(self, inplace=False):
        """Take the trigonometric inverse sine of the data element-wise.

        Units are ignored in the calculation. The result has units of radians.

        .. versionadded:: 3.2.0

        .. seealso::  `sin`, `arccos`, `arctan`, `arcsinh`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> print(d.array)
        [[0.5 0.7]
         [0.9 1.1]]
        >>> e = d.arcsin()
        >>> e.Units
        <Units: radians>
        >>> print(e.array)
        [[0.52359878 0.7753975 ]
         [1.11976951        nan]]

        >>> print(d.array)
        [1.2 1.0 0.8 0.6 --]
        >>> d.arcsin(inplace=True)
        >>> print(d.array)
        [nan 1.5707963267948966 0.9272952180016123 0.6435011087932844 --]
        >>> d.masked_invalid(inplace=True)
        >>> print(d.array)
        [-- 1.5707963267948966 0.9272952180016123 0.6435011087932844 --]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        # Data.func is used instead of the Dask built-in in this case because
        # arcsin has a restricted domain therefore it is necessary to use our
        # custom logic implemented via the `preserve_invalid` keyword to func.
        d.func(
            np.arcsin,
            units=_units_radians,
            inplace=True,
            preserve_invalid=True,
        )

        return d

    @_inplace_enabled(default=False)
    def arcsinh(self, inplace=False):
        """Take the inverse hyperbolic sine of the data element-wise.

        Units are ignored in the calculation. The result has units of radians.

        .. versionadded:: 3.1.0

        .. seealso:: `sinh`, `arccosh`, `arctanh`, `arcsin`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> print(d.array)
        [[0.5 0.7]
         [0.9 1.1]]
        >>> e = d.arcsinh()
        >>> e.Units
        <Units: radians>
        >>> print(e.array)
        [[0.48121183 0.65266657]
         [0.80886694 0.95034693]]

        >>> print(d.array)
        [1.2 1.0 0.8 0.6 --]
        >>> d.arcsinh(inplace=True)
        >>> print(d.array)
        [1.015973134179692 0.881373587019543 0.732668256045411 0.5688248987322475
         --]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        dx = d.to_dask_array()
        dx = da.arcsinh(dx)
        d._set_dask(dx)

        d.override_units(_units_radians, inplace=True)

        return d

    @_inplace_enabled(default=False)
    def arccos(self, inplace=False):
        """Take the trigonometric inverse cosine of the data element-
        wise.

        Units are ignored in the calculation. The result has units of radians.

        .. versionadded:: 3.2.0

        .. seealso:: `cos`, `arcsin`, `arctan`, `arccosh`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> print(d.array)
        [[0.5 0.7]
         [0.9 1.1]]
        >>> e = d.arccos()
        >>> e.Units
        <Units: radians>
        >>> print(e.array)
        [[1.04719755 0.79539883]
         [0.45102681        nan]]

        >>> print(d.array)
        [1.2 1.0 0.8 0.6 --]
        >>> d.arccos(inplace=True)
        >>> print(d.array)
        [nan 0.0 0.6435011087932843 0.9272952180016123 --]
        >>> d.masked_invalid(inplace=True)
        >>> print(d.array)
        [-- 0.0 0.6435011087932843 0.9272952180016123 --]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        # Data.func is used instead of the Dask built-in in this case because
        # arccos has a restricted domain therefore it is necessary to use our
        # custom logic implemented via the `preserve_invalid` keyword to func.
        d.func(
            np.arccos,
            units=_units_radians,
            inplace=True,
            preserve_invalid=True,
        )

        return d

    @_inplace_enabled(default=False)
    def arccosh(self, inplace=False):
        """Take the inverse hyperbolic cosine of the data element-wise.

        Units are ignored in the calculation. The result has units of radians.

        .. versionadded:: 3.2.0

        .. seealso::  `cosh`, `arcsinh`, `arctanh`, `arccos`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> print(d.array)
        [[0.5 0.7]
         [0.9 1.1]]
        >>> e = d.arccosh()
        >>> e.Units
        <Units: radians>
        >>> print(e.array)
        [[       nan        nan]
         [       nan 0.44356825]]

        >>> print(d.array)
        [1.2 1.0 0.8 0.6 --]
        >>> d.arccosh(inplace=True)
        >>> print(d.array)
        [0.6223625037147786 0.0 nan nan --]
        >>> d.masked_invalid(inplace=True)
        >>> print(d.array)
        [0.6223625037147786 0.0 -- -- --]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        # Data.func is used instead of the Dask built-in in this case because
        # arccosh has a restricted domain therefore it is necessary to use our
        # custom logic implemented via the `preserve_invalid` keyword to func.
        d.func(
            np.arccosh,
            units=_units_radians,
            inplace=True,
            preserve_invalid=True,
        )

        return d

    def allclose(self, y, rtol=None, atol=None):
        """Whether an array is element-wise equal within a tolerance.

        Return True if the data is broadcastable to array *y* and
        element-wise equal within a tolerance.

        {{equals tolerance}}

        .. seealso:: `all`, `any`, `isclose`

        :Parameters:

            y: data_like
                The data to compare.

            {{rtol: number, optional}}

            {{atol: number, optional}}

        :Returns:

            `Data`
                A scalar boolean array that is `True` if the two arrays
                are equal within the given tolerance, or `False`
                otherwise.

        **Examples**

        >>> d = cf.Data([1000, 2500], 'metre')
        >>> e = cf.Data([1, 2.5], 'km')
        >>> bool(d.allclose(e))
        True

        >>> d = cf.Data(['ab', 'cdef'])
        >>> bool(d.allclose([[['ab', 'cdef']]]))
        True

        >>> d = cf.Data([[1000, 2500], [1000, 2500]], 'metre')
        >>> e = cf.Data([1, 2.5], 'km')
        >>> bool(d.allclose(e))
        True

        >>> d = cf.Data([1, 1, 1], 's')
        >>> bool(d.allclose(1))
        True

        """
        return self.isclose(y, rtol=rtol, atol=atol).all()

    def argmax(self, axis=None, unravel=False):
        """Return the indices of the maximum values along an axis.

        If no axis is specified then the returned index locates the
        maximum of the whole data.

        In case of multiple occurrences of the maximum values, the
        indices corresponding to the first occurrence are returned.

        **Performance**

        If the data index is returned as a `tuple` (see the *unravel*
        parameter) then all delayed operations are computed.

        .. versionadded:: 3.0.0

        .. seealso:: `argmin`

        :Parameters:

            axis: `int`, optional
                The specified axis over which to locate the maximum
                values. By default the maximum over the flattened data
                is located.

            unravel: `bool`, optional
                If True then when locating the maximum over the whole
                data, return the location as an integer index for each
                axis as a `tuple`. By default an index to the
                flattened array is returned in this case. Ignored if
                locating the maxima over a subset of the axes.

        :Returns:

            `Data` or `tuple` of `int`
                The location of the maximum, or maxima.

        **Examples**

        >>> d = cf.Data(np.arange(6).reshape(2, 3))
        >>> print(d.array)
        [[0 1 2]
         [3 4 5]]
        >>> a = d.argmax()
        >>> a
        <CF Data(): 5>
        >>> print(a.array)
        5

        >>> index = d.argmax(unravel=True)
        >>> index
        (1, 2)
        >>> d[index]
        <CF Data(1, 1): [[5]]>

        >>> d.argmax(axis=0)
        <CF Data(3): [1, 1, 1]>
        >>> d.argmax(axis=1)
        <CF Data(2): [2, 2]>

        Only the location of the first occurrence is returned:

        >>> d = cf.Data([0, 4, 2, 3, 4])
        >>> d.argmax()
        <CF Data(): 1>

        >>> d = cf.Data(np.arange(6).reshape(2, 3))
        >>> d[1, 1] = 5
        >>> print(d.array)
        [[0 1 2]
         [3 5 5]]
        >>> d.argmax(axis=1)
        <CF Data(2): [2, 1]>

        """
        dx = self.to_dask_array()
        a = dx.argmax(axis=axis)

        if unravel and (axis is None or self.ndim <= 1):
            # Return a multidimensional index tuple
            return tuple(np.array(da.unravel_index(a, self.shape)))

        return type(self)(a)

    def argmin(self, axis=None, unravel=False):
        """Return the indices of the minimum values along an axis.

        If no axis is specified then the returned index locates the
        minimum of the whole data.

        In case of multiple occurrences of the minimum values, the
        indices corresponding to the first occurrence are returned.

        **Performance**

        If the data index is returned as a `tuple` (see the *unravel*
        parameter) then all delayed operations are computed.

        .. versionadded:: 3.15.1

        .. seealso:: `argmax`

        :Parameters:

            axis: `int`, optional
                The specified axis over which to locate the minimum
                values. By default the minimum over the flattened data
                is located.

            unravel: `bool`, optional
                If True then when locating the minimum over the whole
                data, return the location as an integer index for each
                axis as a `tuple`. By default an index to the
                flattened array is returned in this case. Ignored if
                locating the minima over a subset of the axes.

        :Returns:

            `Data` or `tuple` of `int`
                The location of the minimum, or minima.

        **Examples**

        >>> d = cf.Data(np.arange(5, -1, -1).reshape(2, 3))
        >>> print(d.array)
        [[5 4 3]
         [2 1 0]]
        >>> a = d.argmin()
        >>> a
        <CF Data(): 5>
        >>> print(a.array)
        5

        >>> index = d.argmin(unravel=True)
        >>> index
        (1, 2)
        >>> d[index]
        <CF Data(1, 1): [[0]]>

        >>> d.argmin(axis=0)
        <CF Data(3): [1, 1, 1]>
        >>> d.argmin(axis=1)
        <CF Data(2): [2, 2]>

        Only the location of the first occurrence is returned:

        >>> d = cf.Data([4, 0, 2, 3, 0])
        >>> d.argmin()
        <CF Data(): 1>

        >>> d = cf.Data(np.arange(5, -1, -1).reshape(2, 3))
        >>> d[1, 1] = 0
        >>> print(d.array)
        [[5 4 3]
         [2 0 0]]
        >>> d.argmin(axis=1)
        <CF Data(2): [2, 1]>

        """
        dx = self.to_dask_array()
        a = dx.argmin(axis=axis)

        if unravel and (axis is None or self.ndim <= 1):
            # Return a multidimensional index tuple
            return tuple(np.array(da.unravel_index(a, self.shape)))

        return type(self)(a)

    @_inplace_enabled(default=False)
    def convert_reference_time(
        self,
        units=None,
        calendar_months=False,
        calendar_years=False,
        inplace=False,
    ):
        """Convert reference time data values to have new units.

        Conversion is done by decoding the reference times to
        date-time objects and then re-encoding them for the new units.

        Any conversions are possible, but this method is primarily for
        conversions which require a change in the date-times
        originally encoded. For example, use this method to
        reinterpret data values in units of "months" since a reference
        time to data values in "calendar months" since a reference
        time. This is often necessary when units of "calendar months"
        were intended but encoded as "months", which have special
        definition. See the note and examples below for more details.

        .. note:: It is recommended that the units "year" and "month"
                  be used with caution, as explained in the following
                  excerpt from the CF conventions: "The Udunits
                  package defines a year to be exactly 365.242198781
                  days (the interval between 2 successive passages of
                  the sun through vernal equinox). It is not a
                  calendar year. Udunits includes the following
                  definitions for years: a common_year is 365 days, a
                  leap_year is 366 days, a Julian_year is 365.25 days,
                  and a Gregorian_year is 365.2425 days. For similar
                  reasons the unit ``month``, which is defined to be
                  exactly year/12, should also be used with caution.

        **Performance**

        For conversions which do not require a change in the
        date-times implied by the data orginal values, this method
        will be considerably slower than a simple reassignment of the
        units. For example, if the original units are ``'days since
        2000-12-1'`` then ``d.Units = cf.Units('days since
        1901-1-1')`` will give the same result and be considerably
        faster than ``d.convert_reference_time(cf.Units('days since
        1901-1-1'))``.

        .. versionadded:: 3.14.0

        .. seeealso:: `change_calendar`, `datetime_array`, `Units`

        :Parameters:

            units: `Units`, optional
                The reference time units to convert to. By default the
                units are days since the original reference time in
                the original calendar.

                *Parameter example:*
                  If the original units are ``'months since
                  2000-1-1'`` in the Gregorian calendar then the
                  default units to convert to are ``'days since
                  2000-1-1'`` in the Gregorian calendar.

            calendar_months: `bool`, optional
                If True then treat units of ``'months'`` as if they
                were calendar months (in whichever calendar is
                originally specified), rather than a 12th of the
                interval between two successive passages of the sun
                through vernal equinox (i.e. 365.242198781/12 days).

            calendar_years: `bool`, optional
                If True then treat units of ``'years'`` as if they
                were calendar years (in whichever calendar is
                originally specified), rather than the interval
                between two successive passages of the sun through
                vernal equinox (i.e. 365.242198781 days).

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The data with converted reference time values, or
                `None` if the operation was in-place.

        **Examples**

        >>> d = cf.Data([1, 2, 3, 4], units="months since 2004-1-1")
        >>> d.Units
        <Units: months since 2004-1-1>
        >>> print(d.datetime_array)
        [cftime.DatetimeGregorian(2003, 12, 1, 0, 0, 0, 0, has_year_zero=False)
         cftime.DatetimeGregorian(2003, 12, 31, 10, 29, 3, 831223, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 1, 30, 20, 58, 7, 662446, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 3, 1, 7, 27, 11, 493670, has_year_zero=False)]
        >>> print(d.array)
        [0 1 2 3]
        >>> e = d.convert_reference_time(calendar_months=True)
        >>> e.Units
        <Units: days since 2004-1-1>
        >>> print(e.datetime_array)
        [cftime.DatetimeGregorian(2003, 12, 1, 0, 0, 0, 0, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 1, 1, 0, 0, 0, 0, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 2, 1, 0, 0, 0, 0, has_year_zero=False)
         cftime.DatetimeGregorian(2004, 3, 1, 0, 0, 0, 0, has_year_zero=False)]
        >>> print(e.array)
        [ 0 31 62 91]

        """
        units0 = self.Units

        if not units0.isreftime:
            raise ValueError(
                f"{self.__class__.__name__} must have reference time units. "
                f"Got {units0!r}"
            )

        d = _inplace_enabled_define_and_cleanup(self)

        if units is None:
            # By default, set the target units to "days since
            # <reference time of units0>, calendar=<calendar of
            # units0>"
            units = self._Units_class(
                "days since " + units0.units.split(" since ")[1],
                calendar=units0._calendar,
            )
        elif not getattr(units, "isreftime", False):
            raise ValueError(
                f"New units must be reference time units, not {units!r}"
            )

        units0_since_reftime = units0._units_since_reftime
        if units0_since_reftime in _month_units:
            if calendar_months:
                units0 = self._Units_class(
                    "calendar_" + units0.units, calendar=units0._calendar
                )
            else:
                units0 = self._Units_class(
                    "days since " + units0.units.split(" since ")[1],
                    calendar=units0._calendar,
                )
                d.Units = units0
        elif units0_since_reftime in _year_units:
            if calendar_years:
                units0 = self._Units_class(
                    "calendar_" + units0.units, calendar=units0._calendar
                )
            else:
                units0 = self._Units_class(
                    "days since " + units0.units.split(" since ")[1],
                    calendar=units0._calendar,
                )
                d.Units = units0

        # 'cf_rt2dt' its own call to 'cfdm_to_memory', so we can set
        # '_force_to_memory=False'.
        dx = d.to_dask_array(_force_to_memory=False)

        # Convert to the correct date-time objects
        dx = dx.map_blocks(cf_rt2dt, units=units0, dtype=object)

        # Convert the date-time objects to reference times
        dx = dx.map_blocks(cf_dt2rt, units=units, dtype=float)

        d._set_dask(dx)
        d.override_units(units, inplace=True)

        return d

    def set_units(self, value):
        """Set the units.

        .. seealso:: `override_units`, `del_units`, `get_units`,
                     `has_units`, `Units`

        :Parameters:

            value: `str`
                The new units.

        :Returns:

            `None`

        **Examples**

        >>> d.set_units('watt')
        >>> d.get_units()
        'watt'
        >>> d.del_units()
        >>> d.get_units()
        ValueError: Can't get non-existent units
        >>> print(d.get_units(None))
        None

        """
        self.Units = self._Units_class(value, self.get_calendar(default=None))

    @_inplace_enabled(default=False)
    def masked_where(self, condition, inplace=False):
        """Mask the data where a condition is met.

        ``d.masked_where(condition)`` is equivalent to
        ``d.where(condition, cf.masked)``.

        **Performance**

        `masked_where` causes all delayed operations to be executed.

        .. versionadded:: 3.16.3

        .. seealso:: `mask`, `masked_values`, `where`

        :Parameters:

            condition: array_like
                The masking condition. The data is masked where
                *condition* is True. Any masked values already in the
                data are also masked in the result.

            {{inplace: `bool`, optional}}

        :Returns:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The result of masking the data, or `None` if the
                operation was in-place.

        **Examples**

        >>> d = cf.Data([1, 2, 3, 4, 5])
        >>> e = d.masked_where([0, 1, 0, 1, 0])
        >>> print(e.array)
        [1 -- 3 -- 5]

        """
        return self.where(condition, masked, None, inplace=inplace)

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def max(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate maximum values.

        Calculates the maximum value or the maximum values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `maximum_absolute_value`, `min`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.max()
        <CF Data(1, 1): [[11]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().max,
            d,
            axis=axes,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )

        return d

    @_inplace_enabled(default=False)
    def maximum_absolute_value(
        self, axes=None, squeeze=False, mtol=1, split_every=None, inplace=False
    ):
        """Calculate maximum absolute values.

        Calculates the maximum absolute value or the maximum absolute
        values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `max`, `minimum_absolute_value`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[-99 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.maximum_absolute_value()
        <CF Data(1, 1): [[99]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().max_abs,
            d,
            axis=axes,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def min(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate minimum values.

        Calculates the minimum value or the minimum values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `max`, `minimum_absolute_value`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.min()
        <CF Data(1, 1): [[0]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().min,
            d,
            axis=axes,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_inplace_enabled(default=False)
    def minimum_absolute_value(
        self, axes=None, squeeze=False, mtol=1, split_every=None, inplace=False
    ):
        """Calculate minimum absolute values.

        Calculates the minimum absolute value or the minimum absolute
        values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `maximum_absolute_value`, `min`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[0, 0] = -99
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[-99 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.minimum_absolute_value()
        <CF Data(1, 1): [[1]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().min_abs,
            d,
            axis=axes,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def mean(
        self,
        axes=None,
        weights=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate mean values.

        Calculates the mean value or the mean values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `mean_abslute_value`, `sd`, `sum`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.mean()
        <CF Data(1, 1): [[5.636363636363637]] K>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.mean(weights=w)
        <CF Data(1, 1): [[5.878787878787879]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().mean,
            d,
            axis=axes,
            weights=weights,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_inplace_enabled(default=False)
    def mean_absolute_value(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        weights=None,
        split_every=None,
        inplace=False,
    ):
        """Calculate mean absolute values.

        Calculates the mean absolute value or the mean absolute values
        along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `mean`, `sd`, `sum`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[0, 0] = -99
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[-99 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.mean_absolute_value()
        <CF Data(1, 1): [[14.636363636363637]] K>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.mean_absolute_value(weights=w)
        <CF Data(1, 1): [[11.878787878787879]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().mean_abs,
            d,
            axis=axes,
            weights=weights,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_inplace_enabled(default=False)
    def integral(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        weights=None,
        split_every=None,
        inplace=False,
    ):
        """Calculate summed values.

        Calculates the sum value or the sum values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `mean`, `sd`, `sum`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.integral()
        <CF Data(1, 1): [[62]] K>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.integral(weights=w)
        <CF Data(1, 1): [[97.0]] K>

        >>> d.integral(weights=cf.Data(w, 'm'))
        <CF Data(1, 1): [[97.0]] m.K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, weights = collapse(
            Collapse().sum,
            d,
            axis=axes,
            weights=weights,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )

        new_units = None
        if weights is not None:
            weights_units = getattr(weights, "Units", None)
            if weights_units:
                units = self.Units
                if units:
                    new_units = units * weights_units
                else:
                    new_units = weights_units

        if new_units is not None:
            d.override_units(new_units, inplace=True)

        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def sample_size(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate sample size values.

        The sample size is the number of non-missing values.

        Calculates the sample size value or the sample size values
        along axes.

        .. seealso:: `sum_of_weights`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.sample_size()
        <CF Data(1, 1): [[11]]>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().sample_size,
            d,
            axis=axes,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        d.override_units(_units_None, inplace=True)

        return d

    @property
    def binary_mask(self):
        """A binary (0 and 1) mask of the data array.

        The binary mask's data array comprises dimensionless 32-bit
        integers and has 0 where the data array has missing data and 1
        otherwise.

        .. seealso:: `mask`

        :Returns:

            `Data`
                The binary mask.

        **Examples**

        >>> d = cf.Data([[0, 1, 2, 3]], 'm')
        >>> m = d.binary_mask
        >>> m
        <CF Data(1, 4): [[0, ..., 0]] 1>
        >>> print(m.array)
        [[0 0 0 0]]
        >>> d[0, 1] = cf.masked
        >>> print(d.binary_mask.array)
        [[0 1 0 0]]

        """
        m = self.mask
        m.dtype = "int32"
        m.override_units(_units_1, inplace=True)
        return m

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def clip(self, a_min, a_max, units=None, inplace=False, i=False):
        """Clip (limit) the values in the data array in place.

        Given an interval, values outside the interval are clipped to
        the interval edges. For example, if an interval of [0, 1] is
        specified then values smaller than 0 become 0 and values
        larger than 1 become 1.

        :Parameters:

            a_min: number
                Minimum value. If `None`, clipping is not performed on
                lower interval edge. Not more than one of `a_min` and
                `a_max` may be `None`.

            a_max: number
                Maximum value. If `None`, clipping is not performed on
                upper interval edge. Not more than one of `a_min` and
                `a_max` may be `None`.

            units: `str` or `Units`
                Specify the units of *a_min* and *a_max*. By default the
                same units as the data are assumed.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The clipped data. If the operation was in-place then
                `None` is returned.


        **Examples**

        >>> d = cf.Data(np.arange(12).reshape(3, 4), 'm')
        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> print(d.clip(2, 10).array)
        [[ 2  2  2  3]
         [ 4  5  6  7]
         [ 8  9 10 10]]
        >>> print(d.clip(0.003, 0.009, 'km').array)
        [[3. 3. 3. 3.]
         [4. 5. 6. 7.]
         [8. 9. 9. 9.]]

        """
        if units is not None:
            # Convert the limits to the same units as the data array
            units = self._Units_class(units)
            self_units = self.Units
            if self_units != units:
                a_min = Units.conform(np.asanyarray(a_min), units, self_units)
                a_max = Units.conform(np.asanyarray(a_max), units, self_units)

        d = _inplace_enabled_define_and_cleanup(self)
        dx = self.to_dask_array()
        dx = da.clip(dx, a_min, a_max)
        d._set_dask(dx)
        return d

    @classmethod
    def arctan2(cls, x1, x2):
        """Element-wise arc tangent of ``x1/x2`` with correct quadrant.

        The quadrant (i.e. branch) is chosen so that ``arctan2(y, x)``
        is the signed angle in radians between the ray ending at the
        origin and passing through the point ``(1, 0)``, and the ray
        ending at the origin and passing through the point ``(x, y)``.
        (Note the role reversal: the "y-coordinate" is the first
        function parameter, the "x-coordinate" is the second.) By IEEE
        convention, this function is defined for ``x = +/-0`` and for
        either or both of ``y = +/-inf`` and ``x = +/-inf`` (see Notes
        for specific values).

        `arctan2` is identical to the ``atan2`` function of the
        underlying C library. The following special values are defined
        in the C standard:

        ======  ======  ===================
        *x1*    *x2*    ``arctan2(x1, x2)``
        ======  ======  ===================
        +/- 0   +0      +/- 0
        +/- 0   -0      +/- pi
        > 0     +/-inf  +0 / +pi
        < 0     +/-inf  -0 / -pi
        +/-inf  +inf    +/- (pi/4)
        +/-inf  -inf    +/- (3*pi/4)
        ======  ======  ===================

        Note that ``+0`` and ``-0`` are distinct floating point
        numbers, as are ``+inf`` and ``-inf``.

        .. versionadded:: 3.16.0

        .. seealso:: `arctan`, `tan`

        :Parameters:

            x1: array_like
                Y coordinates.

            x2: array_like
                X coordinates. *x1* and *x2* must be broadcastable to
                a common shape (which becomes the shape of the
                output).

        :Returns:

            `Data`
                Array of angles in radians, in the range ``(-pi,
                pi]``.

        **Examples**

        >>> import numpy as np
        >>> x = cf.Data([-1, +1, +1, -1])
        >>> y = cf.Data([-1, -1, +1, +1])
        >>> print((cf.Data.arctan2(y, x) * 180 / np.pi).array)
        [-135.0 -45.0 45.0 135.0]
        >>> x[1] = cf.masked
        >>> y[1] = cf.masked
        >>> print((cf.Data.arctan2(y, x) * 180 / np.pi).array)
        [-135.0 -- 45.0 135.0]

        >>> print(cf.Data.arctan2([0, 0, np.inf], [+0., -0., np.inf]).array)
        [0.0 3.141592653589793 0.7853981633974483]

        >>> print((cf.Data.arctan2([1, -1], [0, 0]) * 180 / np.pi).array)
        [90.0 -90.0]

        """
        try:
            y = x1.to_dask_array()
        except AttributeError:
            y = cls.asdata(x1).to_dask_array()

        try:
            x = x2.to_dask_array()
        except AttributeError:
            x = cls.asdata(x2).to_dask_array()

        mask = da.ma.getmaskarray(y) | da.ma.getmaskarray(x)
        y = da.ma.filled(y, 1)
        x = da.ma.filled(x, 1)

        dx = da.arctan2(y, x)
        dx = da.ma.masked_array(dx, mask=mask)

        return cls(dx, units=_units_radians)

    @_inplace_enabled(default=False)
    def compressed(self, inplace=False):
        """Return all non-masked values in a one dimensional data array.

        Not to be confused with compression by convention (see the
        `uncompress` method).

        .. versionadded:: 3.2.0

        .. seealso:: `flatten`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The non-masked values, or `None` if the operation was
                in-place.

        **Examples**

        >>> d = cf.Data(numpy.arange(12).reshape(3, 4), 'm')
        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> print(d.compressed().array)
        [ 0  1  2  3  4  5  6  7  8  9 10 11]
        >>> d[1, 1] = cf.masked
        >>> d[2, 3] = cf.masked
        >>> print(d.array)
        [[0  1  2  3]
         [4 --  6  7]
         [8  9 10 --]]
        >>> print(d.compressed().array)
        [ 0  1  2  3  4  6  7  8  9 10]

        >>> d = cf.Data(9)
        >>> print(d.compressed().array)
        [9]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        dx = d.to_dask_array()
        dx = da.blockwise(
            np.ma.compressed,
            "i",
            dx.ravel(),
            "i",
            adjust_chunks={"i": lambda n: np.nan},
            dtype=dx.dtype,
            meta=np.array((), dtype=dx.dtype),
        )

        d._set_dask(dx)
        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def cos(self, inplace=False, i=False):
        """Take the trigonometric cosine of the data element-wise.

        Units are accounted for in the calculation. If the units are not
        equivalent to radians (such as Kelvin) then they are treated as if
        they were radians. For example, the cosine of 90 degrees_east
        is 0.0, as is the cosine of 1.57079632 kg m-2.

        The output units are changed to '1' (nondimensional).

        .. seealso:: `arccos`, `sin`, `tan`, `cosh`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d.Units
        <Units: degrees_east>
        >>> print(d.array)
        [[-90 0 90 --]]
        >>> e = d.cos()
        >>> e.Units
        <Units: 1>
        >>> print(e.array)
        [[0.0 1.0 0.0 --]]

        >>> d.Units
        <Units: m s-1>
        >>> print(d.array)
        [[1 2 3 --]]
        >>> d.cos(inplace=True)
        >>> d.Units
        <Units: 1>
        >>> print(d.array)
        [[0.540302305868 -0.416146836547 -0.9899924966 --]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if d.Units.equivalent(_units_radians):
            d.Units = _units_radians

        dx = d.to_dask_array()
        dx = da.cos(dx)
        d._set_dask(dx)

        d.override_units(_units_1, inplace=True)

        return d

    def count(self, axis=None, keepdims=True, split_every=None):
        """Count the non-masked elements of the data.

        .. seealso:: `count_masked`

        :Parameters:

            axis: (sequence of) `int`, optional
                Axis or axes along which the count is performed. The
                default (`None`) performs the count over all the
                dimensions of the input array. *axis* may be negative,
                in which case it counts from the last to the first
                axis.

            {{collapse keepdims: `bool`, optional}}

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `Data`
                The count of non-missing elements.

        **Examples**

        >>> d = cf.Data(numpy.arange(12).reshape(3, 4))
        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> d.count()
        <CF Data(1, 1): [[12]]>

        >>> d[0, :] = cf.masked
        >>> print(d.array)
        [[-- -- -- --]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> d.count()
        <CF Data(1, 1): [[8]]>

        >>> print(d.count(0).array)
        [[2 2 2 2]]
        >>> print(d.count(1).array)
        [[0]
         [4]
         [4]]
        >>> print(d.count([0, 1], keepdims=False).array)
        8

        """
        d = self.copy(array=False)
        dx = self.to_dask_array()
        dx = da.ma.count(
            dx, axis=axis, keepdims=keepdims, split_every=split_every
        )
        d._set_dask(dx)
        d.hardmask = self._DEFAULT_HARDMASK
        d.override_units(_units_None, inplace=True)
        return d

    def count_masked(self, split_every=None):
        """Count the masked elements of the data.

        .. seealso:: `count`

        :Parameters:

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `Data`
                The count of missing elements.

        **Examples**

        >>> d = cf.Data(numpy.arange(12).reshape(3, 4))
        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> d.count_masked()
        <CF Data(1, 1): [[0]]>

        >>> d[0, :] = cf.masked
        >>> print(d.array)
        [[-- -- -- --]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> d.count_masked()
        <CF Data(1, 1): [[4]]>

        """
        return self.size - self.count(split_every=split_every)

    def cyclic(self, axes=None, iscyclic=True):
        """Get or set the cyclic axes.

        Some methods treat the first and last elements of a cyclic
        axis as adjacent and physically connected, such as
        `convolution_filter`, `__getitem__` and `__setitem__`. Some
        methods may make a cyclic axis non-cyclic, such as `halo`.

        :Parameters:

            axes: (sequence of) `int`, optional
                Select the axes to have their cyclicity set. By
                default, or if *axes* is `None` or an empty sequence,
                no axes are modified.

            iscyclic: `bool`
                Specify whether to make the axes cyclic or
                non-cyclic. By default (True), the axes are set as
                cyclic.

        :Returns:

            `set`
                The cyclic axes prior to the change, or the current
                cyclic axes if no axes are specified.

        **Examples**

        >>> d = cf.Data(np.arange(12).reshape(3, 4))
        >>> d.cyclic()
        set()
        >>> d.cyclic(0)
        set()
        >>> d.cyclic()
        {0}
        >>> d.cyclic(0, iscyclic=False)
        {0}
        >>> d.cyclic()
        set()
        >>> d.cyclic([0, 1])
        set()
        >>> d.cyclic()
        {0, 1}
        >>> d.cyclic([0, 1], iscyclic=False)
        {0, 1}
        >>> d.cyclic()
        set()

        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> d[0, -1:2]
        Traceback (most recent call last):
            ...
        IndexError: Can't take a cyclic slice of a non-cyclic axis
        >>> d.cyclic(1)
        set()
        >>> d[0, -1:2]
        <CF Data(1, 2): [[3, 0, 1]]>

        """
        cyclic_axes = self._cyclic
        data_axes = self._axes

        old = set([data_axes.index(axis) for axis in cyclic_axes])

        if axes is None:
            return old

        axes = [data_axes[i] for i in self._parse_axes(axes)]

        # Never change the value of the _cyclic attribute in-place
        if iscyclic:
            self._cyclic = cyclic_axes.union(axes)
        else:
            self._cyclic = cyclic_axes.difference(axes)

        return old

    @property
    def year(self):
        """The year of each date-time value.

        Only applicable for data with reference time units. The
        returned `Data` will have the same mask hardness as the
        original array.

        .. seealso:: `~cf.Data.month`, `~cf.Data.day`, `~cf.Data.hour`,
                     `~cf.Data.minute`, `~cf.Data.second`

        **Examples**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data(1, 2): [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.year
        <CF Data(1, 2): [[2000, 2001]] >

        """
        return YMDhms(self, "year")

    @property
    def month(self):
        """The month of each date-time value.

        Only applicable for data with reference time units. The
        returned `Data` will have the same mask hardness as the
        original array.

        .. seealso:: `~cf.Data.year`, `~cf.Data.day`, `~cf.Data.hour`,
                     `~cf.Data.minute`, `~cf.Data.second`

        **Examples**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data(1, 2): [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.month
        <CF Data(1, 2): [[12, 1]] >

        """
        return YMDhms(self, "month")

    @property
    def day(self):
        """The day of each date-time value.

        Only applicable for data with reference time units. The
        returned `Data` will have the same mask hardness as the
        original array.

        .. seealso:: `~cf.Data.year`, `~cf.Data.month`, `~cf.Data.hour`,
                     `~cf.Data.minute`, `~cf.Data.second`

        **Examples**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data(1, 2): [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.day
        <CF Data(1, 2): [[30, 3]] >

        """
        return YMDhms(self, "day")

    @property
    def hour(self):
        """The hour of each date-time value.

        Only applicable for data with reference time units. The
        returned `Data` will have the same mask hardness as the
        original array.

        .. seealso:: `~cf.Data.year`, `~cf.Data.month`, `~cf.Data.day`,
                     `~cf.Data.minute`, `~cf.Data.second`

        **Examples**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data(1, 2): [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.hour
        <CF Data(1, 2): [[22, 4]] >

        """
        return YMDhms(self, "hour")

    @property
    def minute(self):
        """The minute of each date-time value.

        Only applicable for data with reference time units. The
        returned `Data` will have the same mask hardness as the
        original array.

        .. seealso:: `~cf.Data.year`, `~cf.Data.month`, `~cf.Data.day`,
                     `~cf.Data.hour`, `~cf.Data.second`

        **Examples**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data(1, 2): [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.minute
        <CF Data(1, 2): [[19, 4]] >

        """
        return YMDhms(self, "minute")

    @property
    def second(self):
        """The second of each date-time value.

        Only applicable for data with reference time units. The
        returned `Data` will have the same mask hardness as the
        original array.

        .. seealso:: `~cf.Data.year`, `~cf.Data.month`, `~cf.Data.day`,
                     `~cf.Data.hour`, `~cf.Data.minute`

        **Examples**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data(1, 2): [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.second
        <CF Data(1, 2): [[12, 48]] >

        """
        return YMDhms(self, "second")

    def unique(self, split_every=None):
        """The unique elements of the data.

        Returns the sorted unique elements of the array.

        :Parameters:

            {{split_every: `int` or `dict`, optional}}

        :Returns:

            `Data`
                The unique values in a 1-d array.

        **Examples**

        >>> d = cf.Data([[4, 2, 1], [1, 2, 3]], 'metre')
        >>> print(d.array)
        [[4 2 1]
         [1 2 3]]
        >>> e = d.unique()
        >>> e
        <CF Data(4): [1, ..., 4] metre>
        >>> print(e.array)
        [1 2 3 4]
        >>> d[0, 0] = cf.masked
        >>> print(d.array)
        [[-- 2 1]
         [1 2 3]]
        >>> e = d.unique()
        >>> print(e.array)
        [1 2 3 --]

        """
        d = self.copy()

        # Soften the hardmask so that the result doesn't contain a
        # seperate missing value for each input chunk that contains
        # missing values. For any number greater than 0 of missing
        # values in the original data, we only want one missing value
        # in the result.
        d.soften_mask()

        # The applicable chunk function will have its own call to
        # 'cfdm_to_memory', so we can set '_force_to_memory=False'.
        dx = d.to_dask_array(_force_to_memory=False)
        dx = Collapse().unique(dx, split_every=split_every)

        d._set_dask(dx)

        d.hardmask = self._DEFAULT_HARDMASK

        return d

    @_display_or_return
    def dump(self, display=True, prefix=None):
        """Return a string containing a full description of the
        instance.

        :Parameters:

            display: `bool`, optional
                If False then return the description as a string. By
                default the description is printed, i.e. ``d.dump()`` is
                equivalent to ``print(d.dump(display=False))``.

            prefix: `str`, optional
               Set the common prefix of component names. By default the
               instance's class name is used.

        :Returns:

            `None` or `str`
                A string containing the description.

        """
        if prefix is None:
            prefix = self.__class__.__name__

        string = [f"{prefix}.shape = {self.shape}"]

        if self.size == 1:
            string.append(f"{prefix}.first_datum = {self.datum(0)}")
        else:
            string.append(f"{prefix}.first_datum = {self.datum(0)}")
            string.append(f"{prefix}.last_datum  = {self.datum(-1)}")

        for attr in ("fill_value", "Units"):
            string.append(f"{prefix}.{attr} = {getattr(self, attr)!r}")

        return "\n".join(string)

    def ndindex(self):
        """Return an iterator over the N-dimensional indices of the data
        array.

        At each iteration a tuple of indices is returned, the last
        dimension is iterated over first.

        :Returns:

            `itertools.product`
                An iterator over tuples of indices of the data array.

        **Examples**

        >>> d = cf.Data(np.arange(6).reshape(2, 3))
        >>> print(d.array)
        [[0 1 2]
         [3 4 5]]
        >>> for i in d.ndindex():
        ...     print(i, d[i])
        ...
        (0, 0) [[0]]
        (0, 1) [[1]]
        (0, 2) [[2]]
        (1, 0) [[3]]
        (1, 1) [[4]]
        (1, 2) [[5]]

        >>> d = cf.Data(9)
        >>> for i in d.ndindex():
        ...     print(i, d[i])
        ...
        () 9

        """
        return product(*[range(0, r) for r in self.shape])

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def exp(self, inplace=False, i=False):
        """Take the exponential of the data array.

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        """
        d = _inplace_enabled_define_and_cleanup(self)

        units = self.Units
        if units and not units.isdimensionless:
            raise ValueError(
                "Can't take exponential of dimensional "
                f"quantities: {units!r}"
            )

        if d.Units:
            d.Units = _units_1

        dx = d.to_dask_array()
        dx = da.exp(dx)
        d._set_dask(dx)

        return d

    @_deprecated_kwarg_check("size", version="3.14.0", removed_at="5.0.0")
    @_inplace_enabled(default=False)
    @_manage_log_level_via_verbosity
    def halo(
        self,
        depth,
        axes=None,
        tripolar=None,
        fold_index=-1,
        inplace=False,
        verbose=None,
        size=None,
    ):
        """Expand the data by adding a halo.

        The halo contains the adjacent values up to the given
        depth(s). See the example for details.

        The halo may be applied over a subset of the data dimensions
        and each dimension may have a different halo size (including
        zero). The halo region is populated with a copy of the
        proximate values from the original data.

        **Cyclic axes**

        A cyclic axis that is expanded with a halo of at least size 1
        is no longer considered to be cyclic.

        **Tripolar domains**

        Data for global tripolar domains are a special case in that a
        halo added to the northern end of the "Y" axis must be filled
        with values that are flipped in "X" direction. Such domains
        need to be explicitly indicated with the *tripolar* parameter.

        .. versionadded:: 3.5.0

        :Parameters:

            depth: `int` or `dict`
                Specify the size of the halo for each axis.

                If *depth* is a non-negative `int` then this is the
                halo size that is applied to all of the axes defined
                by the *axes* parameter.

                Alternatively, halo sizes may be assigned to axes
                individually by providing a `dict` for which a key
                specifies an axis (defined by its integer position in
                the data) with a corresponding value of the halo size
                for that axis. Axes not specified by the dictionary
                are not expanded, and the *axes* parameter must not
                also be set.

                *Parameter example:*
                  Specify a halo size of 1 for all otherwise selected
                  axes: ``depth=1``.

                *Parameter example:*
                  Specify a halo size of zero ``depth=0``. This
                  results in no change to the data shape.

                *Parameter example:*
                  For data with three dimensions, specify a halo size
                  of 3 for the first dimension and 1 for the second
                  dimension: ``depth={0: 3, 1: 1}``. This is
                  equivalent to ``depth={0: 3, 1: 1, 2: 0}``.

                *Parameter example:*
                  Specify a halo size of 2 for the first and last
                  dimensions `depth=2, axes=[0, -1]`` or equivalently
                  ``depth={0: 2, -1: 2}``.

            axes: (sequence of) `int`
                Select the domain axes to be expanded, defined by
                their integer positions in the data. By default, or if
                *axes* is `None`, all axes are selected. No axes are
                expanded if *axes* is an empty sequence.

            tripolar: `dict`, optional
                A dictionary defining the "X" and "Y" axes of a global
                tripolar domain. This is necessary because in the
                global tripolar case the "X" and "Y" axes need special
                treatment, as described above. It must have keys
                ``'X'`` and ``'Y'``, whose values identify the
                corresponding domain axis construct by their integer
                positions in the data.

                The "X" and "Y" axes must be a subset of those
                identified by the *depth* or *axes* parameter.

                See the *fold_index* parameter.

                *Parameter example:*
                  Define the "X" and Y" axes by positions 2 and 1
                  respectively of the data: ``tripolar={'X': 2, 'Y':
                  1}``

            fold_index: `int`, optional
                Identify which index of the "Y" axis corresponds to
                the fold in "X" axis of a tripolar grid. The only
                valid values are ``-1`` for the last index, and ``0``
                for the first index. By default it is assumed to be
                the last index. Ignored if *tripolar* is `None`.

            {{inplace: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            size: deprecated at version 3.14.0
                Use the *depth* parameter instead.

        :Returns:

            `Data` or `None`
                The expanded data, or `None` if the operation was
                in-place.

        **Examples**

        >>> d = cf.Data(numpy.arange(12).reshape(3, 4), 'm')
        >>> d[-1, -1] = cf.masked
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2 3]
         [4 -- 6 7]
         [8 9 10 --]]

        >>> e = d.halo(1)
        >>> print(e.array)
        [[0 0 1 2 3 3]
         [0 0 1 2 3 3]
         [4 4 -- 6 7 7]
         [8 8 9 10 -- --]
         [8 8 9 10 -- --]]

        >>> d.equals(e[1:-1, 1:-1])
        True

        >>> e = d.halo(2)
        >>> print(e.array)
        [[0 1 0 1 2 3 2 3]
         [4 -- 4 -- 6 7 6 7]
         [0 1 0 1 2 3 2 3]
         [4 -- 4 -- 6 7 6 7]
         [8 9 8 9 10 -- 10 --]
         [4 -- 4 -- 6 7 6 7]
         [8 9 8 9 10 -- 10 --]]
        >>> d.equals(e[2:-2, 2:-2])
        True

        >>> e = d.halo(0)
        >>> d.equals(e)
        True

        >>> e = d.halo(1, axes=0)
        >>> print(e.array)
        [[0 1 2 3]
         [0 1 2 3]
         [4 -- 6 7]
         [8 9 10 --]
         [8 9 10 --]]

        >>> d.equals(e[1:-1, :])
        True
        >>> f = d.halo({0: 1})
        >>> f.equals(e)
        True

        >>> e = d.halo(1, tripolar={'X': 1, 'Y': 0})
        >>> print(e.array)
        [[0 0 1 2 3 3]
         [0 0 1 2 3 3]
         [4 4 -- 6 7 7]
         [8 8 9 10 -- --]
         [-- -- 10 9 8 8]]

        >>> e = d.halo(1, tripolar={'X': 1, 'Y': 0}, fold_index=0)
        >>> print(e.array)
        [[3 3 2 1 0 0]
         [0 0 1 2 3 3]
         [4 4 -- 6 7 7]
         [8 8 9 10 -- --]
         [8 8 9 10 -- --]]

        """
        from dask.array.core import concatenate

        if size is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "halo",
                {"size": None},
                message="Use the 'depth' parameter instead.",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        d = _inplace_enabled_define_and_cleanup(self)

        ndim = d.ndim
        shape = d.shape

        # Parse the depth and axes parameters
        if isinstance(depth, dict):
            if axes is not None:
                raise ValueError(
                    "Can't set the axes parameter when the "
                    "depth parameter is a dictionary"
                )

            # Check that the dictionary keys are OK and remove size
            # zero depths
            axes = self._parse_axes(tuple(depth))
            depth = {i: size for i, size in depth.items() if size}
        else:
            if axes is None:
                axes = list(range(ndim))
            else:
                axes = d._parse_axes(axes)

            depth = {i: depth for i in axes}

        # Return if all axis depths are zero
        if not any(depth.values()):
            return d

        # Parse the tripolar parameter
        if tripolar:
            if fold_index not in (0, -1):
                raise ValueError(
                    "fold_index parameter must be -1 or 0. "
                    f"Got {fold_index!r}"
                )

            # Find the X and Y axes of a tripolar grid
            tripolar = tripolar.copy()
            X_axis = tripolar.pop("X", None)
            Y_axis = tripolar.pop("Y", None)

            if tripolar:
                raise ValueError(
                    f"Can not set key {tripolar.popitem()[0]!r} in the "
                    "tripolar dictionary."
                )

            if X_axis is None:
                raise ValueError("Must provide a tripolar 'X' axis.")

            if Y_axis is None:
                raise ValueError("Must provide a tripolar 'Y' axis.")

            X = d._parse_axes(X_axis)
            Y = d._parse_axes(Y_axis)

            if len(X) != 1:
                raise ValueError(
                    "Must provide exactly one tripolar 'X' axis. "
                    f"Got {X_axis!r}"
                )

            if len(Y) != 1:
                raise ValueError(
                    "Must provide exactly one tripolar 'Y' axis. "
                    f"Got {Y_axis!r}"
                )

            X_axis = X[0]
            Y_axis = Y[0]

            if X_axis == Y_axis:
                raise ValueError(
                    "Tripolar 'X' and 'Y' axes must be different. "
                    f"Got {X_axis!r}, {Y_axis!r}"
                )

            for A, axis in zip(("X", "Y"), (X_axis, Y_axis)):
                if axis not in axes:
                    raise ValueError(
                        "If dimensions have been identified with the "
                        "axes or depth parameters then they must include "
                        f"the tripolar {A!r} axis: {axis!r}"
                    )

            tripolar = Y_axis in depth

        # Create the halo
        dx = d.to_dask_array()

        indices = [slice(None)] * ndim
        for axis, size in sorted(depth.items()):
            if not size:
                continue

            if size > shape[axis]:
                raise ValueError(
                    f"Halo depth {size} is too large for axis of size "
                    f"{shape[axis]}"
                )

            left_indices = indices[:]
            right_indices = indices[:]

            left_indices[axis] = slice(0, size)
            right_indices[axis] = slice(-size, None)

            left = dx[tuple(left_indices)]
            right = dx[tuple(right_indices)]

            dx = concatenate([left, dx, right], axis=axis)

        d._set_dask(dx)

        # Special case for tripolar: The northern Y axis halo contains
        # the values that have been flipped in the X direction.
        if tripolar:
            # Make sure that we can overwrite any missing values in
            # the northern Y axis halo
            d.soften_mask()

            indices1 = indices[:]
            if fold_index == -1:
                # The last index of the Y axis corresponds to the fold
                # in X axis of a tripolar grid
                indices1[Y_axis] = slice(-depth[Y_axis], None)
            else:
                # The first index of the Y axis corresponds to the
                # fold in X axis of a tripolar grid
                indices1[Y_axis] = slice(0, depth[Y_axis])

            indices2 = indices1[:]
            indices2[X_axis] = slice(None, None, -1)

            dx = d.to_dask_array()
            dx[tuple(indices1)] = dx[tuple(indices2)]

            d._set_dask(dx)

            # Reset the mask hardness
            d.hardmask = self.hardmask

        # Set expanded axes to be non-cyclic
        d.cyclic(axes=tuple(depth), iscyclic=False)

        return d

    def flat(self, ignore_masked=True):
        """Return a flat iterator over elements of the data array.

        **Performance**

        Any delayed operations and/or disk interactions will be
        executed during *each* iteration, possibly leading to poor
        performance. If possible, consider bringing the values into
        memory first with `persist` or using ``d.array.flat``.

        .. seealso:: `flatten`, `persist`

        :Parameters:

            ignore_masked: `bool`, optional
                If False then masked and unmasked elements will be
                returned. By default only unmasked elements are
                returned

        :Returns:

            generator
                An iterator over elements of the data array.

        **Examples**

        >>> d = cf.Data([[1, 2], [3,4]], mask=[[0, 1], [0, 0]])
        >>> print(d.array)
        [[1 --]
         [3 4]]
        >>> list(d.flat())
        [1, 3, 4]
        >>> list(d.flat(ignore_masked=False))
        [1, masked, 3, 4]

        """
        mask = self.mask

        if ignore_masked:
            for index in self.ndindex():
                if not mask[index]:
                    yield self[index].array.item()
        else:
            for index in self.ndindex():
                if not mask[index]:
                    yield self[index].array.item()
                else:
                    yield masked

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def floor(self, inplace=False, i=False):
        """Return the floor of the data array.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `rint`, `trunc`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d = cf.Data([-1.9, -1.5, -1.1, -1, 0, 1, 1.1, 1.5 , 1.9])
        >>> print(d.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(d.floor().array)
        [-2. -2. -2. -1.  0.  1.  1.  1.  1.]

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = d.to_dask_array()
        d._set_dask(da.floor(dx))
        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def outerproduct(self, a, inplace=False, i=False):
        """Compute the outer product with another data array.

        The axes of result will be the combined axes of the two input
        arrays.

        .. seealso:: `np.multiply.outer`

        :Parameters:

            a: array_like
                The data with which to form the outer product.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The outer product, or `None` if the operation was
                in-place.

        **Examples**

        >>> d = cf.Data([1, 2, 3], 'm')
        >>> d
        <CF Data(3): [1, 2, 3] m>
        >>> f = d.outerproduct([4, 5, 6, 7])
        >>> f
        <CF Data(3, 4): [[4, ..., 21]] m>
        >>> print(f.array)
        [[ 4  5  6  7]
         [ 8 10 12 14]
         [12 15 18 21]]

        >>> e = cf.Data([[4, 5, 6, 7], [6, 7, 8, 9]], 's-1')
        >>> e
        <CF Data(2, 4): [[4, ..., 9]] s-1>
        >>> f = d.outerproduct(e)
        >>> f
        <CF Data(3, 2, 4): [[[4, ..., 27]]] m.s-1>
        >>> print(f.array)
        [[[ 4  5  6  7]
          [ 6  7  8  9]]

         [[ 8 10 12 14]
          [12 14 16 18]]

         [[12 15 18 21]
          [18 21 24 27]]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        shape = d.shape
        chunksizes0 = d.nc_dataset_chunksizes()

        # Cast 'a' as a Data object so that it definitely has sensible
        # Units. We don't mind if the units of 'a' are incompatible
        # with those of 'self', but if they are then it's nice if the
        # units are conformed.
        a = self.asdata(a)
        try:
            a = conform_units(a, d.Units, message="")
        except ValueError:
            pass

        dx = d.to_dask_array()
        ndim = dx.ndim

        dx = da.ufunc.multiply.outer(dx, a)
        d._set_dask(dx)

        d.override_units(d.Units * a.Units, inplace=True)

        # Include axis names for the new dimensions
        axes = d._axes
        for i, a_axis in enumerate(a._axes):
            axes += (new_axis_identifier(axes),)

        d._axes = axes

        # Make sure that cyclic axes in 'a' are still cyclic in 'd'
        for a_axis in a._cyclic:
            d.cyclic(ndim + a._axes.index(a_axis))

        # Update the dataset chunking strategy
        chunksizes1 = a.nc_dataset_chunksizes()
        if chunksizes0 or chunksizes1:
            if isinstance(chunksizes0, tuple):
                if isinstance(chunksizes1, tuple):
                    chunksizes = chunksizes0 + chunksizes1
                else:
                    chunksizes = chunksizes0 + a.shape

                d.nc_set_dataset_chunksizes(chunksizes)
            elif isinstance(chunksizes1, tuple):
                chunksizes = shape + chunksizes1
                d.nc_set_dataset_chunksizes(chunksizes)

        d._update_deterministic(a)
        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def change_calendar(self, calendar, inplace=False, i=False):
        """Change the calendar of date-time array elements.

        Reinterprets the existing date-times for the new calendar by
        adjusting the underlying numerical values relative to the
        reference date-time defined by the units.

        If a date-time value is not allowed in the new calendar then
        an exception is raised when the data array is accessed.

        .. seealso:: `override_calendar`, `Units`

        :Parameters:

            calendar: `str`
                The new calendar, as recognised by the CF conventions.

                *Parameter example:*
                  ``'proleptic_gregorian'``

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The new data with updated calendar, or `None` if the
                operation was in-place.

        **Examples**

        >>> d = cf.Data([0, 1, 2, 3, 4], 'days since 2004-02-27')
        >>> print(d.array)
        [0 1 2 3 4]
        >>> print(d.datetime_as_string)
        ['2004-02-27 00:00:00' '2004-02-28 00:00:00' '2004-02-29 00:00:00'
         '2004-03-01 00:00:00' '2004-03-02 00:00:00']
        >>> e = d.change_calendar('360_day')
        >>> print(e.array)
        [0 1 2 4 5]
        >>> print(e.datetime_as_string)
        ['2004-02-27 00:00:00' '2004-02-28 00:00:00' '2004-02-29 00:00:00'
        '2004-03-01 00:00:00' '2004-03-02 00:00:00']

        >>> d.change_calendar('noleap').array
        Traceback (most recent call last):
            ...
        ValueError: invalid day number provided in cftime.DatetimeNoLeap(2004, 2, 29, 0, 0, 0, 0, has_year_zero=True)

        """
        d = _inplace_enabled_define_and_cleanup(self)

        units = self.Units
        if not units.isreftime:
            raise ValueError(
                "Can't change calendar of non-reference time "
                f"units: {units!r}"
            )

        d._asdatetime(inplace=True)
        d.override_calendar(calendar, inplace=True)
        d._asreftime(inplace=True)

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def override_units(self, units, inplace=False, i=False):
        """Override the data array units.

        Not to be confused with setting the `Units` attribute to units
        which are equivalent to the original units. This is different
        because in this case the new units need not be equivalent to the
        original ones and the data array elements will not be changed to
        reflect the new units.

        :Parameters:

            units: `str` or `Units`
                The new units for the data array.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The new data, or `None` if the operation was in-place.

        **Examples**

        >>> d = cf.Data(1012.0, 'hPa')
        >>> e = d.override_units('km')
        >>> e.Units
        <Units: km>
        >>> e.datum()
        1012.0
        >>> d.override_units(cf.Units('watts'), inplace=True)
        >>> d.Units
        <Units: watts>
        >>> d.datum()
        1012.0

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d._Units = self._Units_class(units)
        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def override_calendar(self, calendar, inplace=False, i=False):
        """Override the calendar of the data array elements.

        Not to be confused with using the `change_calendar` method or
        setting the `d.Units.calendar`. `override_calendar` is different
        because the new calendar need not be equivalent to the original
        ones and the data array elements will not be changed to reflect
        the new units.

        :Parameters:

            calendar: `str`
                The new calendar.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The new data, or `None` if the operation was in-place.

        **Examples**

        >>> d = cf.Data(1, 'days since 2020-02-28')
        >>> d
        <CF Data(): 2020-02-29 00:00:00>
        >>> d.datum()
        1
        >>> e = d.override_calendar('noleap')
        <CF Data(): 2020-03-01 00:00:00 noleap>
        >>> e.datum()
        1

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d._Units = d._Units_class(d.Units._units, calendar)
        return d

    def datum(self, *index):
        """Return an element of the data array as a standard Python
        scalar.

        The first and last elements are always returned with
        ``d.datum(0)`` and ``d.datum(-1)`` respectively, even if the data
        array is a scalar array or has two or more dimensions.

        The returned object is of the same type as is stored internally.

        .. seealso:: `array`, `datetime_array`

        :Parameters:

            index: *optional*
                Specify which element to return. When no positional
                arguments are provided, the method only works for data
                arrays with one element (but any number of dimensions),
                and the single element is returned. If positional
                arguments are given then they must be one of the
                fdlowing:

                * An integer. This argument is interpreted as a flat index
                  into the array, specifying which element to copy and
                  return.

                  *Parameter example:*
                    If the data array shape is ``(2, 3, 6)`` then:
                    * ``d.datum(0)`` is equivalent to ``d.datum(0, 0, 0)``.
                    * ``d.datum(-1)`` is equivalent to ``d.datum(1, 2, 5)``.
                    * ``d.datum(16)`` is equivalent to ``d.datum(0, 2, 4)``.

                  If *index* is ``0`` or ``-1`` then the first or last data
                  array element respectively will be returned, even if the
                  data array is a scalar array.

                * Two or more integers. These arguments are interpreted as a
                  multidimensional index to the array. There must be the
                  same number of integers as data array dimensions.

                * A tuple of integers. This argument is interpreted as a
                  multidimensional index to the array. There must be the
                  same number of integers as data array dimensions.

                  *Parameter example:*
                    ``d.datum((0, 2, 4))`` is equivalent to ``d.datum(0,
                    2, 4)``; and ``d.datum(())`` is equivalent to
                    ``d.datum()``.

        :Returns:

                A copy of the specified element of the array as a suitable
                Python scalar.

        **Examples**

        >>> d = cf.Data(2)
        >>> d.datum()
        2
        >>> 2 == d.datum(0) == d.datum(-1) == d.datum(())
        True

        >>> d = cf.Data([[2]])
        >>> 2 == d.datum() == d.datum(0) == d.datum(-1)
        True
        >>> 2 == d.datum(0, 0) == d.datum((-1, -1)) == d.datum(-1, 0)
        True

        >>> d = cf.Data([[4, 5, 6], [1, 2, 3]], 'metre')
        >>> d[0, 1] = cf.masked
        >>> print(d)
        [[4 -- 6]
         [1  2 3]]
        >>> d.datum(0)
        4
        >>> d.datum(-1)
        3
        >>> d.datum(1)
        masked
        >>> d.datum(4)
        2
        >>> d.datum(-2)
        2
        >>> d.datum(0, 0)
        4
        >>> d.datum(-2, -1)
        6
        >>> d.datum(1, 2)
        3
        >>> d.datum((0, 2))
        6

        """
        # TODODASKAPI: consider renaming/aliasing to 'item'. Might depend
        # on whether or not the APIs are the same.

        if index:
            n_index = len(index)
            if n_index == 1:
                index = index[0]
                if index == 0:
                    # This also works for scalar arrays
                    index = (slice(0, 1),) * self.ndim
                elif index == -1:
                    # This also works for scalar arrays
                    index = (slice(-1, None),) * self.ndim
                elif isinstance(index, int):
                    if index < 0:
                        index += self.size

                    index = np.unravel_index(index, self.shape)
                elif len(index) == self.ndim:
                    index = tuple(index)
                else:
                    raise ValueError(
                        f"Incorrect number of indices ({n_index}) for "
                        f"{self.ndim}-d {self.__class__.__name__} data"
                    )
            elif n_index != self.ndim:
                raise ValueError(
                    f"Incorrect number of indices ({n_index}) for "
                    f"{self.ndim}-d {self.__class__.__name__} data"
                )

            array = self[index].array

        elif self.size == 1:
            array = self.array

        else:
            raise ValueError(
                f"For size {self.size} data, must provide an index of "
                "the element to be converted to a Python scalar"
            )

        if not np.ma.isMA(array):
            return array.item()

        mask = array.mask
        if mask is np.ma.nomask or not mask.item():
            return array.item()

        return masked

    @_inplace_enabled(default=False)
    def masked_invalid(self, inplace=False):
        """Mask the array where invalid values occur (NaN or inf).

        .. seealso:: `where`, `numpy.ma.masked_invalid`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The masked data, or `None` if the operation was
                in-place.

        **Examples**

        >>> d = cf.Data([0, 1, 2])
        >>> e = cf.Data([0, 2, 0])
        >>> f = d / e
        >>> f
        <CF Data(3): [nan, 0.5, inf]>
        >>> f.masked_invalid()
        <CF Data(3): [--, 0.5, --]>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = self.to_dask_array()
        dx = da.ma.masked_invalid(dx)
        d._set_dask(dx)
        return d

    @classmethod
    def masked_all(
        cls, shape, dtype=None, units=None, calendar=None, chunks="auto"
    ):
        """Return an empty masked array with all elements masked.

        .. seealso:: `empty`, `ones`, `zeros`, `masked_invalid`

        :Parameters:

            shape: `int` or `tuple` of `int`
                The shape of the new array. e.g. ``(2, 3)`` or ``2``.

            dtype: data-type
                The desired output data-type for the array, e.g.
                `numpy.int8`. The default is `numpy.float64`.

            units: `str` or `Units`
                The units for the new data array.

            calendar: `str`, optional
                The calendar for reference time units.

            {{chunks: `int`, `tuple`, `dict` or `str`, optional}}

                .. versionadded:: 3.14.0

        :Returns:

            `Data`
                A masked array with all data masked.

        **Examples**

        >>> d = cf.Data.masked_all((2, 2))
        >>> print(d.array)
        [[-- --]
         [-- --]]

        >>> d = cf.Data.masked_all((), dtype=bool)
        >>> d.array
        masked_array(data=--,
                     mask=True,
               fill_value=True,
                    dtype=bool)

        """
        d = cls.empty(
            shape=shape,
            dtype=dtype,
            units=units,
            calendar=calendar,
            chunks=chunks,
        )
        dx = d.to_dask_array()
        dx = dx.map_blocks(partial(np.ma.array, mask=True, copy=False))
        d._set_dask(dx)
        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def mid_range(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate mid-range values.

        The mid-range is half of the maximum plus the minimum.

        Calculates the mid-range value or the mid-range values along
        axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `max`, `min`, `range`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed array.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.mid_range()
        <CF Data(1, 1): [[5.5]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().mid_range,
            d,
            axis=axes,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def flip(self, axes=None, inplace=False, i=False):
        """Reverse the direction of axes of the data array.

        .. seealso:: `flatten', `insert_dimension`, `squeeze`, `swapaxes`,
                     `transpose`

        :Parameters:

            axes: (sequence of) `int`
                Select the axes. By default all axes are flipped. Each
                axis is identified by its integer position. No axes
                are flipped if *axes* is an empty sequence.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d.flip()
        >>> d.flip(1)
        >>> d.flip([0, 1])
        >>> d.flip([])

        >>> e = d[::-1, :, ::-1]
        >>> d.flip((2, 0)).equals(e)
        True

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if axes is not None and not axes and axes != 0:  # i.e. empty sequence
            return d

        if axes is None:
            iaxes = range(d.ndim)
        else:
            iaxes = d._parse_axes(axes)

        if not iaxes:
            return d

        index = [
            slice(None, None, -1) if i in iaxes else slice(None)
            for i in range(d.ndim)
        ]

        dx = d.to_dask_array()
        dx = dx[tuple(index)]
        d._set_dask(dx)

        return d

    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        from ..functions import inspect

        inspect(self)

    def isclose(self, y, rtol=None, atol=None):
        """Return where data are element-wise equal within a tolerance.

        {{equals tolerance}}

        For numeric data arrays, ``d.isclose(e, rtol, atol)`` is
        equivalent to ``abs(d - e) <= atol + rtol*abs(e)``,
        otherwise it is equivalent to ``d == e``.

        :Parameters:

            y: data_like
                The array to compare.

            atol: `float`, optional
                The absolute tolerance for all numerical comparisons. By
                default the value returned by the `atol` function is used.

            rtol: `float`, optional
                The relative tolerance for all numerical comparisons. By
                default the value returned by the `rtol` function is used.

        :Returns:

             `bool`
                 A boolean array of where the data are close to *y*.

        **Examples**

        >>> d = cf.Data([1000, 2500], 'metre')
        >>> e = cf.Data([1, 2.5], 'km')
        >>> print(d.isclose(e).array)
        [ True  True]

        >>> d = cf.Data(['ab', 'cdef'])
        >>> print(d.isclose([[['ab', 'cdef']]]).array)
        [[[ True  True]]]

        >>> d = cf.Data([[1000, 2500], [1000, 2500]], 'metre')
        >>> e = cf.Data([1, 2.5], 'km')
        >>> print(d.isclose(e).array)
        [[ True  True]
         [ True  True]]

        >>> d = cf.Data([1, 1, 1], 's')
        >>> print(d.isclose(1).array)
        [ True  True  True]

        """
        a = np.empty((), dtype=self.dtype)
        b = np.empty((), dtype=da.asanyarray(y).dtype)
        try:
            # Check if a numerical isclose is possible
            np.isclose(a, b)
        except TypeError:
            # self and y do not have suitable numeric data types
            # (e.g. both are strings)
            return self == y
        else:
            # self and y have suitable numeric data types
            if atol is None:
                atol = self._atol

            if rtol is None:
                rtol = self._rtol

            y = conform_units(y, self.Units)

            dx = da.isclose(self, y, atol=atol, rtol=rtol)

            d = self.copy()
            d._set_dask(dx)
            d.hardmask = self._DEFAULT_HARDMASK
            d.override_units(_units_None, inplace=True)
            d._update_deterministic(y)

            return d

    @_inplace_enabled(default=False)
    def reshape(self, *shape, merge_chunks=True, limit=None, inplace=False):
        """Change the shape of the data without changing its values.

        It assumes that the array is stored in row-major order, and
        only allows for reshapings that collapse or merge dimensions
        like ``(1, 2, 3, 4) -> (1, 6, 4)`` or ``(64,) -> (4, 4, 4)``.

        :Parameters:

            shape: `tuple` of `int`, or any number of `int`
                The new shape for the data, which should be compatible
                with the original shape. If an integer, then the
                result will be a 1-d array of that length. One shape
                dimension can be -1, in which case the value is
                inferred from the length of the array and remaining
                dimensions.

            merge_chunks: `bool`
                When True (the default) merge chunks using the logic
                in `dask.array.rechunk` when communication is
                necessary given the input array chunking and the
                output shape. When False, the input array will be
                rechunked to a chunksize of 1, which can create very
                many tasks. See `dask.array.reshape` for details.

            limit: int, optional
                The maximum block size to target in bytes. If no limit
                is provided, it defaults to a size in bytes defined by
                the `cf.chunksize` function.

        :Returns:

            `Data` or `None`
                 The reshaped data, or `None` if the operation was
                 in-place.

        **Examples**

        >>> d = cf.Data(np.arange(12))
        >>> print(d.array)
        [ 0  1  2  3  4  5  6  7  8  9 10 11]
        >>> print(d.reshape(3, 4).array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> print(d.reshape((4, 3)).array)
        [[ 0  1  2]
         [ 3  4  5]
         [ 6  7  8]
         [ 9 10 11]]
        >>> print(d.reshape(-1, 6).array)
        [[ 0  1  2  3  4  5]
         [ 6  7  8  9 10 11]]
        >>>  print(d.reshape(1, 1, 2, 6).array)
        [[[[ 0  1  2  3  4  5]
           [ 6  7  8  9 10 11]]]]
        >>> print(d.reshape(1, 1, -1).array)
        [[[[ 0  1  2  3  4  5  6  7  8  9 10 11]]]]

        """
        d = _inplace_enabled_define_and_cleanup(self)
        super(Data, d).reshape(
            *shape, merge_chunks=merge_chunks, limit=limit, inplace=True
        )

        # Clear cyclic axes, as we can't help but lose them in this
        # operation
        del d._cyclic

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def rint(self, inplace=False, i=False):
        """Round the data to the nearest integer, element-wise.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `floor`, `trunc`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The rounded data. If the operation was in-place then
                `None` is returned.

        **Examples**

        >>> d = cf.Data([-1.9, -1.5, -1.1, -1, 0, 1, 1.1, 1.5 , 1.9])
        >>> print(d.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(d.rint().array)
        [-2. -2. -1. -1.  0.  1.  1.  2.  2.]

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = d.to_dask_array()
        d._set_dask(da.rint(dx))
        return d

    @_inplace_enabled(default=False)
    def root_mean_square(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        weights=None,
        split_every=None,
        inplace=False,
    ):
        """Calculate root mean square (RMS) values.

        Calculates the RMS value or the RMS values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `mean`, `sum`,

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The collapsed array.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.root_mean_square()
        <CF Data(1, 1): [[6.674238124719146]] K>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.root_mean_square(weights=w)
        <CF Data(1, 1): [[6.871107713616576]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().rms,
            d,
            axis=axes,
            weights=weights,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def round(self, decimals=0, inplace=False, i=False):
        """Evenly round elements of the data array to the given number
        of decimals.

        Values exactly halfway between rounded decimal values are rounded
        to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and
        0.5 round to 0.0, etc. Results may also be surprising due to the
        inexact representation of decimal fractions in the IEEE floating
        point standard and errors introduced when scaling by powers of
        ten.

        .. versionadded:: 1.1.4

        .. seealso:: `ceil`, `floor`, `rint`, `trunc`

        :Parameters:

            decimals : `int`, optional
                Number of decimal places to round to (default: 0). If
                decimals is negative, it specifies the number of positions
                to the left of the decimal point.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d = cf.Data([-1.81, -1.41, -1.01, -0.91, 0.09, 1.09, 1.19, 1.59, 1.99])
        >>> print(d.array)
        [-1.81 -1.41 -1.01 -0.91  0.09  1.09  1.19  1.59  1.99]
        >>> print(d.round().array)
        [-2., -1., -1., -1.,  0.,  1.,  1.,  2.,  2.]
        >>> print(d.round(1).array)
        [-1.8, -1.4, -1. , -0.9,  0.1,  1.1,  1.2,  1.6,  2. ]
        >>> print(d.round(-1).array)
        [-0., -0., -0., -0.,  0.,  0.,  0.,  0.,  0.]

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = d.to_dask_array()
        d._set_dask(da.round(dx, decimals=decimals))
        return d

    def stats(
        self,
        all=False,
        values=True,
        minimum=True,
        mean=True,
        median=True,
        maximum=True,
        range=True,
        mid_range=True,
        standard_deviation=True,
        root_mean_square=True,
        sample_size=True,
        minimum_absolute_value=False,
        maximum_absolute_value=False,
        mean_absolute_value=False,
        mean_of_upper_decile=False,
        sum=False,
        sum_of_squares=False,
        variance=False,
        weights=None,
    ):
        """Calculate statistics of the data.

        By default the minimum, mean, median, maximum, range, mid-range,
        standard deviation, root mean square, and sample size are
        calculated. But this selection may be edited, and other metrics
        are available.

        .. seealso:: `minimum`, `mean`, `median`, `maximum`, `range`,
                     `mid_range`, `standard_deviation`,
                     `root_mean_square`, `sample_size`,
                     `minimum_absolute_value`, `maximum_absolute_value`,
                     `mean_absolute_value`, `mean_of_upper_decile`, `sum`,
                     `sum_of_squares`, `variance`

        :Parameters:

            all: `bool`, optional
                Calculate all possible statistics, regardless of the value
                of individual metric parameters.

            values: `bool`, optional
                If True (the default), returned values for the statistical
                calculations in the output dictionary are computed, else
                each is given in the form of a delayed `Data` operation.

            minimum: `bool`, optional
                Calculate the minimum of the values.

            maximum: `bool`, optional
                Calculate the maximum of the values.

            maximum_absolute_value: `bool`, optional
                Calculate the maximum of the absolute values.

            minimum_absolute_value: `bool`, optional
                Calculate the minimum of the absolute values.

            mid_range: `bool`, optional
                Calculate the average of the maximum and the minimum of
                the values.

            median: `bool`, optional
                Calculate the median of the values.

            range: `bool`, optional
                Calculate the absolute difference between the maximum and
                the minimum of the values.

            sum: `bool`, optional
                Calculate the sum of the values.

            sum_of_squares: `bool`, optional
                Calculate the sum of the squares of values.

            sample_size: `bool`, optional
                Calculate the sample size, i.e. the number of non-missing
                values.

            mean: `bool`, optional
                Calculate the weighted or unweighted mean of the values.

            mean_absolute_value: `bool`, optional
                Calculate the mean of the absolute values.

            mean_of_upper_decile: `bool`, optional
                Calculate the mean of the upper group of data values
                defined by the upper tenth of their distribution.

            variance: `bool`, optional
                Calculate the weighted or unweighted variance of the
                values, with a given number of degrees of freedom.

            standard_deviation: `bool`, optional
                Calculate the square root of the weighted or unweighted
                variance.

            root_mean_square: `bool`, optional
                Calculate the square root of the weighted or unweighted
                mean of the squares of the values.

            {{weights: data_like, `dict`, or `None`, optional}}

        :Returns:

            `dict`
                The statistics, with keys giving the operation names
                and values being the result of the corresponding
                statistical calculation, which are either the computed
                numerical values if *values*` is True, else the
                delayed `Data` operations which encapsulate those.

        **Examples**

        >>> d = cf.Data([[0, 1, 2], [3, -99, 5]], mask=[[0, 0, 0], [0, 1, 0]])
        >>> print(d.array)
        [[0  1  2]
         [3 --  5]]
        >>> d.stats()
        {'minimum': 0,
         'mean': 2.2,
         'median': 2.0,
         'maximum': 5,
         'range': 5,
         'mid_range': 2.5,
         'standard_deviation': 1.7204650534085255,
         'root_mean_square': 2.792848008753788,
         'sample_size': 5}
        >>> d.stats(all=True)
        {'minimum': 0,
         'mean': 2.2,
         'median': 2.0,
         'maximum': 5,
         'range': 5,
         'mid_range': 2.5,
         'standard_deviation': 1.7204650534085255,
         'root_mean_square': 2.792848008753788,
         'minimum_absolute_value': 0,
         'maximum_absolute_value': 5,
         'mean_absolute_value': 2.2,
         'mean_of_upper_decile': 5.0,
         'sum': 11,
         'sum_of_squares': 39,
         'variance': 2.9600000000000004,
         'sample_size': 5}
        >>> d.stats(mean_of_upper_decile=True, range=False)
        {'minimum': 0,
         'mean': 2.2,
         'median': 2.0,
         'maximum': 5,
         'mid_range': 2.5,
         'standard_deviation': 1.7204650534085255,
         'root_mean_square': 2.792848008753788,
         'mean_of_upper_decile': 5.0,
         'sample_size': 5}

        To ask for delayed operations instead of computed values:

        >>> d.stats(values=False)
        {'minimum': <CF Data(): 0>,
         'mean': <CF Data(): 2.2>,
         'median': <CF Data(): 2.0>,
         'maximum': <CF Data(): 5>,
         'range': <CF Data(): 5>,
         'mid_range': <CF Data(): 2.5>,
         'standard_deviation': <CF Data(): 1.7204650534085255>,
         'root_mean_square': <CF Data(): 2.792848008753788>,
         'sample_size': <CF Data(1, 1): [[5]]>}

        """
        no_weights = (
            "minimum",
            "median",
            "maximum",
            "range",
            "mid_range",
            "minimum_absolute_value",
            "maximum_absolute_value",
        )

        out = {}
        for stat in (
            "minimum",
            "mean",
            "median",
            "maximum",
            "range",
            "mid_range",
            "standard_deviation",
            "root_mean_square",
            "minimum_absolute_value",
            "maximum_absolute_value",
            "mean_absolute_value",
            "mean_of_upper_decile",
            "sum",
            "sum_of_squares",
            "variance",
        ):
            if all or locals()[stat]:
                func = getattr(self, stat)
                if stat in no_weights:
                    value = delayed(func)(squeeze=True)
                else:
                    value = delayed(func)(squeeze=True, weights=weights)

                out[stat] = value

        if all or sample_size:
            out["sample_size"] = delayed(lambda: self.sample_size())()

        data_values = compute(out)[0]
        if values:
            # Convert cf.Data objects holding the scalars (or scalar array
            # for the case of sample_size only) to scalar values
            return {op: val.array.item() for op, val in data_values.items()}
        else:
            return data_values

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def swapaxes(self, axis0, axis1, inplace=False, i=False):
        """Interchange two axes of an array.

        .. seealso:: `flatten', `flip`, 'insert_dimension`, `squeeze`,
                     `transpose`

        :Parameters:

            axis0, axis1 : `int`, `int`
                Select the axes to swap. Each axis is identified by its
                original integer position.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The data with swapped axis positions.

        **Examples**

        >>> d = cf.Data([[[1, 2, 3], [4, 5, 6]]])
        >>> d
        <CF Data(1, 2, 3): [[[1, ..., 6]]]>
        >>> d.swapaxes(1, 0)
        <CF Data(2, 1, 3): [[[1, ..., 6]]]>
        >>> d.swapaxes(0, -1)
        <CF Data(3, 2, 1): [[[1, ..., 6]]]>
        >>> d.swapaxes(1, 1)
        <CF Data(1, 2, 3): [[[1, ..., 6]]]>
        >>> d.swapaxes(-1, -1)
        <CF Data(1, 2, 3): [[[1, ..., 6]]]>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = self.to_dask_array()
        dx = da.swapaxes(dx, axis0, axis1)
        d._set_dask(dx)
        return d

    def fits_in_memory(self):
        """Return True if the array is small enough to be retained in
        memory.

        Returns True if the size of the computed array, always
        including space for a full boolean mask, is small enough to be
        retained in available memory.

        **Performance**

        The delayed operations are actually not computed by
        `fits_in_memory`, so it is possible that an intermediate
        operation may require more than the available memory, even if
        the final array does not.

        .. seealso:: `array`, `compute`, `nbytes`, `persist`,
                     `cf.free_memory`

        :Parameters:

            itemsize: deprecated at version 3.14.0
                The number of bytes per word of the master data array.

        :Returns:

            `bool`
                Whether or not the computed array fits in memory.

        **Examples**

        >>> d = cf.Data([1], 'm')
        >>> d.fits_in_memory()
        True

        Create a double precision (8 bytes per word) array that is
        approximately twice the size of the available memory:

        >>> size = int(2 * cf.free_memory() / 8)
        >>> d = cf.Data.empty((size,), dtype=float)
        >>> d.fits_in_memory()
        False
        >>> d.nbytes * (1 + 1/8) > cf.free_memory()
        True

        """
        return self.size * (self.dtype.itemsize + 1) <= free_memory()

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    @_manage_log_level_via_verbosity
    def where(
        self, condition, x=None, y=None, inplace=False, i=False, verbose=None
    ):
        """Assign array elements depending on a condition.

        The elements to be changed are identified by a
        condition. Different values can be assigned according to where
        the condition is True (assignment from the *x* parameter) or
        False (assignment from the *y* parameter).

        **Missing data**

        Array elements may be set to missing values if either *x* or
        *y* are the `cf.masked` constant, or by assignment from any
        missing data elements in *x* or *y*.

        If the data mask is hard (see the `hardmask` attribute) then
        missing data values in the array will not be overwritten,
        regardless of the content of *x* and *y*.

        If the *condition* contains missing data then the
        corresponding elements in the array will not be assigned to,
        regardless of the contents of *x* and *y*.

        **Broadcasting**

        The array and the *condition*, *x* and *y* parameters must all
        be broadcastable across the original array, such that the size
        of the result is identical to the original size of the
        array. Leading size 1 dimensions of these parameters are
        ignored, thereby also ensuring that the shape of the result is
        identical to the original shape of the array.

        If *condition* is a `Query` object then for the purposes of
        broadcasting, the condition is considered to be that which is
        produced by applying the query to the array.

        **Performance**

        If any of the shapes of the *condition*, *x*, or *y*
        parameters, or the array, is unknown, then there is a
        possibility that an unknown shape will need to be calculated
        immediately by executing all delayed operations on that
        object.

        .. seealso:: `cf.masked`, `hardmask`, `__setitem__`

        :Parameters:

            condition: array_like or `Query`
                The condition which determines how to assign values to
                the data.

                Assignment from the *x* and *y* parameters will be
                done where elements of the condition evaluate to
                `True` and `False` respectively.

                If *condition* is a `Query` object then this implies a
                condition defined by applying the query to the data.

                *Parameter example:*
                  ``d.where(d < 0, x=-999)`` will set all data
                  values that are less than zero to -999.

                *Parameter example:*
                  ``d.where(True, x=-999)`` will set all data values
                  to -999. This is equivalent to ``d[...] = -999``.

                *Parameter example:*
                  ``d.where(False, y=-999)`` will set all data values
                  to -999. This is equivalent to ``d[...] = -999``.

                *Parameter example:*
                  If ``d`` has shape ``(5, 3)`` then ``d.where([True,
                  False, True], x=-999, y=cf.masked)`` will set data
                  values in columns 0 and 2 to -999, and data values
                  in column 1 to missing data. This works because the
                  condition has shape ``(3,)`` which broadcasts to the
                  data shape.

                *Parameter example:*
                  ``d.where(cf.lt(0), x=-999)`` will set all data
                  values that are less than zero to -999. This is
                  equivalent to ``d.where(d < 0, x=-999)``.

            x, y: array-like or `None`
                Specify the assignment values. Where the condition is
                True assign to the data from *x*, and where the
                condition is False assign to the data from *y*.

                If *x* is `None` (the default) then no assignment is
                carried out where the condition is True.

                If *y* is `None` (the default) then no assignment is
                carried out where the condition is False.

                *Parameter example:*
                  ``d.where(condition)``, for any ``condition``, returns
                  data with identical data values.

                *Parameter example:*
                  ``d.where(cf.lt(0), x=-d, y=cf.masked)`` will change the
                  sign of all negative data values, and set all other data
                  values to missing data.

                *Parameter example:*
                  ``d.where(cf.lt(0), x=-d)`` will change the sign of
                  all negative data values, and leave all other data
                  values unchanged. This is equivalent to, but faster
                  than, ``d.where(cf.lt(0), x=-d, y=d)``

            {{inplace: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The new data with updated values, or `None` if the
                operation was in-place.

        **Examples**

        >>> d = cf.Data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> e = d.where(d < 5, d, 10 * d)
        >>> print(e.array)
        [ 0  1  2  3  4 50 60 70 80 90]

        >>> d = cf.Data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'km')
        >>> e = d.where(d < 5, cf.Data(10000 * d, 'metre'))
        >>> print(e.array)
        [ 0. 10. 20. 30. 40.  5.  6.  7.  8.  9.]

        >>> e = d.where(d < 5, cf.masked)
        >>> print(e.array)
        [-- -- -- -- -- 5 6 7 8 9]

        >>> d = cf.Data([[1, 2,],
        ...              [3, 4]])
        >>> e = d.where([[True, False], [True, True]], d, [[9, 8], [7, 6]])
        >>> print(e.array)
        [[1 8]
         [3 4]]
        >>> e = d.where([[True, False], [True, True]], [[9, 8], [7, 6]])
        >>> print(e.array)
        [[9 2]
         [7 6]]

        The shape of the result must have the same shape as the
        original data:

        >>> e = d.where([True, False], [9, 8])
        >>> print(e.array)
        [[9 2]
         [9 4]]

        >>> d = cf.Data(np.array([[0, 1, 2],
        ...                       [0, 2, 4],
        ...                       [0, 3, 6]]))
        >>> d.where(d < 4, None, -1)
        >>> print(e.array)
        [[ 0  1  2]
         [ 0  2 -1]
         [ 0  3 -1]]

        >>> x, y = np.ogrid[:3, :4]
        >>> print(x)
        [[0]
         [1]
         [2]]
        >>> print(y)
        [[0 1 2 3]]
        >>> condition = x < y
        >>> print(condition)
        [[False  True  True  True]
         [False False  True  True]
         [False False False  True]]
        >>> d = cf.Data(x)
        >>> e = d.where(condition, d, 10 + y)
            ...
        ValueError: where: 'condition' parameter with shape (3, 4) can not be broadcast across data with shape (3, 1) when the result will have a different shape to the data

        >>> d = cf.Data(np.arange(9).reshape(3, 3))
        >>> e = d.copy()
        >>> e[1, 0] = cf.masked
        >>> f = e.where(d > 5, None, -3.1416)
        >>> print(f.array)
        [[-3.1416 -3.1416 -3.1416]
         [-- -3.1416 -3.1416]
         [6.0 7.0 8.0]]
        >>> e.soften_mask()
        >>> f = e.where(d > 5, None, -3.1416)
        >>> print(f.array)
        [[-3.1416 -3.1416 -3.1416]
         [-3.1416 -3.1416 -3.1416]
         [ 6.      7.      8.    ]]

        """
        from .utils import where_broadcastable

        d = _inplace_enabled_define_and_cleanup(self)

        # Missing values could be affected, so make sure that the mask
        # hardness has been applied.
        #
        # 'cf_where' has its own calls to 'cfdm_to_memory', so we can
        # set '_force_to_memory=False'.
        dx = d.to_dask_array(_force_to_memory=False)

        units = d.Units

        # Parse condition
        if getattr(condition, "isquery", False):
            # Condition is a cf.Query object: Make sure that the
            # condition units are OK, and convert the condition to a
            # boolean Data instance with the same shape as the data.
            condition = condition.copy()
            condition.set_condition_units(units)
            condition = condition.evaluate(d)

        condition = type(self).asdata(condition)
        condition = where_broadcastable(d, condition, "condition")
        # 'cf_where' has its own calls to 'cfdm_to_memory', so we can
        # set '_force_to_memory=False'.
        condition = condition.to_dask_array(_force_to_memory=False)

        # If x or y is self then change it to None. This prevents an
        # unnecessary copy; and, at compute time, an unncessary numpy
        # where.
        if x is self:
            x = None

        if y is self:
            y = None

        if x is None and y is None:
            # The data is unchanged regardless of the condition
            return d

        # Parse x and y
        xy = []
        for arg, name in zip((x, y), ("x", "y")):
            if arg is None:
                xy.append(arg)
                continue

            if arg is masked:
                # Replace masked constant with array
                xy.append(scalar_masked_array(self.dtype))
                continue

            arg = type(self).asdata(arg)
            arg = where_broadcastable(d, arg, name)

            arg_units = arg.Units
            if arg_units:
                arg = conform_units(
                    arg,
                    units,
                    message=f"where: {name!r} parameter units {arg_units!r} "
                    f"are not equivalent to data units {units!r}",
                )

            xy.append(arg.to_dask_array())

        x, y = xy

        # Apply the where operation
        dx = da.core.elemwise(cfdm_where, dx, condition, x, y, d.hardmask)
        d._set_dask(dx)

        # Don't know (yet) if 'x' and 'y' have a deterministic names
        d._update_deterministic(False)

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def sin(self, inplace=False, i=False):
        """Take the trigonometric sine of the data element-wise.

        Units are accounted for in the calculation. If the units are not
        equivalent to radians (such as Kelvin) then they are treated as if
        they were radians. For example, the sine of 90 degrees_east
        is 1.0, as is the sine of 1.57079632 radians.

        The output units are changed to '1' (nondimensional).

        .. seealso:: `arcsin`, `cos`, `tan`, `sinh`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d.Units
        <Units: degrees_north>
        >>> print(d.array)
        [[-90 0 90 --]]
        >>> e = d.sin()
        >>> e.Units
        <Units: 1>
        >>> print(e.array)
        [[-1.0 0.0 1.0 --]]

        >>> d.Units
        <Units: m s-1>
        >>> print(d.array)
        [[1 2 3 --]]
        >>> d.sin(inplace=True)
        >>> d.Units
        <Units: 1>
        >>> print(d.array)
        [[0.841470984808 0.909297426826 0.14112000806 --]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if d.Units.equivalent(_units_radians):
            d.Units = _units_radians

        dx = d.to_dask_array()
        dx = da.sin(dx)
        d._set_dask(dx)

        d.override_units(_units_1, inplace=True)

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def sinh(self, inplace=False):
        """Take the hyperbolic sine of the data element-wise.

        Units are accounted for in the calculation. If the units are not
        equivalent to radians (such as Kelvin) then they are treated as if
        they were radians. For example, the the hyperbolic sine of 90
        degrees_north is 2.30129890, as is the hyperbolic sine of
        1.57079632 radians.

        The output units are changed to '1' (nondimensional).

        .. versionadded:: 3.1.0

        .. seealso:: `arcsinh`, `cosh`, `tanh`, `sin`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d.Units
        <Units: degrees_north>
        >>> print(d.array)
        [[-90 0 90 --]]
        >>> e = d.sinh()
        >>> e.Units
        <Units: 1>
        >>> print(e.array)
        [[-2.3012989023072947 0.0 2.3012989023072947 --]]

        >>> d.Units
        <Units: m s-1>
        >>> print(d.array)
        [[1 2 3 --]]
        >>> d.sinh(inplace=True)
        >>> d.Units
        <Units: 1>
        >>> print(d.array)
        [[1.1752011936438014 3.626860407847019 10.017874927409903 --]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if d.Units.equivalent(_units_radians):
            d.Units = _units_radians

        dx = d.to_dask_array()
        dx = da.sinh(dx)
        d._set_dask(dx)

        d.override_units(_units_1, inplace=True)

        return d

    @_inplace_enabled(default=False)
    def cosh(self, inplace=False):
        """Take the hyperbolic cosine of the data element-wise.

        Units are accounted for in the calculation. If the units are not
        equivalent to radians (such as Kelvin) then they are treated as if
        they were radians. For example, the the hyperbolic cosine of 0
        degrees_east is 1.0, as is the hyperbolic cosine of 1.57079632 radians.

        The output units are changed to '1' (nondimensional).

        .. versionadded:: 3.1.0

        .. seealso:: `arccosh`, `sinh`, `tanh`, `cos`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d.Units
        <Units: degrees_north>
        >>> print(d.array)
        [[-90 0 90 --]]
        >>> e = d.cosh()
        >>> e.Units
        <Units: 1>
        >>> print(e.array)
        [[2.5091784786580567 1.0 2.5091784786580567 --]]

        >>> d.Units
        <Units: m s-1>
        >>> print(d.array)
        [[1 2 3 --]]
        >>> d.cosh(inplace=True)
        >>> d.Units
        <Units: 1>
        >>> print(d.array)
        [[1.5430806348152437 3.7621956910836314 10.067661995777765 --]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if d.Units.equivalent(_units_radians):
            d.Units = _units_radians

        dx = d.to_dask_array()
        dx = da.cosh(dx)
        d._set_dask(dx)

        d.override_units(_units_1, inplace=True)

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def tanh(self, inplace=False):
        """Take the hyperbolic tangent of the data element-wise.

        Units are accounted for in the calculation. If the units are not
        equivalent to radians (such as Kelvin) then they are treated as if
        they were radians. For example, the the hyperbolic tangent of 90
        degrees_east is 0.91715234, as is the hyperbolic tangent of
        1.57079632 radians.

        The output units are changed to '1' (nondimensional).

        .. versionadded:: 3.1.0

        .. seealso:: `arctanh`, `sinh`, `cosh`, `tan`, `arctan2`


        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d.Units
        <Units: degrees_north>
        >>> print(d.array)
        [[-90 0 90 --]]
        >>> e = d.tanh()
        >>> e.Units
        <Units: 1>
        >>> print(e.array)
        [[-0.9171523356672744 0.0 0.9171523356672744 --]]

        >>> d.Units
        <Units: m s-1>
        >>> print(d.array)
        [[1 2 3 --]]
        >>> d.tanh(inplace=True)
        >>> d.Units
        <Units: 1>
        >>> print(d.array)
        [[0.7615941559557649 0.9640275800758169 0.9950547536867305 --]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if d.Units.equivalent(_units_radians):
            d.Units = _units_radians

        dx = d.to_dask_array()
        dx = da.tanh(dx)
        d._set_dask(dx)

        d.override_units(_units_1, inplace=True)

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def log(self, base=None, inplace=False, i=False):
        """Takes the logarithm of the data array.

        :Parameters:

            base:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = d.to_dask_array()

        if base is None:
            dx = da.log(dx)
        elif base == 10:
            dx = da.log10(dx)
        elif base == 2:
            dx = da.log2(dx)
        else:
            dx = da.log(dx)
            dx /= da.log(base)

        d._set_dask(dx)

        d.override_units(
            _units_1, inplace=True
        )  # all logarithm outputs are unitless

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def tan(self, inplace=False, i=False):
        """Take the trigonometric tangent of the data element-wise.

        Units are accounted for in the calculation. If the units are not
        equivalent to radians (such as Kelvin) then they are treated as if
        they were radians. For example, the tangents of 45
        degrees_east, 0.78539816 radians and 0.78539816 Kelvin are all
        1.0.

        The output units are changed to '1' (nondimensional).

        .. seealso:: `arctan`, `cos`, `sin`, `tanh`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d.Units
        <Units: degrees_north>
        >>> print(d.array)
        [[-45 0 45 --]]
        >>> e = d.tan()
        >>> e.Units
        <Units: 1>
        >>> print(e.array)
        [[-1.0 0.0 1.0 --]]

        >>> d.Units
        <Units: m s-1>
        >>> print(d.array)
        [[1 2 3 --]]
        >>> d.tan(inplace=True)
        >>> d.Units
        <Units: 1>
        >>> print(d.array)
        [[1.55740772465 -2.18503986326 -0.142546543074 --]]

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if d.Units.equivalent(_units_radians):
            d.Units = _units_radians

        dx = d.to_dask_array()
        dx = da.tan(dx)
        d._set_dask(dx)

        d.override_units(_units_1, inplace=True)

        return d

    def to_memory(self):
        """Bring data on disk into memory.

        Not implemented. Consider using `persist` instead.

        """
        raise NotImplementedError(
            "'Data.to_memory' is not available. "
            "Consider using 'Data.persist' instead."
        )

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def trunc(self, inplace=False, i=False):
        """Return the truncated values of the data array.

        The truncated value of the number, ``x``, is the nearest integer
        which is closer to zero than ``x`` is. In short, the fractional
        part of the signed number ``x`` is discarded.

        .. versionadded:: 1.0

        .. seealso:: `ceil`, `floor`, `rint`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d = cf.Data([-1.9, -1.5, -1.1, -1, 0, 1, 1.1, 1.5 , 1.9])
        >>> print(d.array)
        [-1.9 -1.5 -1.1 -1.   0.   1.   1.1  1.5  1.9]
        >>> print(d.trunc().array)
        [-1. -1. -1. -1.  0.  1.  1.  1.  1.]

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = d.to_dask_array()
        dx = da.trunc(dx)
        d._set_dask(dx)
        return d

    @_deprecated_kwarg_check("out", version="3.14.0", removed_at="5.0.0")
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def func(
        self,
        f,
        units=None,
        out=False,
        inplace=False,
        preserve_invalid=False,
        i=False,
        **kwargs,
    ):
        """Apply an element-wise array operation to the data array.

        :Parameters:

            f: `function`
                The function to be applied.

            units: `Units`, optional

            out: deprecated at version 3.14.0

            {{inplace: `bool`, optional}}

            preserve_invalid: `bool`, optional
                For MaskedArray arrays only, if True any invalid values produced
                by the operation will be preserved, otherwise they are masked.

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples**

        >>> d.Units
        <Units: radians>
        >>> print(d.array)
        [[ 0.          1.57079633]
         [ 3.14159265  4.71238898]]
        >>> import numpy
        >>> e = d.func(numpy.cos)
        >>> e.Units
        <Units: 1>
        >>> print(e.array)
        [[ 1.0  0.0]
         [-1.0  0.0]]
        >>> d.func(numpy.sin, inplace=True)
        >>> print(d.array)
        [[0.0   1.0]
         [0.0  -1.0]]

        >>> d = cf.Data([-2, -1, 1, 2], mask=[0, 0, 0, 1])
        >>> f = d.func(numpy.arctanh, preserve_invalid=True)
        >>> f.array
        masked_array(data=[nan, -inf, inf, --],
                     mask=[False, False, False,  True],
               fill_value=1e+20)
        >>> e = d.func(numpy.arctanh)  # default preserve_invalid is False
        >>> e.array
        masked_array(data=[--, --, --, --],
                     mask=[ True,  True,  True,  True],
               fill_value=1e+20,
                    dtype=float64)

        """
        d = _inplace_enabled_define_and_cleanup(self)

        dx = d.to_dask_array()

        if preserve_invalid:
            # Assume all inputs are masked, as checking for a mask to confirm
            # is expensive. If unmasked, effective mask will be all False.
            dx_mask = da.ma.getmaskarray(dx)  # store original mask
            dx = da.ma.getdata(dx)

        # Step 2: apply operation to data alone
        axes = tuple(range(dx.ndim))
        dx = da.blockwise(f, axes, dx, axes, **kwargs)

        if preserve_invalid:
            # Step 3: reattach original mask onto the output data
            dx = da.ma.masked_array(dx, mask=dx_mask)

        d._set_dask(dx)

        if units is not None:
            d.override_units(units, inplace=True)

        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def range(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate range values.

        The range is the maximum minus the minimum.

        Calculates the range value or the range values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `max`, `min`, `mid_range`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed array.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.range()
        <CF Data(1, 1): [[11]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().range,
            d,
            axis=axes,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def roll(self, axis, shift, inplace=False, i=False):
        """Roll array elements along one or more axes.

        Elements that roll beyond the last position are re-introduced
        at the first.

        .. seealso:: `flatten`, `insert_dimension`, `flip`, `squeeze`,
                     `transpose`

        :Parameters:

            axis: `int`, or `tuple` of `int`
                Axis or axes along which elements are shifted.

                *Parameter example:*
                  Roll the second axis: ``axis=1``.

                *Parameter example:*
                  Roll the last axis: ``axis=-1``.

                *Parameter example:*
                  Roll the first and last axes: ``axis=(0, -1)``.

            shift: `int`, or `tuple` of `int`
                The number of places by which elements are shifted.
                If a `tuple`, then *axis* must be a tuple of the same
                size, and each of the given axes is shifted by the
                corresponding number. If an `int` while *axis* is a
                `tuple` of `int`, then the same value is used for all
                given axes.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The rolled data.

        **Examples**

        >>> d = cf.Data([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        >>> print(d.roll(0, 2).array)
        [10 11  0  1  2  3  4  5  6  7  8  9]
        >>> print(d.roll(0, -2).array)
        [ 2  3  4  5  6  7  8  9 10 11  0  1]

        >>> d2 = d.reshape(3, 4)
        >>> print(d2.array)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]]
        >>> print(d2.roll(0, 1).array)
        [[ 8  9 10 11]
         [ 0  1  2  3]
         [ 4  5  6  7]]
        >>> print(d2.roll(0, -1).array)
        [[ 4  5  6  7]
         [ 8  9 10 11]
         [ 0  1  2  3]]
        >>> print(d2.roll(1, 1).array)
        [[ 3  0  1  2]
         [ 7  4  5  6]
         [11  8  9 10]]
        >>> print(d2.roll(1, -1).array)
        [[ 1  2  3  0]
         [ 5  6  7  4]
         [ 9 10 11  8]]
        >>> print(d2.roll((1, 0), (1, 1)).array)
        [[11  8  9 10]
         [ 3  0  1  2]
         [ 7  4  5  6]]
        >>> print(d2.roll((1, 0), (2, 1)).array)
        [[10 11  8  9]
         [ 2  3  0  1]
         [ 6  7  4  5]]

        """
        # TODODASKAPI - consider matching the numpy/dask api:
        #               "shift,axis=", and the default axis behaviour
        #               of a flattened roll followed by shape
        #               restore

        d = _inplace_enabled_define_and_cleanup(self)

        dx = d.to_dask_array()
        dx = da.roll(dx, shift, axis=axis)
        d._set_dask(dx)

        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def sum(
        self,
        axes=None,
        weights=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate sum values.

        Calculates the sum value or the sum values along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `integral`, `mean`, `sd`,
                     `sum_of_squares`, `sum_of_weights`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.sum()
        <CF Data(1, 1): [[62]] K>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.sum(weights=cf.Data(w, 'm'))
        <CF Data(1, 1): [[97.0]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().sum,
            d,
            axis=axes,
            weights=weights,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )
        return d

    @_inplace_enabled(default=False)
    def sum_of_squares(
        self,
        axes=None,
        weights=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
    ):
        """Calculate sums of squares.

        Calculates the sum of squares or the sum of squares values
        along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `sum`, `sum_of_squares`,
                     `sum_of_weights2`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.sum_of_squares()
        <CF Data(1, 1): [[490]] K2>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.sum_of_squares(weights=w)
        <CF Data(1, 1): [[779.0]] K2>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d.square(inplace=True)
        d.sum(
            axes=axes,
            weights=weights,
            squeeze=squeeze,
            mtol=mtol,
            split_every=split_every,
            inplace=True,
        )
        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def sum_of_weights(
        self,
        axes=None,
        weights=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate sums of weights.

        Calculates the sum of weights or the sum of weights values
        along axes.

        The weights given by the *weights* parameter are internally
        broadcast to the shape of the data, and those weights that are
        missing data, or that correspond to the missing elements of
        the data, are assigned a weight of 0. It is these processed
        weights that are summed.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `sum`, `sum_of_squares`,
                     `sum_of_weights2`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {[inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.sum_of_weights()
        <CF Data(1, 1): [[11]]>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.sum_of_weights(weights=w)
        <CF Data(1, 1): [[16.5]]>

        >>> d.sum_of_weights(weights=cf.Data(w, 'm'))
        <CF Data(1, 1): [[16.5]] m>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, weights = collapse(
            Collapse().sum_of_weights,
            d,
            axis=axes,
            weights=weights,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )

        units = _units_None
        if weights is not None:
            units = getattr(weights, "Units", None)
            if units is None:
                units = _units_None

        d.override_units(units, inplace=True)

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def sum_of_weights2(
        self,
        axes=None,
        weights=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate sums of squares of weights.

        Calculates the sum of squares of weights or the sum of squares
        of weights values along axes.

        The weights given by the *weights* parameter are internally
        broadcast to the shape of the data, and those weights that
        are missing data, or that correspond to the missing elements
        of the data, are assigned a weight of 0. It is these processed
        weights that are squared and summed.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `sum`, `sum_of_squares`,
                     `sum_of_weights`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {[inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.sum_of_weights2()
        <CF Data(1, 1): [[11]]>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.sum_of_weights2(weights=w)
        <CF Data(1, 1): [[26.75]]>

        >>> d.sum_of_weights2(weights=cf.Data(w, 'm'))
        <CF Data(1, 1): [[26.75]] m2>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, weights = collapse(
            Collapse().sum_of_weights2,
            d,
            axis=axes,
            weights=weights,
            keepdims=not squeeze,
            split_every=split_every,
            mtol=mtol,
        )

        units = _units_None
        if weights is not None:
            units = getattr(weights, "Units", None)
            if not units:
                units = _units_None
            else:
                units = units**2

        d.override_units(units, inplace=True)

        return d

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def std(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        weights=None,
        ddof=0,
        split_every=None,
        inplace=False,
        i=False,
    ):
        r"""Calculate standard deviations.

        Calculates the standard deviation of an array or the standard
        deviations along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `mean`, `sum`, `var`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{ddof: number}}

                 By default *ddof* is 0.

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.std()
        <CF Data(1, 1): [[3.5744733184250004]] K>
        >>> d.std(ddof=1)
        <CF Data(1, 1): [[3.7489392439122637]] K>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.std(ddof=1, weights=w)
        <CF Data(1, 1): [[3.7457375639741506]] K>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d.var(
            axes=axes,
            weights=weights,
            squeeze=squeeze,
            mtol=mtol,
            ddof=ddof,
            split_every=split_every,
            inplace=True,
        )
        d.sqrt(inplace=True)
        return d

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def var(
        self,
        axes=None,
        weights=None,
        squeeze=False,
        mtol=1,
        ddof=0,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Calculate variances.

        Calculates the variance of an array or the variance values
        along axes.

        See
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for mathematical definitions.

         ..seealso:: `sample_size`, `mean`, `sd`, `sum`

        :Parameters:

            {{collapse axes: (sequence of) `int`, optional}}

            {{weights: data_like, `dict`, or `None`, optional}}

            {{collapse squeeze: `bool`, optional}}

            {{mtol: number, optional}}

            {{ddof: number}}

                 By default *ddof* is 0.

            {{split_every: `int` or `dict`, optional}}

                .. versionadded:: 3.14.0

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The collapsed data, or `None` if the operation was
                in-place.

        **Examples**

        >>> a = np.ma.arange(12).reshape(4, 3)
        >>> d = cf.Data(a, 'K')
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[0 1 2]
         [3 -- 5]
         [6 7 8]
         [9 10 11]]
        >>> d.var()
        <CF Data(1, 1): [[12.776859504132233]] K2>
        >>> d.var(ddof=1)
        <CF Data(1, 1): [[14.054545454545456]] K2>

        >>> w = np.linspace(1, 2, 3)
        >>> print(w)
        [1.  1.5 2. ]
        >>> d.var(ddof=1, weights=w)
        <CF Data(1, 1): [[14.030549898167004]] K2>

        """
        d = _inplace_enabled_define_and_cleanup(self)
        d, _ = collapse(
            Collapse().var,
            d,
            axis=axes,
            weights=weights,
            keepdims=not squeeze,
            mtol=mtol,
            ddof=ddof,
            split_every=split_every,
        )

        units = d.Units
        if units:
            d.override_units(units**2, inplace=True)

        return d

    def section(
        self, axes, stop=None, chunks=False, min_step=1, mode="dictionary"
    ):
        """Returns a dictionary of sections of the `Data` object.

        Specifically, returns a dictionary of Data objects which are the
        m-dimensional sections of this n-dimensional Data object, where
        m <= n. The dictionary keys are the indices of the sections
        in the original Data object. The m dimensions that are not
        sliced are marked with None as a placeholder making it possible
        to reconstruct the original data object. The corresponding
        values are the resulting sections of type `Data`.

        :Parameters:

            axes: (sequence of) `int`
                This is should be one or more integers of the m indices of
                the m axes that define the sections of the `Data`
                object. If axes is `None` (the default) or an empty
                sequence then all axes are selected.

                Note that the axes specified by the *axes* parameter are
                the one which are to be kept whole. All other axes are
                sectioned.

            stop: `int`, optional
                Deprecated at version 3.14.0.

                Stop after this number of sections and return. If stop is
                None all sections are taken.

            chunks: `bool`, optional
                Deprecated at version 3.14.0. Consider using
                `cf.Data.rechunk` instead.

                If True return sections that are of the maximum possible
                size that will fit in one chunk of memory instead of
                sectioning into slices of size 1 along the dimensions that
                are being sectioned.


            min_step: `int`, optional
                The minimum step size when making chunks. By default this
                is 1. Can be set higher to avoid size 1 dimensions, which
                are problematic for linear regridding.

        :Returns:

            `dict`
                The dictionary of m dimensional sections of the Data
                object.

        **Examples**

        >>> d = cf.Data(np.arange(120).reshape(2, 6, 10))
        >>> d
        <CF Data(2, 6, 10): [[[0, ..., 119]]]>
        >>> d.section([1, 2])
        {(0, None, None): <CF Data(1, 6, 10): [[[0, ..., 59]]]>,
         (1, None, None): <CF Data(1, 6, 10): [[[60, ..., 119]]]>}
        >>> d.section([0, 1], min_step=2)
        {(None, None, 0): <CF Data(2, 6, 2): [[[0, ..., 111]]]>,
         (None, None, 2): <CF Data(2, 6, 2): [[[2, ..., 113]]]>,
         (None, None, 4): <CF Data(2, 6, 2): [[[4, ..., 115]]]>,
         (None, None, 6): <CF Data(2, 6, 2): [[[6, ..., 117]]]>,
         (None, None, 8): <CF Data(2, 6, 2): [[[8, ..., 119]]]>}

        """
        if chunks:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "section",
                {"chunks": chunks},
                message="Consider using Data.rechunk() instead.",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        if stop is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "section",
                {"stop": stop},
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        return _section(self, axes, min_step=min_step)

    @_inplace_enabled(default=False)
    def square(self, dtype=None, inplace=False):
        """Calculate the element-wise square.

        .. versionadded:: 3.14.0

        .. seealso:: `sqrt`, `sum_of_squares`

        :Parameters:

            dtype: data-type, optional
                Overrides the data type of the output arrays. A
                matching precision of the calculation should be
                chosen. For example, a *dtype* of ``'int32'`` is only
                allowed when the input values are integers.

             {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The element-wise square of the data, or `None` if the
                operation was in-place.

        **Examples**

        >>> d = cf.Data([[0, 1, 2.5, 3, 4]], 'K', mask=[[0, 0, 0, 1, 0]])
        >>> print(d.array)
        [[0.0 1.0 2.5 -- 4.0]]
        >>> e = d.square()
        >>> e
        <CF Data(1, 5): [[0.0, ..., 16.0]] K2>
        >>> print(e.array)
        [[0.0 1.0 6.25 -- 16.0]]

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = d.to_dask_array()
        dx = da.square(dx, dtype=dtype)
        d._set_dask(dx)

        units = d.Units
        if units:
            d.override_units(units**2, inplace=True)

        return d

    @_inplace_enabled(default=False)
    def sqrt(self, dtype=None, inplace=False):
        """Calculate the non-negative square root.

        .. versionadded:: 3.14.0

        .. seealso:: `square`

        :Parameters:

            dtype: data-type, optional
                Overrides the data type of the output arrays. A
                matching precision of the calculation should be
                chosen. For example, a *dtype* of ``'int32'` is not
                allowed, even if the input values are perfect squares.

             {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`
                The element-wise positive square root of the data, or
                `None` if the operation was in-place.

        **Examples**

        >>> d = cf.Data([[0, 1, 2, 3, 4]], 'K2', mask=[[0, 0, 0, 1, 0]])
        >>>print(d.array)
        [[0 1 2 -- 4]]
        >>> e = d.sqrt()
        >>> e
        <CF Data(1, 5): [[0.0, ..., 2.0]] K>
        >>> print(e.array)
        [[0.0 1.0 1.4142135623730951 -- 2.0]]

        Negative input values raise a warning but nonetheless result in NaN
        or, if there are already missing values, missing data:

        >>> import warnings
        >>> d = cf.Data([0, 1, -4])
        >>> print(d.array)
        [ 0  1 -4]
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     print(d.sqrt().array)
        ...
        [ 0.  1. nan]

        >>> d = cf.Data([0, 1, -4], mask=[1, 0, 0])
        >>> print(d.array)
        [-- 1 -4]
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     print(d.sqrt().array)
        ...
        [-- 1.0 --]

        """
        d = _inplace_enabled_define_and_cleanup(self)
        dx = d.to_dask_array()
        dx = da.sqrt(dx, dtype=dtype)
        d._set_dask(dx)

        units = d.Units
        if units:
            try:
                d.override_units(units**0.5, inplace=True)
            except ValueError as e:
                raise type(e)(
                    f"Incompatible units for taking a square root: {units!r}"
                )

        return d

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def maximum(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Alias for `max`"""
        return self.max(
            axes=axes,
            squeeze=squeeze,
            mtol=mtol,
            split_every=split_every,
            inplace=inplace,
            i=i,
        )

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def minimum(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Alias for `min`"""
        return self.min(
            axes=axes,
            squeeze=squeeze,
            mtol=mtol,
            split_every=split_every,
            inplace=inplace,
            i=i,
        )

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def sd(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        weights=None,
        ddof=0,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Alias for `std`"""
        return self.std(
            axes=axes,
            squeeze=squeeze,
            weights=weights,
            mtol=mtol,
            ddof=ddof,
            split_every=split_every,
            inplace=inplace,
            i=i,
        )

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def standard_deviation(
        self,
        axes=None,
        squeeze=False,
        mtol=1,
        weights=None,
        ddof=0,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Alias for `std`"""
        return self.std(
            axes=axes,
            squeeze=squeeze,
            weights=weights,
            mtol=mtol,
            ddof=ddof,
            split_every=split_every,
            inplace=inplace,
            i=i,
        )

    @_inplace_enabled(default=False)
    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    def variance(
        self,
        axes=None,
        squeeze=False,
        weights=None,
        mtol=1,
        ddof=0,
        split_every=None,
        inplace=False,
        i=False,
    ):
        """Alias for `var`"""
        return self.var(
            axes=axes,
            squeeze=squeeze,
            weights=weights,
            mtol=mtol,
            ddof=ddof,
            split_every=split_every,
            inplace=inplace,
            i=i,
        )


def _size_of_index(index, size=None):
    """Return the number of elements resulting in applying an index to a
    sequence.

    :Parameters:

        index: `slice` or `list` of `int`
            The index being applied to the sequence.

        size: `int`, optional
            The number of elements in the sequence being indexed. Only
            required if *index* is a slice object.

    :Returns:

        `int`
            The length of the sequence resulting from applying the index.

    **Examples**

    >>> _size_of_index(slice(None, None, -2), 10)
    5
    >>> _size_of_index([1, 4, 9])
    3

    """
    if isinstance(index, slice):
        # Index is a slice object
        start, stop, step = index.indices(size)
        div, mod = divmod(stop - start, step)
        if mod != 0:
            div += 1
        return div
    else:
        # Index is a list of integers
        return len(index)
