"""Regridding functions used within a dask graph."""
import numpy as np


def regrid(
    a,
    weights=None,
    method=None,
    src_shape=None,
    dst_shape=None,
    axis_order=None,
    ref_src_mask=None,
    min_weight=None,
):
    """Regrid an array.

    .. versionadded:: 3.14.0

    .. seealso:: `regrid_weights`, `_regrid`, `cf.Data._regrid`

    :Parameters:

        a: `numpy.ndarray`
            The array to be regridded.

        weights: `numpy.ndarray`
            The weights matrix that defines the regridding operation.

            The weights matrix has J rows and I columns, where J and I
            are the total number of cells in the destination and
            source grids respectively.

            The weights matrix only describes cells defined by the
            regridding dimensions. If the array *a* includes
            non-regridding dimensions then, in essence, the regrid
            operation is carried out separately for each slice of the
            regridding dimensions. For instance, if *a* represents T,
            Z, Y, X dimensions with shape ``(12, 20, 73, 96)`` and is
            to have its Y and X dimension regridded, then the result
            may be thought of as the concatenation of the 240
            individual regrids arising from all of the T and Z
            dimension combinations.

            Each element w_ji is the multiplicative weight that
            defines how much of Vs_i (the value in source grid cell i)
            contributes to Vd_j (the value in destination grid cell
            j).

            The final value of Vd_j is the sum of w_ji * Vs_i for all
            source grid cells i. Note that it is typical that for a
            given j most w_ji will be zero, reflecting the fact only a
            few source grid cells intersect a particular destination
            grid cell. I.e. *weights* is typically a very sparse
            matrix.

            If the destination grid has masked cells, either because
            it spans areas outside of the source grid, or by selection
            (such as ocean cells for land-only data), then the
            corresponding rows in the weights matrix must be be
            entirely missing data.

            For the patch recovery and second-order conservative
            regridding methods, the weights matrix will have been
            constructed taking into account the mask of the source
            grid, which must match the mask of *a* for its regridding
            dimensions.

            For all other regridding methods, the weights matrix will
            have been constructed assuming that no source grid cells
            are masked, and the weights matrix will be modified
            on-the-fly to account for any masked elements of *a* in
            each regridding slice.

            It is assumed that data-type of the weights matrix is same
            as the desired data-type of the regridded data.

            See section 12.3 "Regridding Methods" of
            https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node1.html

        method: `str`
            The name of the regridding method.

        src_shape: sequence of `int`
            The shape of the source grid.

        dst_shape: sequence of `int`
            The shape of the destination grid.

        axis_order, sequence of `int`
            The axis order that transposes *a* so that the regrid axes
            become the trailing dimensions, ordered consistently with
            the order used to create the weights matrix; and the
            non-regrid axes become the leading dimensions.

        ref_src_mask, `numpy.ndarray` or `None`
            If a `numpy.ndarray` with shape *src_shape* then this is
            the reference source grid mask that was used during the
            creation of the weights matrix given by *weights*, and the
            mask of each regrid slice of *a* must therefore be
            identical to *ref_src_mask*. If *ref_src_mask* is a scalar
            array with value `False`, then this is equivalent to a
            reference source grid mask with shape *src_shape* entirely
            populated with `False`.

            If `None` (the default), then the weights matrix will have
            been created assuming no source grid mask, and the mask of
            each regrid slice of *a* is automatically applied to
            *weights* prior to the regridding calculation.

        min_weight: float, optional
            A very small non-negative number. By default *min_weight*
            is ``2.5 * np.finfo("float64").eps``,
            i.e. ``5.551115123125783e-16`. It is used during linear
            and first-order conservative regridding when adjusting the
            weights matrix to account for the data mask. It is ignored
            for all other regrid methods, or if data being regridded
            has no missing values.

            In some cases (described below) for which weights might
            only be non-zero as a result of rounding errors, the
            *min_weight* parameter controls whether or a not cell in
            the regridded field is masked.

            The default value has been chosen empirically as the
            smallest value that produces the same masks as ESMF for
            the use cases defined in the cf test suite.

            **Linear regridding**

            Destination grid cell j will only be masked if a) it is
            masked in destination grid definition; or b) ``w_ji >=
            min_weight`` for those masked source grid cells i for
            which ``w_ji > 0``.

            **Conservative first-order regridding**

            Destination grid cell j will only be masked if a) it is
            masked in destination grid definition; or b) The sum of
            ``w_ji`` for all non-masked source grid cells i is
            strictly less than *min_weight*.

    :Returns:

        `numpy.ndarray`
            The regridded data.

    """
    # ----------------------------------------------------------------
    # Reshape the array into a form suitable for the regridding dot
    # product, i.e. a 2-d array whose right-hand dimension represents
    # are the gathered regridding axes and whose left-hand dimension
    # represent of all the other dimensions.
    # ----------------------------------------------------------------
    a = a.transpose(axis_order)
    non_regrid_shape = a.shape[: a.ndim - len(src_shape)]
    dst_size, src_size = weights.shape
    a = a.reshape(-1, src_size)
    a = a.T

    # ----------------------------------------------------------------
    # Find the source grid mask
    # ----------------------------------------------------------------
    src_mask = None
    variable_mask = False
    if np.ma.is_masked(a):
        # Source data is masked for at least one slice
        mask = np.ma.getmaskarray(a)
        if mask.shape[1] == 1 or (mask == mask[:, 0:1]).all():
            # The source mask is same for all slices
            src_mask = mask[:, 0]
            if not src_mask.any():
                src_mask = None
        else:
            # Source mask varies across slices
            variable_mask = True

        del mask

    if ref_src_mask is not None:
        # A reference source grid mask has already been incorporated
        # into the weights matrix. Therefore, the mask for all slices
        # of 'a' must be the same as this reference mask.
        if variable_mask or (src_mask is None and ref_src_mask.any()):
            raise ValueError(
                f"Can't regrid with the {method!r} method when the source "
                f"data mask varies over different {len(src_shape)}-d "
                "regridding slices"
            )

        if ref_src_mask.dtype != bool or (
            ref_src_mask.shape and ref_src_mask.shape != src_shape
        ):
            raise ValueError(
                f"For {method!r} regridding, "
                "the 'ref_src_mask' parameter must be None or a "
                f"Boolean numpy array with shape () or {src_shape}. Got: "
                f"dtype={ref_src_mask.dtype}, shape={ref_src_mask.shape}"
            )

        if src_mask is not None:
            ref_src_mask = ref_src_mask.reshape(src_mask.shape)
            if (src_mask != ref_src_mask).any():
                raise ValueError(
                    f"Can't regrid with the {method!r} method when the "
                    "source data mask does not match the mask used to "
                    "construct the regrid operator"
                )

    # ----------------------------------------------------------------
    # Regrid the source data
    # ----------------------------------------------------------------
    if min_weight is None:
        min_weight = np.finfo("float64").eps * 2.5

    if variable_mask:
        # Source data is masked and the source mask varies across
        # slices => we have to regrid each slice separately, adjusting
        # the weights for the mask of each slice.
        #
        # However, if the mask of a regrid slice is the same as the
        # mask of its previous regrid slice then we can reuse the
        # weights that already have the correct mask.
        n_slices = a.shape[1]
        regridded_data = np.ma.empty(
            (dst_size, n_slices), dtype=weights.dtype, order="F"
        )
        prev_weights, prev_mask = None, None
        for n in range(n_slices):
            n = slice(n, n + 1)
            a_n = a[:, n]
            regridded_data[:, n], prev_mask, prev_weights = _regrid(
                a_n,
                np.ma.getmaskarray(a_n[:, 0]),
                weights,
                method,
                prev_mask=prev_mask,
                prev_weights=prev_weights,
                min_weight=min_weight,
            )

        a = regridded_data
        del a_n, regridded_data, prev_weights, prev_mask
    else:
        # Source data is either not masked or the source mask is same
        # for all slices => all slices can be regridded
        # simultaneously.
        a, _, _ = _regrid(a, src_mask, weights, method, min_weight=min_weight)
        del _

    # ----------------------------------------------------------------
    # Reshape the regridded data back to its original axis order
    # ----------------------------------------------------------------
    a = a.T
    a = a.reshape(non_regrid_shape + tuple(dst_shape))

    d = {k: i for i, k in enumerate(axis_order)}
    axis_reorder = [i for k, i in sorted(d.items())]
    a = a.transpose(axis_reorder)

    return a


def _regrid(
    a,
    src_mask,
    weights,
    method,
    prev_mask=None,
    prev_weights=None,
    min_weight=None,
):
    """Worker function for `regrid`.

    Modifies the *weights* matrix to account for missing data in *a*,
    and then creates the regridded array by forming the dot product of
    the modified *weights* and *a*.

    .. versionadded:: 3.14.0

    .. seealso:: `regrid`, `regrid_weights`

    :Parameters:

        a: `numpy.ndarray`
            The array to be regridded. Must have shape ``(I, n)``,
            where ``I`` is the number of source grid cells and ``n``
            is the number of regrid slices. Performance will be
            optimised if *a* has Fortran order memory layout
            (column-major order).

        src_mask: `numpy.ndarray` or `None`
            The source grid mask to be applied to the weights
            matrix. If `None` then no source grid cells are masked. If
            a Boolean `numpy` array then it must have shape ``(I,)``,
            where ``I`` is the number of source grid cells, and `True`
            signifies a masked cell.

        weights: `numpy.ndarray`
            The weights matrix that defines the regridding
            operation. Might be modified (not in-place) to account for
            missing data in *a*. Must have shape ``(J, I)``, where
            ``J`` is the number of destination grid cells and ``I`` is
            the number of source grid cells. Performance will be
            optimised if *weights* has C order memory layout
            (row-major order).

            See `regrid` for details.

        min_weight: float, optional
            A very small non-negative number. It is used during linear
            and first-order conservative regridding when adjusting the
            weights matrix to account for the data mask. It is ignored
            for all other regrid methods, or if data being regridded
            has no missing values.

            See `regrid` for details.

        method: `str`
            The name of the regridding method.

        prev_mask: `numpy.ndarray` or `None`
            The source grid mask used by a previous call to `_regrid`.
            See *prev_weights* for details`.

        prev_weights: `numpy.ndarray`, optional
            The weights matrix used by a previous call to `_regrid`,
            possibly modified to account for missing data. If
            *prev_mask* equals *src_mask* then the *prev_weights*
            weights matrix is used to calculate the regridded data,
            bypassing any need to calculate a new weights matrix.
            Ignored if `prev_mask` is `None`.

    :Returns:

        3-`tuple`
            * `numpy.ndarray`: The regridded data.
            * `numpy.ndarray` or `None`: The source grid mask applied
              to the returned weights matrix, always identical to the
              *src_mask* parameter.
            * `numpy.ndarray`: The weights matrix used to regrid *a*.

    """
    if src_mask is None or not src_mask.any():
        # ------------------------------------------------------------
        # Source data is not masked
        # ------------------------------------------------------------
        w = weights
    elif prev_mask is not None and (prev_mask == src_mask).all():
        # ------------------------------------------------------------
        # Source data is masked, but no need to modify the weights
        # since we have been provided with an already-modified weights
        # matrix.
        # ------------------------------------------------------------
        w = prev_weights
    else:
        # ------------------------------------------------------------
        # Source data is masked and we might need to adjust the
        # weights matrix accordingly
        # ------------------------------------------------------------
        if method in ("conservative", "conservative_1st"):
            # 1) First-order conservative method:
            #
            #     w_ji = f_ji * As_i / Ad_j
            #
            # where f_ji is the fraction of source cell i that
            # contributes to destination cell j, As_i is the area of
            # source cell i, and Ad_j is the area of destination cell
            # j.
            #
            # If source grid cell i is masked then w_ji are set to
            # zero for all destination cells j, and the remaining
            # non-zero w_ji are divided by D_j, the fraction of
            # destination cell j that intersects with unmasked cells
            # of the source grid.
            #
            #     D_j = 1 - w_i1j - ... - w_iNj
            #
            # where w_iXj is the unmasked weight for masked source
            # cell i and destination cell j.
            D = 1 - weights[:, src_mask].sum(axis=1, keepdims=True)

            # Get rid of values that are approximately zero, or
            # spuriously negative. These values of 'D' correspond to
            # destination cells that overlap only masked source
            # cells. These weights will be zeroed later on, so it's OK
            # to set their value to 1, i.e. a nice non-zero value that
            # will allow us to divide by 'D' in the next step.
            D = np.where(D < min_weight, 1, D)

            # Divide the weights by 'D'. Note that for destination
            # cells which do not intersect any masked source grid
            # cells, 'D' will now be 1.
            w = weights / D

            # Zero the weights associated with masked source grid
            # cells
            w[:, src_mask] = 0

            # Mask out rows of the weights matrix which contain all
            # zeros. Such a row corresponds to a destination grid cell
            # that does not intersect any unmasked source grid cells.
            #
            # Note: Rows that correspond to masked destination grid
            #       cells will have already been masked.
            w = np.ma.where(
                np.count_nonzero(w, axis=1, keepdims=True), w, np.ma.masked
            )

        elif method in ("linear", "bilinear", "nearest_dtos"):
            # 2) Linear and nearest neighbour methods:
            #
            # Mask out any row j that contains at least one positive
            # (i.e. greater than or equal to 'min_weight') w_ji that
            # corresponds to a masked source grid cell i. Such a row
            # corresponds to a destination grid cell that intersects
            # at least one masked source grid cell.
            if np.ma.isMA(weights):
                where = np.ma.where
            else:
                where = np.where

            j = np.unique(where((weights >= min_weight) & (src_mask))[0])
            if j.size:
                if np.ma.isMA(weights):
                    w = weights.copy()
                else:
                    w = np.ma.array(weights, copy=True)

                w[j, :] = np.ma.masked
            else:
                w = weights

        elif method in (
            "patch",
            "conservative_2nd",
            "nearest_stod",
        ):
            # 3) Patch recovery and second-order conservative methods:
            #
            # A reference source data mask has already been
            # incorporated into the weights matrix, and 'a' is assumed
            # to have the same mask (this is checked in `regrid`).
            w = weights

        else:
            raise ValueError(f"Unknown regrid method: {method!r}")

    # ----------------------------------------------------------------
    # Regrid the data by calculating the dot product of the weights
    # matrix with the source data
    # ----------------------------------------------------------------
    a = np.ma.getdata(a)
    a = w.dot(a)

    return a, src_mask, w


def regrid_weights(
    weights,
    row,
    col,
    src_shape,
    dst_shape,
    dtype=None,
    dst_mask=None,
    start_index=0,
    dense=True,
    order="C",
):
    """Create a weights matrix for use in `regrid`.

    .. versionadded:: 3.14.0

    .. seealso:: `regrid`, `_regrid`, `ESMF.Regrid.get_weights_dict`

    :Parameters:

        weights: `numpy.ndarray`
            The 1-d array of regridding weights for locations in the
            2-d dense weights matrix. The locations are defined by the
            *row* and *col* parameters.

        row, col: `numpy.ndarray`, `numpy.ndarray`
            The 1-d arrays of the row and column indices of the
            regridding weights in the dense weights matrix, which has
            J rows and I columns, where J and I are the total number
            of cells in the destination and source grids
            respectively. See the *start_index* parameter.

        row: `numpy.ndarray`
            The 1-d destination/row indices, as returned by
            `ESMF.Regrid.get_weights_dict`.

        col: `numpy.ndarray`
            The 1-d source/col indices, as returned by
            `ESMF.Regrid.get_weights_dict`.

        src_shape: sequence of `int`
            The shape of the source grid.

        dst_shape: sequence of `int`
            The shape of the destination grid.

        dtype: `str` or dtype, optional
            Typecode or data-type to which the weights are cast. The
            data-type of the weights must be the same as desired
            datatype of the regridded data. If unset then the weights
            data-type is unchanged.

        dst_mask: `numpy.ndarray` or `None`, optional
            A destination grid mask to be applied to the weights
            matrix, in addition to those destination grid cells that
            have no non-zero weights. If `None` (the default) then no
            additional destination grid cells are masked. If a Boolean
            `numpy` array then it must have shape *dst_shape*, and
            `True` signifies a masked cell.

        start_index: `int`, optional
            Specify whether the *row* and *col* parameters use 0- or
            1-based indexing. By default 0-based indexing is used.

        dense: `bool`, optional
            If True (the default) then return the weights as a dense
            `numpy` array. Otherwise return the weights as a `scipy`
            sparse array.

        order: `str`, optional
            If *dense* is True then specify the memory layout of the
            returned weights matrix. ``'C'`` (the default) means C
            order (row-major), and ``'F'`` means Fortran order
            (column-major).

    :Returns:

        `numpy.ndarray`
            The 2-d dense weights matrix, an array with with shape
            ``(J, I)``, where ``J`` is the number of destination grid
            cells and ``I`` is the number of source grid cells.

    """
    from math import prod

    from scipy.sparse import coo_array

    # Create a sparse array for the weights
    src_size = prod(src_shape)
    dst_size = prod(dst_shape)
    shape = [dst_size, src_size]

    if dtype is not None:
        weights = weights.astype(dtype, copy=False)

    if start_index:
        row = row - start_index
        col = col - start_index

    w = coo_array((weights, (row, col)), shape=shape)

    if dense:
        # Convert the sparse array to a dense array of weights, padded
        # with zeros.
        w = w.todense(order=order)

        # Mask out rows that correspond to masked destination
        # cells. Such a row will have all zero values, or be
        # identified by 'dst_mask'.
        not_masked = np.count_nonzero(w, axis=1, keepdims=True)
        if dst_mask is not None:
            if dst_mask.dtype != bool or dst_mask.shape != dst_shape:
                raise ValueError(
                    "The 'dst_mask' parameter must be None or a "
                    f"Boolean numpy array with shape {dst_shape}. Got: "
                    f"dtype={dst_mask.dtype}, shape={dst_mask.shape}"
                )

            not_masked = not_masked.astype(bool, copy=False)
            not_masked &= ~dst_mask.reshape(dst_mask.size, 1)

        if not not_masked.all():
            # Some destination cells are masked
            w = np.ma.where(not_masked, w, np.ma.masked)
    else:
        raise NotImplementedError("Can't yet use sparse arrays")

    return w
