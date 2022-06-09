"""Regridding functions executed during a dask compute."""
from functools import partial

import numpy as np


def regrid(
    a,
    weights=None,
    method=None,
    src_shape=None,
    dst_shape=None,
    dst_dtype=None,
    axis_order=None,
    ref_src_mask=None,
):
    """TODODASK

    .. versionadded:: TODODASK

    :Parmaeters:

        a: `numpy.ndarray`
            The array to be regridded.

        weights: `numpy.ndarray`
            The weights matrix that defines the regridding operation.
            
            The weights matrix has J rows and I columns, where J and I
            are the total number of cells in the destination and
            source grids respectively. Each element w_ji is the
            multiplicative weight that defines how much of V_si (the
            value in source grid cell i) contributes to V_dj (the
            value in destination grid cell j).

            The final value of V_dj is the sum of w_ji * V_si for all
            source grid cells i. Note that it is typical that for a
            given j most w_ji will be zero, reflecting the fact only a
            few source grid cells intersect a particular destination
            grid cell. I.e. *weights* is typically a very sparse
            matrix.
        
            If the destination grid has masked cells, either because
            it spans areas outside of the definition of the source
            grid, or by selection (such as ocean cells in a land-only
            grid), then the corresponding rows in the weights matrix
            must be be entirely missing data.
        
            For the patch recovery and second-order conservative
            regridding methods, the weights matrix will have been
            constructed taking the source grid mask into account,
            which must match the mask of *a*.

            For all other regridding methods, the weights matrix will
            have been constructed assuming that no source grid cells
            are masked. However, the weights matrix will be modified
            on-the-fly to account for any masked elements of *a*.

    :Returns:

        `numpy.ndarray`
            The regridded data.

    """
    # Reshape the array into a form suitable for the regridding dot
    # product
    a = a.transpose(axis_order)
    non_regrid_shape = a.shape[: a.ndim - len(src_shape)]
    dst_size, src_size = weights.shape
    a = a.reshape(-1, src_size)
    a = a.T

    src_mask = None
    variable_mask = False
    if np.ma.is_masked(a):
        # Source data is masked for at least one slice
        mask = np.ma.getmaskarray(a)
        if mask.shape[1] == 1 or set(mask.sum(axis=1).tolist()).issubset(
            (0, mask.shape[1])
        ):
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
                "Can't regrid data with non-constant mask using the "
                f"{method!r} method"
            )

        if src_mask is not None:
            if ref_src_mask.size == src_mask.size:
                ref_src_mask = ref_src_mask.reshape(src_mask.shape)

            if (src_mask != ref_src_mask).any():
                raise ValueError(
                    "Can't regrid data with non-constant mask using the "
                    f"{method!r} method"
                )

    # Regrid the source data
    if variable_mask:
        # Source data is masked and the source mask varies across
        # slices => we have to regrid each slice separately.
        #
        # If the mask of this slice is the same as the mask of the
        # previous slice then we don't need to re-adjust the weights.
        n_slices = a.shape[1]
        regridded_data = np.ma.empty(
            (dst_size, n_slices), dtype=dst_dtype, order="F"
        )
        prev_weights, prev_mask = None, None
        for n in range(n_slices):
            n = slice(n, n + 1)
            data = a[:, n]
            regridded_data[:, n], prev_weights, prev_mask = _regrid(
                data, weights, method, data.mask, prev_weights, prev_mask
            )

        a = regridded_data
        del data, regridded_data, prev_weights, prev_mask
    else:
        # Source data is not masked or the source mask is same for all
        # slices => we can regrid all slices simultaneously.
        a, _, _ = _regrid(a, weights, method, src_mask)
        del _

    # Reshape the regridded data back to its original axis order
    a = a.T
    a = a.reshape(non_regrid_shape + tuple(dst_shape))

    d = {k: i for i, k in enumerate(axis_order)}
    axis_reorder = [i for k, i in sorted(d.items())]
    a = a.transpose(axis_reorder)

    return a


# https://earthsystemmodeling.org/esmpy_doc/nightly/develop/html/


def _regrid(a, weights, method, src_mask, prev_weights=None, prev_mask=None):
    """Worker function for `regrid`.

    Modifies the *weights* matrix to account for missing data in *a*,
    and creates the regridded array by forming the dot product of the
    modifiedx *weights* and *a*.

    .. versionadded:: TODODASK

    .. seealso:: `regrid`, `regrid_weights`

    :Parameters:

        weights: `numpy.ndarray`
            The weights matrix that defines the regridding
            operation. Might be modified (not in-place) to account for
            missing data in *a*.
            
            See `regrid` for details.

        src_mask: `numpy.ndarray` or `None`

        prev_weights: `numpy.ndarray`, optional
            The weights matrix used by a previous call to `_regrid`,
            possibly modifed to account for missing data. If set along
            with `prev_mask` then, if appropriate, this weights matrix
            will be used instead of creating a new one. Ignored if
            `prev_mask` is `None`.

        prev_mask: `numpy.ndarray`, optional
            The source grid mask used by a previous call to `_regrid`.

   ALL THIS ASSUMES -2-d what about 3-d?

    """
    if src_mask is None:
        # ------------------------------------------------------------
        # Source data is not masked
        # ------------------------------------------------------------
        w = weights
    elif not src_mask.any():
        # ------------------------------------------------------------
        # Source data is not masked
        # ------------------------------------------------------------
        w = weights
    elif prev_mask is not None and (prev_mask == src_mask).all():
        # ------------------------------------------------------------
        # Source data is masked, but no need to modify the weights, as
        # we have been provided with an already-modified weights
        # matrix.
        # ------------------------------------------------------------
        w = prev_weights
    else:
        # ------------------------------------------------------------
        # Source data is masked and we need to adjust the weights
        # accordingly
        # ------------------------------------------------------------
        if method in ("conservative", "conservative_1st"):
            # First-order consevative method:
            #
            #     w_ji = f_ji * A_si / A_dj
            #
            # where f_ji is the fraction of source cell i that
            # contributes to destination cell j, A_si is the area of
            # source cell i, and A_dj is the area of destination cell
            # j.
            #
            # If source grid cell i is masked then w_ji are set to
            # zero for all destination cells j, and the remaining
            # non-zero w_ji are divided by D_j, the fraction of
            # destination cell j that intersects with unmasked cells
            # of the source grid.
            #
            #     D_j = 1 - w_i1j - ... - wiNj
            #
            # where w_iXj is the unmasked weight for masked source
            # cell i and desination cell j.
            #
            # See section 12.3 "Regridding Methods" of
            # https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node1.html
            D = 1 - weights[:, src_mask].sum(axis=1, keepdims=True)

            # Get rid of values that are approximately zero, or
            # negative. Very small or negative values can occur as a
            # result of rounding.
            D = np.where(D < np.finfo(D.dtype).eps, 1, D)

            # Divide the weights by 'D' (note that for destination
            # cells which do not intersect any masked source grid
            # cells, 'D' will be 1) and zero weights associated with
            # masked source grid cells.
            w = weights / D
            w[:, src_mask] = 0

            # Mask out rows of the weights matrix which contain all
            # zeros. Such a row corresponds to a destination grid cell
            # that does not intersect any unmasked source grid cells.
            #
            # Note: Rows that correspond to masked destination grid
            #       cells have already been masked.
            w = np.ma.where(
                np.count_nonzero(w, axis=1, keepdims=True), w, np.ma.masked
            )
        elif method in ("linear", "nearest_stod", "nearest_dtos"):
            # Linear and nearest neighbour methods:
            #
            # Mask out any row j that contains at least one positive
            # w_ji that corresponds to a masked source cell i.
            if np.ma.isMA(weights):
                where = np.ma.where
            else:
                where = np.where

            j = np.unique(where((weights > 0) & (src_mask))[0])

            if j.size:
                if np.ma.isMA(weights):
                    w = weights.copy()
                else:
                    w = np.ma.array(weights, copy=True)

                w[j, :] = np.ma.masked
            else:
                w = weights
        elif method in ("patch", "conservative_2nd"):
            # Patch recovery and second-order consevative methods:
            #
            # A reference source data mask has already been
            # incorporated into the weights matrix. The check that the
            # mask of 'a' matches the reference mask has already been
            # done in `regrid`.
            w = weights
        else:
            raise ValueError(f"Unknown regrid method: {method!r}")

    # ----------------------------------------------------------------
    # Regrid the data by calculating the dot product of the weights
    # matrix with the source data
    # ----------------------------------------------------------------
    a = np.ma.getdata(a)
    a = w.dot(a)

    return a, w, src_mask


def regrid_weights(
    weights,
    row,
    col,
    src_shape=None,
    dst_shape=None,
    dst_mask=None,
    quarter=False,
    dense=True,
    order="C",
):
    """TODODASK

    .. versionadded:: TODODASK

    .. seealso:: `regrid`, `_regrid`

    """
    from math import prod

    # Create a sparse array for the weights
    src_size = prod(src_shape)
    dst_size = prod(dst_shape)
    shape = [dst_size, src_size]

    if quarter:
        # Quarter the weights matrix, choosing the top left quarter
        # (which will be equivalent to the botoom right quarter).
        from scipy.sparse import csr_array

        w = csr_array((weights, (row - 1, col - 1)), shape=shape)
        w = w[: dst_size / 2, : src_size / 2]
    else:
        from scipy.sparse import coo_array

        w = coo_array((weights, (row - 1, col - 1)), shape=shape)
        # if dense==False, convert to csr/csc?
        
    if dense:
        # Convert the sparse array to a dense array
        w = w.todense(order=order)

        # Mask out rows that correspond to masked destination
        # cells. Such a row will have all zero values, or be
        # identified by 'dst_mask'.
        not_masked = np.count_nonzero(w, axis=1, keepdims=True)
        if dst_mask is not None:
            not_masked = not_masked.astype(bool, copy=False)
            not_masked &= ~dst_mask.reshape(dst_mask.size, 1)

        if not not_masked.all():
            # Some destination cells are masked
            w = np.ma.where(not_masked, w, np.ma.masked)
    else:
        raise NotImplementedError("Can't yet use sparse arrays")
    
    return w
