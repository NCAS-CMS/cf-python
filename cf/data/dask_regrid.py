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
):
    """TODODASK

    .. versionadded:: TODODASK

    :Parmaeters:

        a: `numpy.ndarray`
            The array to be regridded.

        return_dict: `bool`, optional
            If True then retun a dictionary containing the
            *base_operator* and the `numpy` array of the regridded
            data, with keys ``'base_operator'`` and ``'array``
            respectively. By default the regridded data is returned as
            a `numpy` array.

    :Returns:

        `numpy.ndarray` or `dict`
            The regridded data or, if *return_dict* is True, a
            dictionary containing the *base_operator* and the
            regridded data.

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
        if mask.shape[1] == 1 or set(np.unique(mask.sum(axis=1))).issubset(
            (0, mask.shape[1])
        ):
            # There is only one slice, or the source mask is same for
            # all slices.
            src_mask = mask[:, 0]
        else:
            # Source mask varies across slices
            variable_mask = True

        del mask

    # Regrid the source data
    if variable_mask:
        # Source data is masked and the source mask varies across
        # slices => regrid each slice sparately.
        #
        # If the mask of this slice is the same as the mask of the
        # previous slice then we don't need to re-adjust the weights.
        n_slices = a.shape[1]
        r = np.ma.empty((dst_size, n_slices), dtype=dst_dtype, order="F")
        prev_weights, prev_mask = None, None
        for n in range(n_slices):
            n = slice(n, n + 1)
            data = a[:, n]
            regridded_data, prev_weights, prev_mask = _regrid(
                data, weights, method, data.mask, prev_weights, prev_mask
            )
            r[:, n] = regridded_data

        a = r
        del data, regridded_data, prev_weights, prev_mask
    else:
        # Source data is not masked or the source mask is same for all
        # slices => regrid all slices simultaneously.
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

    Adjusts the weights matrix to account for missing data in *a*, and
    creates the regridded array by forming the dot product of the
    (modified) weights and *a*.

    **Definition of the weights matrix**
            
    The input *weights* is a j by i matrix where each element is the
    multiplicative weight (between 0 and 1) that defines how much
    V_si, the value in source grid cell i, contributes to V_dj, the
    value in desintation grid j.

        w_ij = f_ij * A_si / A_dj
            
    where f_ij is the fraction of source cell i that contributes to
    destination cell j, A_si is the area of source cell i, and A_dj is
    the rea of destination cell j.
            
    The final value of V_dj is the sum of w_ij * V_si for all source
    grid cells i. Note that it is typical that, for a given j, most
    w_ij are zero, because f_ij will be zero for those source grid
    cells that do not intersect with the desination grid cell.

    See
    https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node1.html
    for details (especially section 12.3 "Regridding Methods").

    If the destination grid is masked, either because it spans areas
    outside of the definition of the source grid, or by selection
    (such as ocean cells), then the corresponding rows in the
    'weights' matrix must be be entirely missing data.

    The *weights* matrix will have been constructed assuming that no
    source grid cells are masked. If the source data *a* do in fact
    include masked elements, then this function adjusts the weights
    on-th-fly to account for them.

    .. versionadded:: TODODASK

    .. seealso:: `weights`


   ALL THIS ASSUMES -2-d what about 3-d?

    """
    if src_mask is None or not src_mask.any():
        # ------------------------------------------------------------
        # Source data is not masked
        # ------------------------------------------------------------
        w = weights
    elif prev_mask is not None and (prev_mask == src_mask).all():
        # ------------------------------------------------------------
        # Source data is masked, but no need to re-calculate the
        # weights since this source data's mask is the same as the
        # previous slice.
        # ------------------------------------------------------------
        w = prev_weights
    else:
        # ------------------------------------------------------------
        # Source data is masked and we need to adjust the weights
        # accordingly
        # ------------------------------------------------------------
        if method in ("conservative", 'conservative_1st'):
            # First-order consevative method:
            #
            # If source grid cell i is masked then w_ij are set to
            # zero for all destination cells j, and the remaining
            # non-zero w_ij are divided by D_j, the fraction of
            # destination cell j that intersects with unmasked cells
            # of the source grid.
            #
            #     D_j = 1 - w_i1j - ... - wiNj
            #
            # where w_iXj is the unmasked weight for masked source
            # cell i and desination cell j.            
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
        elif method in ("linear", 'nearest_stod', 'nearest_dtos'):
            # Linear and nearest neighbour methods:
            #
            # Mask out any destination cell with a positive w_ij that
            # corresponds to a masked source cell. I.e. set row j of
            # 'weights' to all missing data if it contains a positive
            # w_ij that corresponds to a masked source cell.
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
        elif method in ("patch", "conservative_2d"):
            # Patch recovery and and second-order consevative methods:
            #
            # Don't yet know how to adjust the weights matrix for
            # source missing data.
            raise ValueError(
                f"Can't regrid masked data with the {method!} method"
            )

    # ----------------------------------------------------------------
    # Regrid the data by calculating the dot product of the weights
    # matrix with the source data
    # ----------------------------------------------------------------
    a = np.ma.getdata(a)
    a = w.dot(a)

    return a, weights, src_mask


def regrid_weights(
    weights,
    row,
    col,
    src_shape=None,
    dst_shape=None,
    dst_mask=None,
        dense=True,
    order="C",
):
    """TODODASK

    .. versionadded:: TODODASK

    .. seealso:: `regrid`, `_regrid`

    """
    from math import prod
    from scipy.sparse import coo_array
    
    # Create a sparse array for the weights
    src_size = prod(src_shape)
    dst_size = prod(dst_shape)
    w = coo_array((weights, (row - 1, col - 1)), shape=[dst_size, src_size])

    if dense:
        # Convert the sparse array to a dense array
        w = w.toarray(order=order)

        # Mask out rows that correspond to masked destination
        # cells. Such a row will have all zero values, or be
        # identified by 'dst_mask'.
#        mask = np.count_nonzero(w, axis=1, keepdims=True)
#        mask = ~mask.astype(bool)
#        if dst_mask is not None:
#            dst_mask = dst_mask.reshape(dst_mask.size, 1)
#            mask |= dst_mask
#
#        if mask.any():
#            w = np.ma.where(mask, np.ma.masked, w)

        not_masked = np.count_nonzero(w, axis=1, keepdims=True)
        if dst_mask is not None:
            not_masked = not_masked.astype(bool, copy=False)
            not_masked &= ~dst_mask.reshape(dst_mask.size, 1)

        if not not_masked.all():
            # Some destination cells are masked
            w = np.ma.where(not_masked, w, np.ma.masked)

    return w
