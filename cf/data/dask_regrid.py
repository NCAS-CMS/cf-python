"""Regridding functions executed during a dask compute."""
from functools import partial

import numpy as np


def regrid(
    a,
    weights=None,
    method=None,
    src_shape=None,
    dst_shape=None,
    axis_order=None,
    ref_src_mask=None,
):
    """TODODASK

    .. versionadded:: TODODASK

    .. seealso:: `regrid_weights`, `_regrid`, 

    :Parmaeters:

        a: `numpy.ndarray`
            The array to be regridded.

        weights: `numpy.ndarray`
            The weights matrix that defines the regridding operation.
            
            The weights matrix has J rows and I columns, where J and I
            are the total number of cells in the destination and
            source grids respectively. 

            The weights matrix only describes cells defined by the
            regridding dimensions. If the array *a* includes
            non-regridding dimensions then, in essence, the the regrid
            operation is carried out seperately for each slice of the
            regridding dimensions. For instance, if an *a* that
            represents T, Z, Y, X dimensions with shape ``(12, 20, 73,
            96)`` is to have its Y and X dimension regridded, then the
            result may be thought of as the concatenation of the 240
            individual regrids arising from all of the T and Z
            dimension combinations.

            Each element w_ji is the multiplicative weight that
            defines how much of V_si (the value in source grid cell i)
            contributes to V_dj (the value in destination grid cell
            j).

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
            constructed taking into account the mask of the source
            grid, which must match the mask of *a* for each regridding
            slice.

            For all other regridding methods, the weights matrix will
            have been constructed assuming that no source grid cells
            are masked. However, the weights matrix will be modified
            on-the-fly to account for any masked elements of *a* in
            each regridding slice.

            It is assumed that data-type of the weights matrix is same
            as the desired data-type of the regridded data.
    
            See section 12.3 "Regridding Methods" of
            https://earthsystemmodeling.org/docs/release/latest/ESMF_refdoc/node1.html

        method: `str`
            The name of the regridding method.

    :Returns:

        `numpy.ndarray`
            The regridded data.

    """
    # ----------------------------------------------------------------
    # Reshape the array into a form suitable for the regridding dot
    # product
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
        if ref_src_mask is not None and (
            ref_src_mask.dtype != bool or ref_src_mask.shape != src_shape
        ):
            raise ValueEror(
                "The 'ref_src_mask' parameter must be None or a "
                "boolean numpy array with shape {src_shape}. Got: "
                f"dtype={ref_src_mask.dtype}, shape={ref_src_mask.shape}"
            )

        message = (
            f"Can't regrid with the {method!r} method when the "
            "source data mask does not match that used to "
            "construct the regrid operator"
        )
        if variable_mask:
            raise ValueError(message)

        if src_mask is None and ref_src_mask.any():
            raise ValueError(message)

        if src_mask is not None:
            if ref_src_mask.size != src_mask.size:
                raise ValueError(message)

            ref_src_mask = ref_src_mask.reshape(src_mask.shape)
            if (src_mask != ref_src_mask).any():
                raise ValueError(message)

    # ----------------------------------------------------------------
    # Regrid the source data
    # ----------------------------------------------------------------
    if variable_mask:
        # Source data is masked and the source mask varies across
        # slices => we have to regrid each slice separately.
        #
        # However, if the mask of a regrid slice is the same as the
        # mask of its previous regrid slice then we don't need to
        # re-adjust the weights.
        n_slices = a.shape[1]
        regridded_data = np.ma.empty(
            (dst_size, n_slices), dtype=weights.dtype, order="F"
        )
        prev_weights, prev_mask = None, None
        for n in range(n_slices):
            n = slice(n, n + 1)
            a_n = a[:, n]
            regridded_data[:, n], prev_weights, prev_mask = _regrid(
                a_n, weights, method, a_n.mask, prev_weights, prev_mask
            )

        a = regridded_data
        del a_n, regridded_data, prev_weights, prev_mask
    else:
        # Source data is not masked or the source mask is same for all
        # slices => all slices can be regridded simultaneously.
        a, _, _ = _regrid(a, weights, method, src_mask)
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


def _regrid(a, weights, method, src_mask, prev_weights=None, prev_mask=None):
    """Worker function for `regrid`.

    Modifies the *weights* matrix to account for missing data in *a*,
    and creates the regridded array by forming the dot product of the
    modifiedx *weights* and *a*.

    .. versionadded:: TODODASK

    .. seealso:: `regrid`, `regrid_weights`

    :Parameters:

        a: `numpy.ndarray`
            The array to be regridded. Must have shape ``(I, n)`` for
            any positive ``n``, and where ``I`` is the number of
            source grid cells. Performance will be optimised if *a*
            has Fortran order memory layout (column-major order), as
            opposed to C order memory layout (row-major order).

        weights: `numpy.ndarray`
            The weights matrix that defines the regridding
            operation. Might be modified (not in-place) to account for
            missing data in *a*. Must have shape ``(J, I)`` where
            ``J`` is the number of destination grid cells and ``I`` is
            the number of source grid cells. Performance will be
            optimised if *weights* has C order memory layout
            (row-major order), as opposed to Fortran order memory
            layout (column-major order).

            See `regrid` for details.

        method: `str`
            The name of the regridding method.

        src_mask: `numpy.ndarray` or `None`

        prev_weights: `numpy.ndarray`, optional
            The weights matrix used by a previous call to `_regrid`,
            possibly modifed to account for missing data. If set along
            with `prev_mask` then, if appropriate, this weights matrix
            will be used instead of creating a new one. Ignored if
            `prev_mask` is `None`.

        prev_mask: `numpy.ndarray`, optional
            The source grid mask used by a previous call to `_regrid`.

    :Returns:
        
        3-`tuple`:
     
        * `numpy.ndarray`: The regridded data.

        * `numpy.ndarray` or `scipy._sparray`: the weights matrix used
                                               to carry out the
                                               regridding.

        * `numpy.ndarray` or `None`: The source grid mask applied to
                                     the returned weights matrix,
                                     always identical to the
                                     *src_mask* parameter.

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
            # w_ji that corresponds to a masked source grid cell i.
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
            # incorporated into the weights matrix, and 'a' is assumed
            # to have the same mask (although this is not checked).
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
    src_shape,
    dst_shape,
    dtype=None,
    dst_mask=None,
    quarter=False,
    dense=True,
    order="C",
):
    """TODODASK

    .. versionadded:: TODODASK

    .. seealso:: `regrid`, `_regrid`

    :Parameters:

        weights: `numpy.ndarray`
          
        row: `numpy.ndarray`
          
        col: `numpy.ndarray`

        src_shape: sequence of `int`

        dst_shape: sequence of `int`

        dtype: `str` or dtype, optional
            Typecode or data-type to which the weights are cast. The
            data-type of the weights must be the same as desired
            datatype of the regridded data. If unset then the weights
            data-type is unchanged.

        dst_mask: `numpy.ndarray` or `None`, optional
            A destination grid mask to be applied to the weights
            matrix, in addition to those destination grid cells that
            have no non-zero weights. If a boolean `numpy` array then
            it must have shape *dst_shape*.

        quarter: `bool`, optional

        dense: `bool`, optional
            If True (the default) then return the weights as a dense
            `numpy` array. Otherwise return the weights as a `scipy`
            sparse array.

        order: `str`, optional
            If *dense* is True then specify the memory layout of the
            weights matrix. ``'C'`` (the default) means C order,
            ``'F'`` means Fortran order.

    :Returns:
    
        `numpy.ndarray` or `scipy._sparray`
            The weights matrix.

    """
    from math import prod

    # Create a sparse array for the weights
    src_size = prod(src_shape)
    dst_size = prod(dst_shape)
    shape = [dst_size, src_size]

    if dtype is not None:
        weights = weights.astype(dtype, copy=False)

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
            if dst_mask.dtype != bool or dst_mask.shape != dst_shape:
                raise ValueEror(
                    "The 'dst_mask' parameter must be None or a "
                    "boolean numpy array with shape {dst_shape}. Got: "
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
