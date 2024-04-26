"""Regridding functions used within a dask graph."""

import numpy as np


def regrid(
    a,
    weights_dst_mask=None,
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

        weights_dst_mask: 2-`tuple`
            The sparse weights matrix that defines the regridding
            operation; and the mask to be applied to the regridded
            data (as yet unmodified for the source grid mask).

            **weights**

            The dense weights matrix has J rows and I columns, where J
            and I are the total number of cells in the destination and
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
            grid cell. I.e. *weights* is usually a very sparse matrix.

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

            **dst_mask**

            If a `numpy.ndarray` with shape ``(J,)`` then this is the
            reference destination grid mask that was used during the
            creation of the weights. If `None` then there are no
            reference destination grid masked points.

            In either case the reference destination grid mask may get
            updated (not in-place) to account for source grid masked
            points.

        method: `str`
            The name of the regridding method.

        src_shape: sequence of `int`
            The shape of the source grid.

        dst_shape: sequence of `int`
            The shape of the destination grid.

        axis_order: sequence of `int`
            The axis order that transposes *a* so that the regrid axes
            become the trailing dimensions, ordered consistently with
            the order used to create the weights matrix; and the
            non-regrid axes become the leading dimensions.

            *Parameter example:*
              If the regrid axes are in positions 2 and 1 for 4-d
              data: ``[0, 3, 2, 1]``

            *Parameter example:*
              If the regrid axes are in positions 0 and 3 for 4-d
              data: ``[1, 2, 0, 3]``

            *Parameter example:*
              If the regrid axis is in position 0 for 3-d data: ``[1,
              2, 0]``

        ref_src_mask: `numpy.ndarray` or `None`
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
            smallest value that produces the same masks as esmpy for
            the use cases defined in the cf test suite.

            **Linear regridding**

            Destination grid cell j will only be masked if a) it is
            masked in the destination grid definition; or b) ``w_ji >=
            min_weight`` for those masked source grid cells i for
            which ``w_ji > 0``.

            **Conservative first-order regridding**

            Destination grid cell j will only be masked if a) it is
            masked in the destination grid definition; or b) the sum
            of ``w_ji`` for all non-masked source grid cells i is
            strictly less than *min_weight*.

    :Returns:

        `numpy.ndarray`
            The regridded data.

    """
    weights, dst_mask = weights_dst_mask

    # ----------------------------------------------------------------
    # Reshape the array into a form suitable for the regridding dot
    # product, i.e. a 2-d array whose right-hand dimension represents
    # are the gathered regridding axes and whose left-hand dimension
    # represent of all the other dimensions.
    # ----------------------------------------------------------------
    n_src_axes = len(src_shape)
    a = a.transpose(axis_order)
    non_regrid_shape = a.shape[: a.ndim - n_src_axes]
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
        # into the sparse weights matrix. Therefore, the mask for all
        # slices of 'a' must be the same as this reference mask.
        if variable_mask or (src_mask is None and ref_src_mask.any()):
            raise ValueError(
                f"Can't regrid with the {method!r} method when the source "
                f"data mask varies over different {n_src_axes}-d "
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
        # the sparse weights matrix for the mask of each slice.
        #
        # However, if the mask of a regrid slice is the same as the
        # mask of its previous regrid slice then we can reuse the
        # sparse weights matrix that already has the correct mask.
        n_slices = a.shape[1]
        regridded_data = np.ma.empty(
            (dst_size, n_slices), dtype=weights.dtype, order="F"
        )
        prev_weights, prev_src_mask, prev_dst_mask = None, None, None
        for n in range(n_slices):
            n = slice(n, n + 1)
            a_n = a[:, n]
            (
                regridded_data[:, n],
                prev_src_mask,
                prev_dst_mask,
                prev_weights,
            ) = _regrid(
                a_n,
                np.ma.getmaskarray(a_n[:, 0]),
                dst_mask,
                weights,
                method,
                prev_src_mask=prev_src_mask,
                prev_dst_mask=prev_dst_mask,
                prev_weights=prev_weights,
                min_weight=min_weight,
            )

        a = regridded_data
        del a_n, regridded_data, prev_weights, prev_src_mask, prev_dst_mask
    else:
        # Source data is either not masked or the source mask is same
        # for all slices => all slices can be regridded
        # simultaneously.
        a, _, _, _ = _regrid(
            a, src_mask, dst_mask, weights, method, min_weight=min_weight
        )
        del _

    # ----------------------------------------------------------------
    # Reshape the regridded data back to its original axis order
    # ----------------------------------------------------------------
    a = a.T
    a = a.reshape(non_regrid_shape + tuple(dst_shape))

    n_dst_axes = len(dst_shape)

    if n_src_axes == 1 and n_dst_axes == 2:
        # The regridding operation increased the number of data axes
        # by 1 => modify 'axis_order' to contain the new axis.
        #
        # E.g. UGRID -> regular lat-lon could change 'axis_order' from
        #      [0,2,1] to [0,3,1,2]
        raxis = axis_order[-1]
        axis_order = [
            i if i <= raxis else i + n_dst_axes - 1 for i in axis_order
        ]
        axis_order[-1:] = range(raxis, raxis + n_dst_axes)
    elif n_src_axes == 2 and n_dst_axes == 1:
        # The regridding operation decreased the number of data axes
        # by 1 => modify 'axis_order' to remove the removed axis.
        #
        # E.g. regular lat-lon -> UGRID could change 'axis_order' from
        #      [0,2,4,5,1,3] to [0,2,3,4,1], or [0,2,4,5,3,1] to
        #      [0,1,3,4,2]
        raxis0, raxis = axis_order[-2:]
        axis_order = [i if i <= raxis else i - 1 for i in axis_order[:-1]]
    elif n_src_axes == 3 and n_dst_axes == 1:
        # The regridding operation decreased the number of data axes
        # by 2 => modify 'axis_order' to remove the removed axes.
        #
        # E.g. regular Z-lat-lon -> DSG could change 'axis_order' from
        #      [0,2,5,1,3,4] to [0,2,3,1], or [0,2,4,5,3,1] to
        #      [0,1,2,3]
        raxis0, raxis1 = axis_order[-2:]
        if raxis0 > raxis1:
            raxis0, raxis1 = raxis1, raxis0

        new = []
        for i in axis_order[:-2]:
            if i <= raxis0:
                new.append(i)
            elif raxis0 < i <= raxis1:
                new.append(i - 1)
            else:
                new.append(i - 2)

        axis_order = new
    elif n_src_axes != n_dst_axes:
        raise ValueError(
            f"Can't (yet) regrid from {n_src_axes} dimensions to "
            f"{n_dst_axes} dimensions"
        )

    d = {k: i for i, k in enumerate(axis_order)}
    axis_reorder = [i for k, i in sorted(d.items())]

    a = a.transpose(axis_reorder)
    return a


def _regrid(
    a,
    src_mask,
    dst_mask,
    weights,
    method,
    prev_src_mask=None,
    prev_dst_mask=None,
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

        dst_mask: `numpy.ndarray` or `None`
            The reference destination grid mask to be applied to the
            regridded data. If `None` then this is equivalent to a
            mask of all `False`. If a Boolean `numpy` array then it
            must have shape ``(J,)``, where ``J`` is the number of
            destination grid cells, and `True` signifies a masked
            cell.

        weights: `scipy.sparse.spmatrix`
            The sparse weights matrix that defines the regridding
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

        prev_src_mask: `numpy.ndarray` or `None`
            The source grid mask used by a previous call to `_regrid`.
            See *prev_weights* for details.

        prev_dst_mask: `numpy.ndarray` or `None`
            The destination grid mask used by a previous call to
            `_regrid`. If *prev_src_mask* equals *src_mask* then the
            *prev_dst_mask* mask is used to calculate the regridded
            data, bypassing any need to calculate a new destination
            grid mask. Ignored if `prev_src_mask` is `None`.

        prev_weights: `numpy.ndarray`, optional
            The weights matrix used by a previous call to `_regrid`,
            possibly modified to account for missing data. If
            *prev_src_mask* equals *src_mask* then the *prev_weights*
            weights matrix is used to calculate the regridded data,
            bypassing any need to calculate a new weights matrix.
            Ignored if `prev_src_mask` is `None`.

    :Returns:

        4-`tuple`
            * `numpy.ndarray`: The regridded data.
            * `numpy.ndarray` or `None`: The source grid mask applied
                                         to the returned weights
                                         matrix, always identical to
                                         the *src_mask* parameter.
            * `numpy.ndarray` or `None`: The destination grid mask
                                         applied to the regridded
                                         data.
            * `numpy.ndarray`: The weights matrix used to regrid *a*.

    """
    if src_mask is None or not src_mask.any():
        # ------------------------------------------------------------
        # Source data is not masked
        # ------------------------------------------------------------
        pass
    elif prev_src_mask is not None and (prev_src_mask == src_mask).all():
        # ------------------------------------------------------------
        # Source data is masked, but no need to modify the weights
        # since we have been provided with an already-modified weights
        # matrix.
        # ------------------------------------------------------------
        weights = prev_weights
        dst_mask = prev_dst_mask
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
            #     D_j = w_i1j + ... + w_iNj
            #
            # where each w_iXj is the weight for unmasked source cell
            # i and destination cell j.
            dst_size = weights.shape[0]
            if dst_mask is None:
                dst_mask = np.zeros((dst_size,), dtype=bool)
            else:
                dst_mask = dst_mask.copy()

            weights = weights.copy()

            # Note: It is much more efficient to access
            #       'weights.indptr', 'weights.indices', and
            #       'weights.data' directly, rather than iterating
            #       over rows of 'weights' and using 'weights.getrow'.
            count_nonzero = np.count_nonzero
            indptr = weights.indptr.tolist()
            indices = weights.indices
            data = weights.data
            for j, (i0, i1) in enumerate(zip(indptr[:-1], indptr[1:])):
                mask = src_mask[indices[i0:i1]]
                if not count_nonzero(mask):
                    continue

                if mask.all():
                    dst_mask[j] = True
                    continue

                w = data[i0:i1]
                D_j = w[~mask].sum()
                w = w / D_j
                w[mask] = 0
                data[i0:i1] = w

            del indptr

        elif method in ("linear", "bilinear", "nearest_dtos"):
            # 2) Linear and nearest neighbour methods:
            #
            # Mask out any row j that contains at least one positive
            # (i.e. greater than or equal to 'min_weight') w_ji that
            # corresponds to a masked source grid cell i. Such a row
            # corresponds to a destination grid cell that intersects
            # at least one masked source grid cell.
            dst_size = weights.shape[0]
            if dst_mask is None:
                dst_mask = np.zeros((dst_size,), dtype=bool)
            else:
                dst_mask = dst_mask.copy()

            # Note: It is much more efficient to access
            #       'weights.indptr', 'weights.indices', and
            #       'weights.data' directly, rather than iterating
            #       over rows of 'weights' and using 'weights.getrow'.
            count_nonzero = np.count_nonzero
            where = np.where
            indptr = weights.indptr.tolist()
            indices = weights.indices
            pos_data = weights.data >= min_weight
            for j, (i0, i1) in enumerate(zip(indptr[:-1], indptr[1:])):
                mask = src_mask[indices[i0:i1]]
                if not count_nonzero(mask):
                    continue

                if where((mask) & (pos_data[i0:i1]))[0].size:
                    dst_mask[j] = True

            del indptr, pos_data

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
            pass

        else:
            raise ValueError(f"Unknown regrid method: {method!r}")

    # ----------------------------------------------------------------
    # Regrid the data by calculating the dot product of the weights
    # matrix with the source data
    # ----------------------------------------------------------------
    a = np.ma.getdata(a)
    a = weights.dot(a)

    if dst_mask is not None:
        a = np.ma.array(a)
        a[dst_mask] = np.ma.masked

    return a, src_mask, dst_mask, weights


def regrid_weights(operator, dst_dtype=None):
    """Create a weights matrix and destination grid mask for `regrid`.

    .. versionadded:: 3.14.0

    .. seealso:: `regrid`, `_regrid`

    :Parameters:

        operator: `RegridOperator`
            The definition of the source and destination grids and the
            regridding weights.

            .. versionadded:: 3.14.2

        dst_dtype: `str`, dtype, or `None`, optional
            Typecode or data-type to which the weights are cast. The
            data-type of the weights must be the same as desired
            datatype of the regridded data. If `None`, the default,
            then the weights data-type is unchanged.

    :Returns:

        2-`tuple`
            The sparse weights matrix, and the 1-d destination grid
            mask. If the latter is `None` then it is equivalent to a
            mask of all `False`.

    """
    from math import prod

    operator.tosparse()

    weights = operator.weights
    if dst_dtype is not None and weights.dtype != dst_dtype:
        # Convert weights to have the same dtype as the regridded data
        weights = weights.astype(dst_dtype)

    dst_mask = operator.dst_mask
    if dst_mask is not None:
        # Convert dst_mask to a 1-d array
        dst_mask = dst_mask.reshape((prod(operator.dst_shape),))

    return weights, dst_mask
