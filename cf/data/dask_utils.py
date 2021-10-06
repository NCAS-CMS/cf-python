"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

import numpy as np


def cf_harden_mask(a):
    """Harden the mask of a masked `numpy` array.

    Has no effect if the array is not a masked array.

    .. versionadded:: TODODASK

    .. seealso:: `cf.Data.harden_mask`

    :Parameters:

        a: `numpy.ndarray`
            The array to have a hardened mask.

    :Returns:

        `numpy.ndarray`
            The array with hardened mask.

    """
    if np.ma.isMA(a):
        a.harden_mask()

    return a


def cf_soften_mask(a):
    """Soften the mask of a masked `numpy` array.

    Has no effect if the array is not a masked array.

    .. versionadded:: TODODASK

    .. seealso:: `cf.Data.soften_mask`

    :Parameters:

        a: `numpy.ndarray`
            The array to have a softened mask.

    :Returns:

        `numpy.ndarray`
            The array with softened mask.

    """
    if np.ma.isMA(a):
        a.soften_mask()

    return a


def cf_where(array, condition, x, y, hardmask):
    """Set elements of *array* from *x* or *y* depending on *condition*.

    The input *array* is not changed in-place.

    See `where` for details on the expected functionality.

    .. note:: This function correctly sets the mask hardness of the
              output array.

    .. versionadded:: TODODASK

    .. seealso:: `cf.Data.where`

    :Parameters:

        array: numpy.ndarray
            The array to be assigned to.

        condition: numpy.ndarray
            Where False or masked, assign from *y*, otherwise assign
            from *x*.

        x: numpy.ndarray or `None`
            *x* and *y* must not both be `None`.

        y: numpy.ndarray or `None`
            *x* and *y* must not both be `None`.

        hardmask: `bool`
           Set the mask hardness for a returned masked array. If True
           then a returned masked array will have a hardened mask, and
           the mask of the input *array* (if there is one) will be
           applied to the returned array, in addition to any masked
           elements arising from assignments from *x* or *y*.

    :Returns:

        `numpy.ndarray`
            A copy of the input *array* with elements from *y* where
            *condition* is False or masked, and elements from *x*
            elsewhere.

    """
    mask = None

    if np.ma.isMA(array):
        # Do a masked where
        where = np.ma.where
        if hardmask:
            mask = array.mask
    elif np.ma.isMA(x) or np.ma.isMA(y):
        # Do a masked where
        where = np.ma.where
    else:
        # Do a non-masked where
        where = np.where
        hardmask = False

    condition_is_masked = np.ma.isMA(condition)
    if condition_is_masked:
        condition = condition.astype(bool)

    if x is not None:
        # Assign values from x
        if condition_is_masked:
            # Replace masked elements of condition with False, so that
            # masked locations are assigned from array
            c = condition.filled(False)
        else:
            c = condition

        array = where(c, x, array)

    if y is not None:
        # Assign values from y
        if condition_is_masked:
            # Replace masked elements of condition with True, so that
            # masked locations are assigned from array
            c = condition.filled(True)
        else:
            c = condition

        array = where(c, array, y)

    if hardmask:
        if mask is not None and mask.any():
            # Apply the mask from the input array to the result
            array.mask |= mask

        array.harden_mask()

    return array
