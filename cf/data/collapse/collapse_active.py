from functools import wraps
import logging


# --------------------------------------------------------------------
# Define the active functions
# --------------------------------------------------------------------
def active_min(a, **kwargs):
    """Chunk function for minimum values computed by active storage.

    Converts active storage reduction components to the components
    expected by the reduction combine and aggregate functions.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its returned value must be the same as
    the non-active chunk function that it is replacing.

    .. versionadded:: ACTIVEVERSION

    .. seealso:: `actify`

    :Parameters:

        a: `dict`
            The components output from the active storage
            reduction. For instance:

            >>> print(a)
            {'min': array([[[49.5]]], dtype=float32), 'n': 1015808}

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * min: The minimum.

    """
    return {"N": a["n"], "min": a["min"]}


def active_max(a, **kwargs):
    """Chunk function for maximum values computed by active storage.

    Converts active storage reduction components to the components
    expected by the reduction combine and aggregate functions.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its returned value must be the same as
    the non-active chunk function that it is replacing.

    .. versionadded:: ACTIVEVERSION

    .. seealso:: `actify`

    :Parameters:

        a: `dict`
            The components output from the active storage
            reduction. For instance:

            >>> print(a)
            {'max': array([[[2930.4856]]], dtype=float32), 'n': 1015808}

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * max: The maximum.

    """
    return {"N": a["n"], "max": a["max"]}


def active_mean(a, **kwargs):
    """Chunk function for mean values computed by active storage.

    Converts active storage reduction components to the components
    expected by the reduction combine and aggregate functions.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its returned value must be the same as
    the non-active chunk function that it is replacing.

    .. versionadded:: ACTIVEVERSION

    .. seealso:: `actify`

    :Parameters:

        a: `dict`
            The components output from the active storage
            reduction. For instance:

            >>> print(a)
            {'sum': array([[[1.5131907e+09]]], dtype=float32), 'n': 1015808}

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * V1: The sum of ``weights``. Always equal to ``N``
                  because weights have not been set.
            * sum: The un-weighted sum.
            * weighted: True if weights have been set. Always
                        False.

    """
    return {"N": a["n"], "V1": a["n"], "sum": a["sum"], "weighted": False}


def active_sum(a, **kwargs):
    """Chunk function for sum values computed by active storage.

    Converts active storage reduction components to the components
    expected by the reduction combine and aggregate functions.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its returned value must be the same as
    the non-active chunk function that it is replacing.

    .. versionadded:: ACTIVEVERSION

    .. seealso:: `actify`

    :Parameters:

        a: `dict`
            The components output from the active storage
            reduction. For instance:

            >>> print(a)
            {'sum': array([[[1.5131907e+09]]], dtype=float32), 'n': 1015808}

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * sum: The un-weighted sum.

    """
    return {"N": a["n"], "sum": a["sum"]}


# --------------------------------------------------------------------
# Create a map of reduction methods to their corresponding active
# functions
# --------------------------------------------------------------------
active_chunk_functions = {
    "min": active_min,
    "max": active_max,
    "mean": active_mean,
    "sum": active_sum,
}


def actify(a, method, axis=None):
    """Modify a dask array to use active storage reductions.

    The dask graph is inspected to ensure that active storage
    reductions are possible, and if not then the dask array is
    returned unchanged.

    It is assumed that:

    * The method has a corresponding active function defined in the
      `active_chunk_functions` dictionary. If this is not the case
      then an error will occur at definition time.

    * The `!active_storage` attribute of the `Data` object that
      provided the dask array *a* is `True`. If this is not the case
      then an error at compute time is likely.

    .. versionadded:: ACTIVEVERSION

    .. seealso:: `active_storage`

    :Parameters:

        a: `dask.array.Array`
            The array to be collapsed.

        method: `str`
            The name of the reduction method. Must be a key of the
            `active_chunk_functions` dictionary.

        axis: (sequence of) `int`, optional
            Axis or axes along which to operate. By default,
            flattened input is used.

    :Returns:

        (`dask.array.Array`, function) or (`dask.array.Array`, `None`)
            If active storage operations are possible then return the
            modified dask array and the new chunk reduction
            function. Otherwise return the unaltered input array and
            `None`.

    """
    print ('runing actify')
    try:
        from activestorage import Active  # noqa: F401
    except ModuleNotFoundError:
        # The active storage class dependency is not met, so using
        # active storage is not possible.
        print('oops')
        return a, None

    from numbers import Integral

    import dask.array as da
    from dask.array.utils import validate_axis
    from dask.base import collections_to_dsk

    # Parse axis
    if axis is None:
        axis = tuple(range(a.ndim))
    else:
        if isinstance(axis, Integral):
            axis = (axis,)

        if len(axis) != a.ndim:
            # Can't (yet) use active storage to collapse a subset of
            # the axes, so return the input data unchanged.
            return a, None

        axis = validate_axis(axis, a.ndim)

    # Loop round the nodes of the dask graph looking for data
    # definitions that point to files and which support active storage
    # operations, and modify the dask graph when we find them.
    #
    # The elements are traversed in reverse order so that the data
    # defintions come out first, allowing for the potential of a
    # faster short circuit when using active storage is not possible.
    ok_to_actify = True
    dsk = collections_to_dsk((a,), optimize_graph=True)
    for key, value in reversed(dsk.items()):
        try:
            filename = value.get_filename()
        except AttributeError:
            # This dask chunk is not a data definition
            continue

        if not filename:
            # This data definition doesn't have any files, so can't
            # support active storage reductions.
            ok_to_actify = False
            break

        # Still here? Then this chunk is a data definition that points
        # to files, so try to insert an actified copy into the dask
        # graph.
        try:
            dsk[key] = value.actify(method, axis)
        except AttributeError:
            # This data definition doesn't have an 'actify' method,
            # and so doesn't support active storage reductions.
            ok_to_actify = False
            break

    if not ok_to_actify:
        # It turns out that the dask graph is not suitable for active
        # storage reductions, so return the input data unchanged.
        return a, None

    # Still here? Then all data definitions in the dask graph support
    # active storage reductions => redefine the dask array from the
    # actified dask graph, and set the active storage reduction chunk
    # function.
    return (
        da.Array(dsk, a.name, a.chunks, a.dtype, a._meta),
        active_chunk_functions[method],
    )


def active_storage(method):
    """A decorator that enables active storage reductions.

    This decorator is intended for `Collapse` methods. When a
    `Collapse` method is decorated, active storage operations are only
    carried out when the conditions are right. See `Collapse` for
    details.

    .. versionadded:: ACTIVEVERSION

    .. seealso:: `actify`, `cf.data.collapse.Collapse`

    :Parameters:

        method: `str`
            The name of the reduction method. If it is not one of the
            keys of the `active_chunk_functions` dictionary then
            active storage reductions will not occur.

    """

    def decorator(collapse_method):
        @wraps(collapse_method)
        def wrapper(self, *args, **kwargs):
            if (
                kwargs.get("active_storage")
                and method in active_chunk_functions
                and kwargs.get("weights") is None
                and kwargs.get("chunk_function") is None
            ):
                # Attempt to actify the dask array and provide a new
                # chunk function
                a, chunk_function = actify(
                    args[0],
                    method=method,
                    axis=kwargs.get("axis"),
                )
                args = list(args)
                args[0] = a

                if chunk_function is not None:
                    # The dask array has been actified, so update the
                    # chunk function.
                    kwargs["chunk_function"] = chunk_function

            # Create the collapse
            return collapse_method(self, *args, **kwargs)

        return wrapper

    return decorator
