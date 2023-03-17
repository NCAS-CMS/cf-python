from functools import wraps


# --------------------------------------------------------------------
# Define the active functions
# --------------------------------------------------------------------
def active_min(a, **kwargs):
    """Chunk calculations for the minimum.

    TODO Assumes that the calculations have already been done,
    i.e. that *a* is already the minimum.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its return signature must be the same
    as the non-active chunk function that it is replacing.

    .. versionadded:: ACTIVEVERSION

    :Parameters:

        a: `dict`
            TODOACTIVEDOCS

        kwargs: optional
            TODOACTIVEDOCS

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * min: The minimum of `a``.

    """
    return {"N": a["n"], "min": a["min"]}


def active_max(a, **kwargs):
    """Chunk calculations for the maximum.

    TODO Assumes that the calculations have already been done,
    i.e. that *a* is already the maximum.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its return signature must be the same
    as the non-active chunk function that it is replacing.

    .. versionadded:: ACTIVEVERSION

    :Parameters:

        a: `dict`
            TODOACTIVEDOCS

        kwargs: optional
            TODOACTIVEDOCS

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * max: The maximum of `a``.

    """
    return {"N": a["n"], "max": a["max"]}


def active_mean(a, **kwargs):
    """Chunk calculations for the unweighted mean.

    TODO Assumes that the calculations have already been done,
    i.e. that *a* is already the uweighted mean.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its return signature must be the same
    as the non-active chunk function that it is replacing.

    .. versionadded:: ACTIVEVERSION

    :Parameters:

        a: `dict`
            TODOACTIVEDOCS

        kwargs: optional
            TODOACTIVEDOCS

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * V1: The sum of ``weights``. Always equal to ``N``
                  because weights have not been set.
            * sum: The un-weighted sum of ``a``.
            * weighted: True if weights have been set. Always
                        False.

    """
    return {"N": a["n"], "V1": a["n"], "sum": a["sum"], "weighted": False}


def active_sum(a, **kwargs):
    """Chunk calculations for the unweighted sum.

    TODO Assumes that the calculations have already been done,
    i.e. that *a* is already the uweighted sum.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its return signature must be the same
    as the non-active chunk function that it is replacing.

    .. versionadded:: ACTIVEVERSION

    :Parameters:

        a: `dict`
            TODOACTIVEDOCS

        kwargs: optional
            TODOACTIVEDOCS

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * sum: The un-weighted sum of ``a``

    """
    return {"N": a["n"], "sum": a["sum"]}


# --------------------------------------------------------------------
# Create a map of reduction methods to their corresponding active
# functions
# --------------------------------------------------------------------
_active_chunk_functions = {
    "min": active_min,
    "max": active_max,
    "mean": active_mean,
    "sum": active_sum,
}


def actify(a, method, axis=None):
    """TODOACTIVEDOCS.

    It is assumed that:

    * The *method* has an entry in the `_active_chunk_functions`
      dictionary

    * The `!active_storage` attribute of the `Data` object that
      provided the dask array *a* is `True`. If this is not the case
      then an error at compute time is likely.

    .. versionadded:: ACTIVEVERSION

    :Parameters:

        a: `dask.array.Array`
            The array to be collapsed.

        method: `str`
            TODOACTIVEDOCS

        axis: (sequence of) `int`, optional
            TODOACTIVEDOCS

    :Returns:

        (`dask.array.Array`, function) or (`dask.array.Array`, `None`)
            TODOACTIVEDOCS

    """
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

    # Loop round elements of the dask graph, looking for data
    # definitions that point to a file and which support active
    # storage operations. The elements are traversed in reverse order
    # so that the data defintions come out first, allowing for the
    # potential of a faster short circuit when using active storage is
    # not possible.
    ok_to_actify = True
    dsk = collections_to_dsk((a,), optimize_graph=True)
    for key, value in reversed(dsk.items()):
        try:
            filenames = value.get_filenames()
        except AttributeError:
            # This dask chunk is not a data definition
            continue

        if not filenames:
            # This data definition doesn't have any files, so can't
            # support active storage reductions
            ok_to_actify = False
            break

        # Still here? Then this chunk is a data definition that points
        # to files, so try to insert an actified copy into the dask
        # graph.
        try:
            new_value = value.actify(method, axis)
        except AttributeError:
            # This data definition doesn't support active storage
            # reductions
            ok_to_actify = False
            break

        if new_value is None:
            # This data definition wasn't actifiable
            ok_to_actify = False
            break

        dsk[key] = new_value

    if not ok_to_actify:
        # The dask graph is not suitable for active storage
        # reductions, so return the input data unchanged.
        return a, None

    # Still here? Then all data definitions in the dask graph support
    # active storage reductions => redefine the array from the
    # actified dask graph, and define the active storage reduction
    # chunk function.
    return (
        da.Array(dsk, a.name, a.chunks, a.dtype, a._meta),
        _active_chunk_functions[method],
    )


def active_storage(method):
    """A decorator for `Collapse` methods that enables active storage
    operations, when the conditions are right.

    .. versionadded:: ACTIVEVERSION

    .. seealso `cf.data.collapse.Collapse`

    :Parameters:

        method: `str`
            TODOACTIVEDOCS

    """

    def decorator(collapse_method):
        @wraps(collapse_method)
        def wrapper(self, *args, **kwargs):
            if (
                kwargs.get("active_storage")
                and method in _active_chunk_functions
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
