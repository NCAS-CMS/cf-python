from functools import wraps


def active_min(a, **kwargs):
    """Chunk calculations for the minimum.

    TODO Assumes that the calculations have already been done,
    i.e. that *a* is already the minimum.

    This function is intended to be passed to `dask.array.reduction`
    as the ``chunk`` parameter. Its return signature must be the same
    as the non-active chunk function that it is replacing.

    .. versionadded:: TODOACTIVEVER

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

    .. versionadded:: TODOACTIVEVER

    :Parameters:

        a: `dict`
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

    .. versionadded:: TODOACTIVEVER

    :Parameters:

        a: `dict`
            TODOACTIVEDOCS

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * V1: The sum of ``weights``. Always equal to ``N``
                  because weights have not been set.
            * sum: The weighted sum of ``a``.
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

    .. versionadded:: TODOACTIVEVER

    :Parameters:

        a: `dict`
            TODOACTIVEDOCS

    :Returns:

        `dict`
            Dictionary with the keys:

            * N: The sample size.
            * sum: The weighted sum of ``a``

    """
    return {"N": a["n"], "sum": a["sum"]}

# Create a lookup of the active functions
_active_chunk_functions = {
    "min": active_min,
    "max": active_max,
    "mean": active_mean,
    "sum": active_sum,
}


def actify(a, method, axis=None):
    """TODOACTIVEDOCS.

    TODO: Describe the necessary conditions here.

    .. versionadded:: TODOACTIVEVER

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

    if not (method in _active_chunk_functions and method in Active.methods()):
        # The given method is not supported, so return the input data
        # unchanged.
        return a, None

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

    filenames = set()
    chunk_functions = set()

    # Loop round elements of the dask graph, looking for data
    # definitions that point to a file and which support active
    # storage operations. The elements are traversed in reverse order
    # so that the data defintions come out first, allowing for a
    # faster short circuit when using active storage is not possible.
    #
    # It is assumed that `actify` has only been called if has been
    # deterimined externally that it is sensible to do so. A
    # necessary, but not sufficient, condition for this being the case
    # will is the parent `Data` instance's `active_storage` attribute
    # being `True`.
    ok_to_actify = True
    dsk = collections_to_dsk((a,), optimize_graph=True)
    for key, value in reversed(dsk.items()):
        try:
            filenames.add(value.get_filename())
        except AttributeError:
            if hasattr(value, "get_full_value"):
                # This value is a constant fragment (such as might
                # arise from CFA aggregated data), which precludes the
                # use of active stoarge.
                ok_to_actify = False
                break

            continue

        # Still here? Then this value is a file fragment, so try to
        # actify it.
        try:
            value = value.actify(method, axis)
        except AttributeError:
            # This file fragment does not support active storage
            # reductions
            ok_to_actify = False
            break

        chunk_functions.add(_active_chunk_functions[method])

        # Still here? Then update the dask graph dictionary with the
        # actified data definition value.
        dsk[key] = value

    for filename in filenames:
        # TODOACTIVE: Check that Active(filename) supports active
        #             storage. I don't really know how this will work
        #             ...
        if not OK:
            # This file location does not support active storage, so
            # return the input data unchanged.
            return a, None

    # Still here?
    if ok_to_actify:
        # All data definitions in the dask graph support active
        # storage reductions => redefine the array from the actified
        # dask graph, and define the active storage reduction chunk
        # function.
        a = da.Array(dsk, a.name, a.chunks, a.dtype, a._meta)
        chunk_function = _active_chunk_functions[method]
    else:
        chunk_function = None

    return a, chunk_function


def active_storage(method):
    """A decorator for `Collapse` methods that enables active storage
    operations, when the conditions are right.

    .. versionadded:: TODOACTIVEVER

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
