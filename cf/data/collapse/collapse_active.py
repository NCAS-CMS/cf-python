import logging
from functools import wraps

try:
    from activestorage import Active
except ModuleNotFoundError:
    Active = None

from ...functions import active_storage as cf_active_storage
from ...functions import active_storage_url

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Specify which reductions are possible with active storage
# --------------------------------------------------------------------
active_reduction_methods = ("max", "mean", "min", "sum")


def active_reduction(x, method, axis=None, **kwargs):
    """Collapse data in a file with `Active`.

    .. versionadded:: NEXTVERSION

    .. seealso:: `actify`, `cf.data.collapse.Collapse`

    :Parameters:

        a: `dask.array.Array`
            The array to be collapsed.

        method: `str`
            The name of the reduction method. If the method does not
            have a corresponding active function in the
            `active_chunk_functions` dictionary then active
            compuations are not carried out.

        axis: (sequence of) `int`, optional
            Axis or axes along which to operate. By default,
            flattened input is used.

        kwargs: optional
            Extra keyword arguments that define the reduction.

    :Returns:

        `dict`
            The reduced data in component form.

    """
    if not getattr(x, "actified", False):
        raise ValueError(
            "Can't do active reductions when on non-actified data"
        )

    weighted = kwargs.get("weights") is not None
    if weighted:
        raise ValueError(f"Can't do weighted {method!r} active reductions")

    filename = x.get_filename()
    filename = "/".join(filename.split("/")[3:])

    active_kwargs = {
        "uri": filename,
        "ncvar": x.get_address(),
        "storage_options": x.get_storage_options(),
        "active_storage_url": x.get_active_storage_url(),
        "storage_type": "s3",  # Temporary requirement!
    }

    if True:
        print(f"Active(**{active_kwargs})")

    active = Active(**active_kwargs)

    # Provide a file lock
    try:
        lock = x._lock
    except AttributeError:
        pass
    else:
        if lock:
            active.lock = lock

    # Create the output dictionary
    active.method = method
    active.components = True
    d = active[x.index]

    # Reformat the output dictionary
    if method == "max":
        d = {"N": d["n"], "max": d["max"]}
    elif method == "mean":
        d = {"N": d["n"], "sum": d["sum"], "V1": d["n"], "weighted": weighted}
    elif method == "min":
        d = {"N": d["n"], "min": d["min"]}
    elif method == "sum":
        d = {"N": d["n"], "sum": d["sum"]}

    print("DONE!")
    return d


# --------------------------------------------------------------------
# Define the active functions
# --------------------------------------------------------------------
# def active_min(x, dtype=None, computing_meta=False, **kwargs):
#    """Chunk function for minimum values computed by active storage.
#
#    Converts active storage reduction components to the components
#    expected by the reduction combine and aggregate functions.
#
#    This function is intended to be passed to `dask.array.reduction`
#    as the ``chunk`` parameter. Its returned value must be the same as
#    the non-active chunk function that it is replacing.
#
#    .. versionadded:: NEXTVERSION
#
#    .. seealso:: `actify`, `active_storage`
#
#    :Parameters:
#
#        See `dask.array.reductions` for details of the parameters.
#
#    :Returns:
#
#        `dict`
#            Dictionary with the keys:
#
#            * N: The sample size.
#            * min: The minimum ``x``.
#
#    """
#    if computing_meta:
#        return x
#
#    x = active_reduction(x, "min", **kwargs)
#    return {"N": x["n"], "min": x["min"]}
#
#
# def active_max(a, **kwargs):
#    """Chunk function for maximum values computed by active storage.
#
#    Converts active storage reduction components to the components
#    expected by the reduction combine and aggregate functions.
#
#    This function is intended to be passed to `dask.array.reduction`
#    as the ``chunk`` parameter. Its returned value must be the same as
#    the non-active chunk function that it is replacing.
#
#    .. versionadded:: NEXTVERSION
#
#    .. seealso:: `actify`, `active_storage`
#
#    :Parameters:
#
#        a: `dict`
#            The components output from the active storage
#            reduction. For instance:
#
#            >>> print(a)
#            {'max': array([[[2930.4856]]], dtype=float32), 'n': 1015808}
#
#    :Returns:
#
#        `dict`
#            Dictionary with the keys:
#
#            * N: The sample size.
#            * max: The maximum.
#
#    """
#    if computing_meta:
#        return x
#
#    x = active_reduction(x, "max", **kwargs)
#    return {"N": a["n"], "max": a["max"]}
#
#
# def active_mean(a, **kwargs):
#    """Chunk function for mean values computed by active storage.
#
#    Converts active storage reduction components to the components
#    expected by the reduction combine and aggregate functions.
#
#    This function is intended to be passed to `dask.array.reduction`
#    as the ``chunk`` parameter. Its returned value must be the same as
#    the non-active chunk function that it is replacing.
#
#    .. versionadded:: NEXTVERSION
#
#    .. seealso:: `actify`, `active_storage`
#
#    :Parameters:
#
#        a: `dict`
#            The components output from the active storage
#            reduction. For instance:
#
#            >>> print(a)
#            {'sum': array([[[1.5131907e+09]]], dtype=float32), 'n': 1015808}
#
#    :Returns:
#
#        `dict`
#            Dictionary with the keys:
#
#            * N: The sample size.
#            * V1: The sum of ``weights``. Always equal to ``N``
#                  because weights have not been set.
#            * sum: The un-weighted sum.
#            * weighted: True if weights have been set. Always
#                        False.
#
#    """
#    if computing_meta:
#        return x
#
#    x = active_reduction(x, "mean", **kwargs)
#    return {"N": a["n"], "V1": a["n"], "sum": a["sum"], "weighted": False}
#
#
# def active_sum(a, **kwargs):
#    """Chunk function for sum values computed by active storage.
#
#    Converts active storage reduction components to the components
#    expected by the reduction combine and aggregate functions.
#
#    This function is intended to be passed to `dask.array.reduction`
#    as the ``chunk`` parameter. Its returned value must be the same as
#    the non-active chunk function that it is replacing.
#
#    .. versionadded:: NEXTVERSION
#
#    .. seealso:: `actify`, `active_storage`
#
#    :Parameters:
#
#        a: `dict`
#            The components output from the active storage
#            reduction. For instance:
#
#            >>> print(a)
#            {'sum': array([[[1.5131907e+09]]], dtype=float32), 'n': 1015808}
#
#    :Returns:
#
#        `dict`
#            Dictionary with the keys:
#
#            * N: The sample size.
#            * sum: The un-weighted sum.
#
#    """
#    if computing_meta:
#        return x
#
#    x = active_reduction(x, "sum", **kwargs)
#    return {"N": a["n"], "sum": a["sum"]}


# --------------------------------------------------------------------
# Create a map of reduction methods to their corresponding active
# functions
# --------------------------------------------------------------------
# active_chunk_functions = {
#    "min": True, #active_min,
#    "max": active_max,
#    "mean": active_mean,
#    "sum": active_sum,
# }


def actify(a, method, axis=None):
    """Modify a dask array to use active storage reductions.

    The dask graph is inspected to ensure that active storage
    reductions are possible, and if not then the dask array is
    returned unchanged.

    .. note:: It is assumed that the `!active_storage` attribute of
              the `Data` object that provided the dask array *a* is
              `True`. If this is not the case then an error at compute
              time is likely. The value of the `Data` object's
              `!active_storage` attribute is registered via the
              *active_storage* parameter of `Collapse` methods.

    .. versionadded:: NEXTVERSION

    .. seealso:: `active_storage`, `cf.data.collapse.Collapse`

    :Parameters:

        a: `dask.array.Array`
            The array to be collapsed.

        method: `str`
            The name of the reduction method. If the method does not
            have a corresponding active function in the
            `active_chunk_functions` dictionary then active
            compuations are not carried out.

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
    import dask.array as da
    from dask.base import collections_to_dsk

    if Active is None:
        raise AttributeError(
            "Can't actify {self.__class__.__name__} when "
            "activestorage.Active is not available"
        )

    if method not in active_reduction_methods:
        # The method cannot be calculated with active storage, so
        # return the input data unchanged.
        return a

    # Parse axis
    ndim = a.ndim
    if axis is None:
        axis = tuple(range(ndim))
    else:
        from numbers import Integral

        from dask.array.utils import validate_axis

        if isinstance(axis, Integral):
            axis = (axis,)

        axis = validate_axis(axis, ndim)
        if len(axis) != ndim or len(set(axis)) != ndim:
            # Can't (yet) use active storage to collapse a subset of
            # the axes, so return the input data unchanged.
            return a

    # Loop round the nodes of the dask graph looking for data
    # definitions that point to files and which support active storage
    # operations, and modify the dask graph when we find them.
    #
    # The elements are traversed in reverse order so that the data
    # defintions come out first, allowing for the potential of a
    # faster short circuit when using active storage is not possible.
    url = str(active_storage_url())
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
            dsk[key] = value.actify(url)
        except AttributeError:
            # This data definition doesn't support active storage
            # reductions
            ok_to_actify = False
            break

    if not ok_to_actify:
        # It turns out that the dask graph is not suitable for active
        # storage reductions, so return the input data unchanged.
        return a

    # Still here? Then all data definitions in the dask graph support
    # active storage reductions => redefine the dask array from the
    # actified dask graph, and set the active storage reduction chunk
    # function.
    logger.warning(
        "At compute time, data will be collapsed with "
        f"active storage at URL {url}"
    )
    return da.Array(dsk, a.name, a.chunks, a.dtype, a._meta)


def active_storage(method):
    """A decorator that enables active storage reductions.

    This decorator is intended for `Collapse` methods. When a
    `Collapse` method is decorated, active storage operations are only
    carried out when the conditions are right.

    .. versionadded:: NEXTVERSION

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
                cf_active_storage()
                and Active is not None
                and kwargs.get("active_storage")
                and method in active_reduction_methods
                and kwargs.get("weights") is None
                and kwargs.get("chunk_function") is None
                and active_storage_url()
            ):
                # Attempt to actify the dask array and provide a new
                # chunk function
                if args:
                    dask_array = args[0]
                else:
                    dask_array = kwargs.pop("a")

                #                dask_array, chunk_function = actify(
                dask_array = actify(
                    dask_array,
                    method=method,
                    axis=kwargs.get("axis"),
                )
                args = list(args)
                args[0] = dask_array

                # if chunk_function is not None:
                #    # The dask array has been actified, so update the
                #    # chunk function.
                #    kwargs["chunk_function"] = chunk_function

            # Create the collapse
            return collapse_method(self, *args, **kwargs)

        return wrapper

    return decorator
