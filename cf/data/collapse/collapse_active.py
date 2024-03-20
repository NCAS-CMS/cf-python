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


def active_chunk(method, x, **kwargs):
    """Collapse a data in a chunk with active storage.

    .. versionadded:: NEXTVERSION

    .. seealso:: `actify`, `active_storage2`, `cf.data.collapse.Collapse`

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
    if kwargs.get("computing_meta"):
        return x

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

    print("ACTIVE CHUNK DONE!")
    return d


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
    #
    # Performance: The optimisation is essential, but can be slow for
    #              complicated graphs.
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


# --------------------------------------------------------------------
# Decoators
# --------------------------------------------------------------------
def active_storage(method):
    """Decorator for active storage reductions on `Collapse` methods.

    When a `Collapse` method is decorated, active storage operations
    are carried out if the conditions are right.

    .. versionadded:: NEXTVERSION

    .. seealso:: `actify`, `cf.data.collapse.Collapse`

    :Parameters:

        method: `str`
            The name of the reduction method. If it is one of the
            `active_chunk_methods` then active storage reductions
            *might* occur.

    """

    def decorator(collapse_method):
        @wraps(collapse_method)
        def wrapper(self, *args, **kwargs):
            if (
                Active is not None
                and method in active_reduction_methods
                and kwargs.get("active_storage")
                and kwargs.get("weights") is None
                and kwargs.get("chunk_function") is None
                and cf_active_storage()
                and active_storage_url()
            ):
                # Attempt to actify the dask array
                args = list(args)
                if args:
                    dask_array = args.pop(0)
                else:
                    dask_array = kwargs.pop("a")

                dask_array = actify(
                    dask_array,
                    method=method,
                    axis=kwargs.get("axis"),
                )
                args.insert(0, dask_array)

            # Run the collapse method
            return collapse_method(self, *args, **kwargs)

        return wrapper

    return decorator


def active_storage_chunk(method):
    """Decorator for active storage reductions on chunks.

    Intended for the ``cf_*_chunk`` methods in
    cf.data.collapse.dask_collapse`.

    .. versionadded:: NEXTVERSION

    :Parameters:

        method: `str`
            The name of the reduction method. If it is one of the
            `active_chunk_methods` then active storage reductions
            *might* occur.

    """

    def decorator(chunk):
        @wraps(chunk)
        def wrapper(*args, **kwargs):
            if (
                Active is not None
                and method in active_reduction_methods
                and cf_active_storage()
                and active_storage_url()
            ):
                try:
                    # Try doing an active storage reduction
                    return active_chunk(method, *args, **kwargs)
                except ValueError:
                    pass

            # Still here? Then we couldn't do an active storage
            # reduction, so we'll do a local one.
            return chunk(*args, **kwargs)

        return wrapper

    return decorator
