# REVIEW: active: `collapse_active.py`: new module for active storage functionality
import logging
from functools import wraps
from numbers import Integral

try:
    from activestorage import Active
except ModuleNotFoundError:
    Active = None

from ...functions import active_storage as cf_active_storage
from ...functions import active_storage_url

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Specify which reduction methods are possible with active storage
# --------------------------------------------------------------------
active_reduction_methods = ("max", "mean", "min", "sum")


class ActiveStorageError(Exception):
    pass


def active_chunk(method, x, **kwargs):
    """Collapse data in a chunk with active storage.

    .. versionadded:: NEXTVERSION

    .. seealso:: `actify`, `active_storage`, `cf.data.collapse.Collapse`

    :Parameters:

        a: array_like
            The data to be collapsed.

        method: `str`
            The name of the reduction method. If the method does not
            have a corresponding active function in the
            `active_chunk_functions` dictionary then active storage
            computations are not carried out.

        axis: (sequence of) `int`, optional
            Axis or axes along which to operate. By default,
            flattened input is used.

        kwargs: optional
            Extra keyword arguments that define the reduction.

    :Returns:

        `dict`
            The reduced data in component form.

    **Examples**

    >>> d = active_chunk('sum', x)
    >>> d
    {'N': 7008, 'sum': 7006221.66903949}

    """
    # Return None if active storage reduction is not approriate, or
    # raise an ActiveStorageError it is appropriate but can't/didn't
    # work
    if not cf_active_storage():
        return

    weighted = kwargs.get("weights") is not None
    if weighted:
        return

    axis = kwargs.get("axis")
    if axis is not None:
        if isinstance(axis, Integral):
            axis = (axis,)

        if len(axis) < x.ndim:
            return

    try:
        filename = x.get_filename()
    except AttributeError:
        # This Dask chunk is not a data definition
        return
    else:
        if not filename:
            # This data definition doesn't have any files, so can't
            # support active storage reductions.
            return

    if hasattr(x, "actify"):
        url = active_storage_url().value
        if url is None:
            raise ActiveStorageError("No active storage URL")

        x = x.actify(url)

    # Still here? Then do active storage reduction
    if kwargs.get("computing_meta"):
        return x

    #    filename = x.get_filename()
    filename = "/".join(filename.split("/")[3:])

    max_threads = 100

    active_kwargs = {
        "uri": filename,
        "ncvar": x.get_address(),
        "storage_options": x.get_storage_options(),
        "active_storage_url": url,  # x.get_active_storage_url(),
        "storage_type": "s3",  # Temporary requirement!
        "max_threads": max_threads,
    }

    if False:
        print(f"Active(**{active_kwargs})")

    active = Active(**active_kwargs)
    active.method = method
    active.components = True

    import datetime
    import time

    try:
        lock = False  # True #False
        if lock:
            x._lock.acquire()
            start = time.time()
            print("START  LOCKED", x.index(), datetime.datetime.now())
            d = active[x.index()]
            print(
                "FINISH LOCKED",
                x.index(),
                datetime.datetime.now(),
                time.time() - start,
                f"maxT={max_threads}",
            )
            x._lock.release()
        else:
            start = time.time()
            print("START  unlocked", x.index(), datetime.datetime.now())
            d = active[x.index()]
            print(
                "FINISH unlocked",
                x.index(),
                datetime.datetime.now(),
                time.time() - start,
                f"maxT={max_threads}",
            )
    except Exception as error:
        raise ActiveStorageError(error)

    # Reformat the components dictionary to match the output of the
    # corresponding local chunk function
    if method == "max":
        # Local chunk function `cf_max_chunk`
        d = {"N": d["n"], "max": d["max"]}
    elif method == "mean":
        # Local chunk function `cf_mean_chunk`
        d = {"N": d["n"], "sum": d["sum"], "V1": d["n"], "weighted": False}
    elif method == "min":
        # Local chunk function `cf_min_chunk`
        d = {"N": d["n"], "min": d["min"]}
    elif method == "sum":
        # Local chunk function `cf_sum_chunk`
        d = {"N": d["n"], "sum": d["sum"]}
    else:
        raise ActiveStorageError(
            f"Don't know how to reformat {method!r} components"
        )

    return d


def actify(a, method, axis=None):
    """Modify a Dask array to use active storage reductions.

    The Dask graph is inspected to ensure that active storage
    reductions are possible, and if not then the Dask array is
    returned unchanged.

    .. note:: It is assumed that the `!active_storage` attribute of
              the `Data` object that provided the Dask array *a* is
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
            `active_chunk_functions` dictionary then active storage
            computations are not carried out.

        axis: (sequence of) `int`, optional
            Axis or axes along which to operate. By default,
            flattened input is used.

    :Returns:

        (`dask.array.Array`, function) or (`dask.array.Array`, `None`)
            If active storage operations are possible then return the
            modified Dask array and the new chunk reduction
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

    url = active_storage_url().value
    if url is None:
        # TODOACTIVE
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

    # Loop round the nodes of the Dask graph looking for data
    # definitions that i) point to files, and ii) which support active
    # storage operations; and modify the Dask graph when we find them.
    #
    # The elements are traversed in reverse order so that the data
    # definitions will tend to come out first, allowing for the
    # potential of a faster short circuit when using active storage is
    # not possible.
    #
    # Performance: The optimising the graph can be slow for
    #              complicated graphs, but nonetheless is essential to
    #              ensure that unused nodes are not considered.
    ok_to_actify = True
    dsk = collections_to_dsk((a,), optimize_graph=True)
    for key, value in reversed(dsk.items()):
        try:
            filename = value.get_filename()
        except AttributeError:
            # This Dask chunk is not a data definition
            continue

        if not filename:
            # This data definition doesn't have any files, so can't
            # support active storage reductions.
            ok_to_actify = False
            break

        # Still here? Then this chunk is a data definition that points
        # to files, so try to insert an actified copy into the Dask
        # graph.
        try:
            dsk[key] = value.actify(url)
        except AttributeError:
            # This data definition doesn't support active storage
            # reductions
            ok_to_actify = False
            break

    if not ok_to_actify:
        # It turns out that the Dask graph is not suitable for active
        # storage reductions, so return the input data unchanged.
        return a

    # Still here? Then the Dask graph supports active storage
    #             reductions => redefine the Dask array from the
    #             actified Dask graph.
    logger.warning(
        "At compute time, the collapse will be attempted with active "
        f"storage at URL {url}"
    )
    return da.Array(dsk, a.name, a.chunks, a.dtype, a._meta)


# --------------------------------------------------------------------
# Decorators
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
                and kwargs.get("active_storage")
                and cf_active_storage()
                #                and active_storage_url()
                and method in active_reduction_methods
                and kwargs.get("weights") is None
                and kwargs.get("chunk_function") is None
            ):
                # Attempt to actify the Dask array
                args = list(args)
                if args:
                    dx = args.pop(0)
                else:
                    dx = kwargs.pop("a")

                dx = actify(dx, method=method, axis=kwargs.get("axis"))
                args.insert(0, dx)

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

    def decorator(chunk_function):
        @wraps(chunk_function)
        def wrapper(*args, **kwargs):
            #            if args:
            #                x = args[0]
            #            else:
            #                x = kwargs["x"]
            #
            #            if getattr(x, "actified", False):
            try:
                # Try doing an active storage reduction on
                # actified chunk data
                out = active_chunk(method, *args, **kwargs)
            except ActiveStorageError as warning:
                # The active storage reduction failed
                logger.warning(f"{warning}. Reverting to local reduction.")
            else:
                if out is not None:
                    return out

            # Still here? Then do a local reduction.
            return chunk_function(*args, **kwargs)

        return wrapper

    return decorator
