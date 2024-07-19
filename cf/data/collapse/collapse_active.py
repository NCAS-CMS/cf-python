# REVIEW: active: `collapse_active.py`: new module for active storage functionality
import logging
from functools import wraps
from numbers import Integral

try:
    from activestorage import Active
except ModuleNotFoundError:
    pass

from ...functions import active_storage as cf_active_storage
from ...functions import active_storage_url, is_log_level_debug

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Specify which reduction methods are possible with active storage
# --------------------------------------------------------------------
active_reduction_methods = ("max", "mean", "min", "sum")


class ActiveStorageError(Exception):
    pass


def active_chunk_function(method, x, **kwargs):
    """Collapse data in a chunk with active storage.

    If an active storage reduction is not approriate then `None` is
    returned, or else an ActiveStorageError is raised if the active
    storage operation fails.

    .. versionadded:: NEXTVERSION

    .. seealso:: `actify`

    :Parameters:

        method: `str`
            The name of the reduction method (e.g. ``'mean'``).

        x: array_like
            The data to be collapsed.

        kwargs: optional
            Extra keyword arguments that define the reduction.

    :Returns:

        `dict` or `None`
            The reduced data in component form, or else `None` if an
            active storage reduction is not approriate.

    **Examples**

    >>> d = active_chunk_function('sum', x)
    >>> d
    {'N': 7008, 'sum': 7006221.66903949}

    Active storage reduction is not yet possible for variances:

    >>> d = active_chunk_function('variance', x)
    >>> print(d)
    None

    """
    if kwargs.get("computing_meta"):
        print("COMPUTING_META", method, repr(x), kwargs)
        return x
        
    # Return None if active storage reduction is not appropriate
    print(method, repr(x), kwargs)
        
    if not cf_active_storage():
        # Active storage is turned off => do a local reduction
        return

    if method not in active_reduction_methods:
        # Active storage is not available for this method => do a
        # local reduction
        return

    if not getattr(x, "active_storage", False):
        # Active storage operations are not allowed on 'x' => do a
        # local reduction
        return
    
    weighted = kwargs.get("weights") is not None
    if weighted:
        # Active storage is not allowed for weighted reductions => do
        # a local reduction
        return

    axis = kwargs.get("axis")
    if axis is not None:
        if isinstance(axis, Integral):
            axis = (axis,)

        if len(axis) < x.ndim:
            # Active storage is not allowed for reductions over a
            # subset of the axes => do a local reduction
            return

    # Raise an ActiveStorageError if the active storage reduction can't
    # happen or fails
    url = active_storage_url().value
    if url is None:
        # Active storage operations are not possible when an active
        # storage URL has not been set => do a local reduction
        raise ActiveStorageError("No active storage URL")

    # ----------------------------------------------------------------
    # Still here? Set up an Active instance that will carry out the
    # active storage operation.
    # ----------------------------------------------------------------
    index = x.index()

    filename = x.get_filename()        
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

    active = Active(**active_kwargs)
    active.method = method
    active.components = True

    if is_log_level_debug:
        logger.debug(f"Active call: Active(**{active_kwargs})[{index}]")

    import datetime
    import time

    # ----------------------------------------------------------------
    # Execute the active storage operation
    # ----------------------------------------------------------------
    try:
        start = time.time()
        print("START  unlocked", index, datetime.datetime.now())
        d = active[index]
        print(
            "FINISH unlocked",
            datetime.datetime.now(),
            time.time() - start,
            f"maxT={max_threads}",
        )
    except Exception as error:
        # Something went wrong with the active storage operations =>
        # do a local reduction
        print ('565')
        raise ActiveStorageError(error)

    # ----------------------------------------------------------------
    # Reformat the components dictionary to match the output of the
    # corresponding local chunk function
    # ----------------------------------------------------------------
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

    return d


# --------------------------------------------------------------------
# Decorators
# --------------------------------------------------------------------
def actify(method):
    """Decorator for active storage reductions on chunks.

    Intended for to decorate the ``cf_*_chunk`` methods in
    cf.data.collapse.dask_collapse`.

    When a ``cf_*_chunk`` method is decorated, then its computations
    will be carried out in active storage, if that is appropriate and
    possible. Whether or not computations are done in active storage
    is determined by `active_chunk_function`.

    .. versionadded:: NEXTVERSION

    .. seealso:: `active_chunk_function`

    :Parameters:

        method: `str`
            The name of the reduction method.

    """

    def decorator(chunk_function):
        @wraps(chunk_function)
        def wrapper(*args, **kwargs):
            
            #if args: TODO
            #    x = args[0]
            #else:
            #    x = kwargs["x"]
            try:
                # Try doing an active storage reduction                
                print (method, args, kwargs)
                out = active_chunk_function(method, *args, **kwargs)
            except ActiveStorageError as warning:
                # The active storage reduction failed
                logger.warning(
                    "Dask chunk failed in active storage reduction => "
                    f"reverting to local computation: {warning}"
                )
            else:
                if out is not None:
                    # The active storage reduction succeeded                
                    return out

            # Still here? Then using active storage is not
            # appropriate, or else doing the active storage operation
            # failed => do a local computation.
            return chunk_function(*args, **kwargs)

        return wrapper

    return decorator
