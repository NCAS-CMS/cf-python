# REVIEW: active: `collapse_active.py`: new module for active storage functionality
import datetime
import logging
import time
from functools import wraps
from numbers import Integral

try:
    from activestorage import Active
except ModuleNotFoundError:
    pass

from ...functions import (
    active_storage,
    active_storage_max_requests,
    active_storage_url,
    is_log_level_debug,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Specify which reduction methods are possible with active storage
# --------------------------------------------------------------------
active_reduction_methods = ("max", "mean", "min", "sum")


class ActiveStorageError(Exception):
    pass


def active_chunk_function(method, *args, **kwargs):
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
    >>> print(d)
    {'N': 7008, 'sum': 7006221.66903949}

    Active storage reduction is not (yet) possible for variances:

    >>> d = active_chunk_function('variance', x, weights)
    >>> print(d)
    None

    """
    x = args[0]
    if kwargs.get("computing_meta"):
        return x

    # ----------------------------------------------------------------
    # Return None if active storage reduction is not
    # appropriate. Inside `actify` this will trigger a local reduction
    # to be carried out instead.
    # ----------------------------------------------------------------
    if not active_storage():
        # Active storage is turned off
        return

    url = kwargs.get("active_storage_url")
    if url is None:
        url = active_storage_url().value
        if url is None:
            return

    if method not in active_reduction_methods:
        # Active storage is not available for this method
        return

    if not getattr(x, "active_storage", False):
        # Active storage operations are not allowed on 'x'
        return

    if len(args) == 2:
        # Weights, if present, are always passed in as a positional
        # parameter, never as a keyword parameter. See
        # `dask.array.reductions.reduction`.
        weights = args[1]
        if weights is not None:
            # Active storage is not allowed for weighted reductions
            return

    axis = kwargs.get("axis")
    if axis is not None:
        if isinstance(axis, Integral):
            axis = (axis,)

        if len(axis) < x.ndim:
            # Active storage is not allowed for reductions over a
            # subset of the axes
            return

    # ----------------------------------------------------------------
    # Still here? Set up an Active instance that will carry out the
    # active storage operation. If it fails then this will trigger
    # (inside `actify`) a local reduction being carried out instead.
    # ----------------------------------------------------------------
    filename = x.get_filename()
    address = x.get_address()
    max_requests = active_storage_max_requests()

    active_kwargs = {
        "uri": "/".join(filename.split("/")[3:]),
        "ncvar": address,
        "storage_options": x.get_storage_options(),
        "active_storage_url": url,
        "storage_type": "s3",  # Temporary requirement to Active!
        "max_threads": max_requests,
    }

    index = x.index()
    
    debug = is_log_level_debug(logger)
    debug = True
    if debug:
        start = time.time()
        details = (
            f"{method!r} (file={filename}, address={address}, url={url}, "
            f"max_requests={max_requests}, chunk={index})"
        )
#        logger.debug(
        print(
            f"INITIATING active storage reduction {details}: "
            f"{datetime.datetime.now()}"
        )  # prgama: no cover

    active = Active(**active_kwargs)
    active.method = method
    active.components = True

    # Force active storage reduction on remote server 
    active._version = 2

    # ----------------------------------------------------------------
    # Execute the active storage operation by indexing the Active
    # instance
    # ----------------------------------------------------------------
    try:
        d = active[index]
        print ("active.metric_data =",active.metric_data)
    except Exception as error:
        # Something went wrong with the active storage operations =>
        # Raise an ActiveStorageError that will trigger (inside
        # `actify`) a local reduction to be carried out instead.
        if debug:
            print(
#            logger.debug(
                f"FAILED in active storage reduction {details} ({error}): "
                f"{round(time.time() - start, 6):.6f}s "
                "=> reverting to local computation"
            )  # prgama: no cover

        raise
        raise ActiveStorageError()    
    else:
        if debug:
            print(
#            logger.debug(
                f"FINISHED active storage reduction {details}: "
                f"{round(time.time() - start, 6):.6f}s"
            )  # prgama: no cover
    # ----------------------------------------------------------------
    # Active storage reduction was a success. Reformat the resulting
    # components dictionary to match the output of the corresponding
    # local chunk function (e.g. `cf_mean_chunk`).
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

    When a ``cf_*_chunk`` method is decorated, its computations will
    be attempted in active storage. If that is not possible (due to
    configuration settings, limitations on the type of reduction that
    can be done in active storage, or the active storage reduction
    failed) then the computations will be done locally "as usual".

    .. versionadded:: NEXTVERSION

    .. seealso:: `active_chunk_function`

    :Parameters:

        method: `str`
            The name of the reduction method.

    """

    def decorator(chunk_function):
        @wraps(chunk_function)
        def wrapper(*args, **kwargs):
            try:
                # Try doing an active storage reduction
                out = active_chunk_function(method, *args, **kwargs)
            except ActiveStorageError:
                # The active storage reduction failed
                pass
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
