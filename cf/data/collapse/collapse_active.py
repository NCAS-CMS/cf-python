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
    is_log_level_info,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Specify which reduction methods are possible with active storage
# --------------------------------------------------------------------
active_reduction_methods = ("max", "mean", "min", "sum")


class ActiveStorageError(Exception):
    pass


def actify(method):
    """Decorator for active storage reductions on chunks.

    Intended to decorate the ``cf_*_chunk`` methods in
    `cf.data.collapse.dask_collapse`.

    When a ``cf_*_chunk`` method is decorated, its computations will
    be attempted in active storage. If that is not possible (due to
    configuration settings, limitations on the type of reduction that
    can be done in active storage, or the active storage reduction
    failed) then the computations will be done locally "as usual".

    .. versionadded:: 3.16.3

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
            except ActiveStorageError as error:
                # The active storage reduction failed
                logger.warning(
                    f"{error} => reverting to local computation"
                )  # pragma: no cover
            else:
                if out is not None:
                    # The active storage reduction succeeded
                    return out

            # Still here? Then using active storage was not
            # appropriate, or else doing the active storage operation
            # failed => do a local computation.
            return chunk_function(*args, **kwargs)

        return wrapper

    return decorator


def active_chunk_function(method, *args, **kwargs):
    """Collapse data in a chunk with active storage.

    Called by the `actify` decorator function.

    If an active storage reduction is not appropriate then `None` is
    returned.

    If the active storage operation fails then an ActiveStorageError
    is raised.

    If the active storage operation is successful then a dictionary of
    reduction components, similar to that returned by a ``cf_*_chunk``
    method, is returned.

    .. versionadded:: 3.16.3

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
            The reduced data in component form, or `None` if an active
            storage reduction is not appropriate.

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

    # Dask reduction machinery
    if kwargs.get("computing_meta"):
        return x

    # ----------------------------------------------------------------
    # Return None if active storage reduction is not appropriate.
    # Inside `actify`, this will trigger a local reduction to be
    # carried out instead.
    # ----------------------------------------------------------------
    if not active_storage():
        # Active storage is turned off
        return

    url = kwargs.get("active_storage_url")
    if url is None:
        url = active_storage_url().value
        if url is None:
            # Active storage is not possible when no active storage
            # server URL has been provided
            return

    if method not in active_reduction_methods:
        # Active storage is not (yet) available for this method
        return

    if not getattr(x, "active_storage", False):
        # The data object 'x' is incompatible with active storage
        # operations. E.g. it is a UMArray object, a numpy array, etc.
        return

    if len(args) == 2:
        # Weights, if present, are always passed in as a positional
        # parameter, never as a keyword parameter (see
        # `dask.array.reductions.reduction` for details).
        weights = args[1]
        if weights is not None:
            # Active storage is not (yet) allowed for weighted
            # reductions
            return

    axis = kwargs.get("axis")
    if axis is not None:
        if isinstance(axis, Integral):
            axis = (axis,)

        if len(axis) < x.ndim:
            # Active storage is not (yet) allowed for reductions over
            # a subset of the axes
            return

    # ----------------------------------------------------------------
    # Still here? Set up an Active instance that will carry out the
    # active storage operation. If the operation fails, for any
    # reason, then this will trigger (inside `actify`) a local
    # reduction being carried out instead.
    # ----------------------------------------------------------------
    filename = x.get_filename()
    address = x.get_address()
    max_requests = active_storage_max_requests()
    active_kwargs = {
        "uri": "/".join(filename.split("/")[3:]),
        "ncvar": address,
        "storage_options": x.get_storage_options(),
        "active_storage_url": url,
        "storage_type": "s3",
        "max_threads": max_requests,
    }
    # WARNING: The "uri", "storage_options", and "storage_type" keys
    #          of the `active_kwargs` dictionary are currently
    #          formatted according to the whims of the `Active` class
    #          (i.e. the pyfive branch of PyActiveStorage). Future
    #          versions of `Active` will have a better API, that will
    #          require improvements to `active_kwargs`.

    index = x.index()

    details = (
        f"{method!r} (file={filename}, address={address}, url={url}, "
        f"Dask chunk={index})"
    )

    info = is_log_level_info(logger)
    if info:
        # Do some detailed logging
        start = time.time()
        logger.info(
            f"STARTED  active storage {details}: {datetime.datetime.now()}"
        )  # pragma: no cover

    active = Active(**active_kwargs)
    active.method = method
    active.components = True

    # Instruct the `Active` class to attempt an active storage
    # reduction on the remote server
    #
    # WARNING: The `_version` API of `Active` is likely to change from
    #          the current version (i.e. the pyfive branch of
    #          PyActiveStorage)
    active._version = 2

    # ----------------------------------------------------------------
    # Execute the active storage operation by indexing the Active
    # instance
    # ----------------------------------------------------------------
    try:
        d = active[index]
    except Exception as error:
        # Something went wrong with the active storage operations =>
        # Raise an ActiveStorageError that will in turn trigger
        # (inside `actify`) a local reduction to be carried out
        # instead.
        raise ActiveStorageError(
            f"FAILED in active storage {details} ({error}))"
        )
    else:
        # Active storage reduction was successful
        if info:
            # Do some detailed logging
            try:
                md = active.metric_data
            except AttributeError:
                logger.info(
                    f"FINISHED active storage {details}: "
                    f"{time.time() - start:6.2f}s"
                )  # pragma: no cover
            else:
                logger.info(
                    f"FINISHED active storage {details}: "
                    f"dataset chunks: {md['dataset chunks']}, "
                    f"load nc (s): {md['load nc time']:6.2f}, "
                    f"indexing (s): {md['indexing time (s)']:6.2f}, "
                    f"reduction (s): {md['reduction time (s)']:6.2f}, "
                    f"selection 2 (s): {md['selection 2 time (s)']:6.2f}, "
                    f"Total: {(time.time() - start):6.2f}s"
                )  # pragma: no cover

    # ----------------------------------------------------------------
    # Active storage reduction was a success. Reformat the resulting
    # components dictionary 'd' to match the output of the
    # corresponding local chunk function (e.g. `cf_mean_chunk`).
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
