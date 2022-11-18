from functools import wraps
from numbers import Integral

import dask.array as da
from dask.array.utils import validate_axis
from dask.base import collections_to_dsk


def actify(a, method, axis=None):
    """TODOACTIVEDOCS.

    .. versionadded:: TODOACTIVEVER

    :Parameters:

        a: `dask.array.Array`
            The array to be collapsed.

        method: `str`
            TODOACTIVEDOCS

        axis: (sequence of) `int`, optional
            TODOACTIVEDOCS

    :Returns:

        `dask.array.Array`, function
            TODOACTIVEDOCS

    """
    chunk_function = None
    #    if not active_storage:
    #        # It has been determined externally that an active storage
    #       # reduction is not possible, so return the input data and
    #       # chunk function unchanged.
    #       return a, chunk_function
    #
    #    # Still here? Then it is assumed that the dask array is of a form
    #    # which might be able to exploit active storage. In particular, it
    #    # is assumed that all data definitions point to files.

    # Parse axis
    if axis is None:
        axis = tuple(range(a.ndim))
    else:
        if isinstance(axis, Integral):
            axis = (axis,)

        if len(axis) != a.ndim:
            # Can't (yet) use active storage to collapse a subset of
            # the axes, so return the input data and chunk function
            # unchanged.
            return a, chunk_function

        axis = validate_axis(axis, a.ndim)

    active_chunk_functions = set()

    # Loop round elements of the dask graph, looking for data
    # definitions that point to a file and which support active
    # storage operations. The elements are traversed in reverse order
    # so that the data defintions come out first, allowing for a
    # faster short circuit when using active storage is not possible.
    #
    # It is assumed that teh graph doesn't have many laters - i.e. it
    # is assumed that this function is called only if has been
    # deterimined extermanlly that it is sensible to do so.

    dsk = collections_to_dsk((a,), optimize_graph=True)
    for key, value in reversed(dsk.items()):
        try:
            value.get_filename()
        except AttributeError:
            # This value is not a data definition (it is assumed that
            # all data definitions point to files).
            continue

        try:
            # Create a new actified data definition value
            value = value.actify(method, axis)
        except (AttributeError, ValueError):
            # This data definition value does not support active
            # storage reductions, or does not support the requested
            # active storage reduction defined by 'method'.
            active_chunk_functions = ()
            break

        try:
            # Get the active storage chunk function
            active_chunk_functions.add(value.get_active_chunk_function())
        except AttributeError:
            # This data definition value does not support active
            # storage reductions
            active_chunk_functions = ()
            break

        # Still here? Then update the dask graph dictionary with the
        # actified data definition value.
        dsk[key] = value

    if len(active_chunk_functions) == 1:
        # All data definitions in the dask graph support active
        # storage reductions with the same chunk function => redefine
        # the array from the actified dask graph, and define the
        # actified reduction chunk function.
        a = da.Array(dsk, a.name, a.chunks, a.dtype, a._meta)
        chunk_function = active_chunk_functions.pop()

    # Return the dask array and chunk function. The array will either
    # be identical to the input or, if it has been determined that
    # active storage operation is possible, then it will have been
    # replaced by its actified version. The chunk function will either
    # be None, or the active storage chunk function provided by the
    # data definitions in each chunk.
    return a, chunk_function


def active_storage(method):
    """A decorator for `Collapse` methods that enables active storage
    operations, when the conditions are right.

    .. versionadded:: TODOACTIVEVER

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
