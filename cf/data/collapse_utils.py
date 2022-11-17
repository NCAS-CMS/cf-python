from functools import wraps
from numbers import Integral

import dask.array as da
from dask.array.utils import validate_axis
from dask.base import collections_to_dsk


def double_precision_dtype(a, default=None, bool_type="i"):
    """Returns the corresponding double precision data type of an array.

    .. versionadded:: TODODASKVER

    :Parameters:

        a: `dask.array.Array` or `None`
            The data. If `None` then the value of *default* is
            returned*.

        default: `str`, optional
            If *a* is `None`, then return this data type.

        bool_type: `str`, optional
            The corresponding double data type kind for Boolean
            data. Defaults to ``'i'``, meaning ``'i8'`` is
            returned. Set to ``'f'` to return ``'f8'`` instead.

    :Returns:

        `str`
            The double precision type.

    **Examples**

    >>> for d in (int, 'int32', float, 'float32', bool):
    ...     print(double_precision_dtype(np.array(1, dtype=d)))
    ...
    i8
    i8
    f8
    f8
    i8

    >>> double_precision_dtype(np.array(1, dtype=bool), bool_type='f')
    'f8'
    >>> double_precision_dtype(None, default="i8")
    'i8'

    """
    if a is None:
        return default

    kind = a.dtype.kind
    if kind == "b":
        return bool_type + "8"

    if kind in "fi":
        return kind + "8"

    raise TypeError(f"Can't collapse data with {a.dtype!r}")


def check_input_dtype(a, allowed="fib"):
    """Check that data has a data type allowed by a collapse method.

    The collapse method is assumed to be defined by the calling
    function.

    .. versionadded:: TODODASKVER

    :Parameters:

        a: `dask.array.Array`
            The data.

        allowed: `str`, optional
            The data type kinds allowed by the collapse
            method. Defaults to ``'fib'``, meaning that only float,
            integer and Boolean data types are allowed.

    :Returns:

        `None`

    """
    if a.dtype.kind not in allowed:
        from inspect import currentframe

        method = currentframe().f_back.f_code.co_name
        raise TypeError(f"Can't calculate {method} of data with {a.dtype!r}")


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
    # so that the data defintions come out first, allowing for a fast
    # short circuit in the common case when using active storage is no
    # feasible.
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

        # Still here? Then update the dask graph in-place with the
        # actified data definition value.
        dsk[key] = value

    if len(active_chunk_functions) == 1:
        # All data definitions in the dask graph support active
        # storage reductions with the same chunk function => redefine
        # the array from the actified dask graph, and redefine the
        # reduction chunk function.
        a = da.Array(dsk, a.name, a.chunks, a.dtype, a._meta)
        chunk_function = active_chunk_functions.pop()

    # Return the data and chunk function. These will either be
    # identical to the inputs or, if it has been determinded that
    # active storage operation is possible, then these the data and
    # chunk function will have been replaced with actified versions.
    return a, chunk_function


def active_storage(method):
    """A decorator for `Collapse` methods that enables active storage
    operations, when the conditions are right."""

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
