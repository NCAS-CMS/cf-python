from functools import wraps


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
    from numbers import Integral
    
    import dask.array as da
    from dask.base import collections_to_dsk
    from dask.array.utils import validate_axis

    if method not in Active.methods():
        # The given method is not recognised by `Active`, so return
        # the input data unchanged.
        return a, None

    # Parse axis
    if axis is None:
        axis = tuple(range(a.ndim))
    else
        if isinstance(axis, Integral):
            axis = (axis,)

        if len(axis) != a.ndim:
            # Can't (yet) use active storage to collapse a subset of
            # the axes, so return the input data unchanged.
            return a, None

        axis = validate_axis(axis, a.ndim)

    filenames = set()
    active_chunk_functions = set()

    # Loop round elements of the dask graph, looking for data
    # definitions that point to a file and which support active
    # storage operations. The elements are traversed in reverse order
    # so that the data defintions come out first, allowing for a
    # faster short circuit when using active storage is not possible.
    #
    # It is assumed that `actify` has only been called if has been
    # deterimined externally that it is sensible to do so. This will
    # be the case if an only if the parent `Data` instance's
    # `active_storage` attribute is `True`.
    dsk = collections_to_dsk((a,), optimize_graph=True)
    for key, value in reversed(dsk.items()):
        try:
            filenames.add(value.get_filename())
        except AttributeError:
            # This value is not a data definition. Note: It is assumed
            # that all data definitions point to files.
            continue

        try:
            # Create a new actified data definition
            value = value.actify(method, axis)
        except (AttributeError, ValueError):
            # Either this data definition does not support active
            # storage reductions (AttributeError), or it does not
            # support the requested active storage reduction defined
            # by 'method' (ValueError).
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

    for filename in filenames:
        # TODOACTIVE: Check that Active(filename) supports active
        #             storage. I don't really know how this will work
        #             ...
        if not OK:
            # This file location does not support active storage, so
            # return the input data unchanged.
            return a, None

    # Still here?
    if len(active_chunk_functions) == 1:
        # All data definitions in the dask graph support active
        # storage reductions with the same chunk function => redefine
        # the array from the actified dask graph, and define the
        # active storage reduction chunk function.
        a = da.Array(dsk, a.name, a.chunks, a.dtype, a._meta)
        chunk_function = active_chunk_functions.pop()
    else:
        chunk_function = None

    return a, chunk_function


def active_storage(method):
    """A decorator for `Collapse` methods that enables active storage
    operations, when the conditions are right.

    .. versionadded:: TODOACTIVEVER

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
