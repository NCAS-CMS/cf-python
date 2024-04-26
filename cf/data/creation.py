"""Functions used during the creation of `Data` objects."""

from functools import lru_cache

import dask.array as da
import numpy as np
from dask.base import is_dask_collection


def to_dask(array, chunks, **from_array_options):
    """Create a `dask` array.

    .. versionadded:: 3.14.0

    :Parameters:

        array: array_like
            The array to be converted to a `dask` array. Examples of
            valid types include anything with a `to_dask_array`
            method, `numpy` arrays, `dask` arrays, `xarray` arrays,
            `cf.Array` subclasses, `list`, `tuple`, scalars.

        chunks: `int`, `tuple`, `dict` or `str`, optional
            Specify the chunking of the returned dask array.  Any
            value accepted by the *chunks* parameter of the
            `dask.array.from_array` function is allowed.

            Ignored if *array* is a `dask` array, which already
            defines its own chunks.

            Might get automatically modified if *array* is a
            compressed `Array` subclass.

        from_array_options: `dict`, optional
            Keyword arguments to be passed to `dask.array.from_array`.

            If *from_array_options* has no ``'lock'`` key then the
            `lock` keyword is set to the `_lock` attribute of *array*
            or, if there is no such attribute, `False`.

            If *from_array_options* has no ``'meta'`` key then the
            `meta` keyword is set to the `_dask_meta` attribute of
            *array* or, if there is no such attribute, `None`.

    :Returns:

        `dask.array.Array`
            The `dask` array representation of the array.

    **Examples**

    >>> cf.data.creation.to_dask([1, 2, 3], 'auto')
    dask.array<array, shape=(3,), dtype=int64, chunksize=(3,), chunktype=numpy.ndarray>
    >>> cf.data.creation.to_dask([1, 2, 3], chunks=2)
    dask.array<array, shape=(3,), dtype=int64, chunksize=(2,), chunktype=numpy.ndarray>
    >>> cf.data.creation.to_dask([1, 2, 3], chunks=2, {'asarray': True})
    dask.array<array, shape=(3,), dtype=int64, chunksize=(2,), chunktype=numpy.ndarray>
    >>> cf.data.creation.to_dask(cf.dt(2000, 1, 1), 'auto')
    dask.array<array, shape=(), dtype=object, chunksize=(), chunktype=numpy.ndarray>
    >>> cf.data.creation.to_dask([cf.dt(2000, 1, 1)], 'auto')
    dask.array<array, shape=(1,), dtype=object, chunksize=(1,), chunktype=numpy.ndarray>

    """
    if is_dask_collection(array):
        return array

    if hasattr(array, "to_dask_array"):
        try:
            return array.to_dask_array(chunks=chunks)
        except TypeError:
            return array.to_dask_array()

    if type(array).__module__.split(".")[0] == "xarray":
        data = getattr(array, "data", None)
        if data is not None:
            return da.asanyarray(data)

    if not isinstance(
        array, (np.ndarray, list, tuple, memoryview) + np.ScalarType
    ) and not hasattr(array, "shape"):
        # 'array' is not of a type that `da.from_array` can cope with,
        # so convert it to a numpy array.
        array = np.asanyarray(array)

    kwargs = from_array_options
    kwargs.setdefault("lock", getattr(array, "_lock", False))
    kwargs.setdefault("meta", getattr(array, "_dask_meta", None))

    try:
        return da.from_array(array, chunks=chunks, **kwargs)
    except NotImplementedError:
        # Try again with 'chunks=-1', in case the failure was due to
        # not being able to use auto rechunking with object dtype.
        return da.from_array(array, chunks=-1, **kwargs)


@lru_cache(maxsize=32)
def generate_axis_identifiers(n):
    """Return new axis identifiers for a given number of axes.

    The names are arbitrary and have no semantic meaning.

    .. versionadded:: 3.14.0

    :Parameters:

        n: `int`
            Generate this number of axis identifiers.

    :Returns:

        `list`
            The new axis identifiers.

    **Examples**

    >>> cf.data.creation.generate_axis_identifiers(0)
    []
    >>> cf.data.creation.generate_axis_identifiers(1)
    ['dim0']
    >>> cf.data.creation.generate_axis_identifiers(3)
    ['dim0', 'dim1', 'dim2']

    """
    return [f"dim{i}" for i in range(n)]
