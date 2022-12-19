"""Functions used during the creation of `Data` objects."""
from functools import lru_cache

import dask.array as da
import numpy as np
from dask.base import is_dask_collection
from dask.utils import SerializableLock


def convert_to_builtin_type(x):
    """Convert a non-JSON-encodable object to a JSON-encodable built-in
    type.

    Possible conversions are:

    ================  =======  ================================
    Input             Output   `numpy` data-types covered
    ================  =======  ================================
    `numpy.bool_`     `bool`   bool
    `numpy.integer`   `int`    int, int8, int16, int32, int64,
                               uint8, uint16, uint32, uint64
    `numpy.floating`  `float`  float, float16, float32, float64
    ================  =======  ================================

    .. versionadded:: 4.0.0

    :Parameters:

        x:
            TODO

    :Returns:

            TODO

    **Examples**

    >>> type(_convert_to_netCDF_datatype(numpy.bool_(True)))
    bool
    >>> type(_convert_to_netCDF_datatype(numpy.array([1.0])[0]))
    double
    >>> type(_convert_to_netCDF_datatype(numpy.array([2])[0]))
    int

    """
    if isinstance(x, np.bool_):
        return bool(x)

    if isinstance(x, np.integer):
        return int(x)

    if isinstance(x, np.floating):
        return float(x)

    raise TypeError(f"{type(x)!r} object is not JSON serializable: {x!r}")


def to_dask(array, chunks, default_chunks=False, **from_array_options):
    """TODODASKDOCS.

    .. versionadded:: TODODASKVER

    :Parameters:

        array: array_like
            TODODASKDOCS.

        chunks: `int`, `tuple`, `dict` or `str`, optional
            Specify the chunking of the returned dask array.

            Any value accepted by the *chunks* parameter of the
            `dask.array.from_array` function is allowed.

        dask_from_array_options: `dict`
            Keyword arguments to be passed to `dask.array.from_array`.

    :Returns:

        `dask.array.Array`

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
        if default_chunks is not False and chunks != default_chunks:
            raise ValueError(
                "Can't define chunks for dask input arrays. Consider "
                "rechunking the dask array before initialisation, "
                "or rechunking the `Data` after initialisation."
            )

        return array

    if hasattr(array, "to_dask_array"):
        try:
            return array.to_dask_array(chunks=chunks)
        except TypeError:
            return array.to_dask_array()

    if not isinstance(
        array, (np.ndarray, list, tuple, memoryview) + np.ScalarType
    ) and not hasattr(array, "shape"):
        # 'array' is not of a type that `da.from_array` can cope with,
        # so convert it to a numpy array.
        array = np.asanyarray(array)

        
    # Set a lock if required
    lock = getattr(array, "_dask_lock", False)
    if lock is True:
        # The input array has requested a lock, but not specified what
        # it should be => so set a lock that coordinates all access to
        # this file, even across multiple dask arrays.
        try:
            filename = array.get_filename(None)
        except AttributeError:
            pass
        else:
            if filename is not None:
                lock = SerializableLock(filename)
        
    kwargs = from_array_options
    kwargs.setdefault("lock", lock)
    kwargs.setdefault("meta", getattr(array, "_dask_meta", None))

    try:
        return da.from_array(array, chunks=chunks, **kwargs)
    except NotImplementedError:
        # Try again with 'chunks=-1', in case the failure was due to
        # not being able to use auto rechunking with object dtype.
        return da.from_array(array, chunks=-1, **kwargs)

@lru_cache(maxsize=32)
def generate_axis_identifiers(n):
    """Return new, unique axis identifiers for a given number of axes.

    The names are arbitrary and have no semantic meaning.

    .. versionadded:: TODODASKVER

    :Parameters:

        n: `int`
            Generate this number of axis identifiers.

    :Returns:

        `list`
            The new axis idenfifiers.

    **Examples**

    >>> generate_axis_identifiers(0)
    []
    >>> generate_axis_identifiers(1)
    ['dim0']
    >>> generate_axis_identifiers(3)
    ['dim0', 'dim1', 'dim2']

    """
    return [f"dim{i}" for i in range(n)]
