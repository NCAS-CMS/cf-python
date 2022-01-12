"""Functions used during the creation of `Data` objects."""
from functools import lru_cache, partial
from uuid import uuid4

import dask.array as da
import numpy as np
from dask.array.core import getter, normalize_chunks, slices_from_chunks
from dask.base import tokenize
from dask import config
from dask.utils import SerializableLock

# from cfdm import GatheredSubarray, RaggedSubarray

from ..units import Units

# from .utils import chunk_positions


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

    **Examples:**

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


def to_dask(array, chunks, dask_from_array_options):
    """TODODASK.

    .. versionadded:: 4.0.0

    """
    if "chunks" in dask_from_array_options:
        raise TypeError(
            "Can't define chunks in the 'dask_from_array_options' "
            "dictionary. Use the 'chunks' parameter instead"
        )

    kwargs = dask_from_array_options.copy()
    kwargs.setdefault("asarray", getattr(array, "dask_asarray", None))
    kwargs.setdefault("lock", getattr(array, "dask_lock", False))

    return da.from_array(array, chunks=chunks, **kwargs)


def compressed_to_dask(array, chunks):
    """TODODASK Create and insert a partition matrix for a compressed
    array.

    .. versionadded:: TODODASK

    :Parameters:

        array: subclass of `CompressedArray`

    :Returns:

        `dask.array.Array`

    """
    compression_type = array.get_compression_type()

    uncompressed_dtype = array.dtype
    uncompressed_shape = array.shape

    # Initialise a dask graph for the uncompressed array, and some
    # dask.array.core.getter arguments
    name = (array.__class__.__name__ + "-" + tokenize(array),)
    dsk = {}
    full_slice = Ellipsis

    context = partial(config.set, scheduler="synchronous")

    compressed_dimensions = array.compressed_dimensions()
    conformed_data = array.conformed_data()
    compressed_data = conformed_data["data"]

    # ----------------------------------------------------------------
    # Set the chunk sizes for the dask array
    # ----------------------------------------------------------------
    print(chunks)
    if chunks != "auto":
        chunks = normalize_chunks(
            chunks, shape=uncompressed_shape, dtype=uncompressed_dtype
        )

    chunks = normalize_chunks(
        array.subarray_shapes(chunks),
        shape=uncompressed_shape,
        dtype=uncompressed_dtype,
    )
    print(chunks)

    Subarray = array.get_Subarray()

    if compression_type.startswith("ragged"):
        # ------------------------------------------------------------
        # Ragged
        # ------------------------------------------------------------
        for u_indices, u_shape, c_indices, chunk_location in zip(
            *array.subarrays(shapes=chunks)
        ):
            subarray = Subarray(
                data=compressed_data,
                indices=c_indices,
                shape=u_shape,
                compressed_dimensions=compressed_dimensions,
                context_manager=context,
            )

            dsk[name + chunk_location] = (
                getter,
                subarray,
                full_slice,
                False,
                False,
            )

    elif compression_type == "gathered":
        # ------------------------------------------------------------
        # Gathered
        # ------------------------------------------------------------
        uncompressed_indices = conformed_data["uncompressed_indices"]

        for u_indices, u_shape, c_indices, chunk_location in zip(
            *array.subarrays(shapes=chunks)
        ):
            subarray = Subarray(
                data=compressed_data,
                indices=c_indices,
                shape=u_shape,
                compressed_dimensions=compressed_dimensions,
                uncompressed_indices=uncompressed_indices,
                context_manager=context,
            )

            dsk[name + chunk_location] = (
                getter,
                subarray,
                full_slice,
                False,
                lock,
            )

    elif compression_type == "subsampled":
        # ------------------------------------------------------------
        # Subsampled
        # ------------------------------------------------------------
        parameters = conformed_data["parameters"]
        dependent_tie_points = conformed_data["dependent_tie_points"]

        for (
            u_indices,
            u_shape,
            c_indices,
            subarea_indices,
            first,
            chunk_location,
        ) in zip(*array.subarrays(shapes=chunks)):
            subarray = Subarray(
                data=tie_points,
                indices=c_indices,
                shape=u_shape,
                compressed_dimensions=compressed_dimensions,
                first=first,
                subarea_indices=subarea_indices,
                parameters=parameters,
                dependent_tie_points=dependent_tie_points,
                context_manager=context,
            )

            dsk[name + chunk_location] = (
                getter,
                subarray,
                full_slice,
                False,
                lock,
            )

    else:
        raise ValueError("TODO 12345")

    return da.Array(dsk, name[0], chunks=chunks, dtype=uncompressed_dtype)


@lru_cache(maxsize=32)
def generate_axis_identifiers(n):
    """Return new, unique axis identifiers for a given number of axes.

    The names are arbitrary and have no semantic meaning.

    .. versionadded:: TODODASK

    :Parameters:

        n: `int`
            Generate this number of axis identifiers.

    :Returns:

        `list`
            The new axis idenfifiers.

    **Examples:**

    >>> generate_axis_identifiers(0)
    []
    >>> generate_axis_identifiers(1)
    ['dim0']
    >>> generate_axis_identifiers(3)
    ['dim0', 'dim1', 'dim2']

    """
    return [f"dim{i}" for i in range(n)]


def threads():
    """Return True if the threaded scheduler executes computations.

    See https://docs.dask.org/en/latest/scheduling.html for details.

    .. versionadded:: TODODASK

    """
    return config.get("scheduler", default=None) in (None, "threads")


def processes():
    """Return True if the multiprocessing scheduler executes
    computations.

    See https://docs.dask.org/en/latest/scheduling.html for details.

    .. versionadded:: TODODASK

    """
    return config.get("scheduler", default=None) == "processes"


def synchronous():
    """Return True if the single-threaded synchronous scheduler executes
    computations computations in the local thread with no parallelism at
    all.

    See https://docs.dask.org/en/latest/scheduling.html for details.

    .. versionadded:: TODODASK

    """
    return config.get("scheduler", default=None) == "synchronous"


def get_lock():
    """TODODASK.

    See https://docs.dask.org/en/latest/scheduling.html for details.

    .. versionadded:: TODODASK

    """
    if threads():
        return SerializableLock()

    if synchronous():
        return False

    if processes():
        raise ValueError("TODODASK - not yet sorted out processes lock")
        # Do we even need one? Can't we have lock=False, here?

    raise ValueError("TODODASK - what now? raise exception? cluster?")
