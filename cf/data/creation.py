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

    .. versionadded:: TODODASK

    :Parameters:

        array: array_like

        chunks: `int`, `tuple`, `dict` or `str`, optional
            Specify the chunking of the returned dask array. See
            `cf.Data.__init__` for details.

        dask_from_array_options: `dict`
            Keyword arguments to pass to `dask.array.from_array`.

    :Returns:

        `dask.array.Array`

    **Examples**

    >>> to_dask([1, 2, 3])
    dask.array<array, shape=(3,), dtype=int64, chunksize=(3,), chunktype=numpy.ndarray>
    >>> to_dask([1, 2, 3], chunks=2)
    dask.array<array, shape=(3,), dtype=int64, chunksize=(2,), chunktype=numpy.ndarray>
    >>> to_dask([1, 2, 3], chunks=2, {'asarray': True})
    dask.array<array, shape=(3,), dtype=int64, chunksize=(2,), chunktype=numpy.ndarray>

    """
    if "chunks" in dask_from_array_options:
        raise TypeError(
            "Can't define 'chunks' in the 'dask_from_array_options' "
            "dictionary. Use the 'chunks' parameter instead."
        )

    kwargs = dask_from_array_options.copy()
    kwargs.setdefault("lock", getattr(array, "_dask_lock", False))

    return da.from_array(array, chunks=chunks, **kwargs)


def compressed_to_dask(array, chunks):
    """TODODASK Create and insert a partition matrix for a compressed
    array.

    .. versionadded:: TODODASK

    :Parameters:

        array: subclass of `CompressedArray`

        chunks: `int`, `tuple`, `dict` or `str`, optional
            Specify the chunking of the returned dask array. See
            `cf.Data.__init__` for details.

    :Returns:

        `dask.array.Array`

    """
    # Initialise a dask graph for the uncompressed array
    name = (array.__class__.__name__ + "-" + tokenize(array),)
    dsk = {}
    full_slice = Ellipsis

    # A context manager that ensures all data accessed from within a
    # `Subarray` instance is done so synchronously, thereby avoiding
    # any "compute within a compute" thread proliferation.
    context = partial(config.set, scheduler="synchronous")

    compressed_dimensions = array.compressed_dimensions()
    conformed_data = array.conformed_data()
    compressed_data = conformed_data["data"]

    # ----------------------------------------------------------------
    # Set the chunk sizes for the dask array.
    #
    # Note: The chunk sizes implied by the input 'chunks' for a
    #       compressed dimension are ignored in favour of those
    #       created by 'array.subarray_shapes'. For subsampled arrays,
    #       such chunk sizes will be incorrect and must be corrected
    #       later.
    #
    # ----------------------------------------------------------------
    uncompressed_dtype = array.dtype
    uncompressed_shape = array.shape
    if chunks != "auto":
        chunks = normalize_chunks(
            chunks, shape=uncompressed_shape, dtype=uncompressed_dtype
        )

    chunks = normalize_chunks(
        array.subarray_shapes(chunks),
        shape=uncompressed_shape,
        dtype=uncompressed_dtype,
    )

    # Get the (cfdm) subarray class
    Subarray = array.get_Subarray()

    compression_type = array.get_compression_type()
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
                False,
            )

    elif compression_type == "subsampled":
        # ------------------------------------------------------------
        # Subsampled
        #
        # Note: The chunks created above are incorrect for the
        #       compressed dimensions, since these chunk sizes are a
        #       function of the tie point indices which haven't yet
        #       been accessed. Therefore, the chunks for the
        #       compressed dimensons must be redefined here.
        #
        # ------------------------------------------------------------

        # Re-initialise the chunks
        dims = list(compressed_dimensions)
        chunks = [[] if i in dims else c for i, c in enumerate(chunks)]
        previous_chunk_location = [-1] * len(chunks)

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
                data=compressed_data,
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
                False,
            )

            # Add correct chunk sizes
            for d in dims[:]:
                previous = previous_chunk_location[d]
                new = chunk_location[d]
                if new > previous:
                    chunks[d].append(u_shape[d])
                    previous_chunk_location[d] = new
                elif new < previous:
                    # No more chunk sizes required for this dimension
                    dims.remove(d)

        chunks = [tuple(c) for c in chunks]

    else:
        raise ValueError(
            f"Can't instantiate 'Data' from {array!r} with unknown "
            f"compression type {compression_type!r}"
        )

    # Return the dask array
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
