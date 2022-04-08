"""Functions used during the creation of `Data` objects."""
from functools import lru_cache, partial

import dask.array as da
import numpy as np
from dask import config
from dask.array.core import getter, normalize_chunks
from dask.base import tokenize
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


def to_dask(array, chunks, **from_array_options):
    """TODODASK.

    .. versionadded:: TODODASK

    :Parameters:

        array: array_like

        chunks: `int`, `tuple`, `dict` or `str`, optional
            Specify the chunking of the returned dask array.

            Any value accepted by the *chunks* parameter of the
            `dask.array.from_array` function is allowed.

        dask_from_array_options: `dict`
            Keyword arguments to be passed to `dask.array.from_array`.

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
    kwargs = from_array_options
    kwargs.setdefault("asarray", getattr(array, "dask_asarray", None))
    kwargs.setdefault("lock", getattr(array, "dask_lock", False))

    return da.from_array(array, chunks=chunks, **kwargs)


def compressed_to_dask(array, chunks):
    """Create a dask array with `Subarray` chunks.

    .. versionadded:: TODODASK

    :Parameters:

        array: subclass of `Array`
            The compressed array.

        chunks: `int`, `tuple`, `dict` or `str`, optional
            Specify the chunking of the returned dask array.

            Any value accepted by the *chunks* parameter of the
            `dask.array.from_array` function is allowed.

            The chunk sizes implied by *chunks* for a dimension that
            has been compressed are ignored and replaced with values
            that are implied by the decompression algorithm, so their
            specification is arbitrary.

    :Returns:

        `dask.array.Array`

    """
    # Initialise a dask graph for the uncompressed array
    name = (array.__class__.__name__ + "-" + tokenize(array),)
    dsk = {}
    full_slice = Ellipsis

    # Create a context manager that is used to ensure that all data
    # accessed from within a `Subarray` instance is done so
    # synchronously, thereby avoiding any "compute within a compute"
    # thread proliferation.
    context = partial(config.set, scheduler="synchronous")

    compressed_dimensions = array.compressed_dimensions()
    conformed_data = array.conformed_data()
    compressed_data = conformed_data["data"]

    # ----------------------------------------------------------------
    # Set the chunk sizes for the dask array.
    #
    # Note: For a dimensions that have been compressed, the chunk
    #       sizes implied by the input *chunks* parameter are
    #       overwritten with those created by `array.subarray_shapes`.
    #
    #       For subsampled arrays, the compressed dimension chunks
    #       created by `array.subarray_shapes` will be incorrect and
    #       must be corrected later.
    # ----------------------------------------------------------------
    uncompressed_dtype = array.dtype
    chunks = normalize_chunks(
        array.subarray_shapes(chunks),
        shape=array.shape,
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
        # ------------------------------------------------------------

        # Re-initialise the chunks
        u_dims = list(compressed_dimensions)
        chunks = [[] if i in u_dims else c for i, c in enumerate(chunks)]

        # For each dimension, initialise the index of the chunk
        # previously created (prior to the chunk currently being
        # created). The value -1 is an arbitrary negative value that is
        # always less than any chunk index, which is always a natural
        # number.
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

            # Add correct chunk sizes for compressed dimensions
            for d in u_dims[:]:
                previous = previous_chunk_location[d]
                new = chunk_location[d]
                if new > previous:
                    chunks[d].append(u_shape[d])
                    previous_chunk_location[d] = new
                elif new < previous:
                    # No more chunk sizes required for this compressed
                    # dimension
                    u_dims.remove(d)

        chunks = [tuple(c) for c in chunks]

    else:
        raise ValueError(
            f"Can't initialise 'Data' from compressed {array!r} with "
            f"unknown compression type {compression_type!r}"
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

    **Examples**

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
