"""Functions used during the creation of `Data` objects."""
from functools import lru_cache
from uuid import uuid4

import numpy as np

import dask.array as da
from dask.array.core import (
    getter,
    normalize_chunks,
    slices_from_chunks,
)
from dask.utils import SerializableLock
from dask.base import tokenize
from dask.config import config

from ..units import Units

from .utils import (
    chunk_shapes,
    chunk_positions,
)

from . import (
    FilledArray,
    GatheredSubarray,
    RaggedContiguousSubarray,
    RaggedIndexedSubarray,
    RaggedIndexedContiguousSubarray,
)


# Cache of axis identities
_cached_axes = {}


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


def compressed_to_dask(array):
    """TODODASK Create and insert a partition matrix for a compressed
    array.

    .. versionadded:: 4.0.0

    .. seealso:: `_set_Array`, `_set_partition_matrix`, `compress`

    :Parameters:

        array: subclass of `CompressedArray`

    :Returns:

        `dask.array.Array`

    """
    compressed_data = array.source()
    compression_type = array.get_compression_type()
    compressed_axes = array.get_compressed_axes()

    dtype = array.dtype
    uncompressed_shape = array.shape
    uncompressed_ndim = array.ndim

    # Initialise a dask graph for the uncompressed array, and some
    # dask.array.core.getter arguments
    token = tokenize(uncompressed_shape, uuid4())
    name = (array.__class__.__name__ + "-" + token,)
    dsk = {}
    full_slice = Ellipsis
    default_asarray = False
    if getattr(compressed_data.source(), "dask_lock", True):
        lock = get_lock()

    if compression_type == "ragged contiguous":
        # ------------------------------------------------------------
        # Ragged contiguous
        # ------------------------------------------------------------
        asarray = getattr(
            RaggedContiguousSubarray, "dask_asarray", default_asarray
        )

        count = array.get_count().dask_array(copy=False)

        if is_small(count):
            count = count.compute()

        # Find the chunk sizes and positions of the uncompressed
        # array. Each chunk will contain the data for one instance,
        # padded with missing values if required.
        chunks = normalize_chunks(
            (1,) + (-1,) * (uncompressed_ndim - 1),
            shape=uncompressed_shape,
            dtype=dtype,
        )
        chunk_shape = chunk_shapes(chunks)
        chunk_position = chunk_positions(chunks)

        #        subarrays = []
        start = 0
        for n in count:
            end = start + int(n)
            subarray = RaggedContiguousSubarray(
                array=compressed_data,
                shape=next(chunk_shape),
                compression={
                    "instance_axis": 0,
                    "instance_index": 0,
                    "c_element_axis": 1,
                    "c_element_indices": slice(start, end),
                },
            )

            dsk[name + next(chunk_position)] = (
                getter,
                subarray,
                full_slice,
                asarray,
                lock,
            )

            start += n

    #            subarrays.append(
    #                da.from_array(subarray, chunks=-1, asarray=asarray, lock=lock)
    #            )
    #
    #       # Concatenate along the instance axis
    #       dx = da.concatenate(subarrays, axis=0)
    #       return dx

    elif compression_type == "ragged indexed":
        # ------------------------------------------------------------
        # Ragged indexed
        # ------------------------------------------------------------
        asarray = getattr(
            RaggedIndexedSubarray, "dask_asarray", default_asarray
        )

        index = array.get_index().dask_array(copy=False)

        _, inverse = da.unique(index, return_inverse=True)

        if is_very_small(index):
            inverse = inverse.compute()

        chunks = normalize_chunks(
            (1,) + (-1,) * (uncompressed_ndim - 1),
            shape=uncompressed_shape,
            dtype=dtype,
        )
        chunk_shape = chunk_shapes(chunks)
        chunk_position = chunk_positions(chunks)

        for i in da.unique(inverse).compute():
            subarray = RaggedIndexedSubarray(
                array=compressed_data,
                shape=next(chunk_shape),
                compression={
                    "instance_axis": 0,
                    "instance_index": 0,
                    "i_element_axis": 1,
                    "i_element_indices": np.where(inverse == i)[0],
                },
            )

            dsk[name + next(chunk_position)] = (
                getter,
                subarray,
                full_slice,
                asarray,
                lock,
            )

    elif compression_type == "ragged indexed contiguous":
        # ------------------------------------------------------------
        # Ragged indexed contiguous
        # ------------------------------------------------------------
        index = array.get_index().dask_array(copy=False)
        count = array.get_count().dask_array(copy=False)

        if is_small(index):
            index = index.compute()
            index_is_dask = False
        else:
            index_is_dask = True

        if is_small(count):
            count = count.compute()

        cumlative_count = count.cumsum(axis=0)

        # Find the chunk sizes and positions of the uncompressed
        # array. Each chunk will contain the data for one profile of
        # one instance, padded with missing values if required.
        chunks = normalize_chunks(
            (1, 1) + (-1,) * (uncompressed_ndim - 2),
            shape=uncompressed_shape,
            dtype=dtype,
        )
        chunk_shape = chunk_shapes(chunks)
        chunk_position = chunk_positions(chunks)

        size0, size1, size2 = uncompressed_shape[:3]

        #        subarrays = []
        for i in range(size0):
            # For all of the profiles in ths instance, find the
            # locations in the count array of the number of
            # elements in the profile
            xprofile_indices = np.where(index == i)[0]
            if index_is_dask:
                xprofile_indices.compute_chunk_sizes()
                if is_small(xprofile_indices):
                    xprofile_indices = xprofile_indices.compute()
            # --- End: if

            # Find the number of actual profiles in this instance
            n_profiles = xprofile_indices.size

            # Loop over profiles in this instance, including "missing"
            # profiles that have ll missing values when uncompressed.
            #            inner_subarrays = []
            for j in range(size1):
                if j >= n_profiles:
                    # This chunk is full of missing data
                    subarray = FilledArray(
                        shape=next(chunk_shape),
                        size=size2,
                        dtype=dtype,
                        fill_value=np.ma.masked,
                    )
                else:
                    # Find the location in the count array of the
                    # number of elements in this profile
                    profile_index = xprofile_indices[j]

                    if profile_index == 0:
                        start = 0
                    else:
                        #                        start = int(count[:profile_index].sum())
                        start = int(cumlative_count[profile_index - 1])

                    stop = start + int(count[profile_index])

                    subarray = RaggedIndexedContiguousSubarray(
                        array=compressed_data,
                        shape=next(chunk_shape),
                        compression={
                            "instance_axis": 0,
                            "instance_index": 0,
                            "i_element_axis": 1,
                            "i_element_index": 0,
                            "c_element_axis": 2,
                            "c_element_indices": slice(start, stop),
                        },
                    )

                asarray = getattr(subarray, "dask_asarray", default_asarray)

                dsk[name + next(chunk_position)] = (
                    getter,
                    subarray,
                    full_slice,
                    asarray,
                    lock,
                )

    #                inner_subarrays.append(
    #                    da.from_array(
    #                        subarray, chunks=-1, asarray=asarray, lock=lock
    #                    )
    #                )
    # --- End: for

    #            # Concatenate along the profile axis for this instance
    #            subarrays.append(da.concatenate(inner_subarrays, axis=1))
    # --- End: for

    #        # Concatenate along the instance axis
    #        dx = da.concatenate(subarrays, axis=0)
    #        return dx

    elif compression_type == "gathered":
        # ------------------------------------------------------------
        # Gathered
        # ------------------------------------------------------------
        asarray = getattr(GatheredSubarray, "dask_asarray", default_asarray)

        compressed_dimension = array.get_compressed_dimension()
        compressed_axes = array.get_compressed_axes()
        indices = array.get_list().dask_array(copy=False)

        #        if is_small(indices):
        #            indices = indices.compute()

        chunks = normalize_chunks(
            [
                -1 if i in compressed_axes else "auto"
                for i in range(uncompressed_ndim)
            ],
            shape=uncompressed_shape,
            dtype=dtype,
        )

        for chunk_slice, chunk_shape, chunk_position in zip(
            slices_from_chunks(chunks),
            chunk_shapes(chunks),
            chunk_positions(chunks),
        ):
            compressed_part = [
                cs
                for i, cs in enumerate(chunk_slice)
                if i not in compressed_axes
            ]
            compressed_part.insert(compressed_dimension, slice(None))

            subarray = GatheredSubarray(
                array=compressed_data,
                shape=chunk_shape,
                compression={
                    "compressed_dimension": compressed_dimension,
                    "compressed_axes": compressed_axes,
                    "compressed_part": compressed_part,
                    "indices": indices,
                },
            )

            dsk[name + chunk_position] = (
                getter,
                subarray,
                full_slice,
                asarray,
                lock,
            )
    # --- End: if

    return da.Array(dsk, name[0], chunks=chunks, dtype=dtype)


@lru_cache(maxsize=32)
def generate_axis_identifiers(n):
    """Return new, unique axis identifiers for a given number of axes.

    The names are arbitrary and have no semantic meaning.

    .. versionadded:: 4.0.0

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

    .. versionadded:: 4.0.0

    """
    return config.get("scheduler") in (None, "threads")


def processes():
    """Return True if the multiprocessing scheduler executes
    computations.

    See https://docs.dask.org/en/latest/scheduling.html for details.

    .. versionadded:: 4.0.0

    """
    return config.get("scheduler") == "processes"


def synchronous():
    """Return True if the single-threaded synchronous scheduler executes
    computations computations in the local thread with no parallelism at
    all.

    See https://docs.dask.org/en/latest/scheduling.html for details.

    .. versionadded:: 4.0.0

    """
    return config.get("scheduler") == "synchronous"


def get_lock():
    """TODODASK.

    See https://docs.dask.org/en/latest/scheduling.html for details.

    .. versionadded:: 4.0.0

    """
    if threads():
        return SerializableLock()

    if synchronous():
        return False

    if processes():
        raise ValueError("TODODASK - not yet sorted out processes lock")

    raise ValueError("TODODASK - what now? raise exception? cluster?")
