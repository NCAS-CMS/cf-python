from functools import reduce
from operator import mul

import numpy as np

from ..functions import parse_indices, get_subspace

from . import abstract

from .utils import is_small


class GatheredSubarray(abstract.CompressedSubarray):
    """An underlying gathered sub-array."""

    def __getitem__(self, indices):
        """x.__getitem__(indices) <==> x[indices]

        Returns a numpy array.

        """
        # The compressed dask array
        compressed_array = self.array.dask_array(copy=False)

        # Initialize the full, uncompressed output array with missing
        # data everywhere
        uarray = np.ma.masked_all(self.shape, dtype=compressed_array.dtype)

        compression = self.compression
        compressed_dimension = compression["compressed_dimension"]
        compressed_axes = compression["compressed_axes"]
        compressed_part = compression["compressed_part"]
        list_array = compression["indices"]

        if is_small(list_array):
            list_array = np.asanyarray(list_array).tolist()

        n_compressed_axes = len(compressed_axes)

        uncompressed_shape = uarray.shape

        partial_uncompressed_shapes = [
            reduce(
                mul, [uncompressed_shape[i] for i in compressed_axes[j:]], 1
            )
            for j in range(1, n_compressed_axes)
        ]

        sample_indices = list(compressed_part)

        u_indices = [slice(None)] * self.ndim

        zeros = [0] * n_compressed_axes

        for j, b in enumerate(list_array):
            sample_indices[compressed_dimension] = j

            # Note that it is important for indices a and b to be
            # integers (rather than the slices a:a+1 and b:b+1) so
            # that these dimensions are dropped from uarray[u_indices]
            u_indices[compressed_axes[0] : compressed_axes[-1] + 1] = zeros
            for i, z in zip(compressed_axes[:-1], partial_uncompressed_shapes):
                if b >= z:
                    (a, b) = divmod(b, z)
                    u_indices[i] = a
            # --- End: for

            u_indices[compressed_axes[-1]] = b

            compressed = compressed_array[tuple(sample_indices)]

            uarray[tuple(u_indices)] = compressed

        # new        u_indices = [slice(None)] * self.ndim
        # new        for i, z in zip(compressed_axes[:-1],
        # new                        partial_uncompressed_shapes):
        # new            u_indices[i] = []
        # new
        # new        u_indices[compressed_axes[-1]] = []
        # new
        # new        for j, b in enumerate(list_array):
        # new            # Note that it is important for indices a and b to be
        # new            # integers (rather than the slices a:a+1 and b:b+1) so
        # new            # that these dimensions are dropped from uarray[u_indices]
        # new            for i, z in zip(compressed_axes[:-1],
        # new                            partial_uncompressed_shapes):
        # new                if b >= z:
        # new                    (a, b) = divmod(b, z)
        # new                    u_indices[i].append(a)
        # new            # --- End: for
        # new
        # new            u_indices[compressed_axes[-1]].append(b)
        # new
        # new        uarray[tuple(u_indices)] = compressed_array[tuple(compressed_part)]

        if indices is Ellipsis:
            return uarray
        else:
            indices = parse_indices(self.shape, indices)
            return get_subspace(uarray, indices)


# --- End: class
