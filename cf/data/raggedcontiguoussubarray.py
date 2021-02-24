import numpy as np

from ..functions import parse_indices, get_subspace

from . import abstract


class RaggedContiguousSubarray(abstract.CompressedSubarray):
    """An underlying contiguous ragged subarray."""

    def __getitem__(self, indices):
        """x.__getitem__(indices) <==> x[indices]

        Returns a numpy array.

        """
        # The compressed array
        array = self.array

        # Initialize the full, uncompressed output array with missing
        # data everywhere
        uarray = np.ma_masked_all(self.shape, dtype=array.dtype)

        u_indices = [slice(None)] * uarray.ndim

        compression = self.compression
        instance_axis = compression["instance_axis"]
        instance_index = compression["instance_index"]
        element_axis = compression["c_element_axis"]
        sample_indices = compression["c_element_indices"]

        u_indices[instance_axis] = instance_index
        u_indices[element_axis] = slice(
            0, sample_indices.stop - sample_indices.start
        )

        uarray[tuple(u_indices)] = array[sample_indices, ...]

        if indices is Ellipsis:
            return uarray

        indices = parse_indices(self.shape, indices)

        return get_subspace(uarray, indices)


# --- End: class
