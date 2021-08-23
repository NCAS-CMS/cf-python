import numpy as np

from ..functions import get_subspace, parse_indices
from . import abstract


class RaggedIndexedSubarray(abstract.CompressedSubarray):
    """An underlying indexed ragged sub-array."""

    def __getitem__(self, indices):
        """x.__getitem__(indices) <==> x[indices]

        Returns a numpy array.

        """
        # The compressed array
        array = self.array

        # Initialize the full, uncompressed output array with missing
        # data everywhere
        uarray = np.ma.masked_all(self.shape, dtype=array.dtype)

        p_indices = [slice(None)] * uarray.ndim

        compression = self.compression

        instance_axis = compression["instance_axis"]
        instance_index = compression["instance_index"]
        element_axis = compression["i_element_axis"]
        sample_indices = compression["i_element_indices"]

        p_indices[instance_axis] = instance_index

        if not isinstance(sample_indices, (list, np.ndarray)):
            sample_indices = np.array(sample_indices)

        p_indices[element_axis] = slice(0, len(sample_indices))

        uarray[tuple(p_indices)] = array[sample_indices, ...]

        if indices is Ellipsis:
            return uarray

        indices = parse_indices(self.shape, indices)

        return get_subspace(uarray, indices)


# --- End: class
