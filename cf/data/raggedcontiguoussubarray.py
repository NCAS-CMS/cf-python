import logging


from numpy import empty as numpy_empty

from numpy.ma import masked_all as numpy_ma_masked_all

from ..functions import parse_indices, get_subspace

from . import abstract


logger = logging.getLogger(__name__)


class RaggedContiguousSubarray(abstract.CompressedSubarray):
    '''TODO

    '''
    def __getitem__(self, indices):
        '''x.__getitem__(indices) <==> x[indices]

    Returns a numpy array.

        '''
        # The compressed array
        array = self.array

        # Initialize the full, uncompressed output array with missing
        # data everywhere
        uarray = numpy_ma_masked_all(self.shape, dtype=array.dtype)

        r_indices = [slice(None)] * array.ndim
        p_indices = [slice(None)] * uarray.ndim

        compression = self.compression
        instance_axis = compression['instance_axis']
        instance_index = compression['instance_index']
        element_axis = compression['c_element_axis']
        sample_indices = compression['c_element_indices']

        p_indices[instance_axis] = instance_index
        p_indices[element_axis] = slice(
            0, sample_indices.stop - sample_indices.start
        )

        uarray[tuple(p_indices)] = array[sample_indices]

        logger.debug(
            f"instance_axis    = {instance_axis}\n"
            f"instance_index   = {instance_index}\n"
            f"element_axis     = {element_axis}\n"
            f"sample_indices   = {sample_indices}\n"
            f"p_indices        = {p_indices}\n"
            f"uarray.shape     = {uarray.shape}\n"
            f"self.array.shape = {array.shape}\n"
        )  # pragma: no cover

        if indices is Ellipsis:
            return uarray
        else:
            logger.debug(f"indices = {indices}")  # pragma: no cover

            indices = parse_indices(self.shape, indices)
            logger.debug(
               f"parse_indices(self.shape, indices) = {indices}"
            )  # pragma: no cover

            return get_subspace(uarray, indices)


# --- End: class
