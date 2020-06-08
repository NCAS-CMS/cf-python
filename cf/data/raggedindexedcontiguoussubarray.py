import logging

import numpy

from ..functions import parse_indices, get_subspace

from . import abstract


logger = logging.getLogger(__name__)


class RaggedIndexedContiguousSubarray(abstract.CompressedSubarray):
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
        uarray = numpy.ma.masked_all(self.shape, dtype=array.dtype)

        r_indices = [slice(None)] * array.ndim
        p_indices = [slice(None)] * uarray.ndim

        compression = self.compression

        instance_axis = compression['instance_axis']
        instance_index = compression['instance_index']
        i_element_axis = compression['i_element_axis']
        i_element_index = compression['i_element_index']
        c_element_axis = compression['c_element_axis']
        c_element_indices = compression['c_element_indices']

        p_indices[instance_axis] = instance_index
        p_indices[i_element_axis] = i_element_index
        p_indices[c_element_axis] = slice(
            0, c_element_indices.stop - c_element_indices.start)

        uarray[tuple(p_indices)] = array[c_element_indices]

        if indices is Ellipsis:
            return uarray
        else:
            logger.debug('indices = {}'.format(indices))

            indices = parse_indices(self.shape, indices)
            logger.debug(
                'parse_indices(self.shape, indices) = {}'.format(indices))

            return get_subspace(uarray, indices)


#    def __repr__(self):
#        '''x.__repr__() <==> repr(x)
#
#        '''
#        return "<CF %s: %s>" % (self.__class__.__name__, str(self.array))


# --- End: class
