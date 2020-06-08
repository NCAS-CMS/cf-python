import logging

from os import close
from operator import mul

import numpy

from ..functions import parse_indices, get_subspace

from . import abstract


logger = logging.getLogger(__name__)


class RaggedIndexedSubarray(abstract.CompressedSubarray):
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
        element_axis = compression['i_element_axis']
        sample_indices = compression['i_element_indices']

        p_indices[instance_axis] = instance_index
        p_indices[element_axis] = slice(0, len(sample_indices))

        uarray[tuple(p_indices)] = array[sample_indices]

        logger.debug('instance_axis    = {}'.format(instance_axis))
        logger.debug('instance_index   = {}'.format(instance_index))
        logger.debug('element_axis     = {}'.format(element_axis))
        logger.debug('sample_indices   = {}'.format(sample_indices))
        logger.debug('p_indices        = {}'.format(p_indices))
        logger.debug('uarray.shape     = {}'.format(uarray.shape))
        logger.debug('self.array.shape = {}'.format(array.shape))

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
