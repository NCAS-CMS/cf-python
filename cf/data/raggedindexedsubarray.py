from os       import close
from operator import mul

import numpy

from ..functions import parse_indices, get_subspace

from . import abstract

_debug = False


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

        instance_axis  = compression['instance_axis']
        instance_index = compression['instance_index']
        element_axis   = compression['i_element_axis']
        sample_indices = compression['i_element_indices']
        
        p_indices[instance_axis] = instance_index
        p_indices[element_axis]  = slice(0, len(sample_indices))
        
        uarray[tuple(p_indices)] = array[sample_indices]
        
        if _debug:
            print('instance_axis    =', instance_axis ) 
            print('instance_index   =', instance_index)
            print('element_axis     =', element_axis  )
            print('sample_indices   =', sample_indices)
            print('p_indices        =', p_indices     )
            print('uarray.shape     =', uarray.shape  )
            print('self.array.shape =', array.shape   )

        if indices is Ellipsis:
            return uarray
        else:
            if _debug:
                print('indices =', indices)

            indices = parse_indices(self.shape, indices)
            if _debug:
                print('parse_indices(self.shape, indices) =', indices)
                
            return get_subspace(uarray, indices)


    def __repr__(self):
        '''x.__repr__() <==> repr(x)

        '''
        return "<CF %s: %s>" % (self.__class__.__name__, str(self.array))


#--- End: class
