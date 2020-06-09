from functools import reduce
from os        import close
from operator  import mul

import numpy

from ..functions import parse_indices, get_subspace

from . import abstract


class GatheredSubarray(abstract.CompressedSubarray):
    '''TODO

    '''
    def __getitem__(self, indices):
        '''x.__getitem__(indices) <==> x[indices]

    Returns a numpy array.

        '''
        # The compressed array
        compressed_array = self.array

        # Initialize the full, uncompressed output array with missing
        # data everywhere
        uarray = numpy.ma.masked_all(self.shape, dtype=compressed_array.dtype)

        compression = self.compression
        compressed_dimension = compression['compressed_dimension']
        compressed_axes = compression['compressed_axes']
        compressed_part = compression['compressed_part']
        list_array = compression['indices']


#        print ('compressed_part=',compressed_part)
        # Initialise the uncomprssed array
        n_compressed_axes = len(compressed_axes)

        uncompressed_shape = self.shape
        partial_uncompressed_shapes = [
            reduce(mul, [uncompressed_shape[i]
                         for i in compressed_axes[j:]], 1)
            for j in range(1, n_compressed_axes)]

        sample_indices = list(compressed_part)
        u_indices = [slice(None)] * self.ndim

        full = [slice(None)] * compressed_array.ndim

        zeros = [0] * n_compressed_axes
        for j, b in enumerate(list_array):
            # print('b=', b, end=", ")
            sample_indices[compressed_dimension] = slice(j, j+1)

            # Note that it is important for indices a and b to be
            # integers (rather than the slices a:a+1 and b:b+1) so
            # that these dimensions are dropped from uarray[u_indices]
            u_indices[compressed_axes[0]:compressed_axes[-1]+1] = zeros
            for i, z in zip(compressed_axes[:-1], partial_uncompressed_shapes):
                if b >= z:
                    (a, b) = divmod(b, z)
                    u_indices[i] = a
            # --- End: for
            u_indices[compressed_axes[-1]] = b

#            print ('sample_indices=', sample_indices, compressed_array.shape,
#                   end=", ")
            compressed = compressed_array[tuple(sample_indices)].array
#            print (compressed.shape)
            sample_indices2 = full[:]
            sample_indices2[compressed_dimension] = 0
            compressed = compressed[tuple(sample_indices2)]

#            print ('u_indices=', u_indices, uarray[tuple(u_indices)].shape,
#                   compressed.shape)
            uarray[tuple(u_indices)] = compressed
        # --- End: for

        if indices is Ellipsis:
            return uarray
        else:
            indices = parse_indices(self.shape, indices)
            return get_subspace(uarray, indices)


#    def __repr__(self):
#        '''x.__repr__() <==> repr(x)
#
#        '''
#        return "<CF %s: %s>" % (self.__class__.__name__, str(self.array))


# --- End: class
