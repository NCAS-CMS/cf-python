from os       import close
from tempfile import mkstemp
from tempfile import mkdtemp

from numpy import load    as numpy_load
from numpy import ndarray as numpy_ndarray
from numpy import save    as numpy_save

from numpy.ma import array      as numpy_ma_array
from numpy.ma import is_masked  as numpy_ma_is_masked

import cfdm

from . import abstract

from ..functions import parse_indices, get_subspace
from ..constants import CONSTANTS


class CachedArray(abstract.FileArray):
    '''A indexable N-dimensional array supporting masked values.

    The array is stored on disk in a temporary file until it is
    accessed. The directory containing the temporary file may be found
    and set with the `cf.TEMPDIR` function.

    '''
    def __init__(self, array):
        '''**Initialization**

    :Parameters:

        array: numpy array
            The array to be stored on disk in a temporary file.

    **Examples:**

    >>> f = CachedArray(numpy.array([1, 2, 3, 4, 5]))
    >>> f = CachedArray(numpy.ma.array([1, 2, 3, 4, 5]))

        '''
        super().__init__()

        # ------------------------------------------------------------
        # Use mkstemp because we want to be responsible for deleting
        # the temporary file when done with it.
        # ------------------------------------------------------------
        _partition_dir = mkdtemp(
            prefix='cf_cachedarray_', dir=CONSTANTS['TEMPDIR'])
        fd, _partition_file = mkstemp(prefix='cf_cachedarray_', suffix='.npy',
                                      dir=_partition_dir)
        close(fd)

        # The name of the temporary file storing the array
        self._set_component('_partition_dir', _partition_dir)
        self._set_component('_partition_file', _partition_file)

        # Numpy data type of the array
        self._set_component('dtype', array.dtype)

        # Tuple of the array's dimension sizes
        self._set_component('shape', array.shape)

        # Number of elements in the array
        self._set_component('size', array.size)

        # Number of dimensions in the array
        self._set_component('ndim', array.ndim)

        if numpy_ma_is_masked(array):
            # Array is a masked array. Save it as record array with
            # 'data' and 'mask' elements because this seems much
            # faster than using numpy.ma.dump.
            self._set_component('_masked_as_record', True)
            numpy_save(_partition_file, array.toflex())
        else:
            self._set_component('_masked_as_record', False)
            if hasattr(array, 'mask'):
                # Array is a masked array with no masked elements
                numpy_save(_partition_file, array.view(numpy_ndarray))
            else:
                # Array is not a masked array.
                numpy_save(_partition_file, array)

    def __getitem__(self, indices):
        '''x.__getitem__(indices) <==> x[indices]

    Returns a numpy array.

        '''
        array = numpy_load(self._partition_file)

        indices = parse_indices(array.shape, indices)

        array = get_subspace(array, indices)

        if self._get_component('_masked_as_record'):
            # Convert a record array to a masked array
            array = numpy_ma_array(array['_data'], mask=array['_mask'],
                                   copy=False)
            array.shrink_mask()

        # Return the numpy array
        return array

    def __str__(self):
        '''x.__str__() <==> str(x)

        '''
        return '%s in %s' % (self.shape, self._partition_file)

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def _partition_dir(self):
        '''TODO

        '''
        return self._get_component('_partition_dir')

    @property
    def _partition_file(self):
        '''TODO

        '''
        return self._get_component('_partition_file')


# --- End: class
