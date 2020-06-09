import atexit
import logging

from copy      import deepcopy
from functools import reduce
from sys       import getrefcount
from os        import close
from os        import remove
from os        import rmdir
from os.path   import isfile
from operator  import mul
from itertools import product as itertools_product
from tempfile  import mkstemp

from numpy import array       as numpy_array
from numpy import bool_       as numpy_bool_
from numpy import dtype       as numpy_dtype
from numpy import expand_dims as numpy_expand_dims
from numpy import ndarray     as numpy_ndarray
from numpy import number      as numpy_number
from numpy import transpose   as numpy_transpose
from numpy import vectorize   as numpy_vectorize

from numpy.ma import is_masked   as numpy_ma_is_masked
from numpy.ma import isMA        as numpy_ma_isMA
from numpy.ma import masked_all  as numpy_ma_masked_all
from numpy.ma import MaskedArray as numpy_ma_MaskedArray
from numpy.ma import nomask      as numpy_ma_nomask

from numpy.ma.core import MaskedConstant as numpy_ma_core_MaskedConstant

# from cfunits import Units

from ..units     import Units
from ..functions import get_subspace, FREE_MEMORY, FM_THRESHOLD
from ..functions import inspect as cf_inspect
from ..constants import CONSTANTS

# from .filearray import  (_TempFileArray #, SharedMemoryArray,
#                          _shared_memory_array,FileArray)
from .cachedarray import CachedArray

from .abstract import FileArray


logger = logging.getLogger(__name__)

_Units_conform = Units.conform

_dtype_object = numpy_dtype(object)

# --------------------------------------------------------------------
# Dictionary of partitions' temporary files containing the full path
# of the directory containing tuples of the temporary file and its
# lock files, the full path of the lock file on this PE and a set of
# the full path of the lock files on other PEs.
#
# For example:
# >>> _temporary_files
# {'/tmp/cf_array_7hoQk5/cf_array_RmC7NJ.npy': ('/tmp/cf_array_7hoQk5',
#  '/tmp/cf_array_7hoQk5/cf_array_RmC7NJ.npy_i043rk',
#  set('/tmp/cf_array_7hoQk5/cf_array_RmC7NJ.npy_cvUcib')),
#  '/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy': ('/tmp/cf_array_CrbJVf',
#  '/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy_n4BKBF',
#  set('/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy_xQ5LVo'))}
# --------------------------------------------------------------------
_temporary_files = {}


def _lock_files_present(lock_files):
    lock_files_present = False
    for filename in lock_files:
        if isfile(filename):
            lock_files_present = True
            break
    # --- End: for

    return lock_files_present


def _remove_temporary_files(filename=None):
    '''Remove temporary partition files from disk.

    The removed files' names are deleted from the _temporary_files
    set.

    The intended use is to delete individual files as part of the
    garbage collection process and to delete all files when python
    exits.

    This is quite brutal and may break partitions if used unwisely. It
    is not recommended to be used as a general tidy-up function.

    .. seealso:: `__del__`

    :Parameters:

        filename: `str`, optional
            The name of file to remove. The file name must be in the
            _temporary_files set. By default all files named in the
            _temporary_files set are removed.

    :Returns:

        `None`

    **Examples:**

    >>> _remove_temporary_files()

    >>> _temporary_files
    {'/tmp/cf_array_7hoQk5/cf_array_RmC7NJ.npy': ('/tmp/cf_array_7hoQk5',
     '/tmp/cf_array_7hoQk5/cf_array_RmC7NJ.npy_i043rk',
     set('/tmp/cf_array_7hoQk5/cf_array_RmC7NJ.npy_cvUcib')),
     '/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy': ('/tmp/cf_array_CrbJVf',
     '/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy_n4BKBF',
     set('/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy_xQ5LVo'))}
    >>> _remove_temporary_files('/tmp/cf_array_7hoQk5/cf_array_RmC7NJ.npy')
    >>> _temporary_files
    {'/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy': ('/tmp/cf_array_CrbJVf',
     '/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy_n4BKBF',
     set('/tmp/cf_array_CrbJVf/cf_array_INZQQc.npy_xQ5LVo'))}
    >>> _remove_temporary_files()
    >>> _temporary_files
    {}

    '''
    if filename is not None:
        if filename in _temporary_files:
            # If this condition is not met then probably
            # _remove_temporary_files() has already been run at
            # exit
            dirname, _lock_file, _other_lock_files = _temporary_files[filename]
            try:
                remove(_lock_file)
            except OSError:
                pass

            # Only remove the temporary file if it is not being
            # used by any other ranks
            if not _lock_files_present(_other_lock_files):
                # Remove the given temporary file
                try:
                    remove(filename)
                    rmdir(dirname)
                except OSError:
                    pass
                del _temporary_files[filename]
        # --- End: if

        return

    # Still here? Then remove all temporary files and lock files
    for filename in _temporary_files:
        try:
            remove(filename)
        except OSError:
            pass
        dirname, _lock_file, _other_lock_files = _temporary_files[filename]
        try:
            remove(_lock_file)
        except OSError:
            pass
        for lock_file in _other_lock_files:
            try:
                remove(lock_file)
            except OSError:
                pass
        # --- End: for

        try:
            rmdir(dirname)
        except OSError:
            pass
    # --- End: for

    _temporary_files.clear()


# --------------------------------------------------------------------
# Instruction to remove all of the temporary files from all partition
# arrays at exit.
# --------------------------------------------------------------------
atexit.register(_remove_temporary_files)

# --------------------------------------------------------------------
# Create a deep copy function for numpy arrays which contain object
# types
# --------------------------------------------------------------------
_copy = numpy_vectorize(deepcopy, otypes=[object])


class Partition:
    '''A partition of a master data array.

    The partition spans all or part of exactly one subarray of the
    master data array

    '''
    # Counters for the number of partitions pointing to each open
    # file.  A file is only closed when the count reaches zero for
    # that file.  The key is the absolute path to the file. The
    # corresponding value is the counter.
    file_counter = {}

    def __init__(self, subarray=None, flip=None, location=None,
                 shape=None, Units=None, part=None, axes=None,
                 fill=None):
        '''**Initialization**

    :Parameters:

        subarray: numpy array-like, optional
            The subarray for the partition. Must be a numpy array or
            any array storing object with a similar interface. DO NOT
            UPDATE INPLACE.

        location: `list`, optional
            The location of the partition's data array in the master
            array. DO NOT UPDATE INPLACE.

        axes: `list`, optional
            The identities of the axes of the partition's subarray. If
            the partition's subarray a scalar array then it is an
            empty list. DO NOT UPDATE INPLACE.

        part: `list`, optional
            The part of the partition's subarray which comprises its
            data array. If the partition's data array is to the whole
            subarray then *part* may be an empty list. DO NOT UPDATE
            INPLACE.

        shape: `list`, optional
            The shape of the partition's data array as a subspace of
            the master array. If the master array is a scalar array
            then *shape* is an empty list. By default the shape is
            inferred from *location*. DO NOT UPDATE INPLACE.

        Units: `Units`, optional
            The units of the partition's subarray. DO NOT UPDATE
            INPLACE.

    **Examples:**

    >>> p = Partition(subarray   = numpy.arange(20).reshape(2,5,1),
    ...               location   = [(0, 6), (1, 3), (4, 5)],
    ...               axes       = ['dim1', 'dim0', 'dim2'],
    ...               part       = [],
    ...               Units      = cf.Units('K'))

    >>> p = Partition(subarray   = numpy.arange(20).reshape(2,5,1),
    ...               location   = [(0, 6), (1, 3), (4, 5)],
    ...               axes       = ['dim1', 'dim0', 'dim2'],
    ...               shape      = [5, 2, 1],
    ...               part       = [slice(None, None, -1), [0,1,3,4],
    ...                             slice(None)],
    ...               Units      = cf.Units('K'))

    >>> p = Partition(subarray   = numpy.array(4),
    ...               location   = [(4, 5)],
    ...               axes       = ['dim1'],
    ...               part       = [],
    ...               Units      = cf.Units('K'))

        '''

        self._subarray = None

        self.axes = axes          # DO NOT UPDATE INPLACE
        self.flip = flip          # DO NOT UPDATE INPLACE
        self.part = part          # DO NOT UPDATE INPLACE
        self.location = location  # DO NOT UPDATE INPLACE
        self.shape = shape        # DO NOT UPDATE INPLACE
        self.Units = Units        # DO NOT UPDATE INPLACE
        self.subarray = subarray  # DO NOT UPDATE INPLACE
        self.fill = fill          # DO NOT UPDATE INPLACE

        if shape is None and location is not None:
            self.shape = [i[1]-i[0] for i in location]

        self._write_to_disk = None

    def __deepcopy__(self, memo):
        '''Used if copy.deepcopy is called on the variable.

        '''
        return self.copy()

    def __del__(self):
        '''Called when the partition's reference count reaches zero.

    If the partition contains a temporary file which is not referenced
    by any other partition then the temporary file is removed from
    disk.

    If the partition contains a non-temporary file which is not
    referenced by any other partition then the file is closed.

        '''
#        subarray = getattr(self, '_subarray', None)
        subarray = self._subarray

        # If the subarray is unique it will have 2 references to
        # it plus 1 within this method, making 3. If it has more
        # than 3 references to it then it is not unique.
        if getrefcount is not None:
            self._decrement_file_counter()
            if subarray is None or getrefcount(subarray) > 3:
                return
        else:
            # getrefcount has itself been deleted or is in the process
            # of being torn down
            return

        _partition_file = getattr(subarray, '_partition_file', None)
        if _partition_file is not None:
            # This partition contains a temporary file which is not
            # referenced by any other partition on this process, so if
            # there are no lock files present remove the file from
            # disk.
            _remove_temporary_files(_partition_file)

        else:
            try:
                if (FileArray is not None and isinstance(subarray, FileArray)):
                    try:
                        filename = subarray.get_filename()
                    except:
                        filename = None

                    if self.file_counter.get(filename, 999) <= 0:
                        # This partition contains a non-temporary file
                        # which is not referenced by any other
                        # partitions, so close the file.
                        subarray.close()
            except:
                # If we're here then it is likely that FileArray has been
                # torn down, so just do nothing.
                pass
        # --- End: if

#    def __getstate__(self):
#        '''
#
#    Called when pickling.
#
#    :Parameters:
#
#        None
#
#    :Returns:
#
#        `dict`
#            A dictionary of the instance's attributes
#
#    **Examples:**
#
#        '''
#        return dict([(attr, getattr(self, attr))
#                     for attr in self.__slots__ if hasattr(self, attr)])
#
#    def __setstate__(self, odict):
#        '''
#
#    Called when unpickling.
#
#    :Parameters:
#
#        odict : dict
#            The output from the instance's `__getstate__` method.
#
#    :Returns:
#
#        None
#
#        '''
#        for attr, value in odict.items():
#            setattr(self, attr, value)

    def __str__(self):
        '''x.__str__() <==> str(x)

        '''
        return '%s: %s' % (self.__class__.__name__, self.__dict__)

    def _add_to_file_counter(self, i):
        '''Add i to the count of subarrays referencing the file of this
    partition's subarray.

    Only do this if self._subarray is an instance of FileArray, but
    not a temporary FileArray.

    :Parameters:

        i: `int`

    :Returns:

        `None`

        '''
#        subarray = getattr(self, '_subarray', None)
        subarray = self._subarray

        if subarray is None:
            return

        try:
            if (isinstance(subarray, FileArray) and
                    not isinstance(subarray, CachedArray)):
                try:
                    filename = subarray.get_filename()
                except:
                    filename = None

                if filename is None:
                    return

                file_counter = self.file_counter
#                count = file_counter.get(filename, 0)
#                file_counter[filename] = count + i
#                if file_counter[filename] <= 0:
                count = file_counter.get(filename, 0) + i
                if count <= 0:
                    # Remove the file from the dictionary if its count has
                    # dropped to zero
                    file_counter.pop(filename, None)
                else:
                    file_counter[filename] = count
        except:
            # If we're here then it is likely that FileArray has been
            # torn down, so just do nothing.
            pass

    def _increment_file_counter(self):
        '''Add 1 to the Partition.file_counter if self._subarray is an
    instance of FileArray and not a temporary FileArray.

    :Returns:

        `None`

        '''
        self._add_to_file_counter(1)

    def _decrement_file_counter(self):
        '''Subtract 1 from the Partition.file_counter if self._subarray is an
    instance of FileArray and not a temporary FileArray.

    :Returns:

        `None`

        '''
        self._add_to_file_counter(-1)

    def _configure_auxiliary_mask(self, auxiliary_mask):
        '''Add the auxiliary mask to the config dictionary.

    Assumes that ``self.config`` already exists.

    :Parameters:

        auxiliary_mask: `list` of `Data`

    :Returns:

        `None`

    **:Examples:**

    >>> p._configure_auxiliary_mask([mask_component1, mask_component2])

        '''
        indices = self.indices

        new = [mask[tuple([(slice(None) if n == 1 else index)
                           for n, index in zip(mask.shape, indices)])]
               for mask in auxiliary_mask]

#        # If the partition is to be parallelised then get rid of mask
#        # components which are all False so the mask component does
#        # not get copied to the child process
#        if not config['serial']:
#            new = [mask for mask in new if not mask.any()]

        self.config['auxiliary_mask'] = new

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def indices(self):
        '''The indices of the master array which correspond to this
    partition's data array.

    :Returns:

        `tuple`
            A tuple of slice objects or, if the master data array is a
            scalar array, an empty tuple.

    **Examples:**

    >>> p.location
    [(0, 5), (2, 9)]
    >>> p.indices
    (slice(0, 5), slice(2, 9))

    >>> p.location
    [()]
    >>> p.indices
    ()

        '''
        return tuple([slice(*r) for r in self.location])

    @property
    def in_memory(self):
        '''True if and only if the partition's subarray is in memory as
    opposed to on disk.

    **Examples:**

    >>> p.in_memory
    False

        '''
        return hasattr(self._subarray, '__array_interface__')

#    @property
#    def in_shared_memory(self):
#        '''
#
#    True if and only if the partition's subarray is a shared memory array.
#
#    .. seealso:: `array`, `in_memory`, `on_disk`, `to_shared_memory`
#
#    **Examples:**
#
#    >>> p.in_shared_memory
#    True
#    >>> p.to_disk()
#    >>> p.in_shared_memory
#    False
#
#        '''
#        return isinstance(self._subarray, SharedMemoryArray)

    @property
    def in_cached_file(self):
        '''True if and only if the partition's subarray is on disk in a
    temporary file.

    .. seealso:: `array`, `in_memory`, `in_shared_memory`, `on_disk`,
                 `to_disk`

    **Examples:**

    >>> p.in_cached_file
    False

        '''
        return isinstance(self._subarray, CachedArray)

    @property
    def on_disk(self):
        '''True if and only if the partition's subarray is on disk as
    opposed to in memory.

    **Examples:**

    >>> p.on_disk
    True
    >>> p.to_memory()
    >>> p.on_disk
    False

        '''
        return isinstance(self._subarray, FileArray)

    @property
    def in_file(self):
        '''True if and only if the partition's subarray is on disk as
    opposed to in memory.

    **Examples:**

    >>> p.on_disk
    True
    >>> p.to_disk()
    >>> p.on_disk
    False

        '''
        return self.on_disk and not self.in_cached_file

    @property
    def dtype(self):
        '''The data type of the master array

        '''
        return self.config['dtype']

    @property
    def isscalar(self):
        '''True if and only if the partition's data array is a scalar array.

    **Examples:**

    >>> p.axes
    []
    >>> p.isscalar
    True

    >>> p.axes
    ['dim2']
    >>> p.isscalar
    False

        '''
        return not self.axes

    @property
    def nbytes(self):
        '''The size in bytes of the subarray.

    The size takes into account the datatype and assumes that there is
    a boolean mask, unless it can be ascertained that there isn't one.

        '''
        dtype = self.config['dtype']
        if dtype is None:
            return None

        size = reduce(mul, self.shape, 1)
        nbytes = size * dtype.itemsize

        if getattr(self, 'masked', True):
            nbytes += size

        return nbytes

    @property
    def ndim(self):
        '''TODO

        '''
        return len(self.shape)

    @property
    def size(self):
        '''Number of elements in the partition's data array (not its subarray).

    **Examples:**

    >>> p.shape
    (73, 48)
    >>> p.size
    3504

        '''
        return reduce(mul, self.shape, 1)

    @property
    def subarray(self):
        '''TODO

        '''
        return self._subarray

    @subarray.setter
    def subarray(self, value):
        self._decrement_file_counter()
        self._subarray = value
        self._increment_file_counter()
        self._in_place_changes = False

    @subarray.deleter
    def subarray(self):
        self._decrement_file_counter()
        self._subarray = None
        self._in_place_changes = True

#    @property
#    def subarray_in_external_file(self):
#        '''
#
#    True if and only if the partition's subarray is in an external file.
#
#    **Examples:**
#
#    >>> p.subarray_in_external_file
#    False
#
#        '''
#
#        return not (self.in_memory or isinstance(self.subarray, FileArray))

    def change_axis_names(self, axis_map):
        '''Change the axis names.

    The axis names are arbitrary, so mapping them to another arbitrary
    collection does not change the data array values, units, nor axis
    order.

    :Parameters:

        axis_map: `dict`

    :Returns:

        `None`

    **Examples:**

    >>> p.axes
    ['dim0', 'dim1']
    >>> p._change_axis_names({'dim0': 'dim2', 'dim1': 'dim0'})
    >>> p.axes
    ['dim2', 'dim0']

    >>> p.axes
    ['dim0', 'dim1']
    >>> p._change_axis_names({'dim0': 'dim1'})
    >>> p.axes
    ['dim1', 'dim2']

        '''
        axes = self.axes

        # Partition axes
        self.axes = [axis_map[axis] for axis in axes]

        # Flipped axes
        flip = self.flip
        if flip:
            self.flip = [axis_map[axis] for axis in flip]

    def close(self, **kwargs):
        '''Close the partition after it has been conformed.

    The partition should usually be closed after its `array` method
    has been called to prevent memory leaks.

    Closing the partition does one of the following, depending on the
    values of the partition's `!_original` attribute and on the
    *keep_in_memory* argument:

    * Nothing.

    * Stores the partition's data array in a temporary file.

    * Reverts the entire partition to a previous state.

    :Parameters:

        to_disk: `bool`, optional
            If True then revert to file pointer or write to disk
            regardless. Ignored if False

        in_memory: `bool`, optional
            If True then keep in memory, if possible,
            regardless. Ignored if False

    :Returns:

        None

    **Examples:**

    >>> p.array(...)
    >>> p.close()

        '''
        config = getattr(self, 'config', None)

        if config is None:
            return

        if kwargs:
            config.update(kwargs)

        original = getattr(self, '_original', None)
        logger.debug('Partition.close: original = {}'.format(original))

        if not original:
            originally_on_disk = False
            original_subarray = None
        else:
            originally_on_disk = not original.in_memory
            original_subarray = original._subarray

        config = self.config
        logger.debug(' config = {}'.format(config))

        if config['serial']:
            # --------------------------------------------------------
            # SERIAL
            # --------------------------------------------------------
            logger.debug('  serial')

            if config['readonly']:
                logger.debug('   readonly=True')

                if originally_on_disk:
                    logger.debug('    subarray originally on disk')

                    if config.get('to_disk', False):
                        # 1.1.1.1 The original subarray was on disk,
                        #         we don't want to keep the current
                        #         subarray in memory, and we are happy
                        #         to discard any changes that may have
                        #         been made to the subaray.
                        logger.debug('    1.1.1.1 revert')
                        self.revert()
                    elif FREE_MEMORY() <= FM_THRESHOLD():
                        # 1.1.1.2 The original subarray was on disk,
                        #         we are happy to keep the current
                        #         subarray in memory, but there is not
                        #         enough free memory to do so.
                        logger.debug('    1.1.1.2 revert ({} <= {})'.format(
                            FREE_MEMORY(), FM_THRESHOLD()))
                        self.revert()
                    else:
                        # 1.1.1.3 The original subarray was on disk
                        #         and there is enough memory to keep
                        #         the current subarray in memory
                        if (config['unique_subarray'] and
                                isinstance(original_subarray, CachedArray)):
                            # The original subarray was a temporary
                            # file which is not referenced by any
                            # other partitions
                            _remove_temporary_files(
                                original_subarray._partition_file)

                        del self.masked
                        logger.debug(
                            '    1.1.1.3 del masked ({} > {})'.format(
                                FREE_MEMORY(), FM_THRESHOLD())
                        )

                else:
                    logger.debug('   subarray originally in memory')
                    if config.get('to_disk', False):
                        # 1.1.2.1 Original subarray was in memory and
                        #         we don't want to keep the current
                        #         subarray in memory
                        logger.debug('    1.1.2.1 to_disk')
                        self.to_disk(reopen=False)
                    elif FREE_MEMORY() <= FM_THRESHOLD():
                        # 1.1.2.2 Original subarray was in memory and
                        #         unique but there is not enough
                        #         memory to keep the current subarray
                        logger.debug('    1.1.2.2 to_disk')
                        self.to_disk(reopen=False)
                    else:
                        # 1.1.2.3 Original subarray was in memory and
                        #         unique and there is enough memory to
                        #         keep the current subarray in memory
                        logger.debug('    1.1.2.3 pass')
                        pass
            else:
                # config['readonly'] is False
                if originally_on_disk:
                    if config.get('to_disk', False):
                        # 1.2.1.1 Original subarray was on disk and
                        #         there and we don't want to keep the
                        #         array
                        if (config['unique_subarray'] and
                                isinstance(original_subarray, CachedArray)):
                            # Original subarray was a temporary file
                            # on disk which is not referenced by any
                            # other partitions
                            _remove_temporary_files(
                                original_subarray._partition_file)

                        logger.debug('    1.2.1.1 to_disk')
                        self.to_disk(reopen=False)
                    elif FREE_MEMORY() <= FM_THRESHOLD():
                        # 1.2.1.2 Original subarray was on disk but
                        #         there is not enough memory to keep
                        #         it
                        if (config['unique_subarray'] and
                                isinstance(original_subarray, CachedArray)):
                            # Original subarray was a temporary file
                            # on disk which is not referenced by any
                            # other partitions
                            _remove_temporary_files(
                                original_subarray._partition_file)

                        logger.debug('    1.2.1.2 to_disk')
                        self.to_disk(reopen=False)
                    else:
                        # 1.2.1.3 Original subarray was on disk and
                        #         there is enough memory to keep it
                        logger.debug('    1.2.1.3 pass')
                        del self.masked
                        pass
                else:
                    if config.get('to_disk', False):
                        # 1.2.2.1 Original subarray was in memory but
                        #         we don't want to keep it
                        logger.debug('    1.2.2.1 to_disk')
                        self.to_disk(reopen=False)
                    elif FREE_MEMORY() <= FM_THRESHOLD():
                        # 1.2.2.2 Original subarray was an in memory
                        #         but there is not enough memory to
                        #         keep it
                        logger.debug('    1.2.2.2 to_disk')
                        self.to_disk(reopen=False)
                    else:
                        # 1.2.2.3 Original subarray was in memory and
                        #         there is enough memory to keep it
                        logger.debug('    1.2.2.3 del masked')
                        del self.masked
        else:
            logger.debug('Partition.close: parallel')
            # --------------------------------------------------------
            # PARALLEL
            # --------------------------------------------------------
            pass

#        if hasattr(self, '_original'):
#            del self._original

#        print(hasattr(self, 'config')),
        try:
            del self.config
        except AttributeError:
            pass

    def copy(self):
        '''Return a deep copy.

    ``p.copy()`` is equivalent to ``copy.deepcopy(p)``.

    :Returns:

            A deep copy.

    **Examples:**

    >>> q = p.copy()

        '''
        new = Partition.__new__(Partition)
        new.__dict__ = self.__dict__.copy()

        self._increment_file_counter()

        return new

    @property
    def array(self):
        '''Returns the partition's data array.

    After a partition has been conformed, the partition must be closed
    (with the `close` method) before another partition is conformed,
    otherwise a memory leak could occur. For example:

    >>> for partition in partition_array.flat:
    ...    # Open the partition
    ...    partition.open(config)
    ...
    ...    # Get the data array as a numpy array
    ...    array = partition.array
    ...
    ...    # < Some code to operate on the aray >
    ...
    ...    # Close the partition
    ...    partition.close()
    ...
    ...    # Now move on to conform the next partition
    ...
    >>>

        '''
        config = self.config

        unique_array = config['unique_subarray']

        p_axes = self.axes
        p_flip = self.flip
        p_part = self.part
        p_units = self.Units
        p_shape = self.shape
        p_location = self.location
        subarray = self._subarray

        len_p_axes = len(p_axes)

        if not self.in_memory:
            # --------------------------------------------------------
            # The subarray is not in memory.
            #
            # It could be in a file on disk or implied by a FileArray
            # object, etc.
            # --------------------------------------------------------
            self._original = self.copy()

            unique_array = True
            update = True
            copy = False

            if not p_part:
                indices = Ellipsis
            else:
                indices = tuple(p_part)

            # Read from a file into a numpy array
            p_data = subarray[indices]

            # We've just copied p_data from disk, so in place changes
            # are not possible
            in_place_changes = False
        else:
            # --------------------------------------------------------
            # The subarray is in memory
            # --------------------------------------------------------
            update = config['update']

            if p_part:
                p_data = get_subspace(subarray, p_part)
            elif not unique_array:
                p_data = subarray.view()
            else:
                p_data = subarray

            copy = config['extra_memory']

            # In place changes to p_data might be possible if we're not
            # copying the data
            in_place_changes = not copy

        if not p_data.ndim and isinstance(p_data, (numpy_number, numpy_bool_)):
            # --------------------------------------------------------
            # p_data is a numpy number (like numpy.int64) which does
            # not support assignment, so convert it to a numpy array.
            # --------------------------------------------------------
            p_data = numpy_array(p_data)
            # We've just copied p_data, so in place changes are
            # not possible
            copy = False
            in_place_changes = False

        masked = numpy_ma_isMA(p_data)
        if masked:
            # The p_data is a masked array
            if (p_data.mask is numpy_ma_nomask or
                    not numpy_ma_is_masked(p_data)):
                # There are no missing data points so recast as an
                # unmasked numpy array
                p_data = p_data.data
                masked = False
        # --- End: if

        if masked:
            # Set the hardness of the mask
            if config['hardmask']:
                p_data.harden_mask()
            else:
                p_data.soften_mask()
        # --- End: if

        self.masked = masked

        # ------------------------------------------------------------
        # Make sure that the data array has the correct units. This
        # process will deep copy the data array if required (e.g. if
        # another partition is referencing this numpy array), even if
        # the units are already correct.
        # ------------------------------------------------------------
        func = config.get('func')
        units = config['units']
        if func is None:
            if not p_units.equals(units) and bool(p_units) is bool(units):
                func = Units.conform

        if func is not None:
            inplace = not copy
            p_data = func(p_data, p_units, units, inplace)
            p_units = units

            if not inplace:
                # We've just copied p_data, so in place changes are
                # not possible
                copy = False
                in_place_changes = False
        # --- End: if

        flip = config.get('flip', None)
        if flip or p_flip:
            flip_axes = set(p_flip).symmetric_difference(flip)
        else:
            flip_axes = None

        axes = config['axes']

        if p_data.size > 1:
            # --------------------------------------------------------
            # Flip axes
            # --------------------------------------------------------
            if flip_axes:
                indices = [(slice(None, None, -1) if axis in flip_axes
                            else slice(None))
                           for axis in p_axes]
                p_data = p_data[tuple(indices)]

            # --------------------------------------------------------
            # Transpose axes
            # --------------------------------------------------------
            if p_axes != axes:
                iaxes = [p_axes.index(axis) for axis in axes if axis in p_axes]

                if len_p_axes > len(iaxes):
                    for i in range(len_p_axes):
                        if i not in iaxes:
                            # iaxes.append(i)
                            iaxes.insert(i, i)
                # --- End: if

                p_data = numpy_transpose(p_data, iaxes)
        # --- End: if

        # ------------------------------------------------------------
        # Remove excessive/insert missing size 1 axes
        # ------------------------------------------------------------
        if p_shape != p_data.shape:
            # if len_p_axes != len(p_shape):
            p_data = p_data.reshape(p_shape)

        # ------------------------------------------------------------
        # Apply the auxiliary mask
        # ------------------------------------------------------------
        auxiliary_mask = config['auxiliary_mask']
        if auxiliary_mask:
            for mask in auxiliary_mask:
                if mask.any():
                    if not masked:
                        p_data = p_data.view(numpy_ma_MaskedArray)
                        masked = True

                    p_data.mask = (mask | p_data.mask).array
            # --- End: for

            self.masked = True

        # ------------------------------------------------------------
        # Convert the array's data type
        # ------------------------------------------------------------
        p_dtype = p_data.dtype
        dtype = config.get('dtype', None)
        if dtype is not None and dtype != p_dtype:
            try:
                p_data = p_data.astype(dtype)  # Note: returns a copy
            except ValueError:
                raise ValueError(
                    "Can't recast partition array from {} to {}".format(
                        p_dtype.name, dtype.name))
            else:
                # We've just copied p_data, so in place changes are
                # not possible
                copy = False
                in_place_changes = False
        # --- End: if

        # ------------------------------------------------------------
        # Copy the array
        # -----------------------------------------------------------
        if copy:
            if p_dtype.char != 'O':
                if not masked or p_data.ndim > 0:
                    p_data = p_data.copy()
                else:
                    # This is because numpy.ma.copy doesn't work for
                    # scalar arrays (at the moment, at least)
                    p_data = numpy_ma_masked_all((), p_data.dtype)

                # We've just copied p_data, so in place changes are
                # not possible
                in_place_changes = False
            else:
                # whilst netCDF4.netcdftime.datetime is mucking bout,
                # don't copy!!!!
                # p_data = _copy(p_data)
                pass
        # --- End: if

        # ------------------------------------------------------------
        # Update the partition
        # ------------------------------------------------------------
        if update:
            self.subarray = p_data  # ?? DCH CHECK
            self.Units = p_units
            self.part = []
            self.axes = axes
            self.flip = flip
            self.flatten = []
            self.shape = p_shape
            self.location = p_location

            self._in_place_changes = in_place_changes

        # ------------------------------------------------------------
        # Return the numpy array
        # ------------------------------------------------------------
        return p_data

    @property
    def isdt(self):
        '''True if the subarray contains date-time objects.

    **Examples:**

    >>> p.Units.isreftime
    True
    >>> p.subarray.dtype == numpy.dtype(object)
    True
    >>> p.isdt
    True

        '''
        return self.Units.isreftime and self._subarray.dtype == _dtype_object

    def file_close(self):
        '''Close the file containing the subarray, if there is one.

    :Returns:

        `None`

    **Examples:**

    >>> p.file_close()

        '''
        if self.on_disk:
            self._subarray.close()

#    def flat(self):
#        '''
#
#    Return an iterator that yields the partition itself.
#
#    This is provided as a convienience to make it easier to iterate
#    through a partition matrix.
#
#    :Returns:
#
#        generator
#            An iterator that yields the partition itself.
#
#    **Examples:**
#
#    >>> type(p.flat())
#    <generator object flat at 0x519a0a0>
#    >>> for q in p.flat():
#    ...    print(q is p)
#    True
#
#    '''
#            yield self
#
#
#    def ndindex(self):
#        '''
#
#    Return an iterator over the N-dimensional indices of the partition's
#    data array.
#
#    At each iteration a tuple of indices is returned, the last dimension
#    is iterated over first.
#
#    :Returns:
#
#        generator
#            An iterator over indices of the partition's data array.
#
#    **Examples:**
#
#    >>> p.shape
#    [2, 1, 3]
#    >>> for index in p.ndindex():
#    ...     print(index)
#    ...
#    (0, 0, 0)
#    (0, 0, 1)
#    (0, 0, 2)
#    (1, 0, 0)
#    (1, 0, 1)
#    (1, 0, 2)
#
#    >>> p.shape
#    []
#    >>> for index in p.ndindex():
#    ...     print(index)
#    ...
#    ()
#
#        '''
#        return itertools_product(*[range(0, r) for r in self.shape])

    def inspect(self):
        '''Inspect the object for debugging.

    .. seealso:: `cf.inspect`

    :Returns:

        None

    **Examples:**

    >>> f.inspect()

        '''
        print(cf_inspect(self))

    def master_ndindex(self):  # itermaster_indices(self):
        '''Return an iterator over indices of the master array which are
    spanned by the data array.

    :Returns:

        generator
            An iterator over indices of the master array which are
            spanned by the data array.

    **Examples:**

    >>> p.location
    [(3, 5), (0, 1), (0, 3)]
    >>> for index in p.master_ndindex():
    ...     print(index)
    ...
    (3, 0, 0)
    (3, 0, 1)
    (3, 0, 2)
    (4, 0, 0)
    (4, 0, 1)
    (4, 0, 2)

        '''
        return itertools_product(
            *[range(*r) for r in self.location])  # TODO check

    def new_part(self, indices, master_axis_to_position, master_flip):
        '''Update the `!part` attribute in-place for new indices of the master
    array.

    :Parameters:

        indices: `list`

        master_axis_to_position: `dict`

        master_flip: `list`

    :Returns:

        None

    **Examples:**

    >>> p.new_part(indices, dim2position, master_flip)

        '''
        shape = self.shape

        if indices == [slice(0, stop, 1) for stop in shape]:
            return

        # ------------------------------------------------------------
        # If a dimension runs in the wrong direction then change its
        # index to account for this.
        #
        # For example, if a dimension with the wrong direction has
        # size 10 and its index is slice(3,8,2) then after the
        # direction is set correctly, the index needs to changed to
        # slice(6,0,-2):
        #
        # >>> a = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        # >>> a[slice(3, 8, 2)]
        # [6, 4, 2]
        # >>> a.reverse()
        # >>> print(a)
        # >>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # >>> a[slice(6, 0, -2)]
        # [6, 4, 2]
        # ------------------------------------------------------------

        if self._subarray.size > 1:
            indices = indices[:]

            p_flip = self.flip

            for axis, i in master_axis_to_position.items():

                if ((axis not in p_flip and axis not in master_flip) or
                        (axis in p_flip and axis in master_flip)):
                    # This axis runs in the correct direction
                    continue

                # Still here? Then this axis runs in the wrong
                # direction.

                # Reset the direction
                p_flip = p_flip[:]
                if axis in self.flip:
                    p_flip.remove(axis)
                else:
                    p_flip.append(axis)

                # Modify the index to account for the changed
                # direction
                size = shape[i]

                if isinstance(indices[i], slice):
                    start, stop, step = indices[i].indices(size)
                    # Note that step is assumed to be always +ve here
                    div, mod = divmod(stop-start-1, step)
                    start = size - 1 - start
                    stop = start - div*step - 1
                    if stop < 0:
                        stop = None
                    indices[i] = slice(start, stop, -step)
                else:
                    size -= 1
                    indices[i] = [size-j for j in indices[i]]
            # --- End: for

            self.flip = p_flip
        # --- End: if

        slice_None = slice(None)

        # Reorder the new indices
        indices = [(indices[master_axis_to_position[axis]]
                    if axis in master_axis_to_position else
                    slice_None)
                   for axis in self.axes]

        part = self.part

        if not part:
            self.part = indices
            return

        # Still here? update an existing part
        p_part = []
        for part_index, index, size in zip(part,
                                           indices,
                                           self._subarray.shape):

            if index == slice_None:
                p_part.append(part_index)
                continue

            if isinstance(part_index, slice):
                if isinstance(index, slice):

                    start, stop, step = part_index.indices(size)

                    size1, mod = divmod(stop-start-1, step)

                    start1, stop1, step1 = index.indices(size1+1)

                    size2, mod = divmod(stop1-start1, step1)

                    if mod != 0:
                        size2 += 1

                    start += start1 * step
                    step *= step1
                    stop = start + (size2-1)*step

                    if step > 0:
                        stop += 1
                    else:
                        stop -= 1
                    if stop < 0:
                        stop = None
                    p_part.append(slice(start, stop, step))

                    continue
                else:
                    new_part = list(range(*part_index.indices(size)))
                    new_part = [new_part[i] for i in index]
            else:
                if isinstance(index, slice):
                    new_part = part_index[index]
                else:
                    new_part = [part_index[i] for i in index]
            # --- End: if

            # Still here? Then the new element of p_part is a list of
            # integers, so let's see if we can convert it to a slice
            # before appending it.
            new_part0 = new_part[0]
            if len(new_part) == 1:
                # Convert a single element list to a slice object
                new_part = slice(new_part0, new_part0+1, 1)
            else:
                step = new_part[1] - new_part0
                if step:
                    if step > 0:
                        start, stop = new_part0, new_part[-1]+1
                    else:
                        start, stop = new_part0, new_part[-1]-1
                        if new_part == list(range(start, stop, step)):
                            if stop < 0:
                                stop = None
                            new_part = slice(start, stop, step)
            # --- End: if

            p_part.append(new_part)
        # --- End: for

        self.part = p_part

    def extra_memory(self):
        '''The extra memory required to access the array.

        '''
        if not self.in_memory:
            # --------------------------------------------------------
            # The subarray is on disk so getting the partition's data
            # array will require extra memory
            # --------------------------------------------------------
            extra_memory = True
        else:
            # --------------------------------------------------------
            # The subarray is already in memory
            # --------------------------------------------------------
            config = self.config

            p_part = self.part
            if p_part:
                extra_memory = True
            elif not config['unique_subarray']:
                extra_memory = True
            else:
                p_data = self._subarray

                if not numpy_ma_isMA(p_data):
                    # The p_data is not a masked array
                    extra_memory = isinstance(p_data.base, numpy_ndarray)
                else:
                    # The p_data is a masked array
                    memory_overlap = isinstance(
                        p_data.data.base, numpy_ndarray)
                    if not (p_data.mask is numpy_ma_nomask or
                            not numpy_ma_is_masked(p_data)):
                        # There is at least one missing data point
                        memory_overlap |= isinstance(
                            p_data.mask.base, numpy_ndarray)

                    extra_memory = memory_overlap
                # --- End: if

                p_dtype = p_data.dtype

                if not extra_memory:
                    if config['func'] is not None:
                        extra_memory = True
                    else:
                        p_units = self.Units
                        units = config['units']
                        if (not p_units.equals(units) and
                                bool(p_units) is bool(units) and
                                not (
                                    p_data.flags['C_CONTIGUOUS'] and
                                    p_dtype.kind == 'f'
                                )):
                            extra_memory = True

                # ------------------------------------------------------------
                # Extra memory is required if the dtype needs changing
                # ------------------------------------------------------------
                if not extra_memory:
                    dtype = config['dtype']
                    if dtype is not None and dtype != p_data.dtype:
                        extra_memory = True
        # --- End: if

        # ------------------------------------------------------------
        # Amount of extra memory (in bytes) required to access the
        # array
        # ------------------------------------------------------------
        return self.nbytes if extra_memory else 0

    def open(self, config):
        '''Open the partition prior to getting its array.

    :Parameters:

        config: `dict`

    :Returns:

        `None`

        '''
        unique_subarray = getrefcount(self._subarray) <= 2

        config = config.copy()
        config['unique_subarray'] = unique_subarray

        self.config = config

        if config.get('auxiliary_mask'):
            self._configure_auxiliary_mask(config['auxiliary_mask'])

        self.config['extra_memory'] = self.extra_memory()

        self._in_place_changes = True
        self.masked = True

        if hasattr(self, 'output'):
            del self.output

        return config

    def overlaps(self, indices):
        '''Return True if the subarray overlaps a subspace of the master array.

    :Parameters:

       indices: sequence
           Indices describing a subset of the master array. Each index
           is either a `slice` object or a `list`. If the sequence is
           empty then it is assumed that the master array is a scalar
           array.

    :Returns:

        p_indices, shape : `list`, `list` or `None`, `None`
            If the partition overlaps the *indices* then return a list
            of indices which will subset the partition's data to where
            it overlaps the master indices and the subsetted
            partition's shape as a list. Otherwise return `None`,
            `None`.

    **Examples:**

    >>> indices = (slice(None), slice(5, 1, -2), [1, 3, 4, 8])
    >>> p.overlaps(indices)
    (slice(), ddfsfsd), [3, 5, 4]

        '''
        p_indices = []
        shape = []

        if not indices:
            return p_indices, shape

        for index, (r0, r1), size in zip(indices, self.location, self.shape):
            if isinstance(index, slice):
                stop = size
                if index.stop < r1:
                    stop -= (r1 - index.stop)

                start = index.start - r0
                if start < 0:
                    start %= index.step   # start is now +ve

                if start >= stop:
                    # This partition does not span the slice
                    return None, None

                # Still here?
                step = index.step
                index = slice(start, stop, step)
                index_size, rem = divmod(stop-start, step)
                if rem:
                    index_size += 1

            else:

                # Still here?
                index = [i - r0 for i in index if r0 <= i < r1]
                index_size = len(index)
                if index_size == 0:
                    return None, None
                elif index_size == 1:
                    index = slice(index[0], index[0]+1)
                else:
                    index0 = index[0]
                    step = index[1] - index0
                    if step > 0:
                        start, stop = index0, index[-1]+1
                    elif step < 0:
                        start, stop = index0, index[-1]-1
                    if index == list(range(start, stop, step)):
                        # Replace the list with a slice object
                        if stop < 0:
                            stop = None
                        index = slice(start, stop, step)
            # --- End: if

            p_indices.append(index)
            shape.append(index_size)
        # --- End: for

        # Still here? Then this partition does span the slice and the
        # elements of this partition specified by p_indices are in the
        # slice.
        return p_indices, shape

#    def parallelise(self, use_shared_memory=1, from_disk=True):
#        '''
#        '''
#        config = self.config
#
#        if 1 <= use_shared_memory <= 2:
#            self.to_shared_memory(from_disk=from_disk)
#        elif use_shared_memory >= 3:
#            if config['unique_subarray']:
#                self.to_shared_memory(from_disk=from_disk)
#
#        if not self.in_memory or self.in_shared_memory:
#            # Work on this partition in parallel
#            config['serial'] = False
#            return True
#
#        # Still here? Then work on this partition in serial.
#        return False

    def to_disk(self, reopen=True):
        '''Move the partition's subarray to a temporary file on disk.

    .. note:: It is assumed that the partition's subarray is currently
              in memory, but this is not checked.

    :Parameters:

        reopen: `bool`, optional

    :Returns:

        `bool`
            True if the subarray was moved to disk, False otherwise.

    **Examples:**

    >>> p.to_disk()
    >>> p.to_disk(reopen=False)

        '''
#        try:
        tfa = CachedArray(self.array)
#        except:
#            return False

        fd, _lock_file = mkstemp(prefix=tfa._partition_file + '_',
                                 dir=tfa._partition_dir)
        close(fd)

        self.subarray = tfa
        _temporary_files[tfa._partition_file] = (tfa._partition_dir,
                                                 _lock_file, set())

        if reopen:
            # Re-open the partition
            self.open(self.config)

        return True

#    def to_shared_memory(self, from_disk=True):
#        '''
#
#    :Parameters:
#
#        from_disk : bool, optional
#
#    :Returns:
#
#        `bool`
#            Whether or not the subarray is in shared memory.
#
#        '''
#        if self.in_shared_memory:
#            return True
#
#        if not from_disk and not self.on_disk:
#            return False
#
#        if not self.fits_in_memory:
#            # There is not enough space for the new shared memory
#            # array
#            return False
#
#        try:
#            self.subarray = SharedMemoryArray(self.array)
#        except:
#            return False
#
#        # Re-open the partition
#        self.open(self.config)
#
#        return True

    def revert(self):
        '''Completely update the partition with another partition's attributes
    in place.

    The updated partition is always dependent of the other partition.

    :Parameters:

        other: `Partition`

    :Returns:

        `None`

    **Examples:**

    >>> p.revert()

        '''
        original = getattr(self, '_original', None)
        if not original:
            return

        if hasattr(self, 'output'):
            output = self.output
            keep_output = True
        else:
            keep_output = False

        del self._original

        self.__dict__ = original.__dict__

        if keep_output:
            self.output = output

    def update_inplace_from(self, other):
        '''Completely update the partition with another partition's
    attributes in place.

    :Parameters:

        other: `Partition`

    :Returns:

        `None`

    **Examples:**

    >>> p.update_inplace_from(q)

        '''
        self.__dict__ = other.__dict__.copy()

    # def set_from_child_process(self, other):
    #     '''
    #     '''
    #     if other.in_shared_memory and self.in_shared_memory:
    #         # If BOTH subarrays are in shared memory
    #         _shared_memory_array[id(other.subarray)] = (
    #              _shared_memory_array[id(self.subarray)])
    #     if other.in_cached_file:
    #         # The other subarray is in a temporary file
    #         _temporary_files.add(other._subarray._partition_file)

    #     self.__dict__ = other.__dict__

    def _register_temporary_file(self):
        '''Register a temporary file on this rank that has been created on
    another rank

        '''
        _partition_file = self._subarray._partition_file
        _partition_dir = self._subarray._partition_dir
        if _partition_file not in _temporary_files:
            fd, _lock_file = mkstemp(prefix=_partition_file + '_',
                                     dir=_partition_dir)
            close(fd)
            _temporary_files[_partition_file] = (_partition_dir,
                                                 _lock_file, set())
        else:
            _, _lock_file, _ = _temporary_files[_partition_file]

        return _lock_file

    def _update_lock_files(self, lock_files):
        '''Add the lock files listed in lock_files to the list of lock files
    managed by other ranks

        '''
        _, _lock_file, _other_lock_files = _temporary_files[
            self._subarray._partition_file]
        _other_lock_files.update(set(lock_files))
        if _lock_file in _other_lock_files:
            # If the lock file managed by this rank is in the list of
            # lock files managed by other ranks, remove it from there
            _other_lock_files.remove(_lock_file)


# --- End: class
