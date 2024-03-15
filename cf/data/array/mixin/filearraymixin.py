from math import ceil
from os import sep
from os.path import basename, dirname, join

import numpy as np

from ....functions import _DEPRECATION_ERROR_ATTRIBUTE, abspath


class FileArrayMixin:
    """Mixin class for an array stored in a file.

    .. versionadded:: 3.14.0

    """

    def __array__(self, *dtype):
        """Convert the ``{{class}}` into a `numpy` array.

        TODO stored indices

        .. versionadded:: (cfdm) NEXTVERSION

        :Parameters:

            dtype: optional
                Typecode or data-type to which the array is cast.

        :Returns:

            `numpy.ndarray`
                An independent numpy array of the data.

        """
        array = np.asanyarray(self._get_array())
        if not dtype:
            return array
        else:
            return array.astype(dtype[0], copy=False)

    def __dask_tokenize__(self):
        """Return a value fully representative of the object.

        .. versionadded:: 3.15.0

        """
        return (
            self.__class__,
            self.shape,
            self.get_filenames(),
            self.get_addresses(),
        )

    def __getitem__(self, index)
        """TODO Returns a subspace of the array as a numpy array.

        x.__getitem__(indices) <==> x[indices]

        The indices that define the subspace must be either `Ellipsis` or
        a sequence that contains an index for each dimension. In the
        latter case, each dimension's index must either be a `slice`
        object or a sequence of two or more integers.

        Indexing is similar to numpy indexing. The only difference to
        numpy indexing (given the restrictions on the type of indices
        allowed) is:

          * When two or more dimension's indices are sequences of integers
            then these indices work independently along each dimension
            (similar to the way vector subscripts work in Fortran).

        .. versionadded:: NEXTVERSION

        """
        shape0 = self.shape
        index = parse_indices(shape0, index, keepdims=False, bool_as_int=True)
        
        index0 = self._get_component('index', None)
        if index0 is None:
            self._set_component('index', index, copy=False)
            return
                
        new_index = []
        for ind0, ind, size0 in zip(index0, index, shape0):            
            if index == slice(None):
                new_index.append(ind0)
                new_shape.apepend(size0)
                continue
            
            if isinstance(ind0, slice):
                if isinstance(ind, slice):
                    # 'ind0' is slice, 'ind' is slice
                    start, stop, step = ind0.indices(size0)
                    size1, mod = divmod(stop - start - 1, step)
                    start1, stop1, step1 = ind.indices(size1 + 1)
                    size2, mod = divmod(stop1 - start1, step1)

                    if mod != 0:
                        size2 += 1

                    start += start1 * step
                    step *= step1
                    stop = start + (size2 - 1) * step

                    if step > 0:
                        stop += 1
                    else:
                        stop -= 1
                        
                    if stop < 0:
                        stop = None
                        
                    new = slice(start, stop, step)
                    new_size = ceil((stop - start)/step)
                else:
                    # 'ind0' is slice, 'ind' is numpy array of int
                    new = np.arange(*ind0.indices(size0))[ind]
                    new_size = new.size
            else:
                # 'ind0' is numpy array of int
                new = ind0[ind]
                new_size = new.size
                    
            new_index.append(new)
            new_shape.apepend(new_size)

        self._set_component('index', tuple(new_index), copy=False)
        self._set_component('shape', tuple(new_shape), copy=False)
        
    def _get_array(self)
        """Returns a subspace of the array as a numpy array.

        x.__getitem__(indices) <==> x[indices]

        The indices that define the subspace must be either `Ellipsis` or
        a sequence that contains an index for each dimension. In the
        latter case, each dimension's index must either be a `slice`
        object or a sequence of two or more integers.

        Indexing is similar to numpy indexing. The only difference to
        numpy indexing (given the restrictions on the type of indices
        allowed) is:

          * When two or more dimension's indices are sequences of integers
            then these indices work independently along each dimension
            (similar to the way vector subscripts work in Fortran).

        .. versionadded:: NEXTVERSION

        """
        return NotImplementedError(
            f"Must implement {self.__class__.__name__}._get_array"
        )

    @property
    def _dask_meta(self):
        """The metadata for the containing dask array.

        This is the kind of array that will result from slicing the
        file array.

        .. versionadded:: 3.14.0

        .. seealso:: `dask.array.from_array`

        """
        return np.array((), dtype=self.dtype)

    @property
    def filename(self):
        """The name of the file containing the array.

        Deprecated at version 3.14.0. Use method `get_filename` instead.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "filename",
            message="Use method 'get_filename' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def index(self):
        """TODO

        .. versionadded:: NEXTVERSION

        """
        ind = self._get_component('index', None)
        if ind is None:
            ind = parse_indices(self.shape, (Ellipsis,), keepdims=False, bool_ti_int=True)
            self._set_component('index', ind, copy=False)

        return ind

    def del_file_location(self, location):
        """Remove reference to files in the given location.

        .. versionadded:: 3.15.0

        :Parameters:

            location: `str`
                 The file location to remove.

        :Returns:

            `{{class}}`
                A new {{class}} with reference to files in *location*
                removed.

        **Examples**

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file2')
        >>> a.get_addresses()
        ('tas1', 'tas2')
        >>> b = a.del_file_location('/data1')
        >>> b = get_filenames()
        ('/data2/file2',)
        >>> b.get_addresses()
        ('tas2',)

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file1', '/data2/file2')
        >>> a.get_addresses()
        ('tas1', 'tas1', 'tas2')
        >>> b = a.del_file_location('/data2')
        >>> b.get_filenames()
        ('/data1/file1',)
        >>> b.get_addresses()
        ('tas1',)

        """
        location = abspath(location).rstrip(sep)

        new_filenames = []
        new_addresses = []
        for filename, address in zip(
            self.get_filenames(), self.get_addresses()
        ):
            if dirname(filename) != location:
                new_filenames.append(filename)
                new_addresses.append(address)

        if not new_filenames:
            raise ValueError(
                "Can't delete a file location when it results in there "
                "being no files"
            )

        a = self.copy()
        a._set_component("filename", tuple(new_filenames), copy=False)
        a._set_component("address", tuple(new_addresses), copy=False)
        return a

    def file_locations(self):
        """The locations of the files, any of which may contain the data.

        .. versionadded:: 3.15.0

        :Returns:

            `tuple`
                The file locations, one for each file, as absolute
                paths with no trailing path name component separator.

        **Examples**

        >>> a.get_filenames()
        ('/data1/file1',)
        >>> a.file_locations()
        ('/data1,)

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file2')
        >>> a.file_locations()
        ('/data1', '/data2')

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file2', '/data1/file2')
        >>> a.file_locations()
        ('/data1', '/data2', '/data1')

        """
        return tuple(map(dirname, self.get_filenames()))

    def add_file_location(self, location):
        """Add a new file location.

        All existing files are additionally referenced from the given
        location.

        .. versionadded:: 3.15.0

        :Parameters:

            location: `str`
                The new location.

        :Returns:

            `{{class}}`
                A new {{class}} with all previous files additionally
                referenced from *location*.

        **Examples**

        >>> a.get_filenames()
        ('/data1/file1',)
        >>> a.get_addresses()
        ('tas',)
        >>> b = a.add_file_location('/home')
        >>> b.get_filenames()
        ('/data1/file1', '/home/file1')
        >>> b.get_addresses()
        ('tas', 'tas')

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file2',)
        >>> a.get_addresses()
        ('tas', 'tas')
        >>> b = a.add_file_location('/home/')
        >>> b = get_filenames()
        ('/data1/file1', '/data2/file2', '/home/file1', '/home/file2')
        >>> b.get_addresses()
        ('tas', 'tas', 'tas', 'tas')

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file1',)
        >>> a.get_addresses()
        ('tas1', 'tas2')
        >>> b = a.add_file_location('/home/')
        >>> b.get_filenames()
        ('/data1/file1', '/data2/file1', '/home/file1')
        >>> b.get_addresses()
        ('tas1', 'tas2', 'tas1')

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file1',)
        >>> a.get_addresses()
        ('tas1', 'tas2')
        >>> b = a.add_file_location('/data1')
        >>> b.get_filenames()
        ('/data1/file1', '/data2/file1')
        >>> b.get_addresses()
        ('tas1', 'tas2')

        """
        location = abspath(location).rstrip(sep)

        filenames = self.get_filenames()
        addresses = self.get_addresses()

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a fully qualified URI.
        new_filenames = list(filenames)
        new_addresses = list(addresses)
        for filename, address in zip(filenames, addresses):
            new_filename = join(location, basename(filename))
            if new_filename not in new_filenames:
                new_filenames.append(new_filename)
                new_addresses.append(address)

        a = self.copy()
        a._set_component("filename", tuple(new_filenames), copy=False)
        a._set_component(
            "address",
            tuple(new_addresses),
            copy=False,
        )
        return a
