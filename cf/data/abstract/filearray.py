from ...functions import inspect as cf_inspect

from .array import Array


class FileArray(Array):
    '''A sub-array stored in a file.

    .. note:: Subclasses must define the following methods:
              `!__getitem__`, `!__str__`, `!close` and `!open`.

    '''
    def __getitem__(self, indices):
        '''
        '''
        pass

    def __str__(self):
        '''x.__str__() <==> str(x)

        '''
        return "%s in %s" % (self.shape, self.file)

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def dtype(self):
        '''Data-type of the data elements.

    **Examples:**

    >>> a.dtype
    dtype('float64')
    >>> print(type(a.dtype))
    <type 'numpy.dtype'>

        '''
        return self._get_component('dtype')

    @property
    def ndim(self):
        '''Number of array dimensions

    **Examples:**

    >>> a.shape
    (73, 96)
    >>> a.ndim
    2
    >>> a.size
    7008

    >>> a.shape
    (1, 1, 1)
    >>> a.ndim
    3
    >>> a.size
    1

    >>> a.shape
    ()
    >>> a.ndim
    0
    >>> a.size
    1

        '''
        return self._get_component('ndim')

    @property
    def shape(self):
        '''Tuple of array dimension sizes.

    **Examples:**

    >>> a.shape
    (73, 96)
    >>> a.ndim
    2
    >>> a.size
    7008

    >>> a.shape
    (1, 1, 1)
    >>> a.ndim
    3
    >>> a.size
    1

    >>> a.shape
    ()
    >>> a.ndim
    0
    >>> a.size
    1

        '''
        return self._get_component('shape')

    @property
    def size(self):
        '''Number of elements in the array.

    **Examples:**

    >>> a.shape
    (73, 96)
    >>> a.size
    7008
    >>> a.ndim
    2

    >>> a.shape
    (1, 1, 1)
    >>> a.ndim
    3
    >>> a.size
    1

    >>> a.shape
    ()
    >>> a.ndim
    0
    >>> a.size
    1

        '''
        return self._get_component('size')

    @property
    def filename(self):
        '''TODO

    **Examples:**

    TODO

        '''
        return self._get_component('filename')

    @property
    def array(self):
        '''Return an independent numpy array containing the data.

    :Returns:

        `numpy.ndarray`
            An independent numpy array of the data.

    **Examples:**

    >>> n = numpy.asanyarray(a)
    >>> isinstance(n, numpy.ndarray)
    True

        '''
        return self[...]

    def inspect(self):
        '''Inspect the object for debugging.

    .. seealso:: `cf.inspect`

    :Returns:

        `None`

        '''
        print(cf_inspect(self))  # pragma: no cover

    def get_filename(self):
        '''The name of the file containing the array.

    **Examples:**

    >>> a.get_filename()
    'file.nc'

        '''
        return self._get_component('filename')

    def close(self):
        pass

    def open(self):
        pass


# --- End: class

# Array.register(FileArray)
