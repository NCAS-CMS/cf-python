from numpy import empty as numpy_empty
from numpy import full as numpy_full

from numpy.ma import masked_all as numpy_ma_masked_all

from . import abstract

from ..functions import parse_indices
from ..constants import masked as cf_masked


class FilledArray(abstract.Array):
    """An underlying filled array."""

    def __init__(
        self,
        dtype=None,
        ndim=None,
        shape=None,
        size=None,
        fill_value=None,
        masked_all=False,
    ):
        """**Initialization**

            :Parameters:

                dtype : numpy.dtype
                    The numpy data type of the data array.

                ndim : int
                    Number of dimensions in the data array.

                shape : tuple
                    The data array's dimension sizes.

                size : int
                    Number of elements in the data array.

                fill_value : scalar, optional

        #        masked_all: `bool`

        """
        super().__init__(
            dtype=dtype,
            ndim=ndim,
            shape=shape,
            size=size,
            fill_value=fill_value,
        )

    def __getitem__(self, indices):
        """x.__getitem__(indices) <==> x[indices]

        Returns a numpy array.

        """
        if indices is Ellipsis:
            array_shape = self.shape
        else:
            array_shape = []
            for index in parse_indices(self.shape, indices):
                if isinstance(index, slice):
                    step = index.step
                    if step == 1:
                        array_shape.append(index.stop - index.start)
                    elif step == -1:
                        stop = index.stop
                        if stop is None:
                            array_shape.append(index.start + 1)
                        else:
                            array_shape.append(index.start - index.stop)
                    else:
                        stop = index.stop
                        if stop is None:
                            stop = -1

                        a, b = divmod(stop - index.start, step)
                        if b:
                            a += 1
                        array_shape.append(a)
                else:
                    array_shape.append(len(index))
        # --- End: if

        if self.fill_value() is cf_masked:
            return numpy_ma_masked_all(array_shape, dtype=self.dtype)
        elif self.fill_value() is not None:
            return numpy_full(
                array_shape, fill_value=self.fill_value(), dtype=self.dtype
            )
        else:
            return numpy_empty(array_shape, dtype=self.dtype)

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def dtype(self):
        """Data-type of the data elements.

        **Examples:**

        >>> a.dtype
        dtype('float64')
        >>> print(type(a.dtype))
        <type 'numpy.dtype'>

        """
        return self._get_component("dtype")

    @property
    def ndim(self):
        """Number of array dimensions.

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

        """
        return self._get_component("ndim")

    @property
    def shape(self):
        """Tuple of array dimension sizes.

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

        """
        return self._get_component("shape")

    @property
    def size(self):
        """Number of elements in the array.

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

        """
        return self._get_component("size")

    def fill_value(self):
        """Return the data array missing data value."""
        return self._get_component("fill_value")

    @property
    def array(self):
        """An independent numpy array containing the data."""
        return self[...]

    def reshape(self, newshape):
        """Give a new shape to the array."""
        new = self.copy()
        new.shape = newshape
        new.ndim = len(newshape)
        return new

    def resize(self, newshape):
        """Change the shape and size of the array in-place."""
        self.shape = newshape
        self.ndim = len(newshape)

    def view(self):
        """Return a view of the entire array."""
        return self[...]


# --- End: class
