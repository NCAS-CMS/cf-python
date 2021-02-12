import abc

from functools import reduce
from operator import mul
from sys import getrefcount

from ...functions import inspect as cf_inspect


class CompressedSubarray(abc.ABC):
    """Abstract base class for a compressed sub-array container."""

    def __init__(self, array, shape, compression):
        """**Initialization**

        :Parameters:

            array:

            shape: `tuple`
                The shape of the uncompressed array

            compression: `dict`

        """
        # DO NOT CHANGE IN PLACE
        self.array = array

        # DO NOT CHANGE IN PLACE
        self.compression = compression

        # DO NOT CHANGE IN PLACE
        self.shape = tuple(shape)

        # DO NOT CHANGE IN PLACE
        self.ndim = len(shape)

        # DO NOT CHANGE IN PLACE
        self.size = reduce(mul, shape, 1)

    @abc.abstractmethod
    def __getitem__(self, indices):
        """x.__getitem__(indices) <==> x[indices]

        Returns a numpy array.

        """
        raise NotImplementedError()  # pragma: no cover

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        array = self.array
        shape = str(array.shape)
        shape = shape.replace(",)", ")")

        return "<CF {}{}: {}>".format(
            self.__class__.__name__, shape, str(array)
        )

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def file(self):
        """The file on disk which contains the compressed array, or
        `None` of the array is in memory.

        **Examples:**

         >>> self.file
         '/home/foo/bar.nc'

        """
        return getattr(self.array, "file", None)

    def close(self):
        """Close all referenced open files.

        :Returns:

            `None`

        **Examples:**

        >>> f.close()

        """
        if self.on_disk():
            self.array.close()

    def copy(self):
        """Replace the abstract base class with a deep copy."""
        C = self.__class__
        new = C.__new__(C)
        new.__dict__ = self.__dict__.copy()
        return new

    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))

    def on_disk(self):
        """True if and only if the compressed array is on disk as
        opposed to in memory.

        **Examples:**

        >>> a.on_disk()
        True

        """
        return not hasattr(self.array, "__array_interface__")

    def unique(self):
        """True if there is only one permanent reference to the array
        instance."""
        # Note, from the Python docs for sys.getrefcount:
        # "The count returned is generally one higher than you might expect,
        # because it includes the (temporary) reference as an argument to
        # getrefcount", hence <= 2 to test for uniqueness rather than <= 1.
        return getrefcount(self.array) <= 2


# --- End: class
