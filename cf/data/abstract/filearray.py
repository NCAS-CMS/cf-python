import numpy as np

from ...functions import inspect as cf_inspect
from .array import Array


class FileArray(Array):
    """An array stored in a file."""

    def __getitem__(self, indices):
        """Return a subspace of the array.

        x.__getitem__(indices) <==> x[indices]

        Returns a subspace of the array as an independent numpy array.

        """
        raise NotImplementedError(
            f"Must implement {self.__class__.__name__}.__getitem__"
        )  # pragma: no cover

    def __str__(self):
        """x.__str__() <==> str(x)"""
        return f"<{self.__class__.__name__}: {self.shape} in {self.file}>"

    @property
    def _dask_lock(self):
        """TODODASKDOCS.

        Concurrent reads are assumed to be not supported.

        .. versionadded:: TODODASKVER

        """
        return True

    @property
    def _dask_meta(self):
        """TODODASKDOCS.

        .. versionadded:: TODODASKVER

        """
        return np.array((), dtype=self.dtype)

    @property
    def array(self):
        """Return an independent numpy array containing the data.

        :Returns:

            `numpy.ndarray`
                An independent numpy array of the data.

        **Examples**

        >>> n = numpy.asanyarray(a)
        >>> isinstance(n, numpy.ndarray)
        True

        """
        return self[...]

    @property
    def dtype(self):
        """Data-type of the array."""
        return self._get_component("dtype")

    @property
    def filename(self):
        """The name of the file containing the array."""
        return self._get_component("filename")

    @property
    def shape(self):
        """Shape of the array."""
        return self._get_component("shape")

    def close(self):
        """Close the dataset containing the data."""
        raise NotImplementedError(
            f"Must implement {self.__class__.__name__}.close"
        )  # pragma: no cover

    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))  # pragma: no cover

    def get_filename(self):
        """Return the name of the file containing the array.

        :Returns:

            `str`
                The file name.

        **Examples**

        >>> a.get_filename()
        'file.nc'

        """
        return self._get_component("filename")

    def open(self):
        """Returns an open dataset containing the data array."""
        raise NotImplementedError(
            f"Must implement {self.__class__.__name__}.open"
        )  # pragma: no cover
