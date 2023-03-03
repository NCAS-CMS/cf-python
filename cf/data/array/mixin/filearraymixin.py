import numpy as np

from ....functions import _DEPRECATION_ERROR_ATTRIBUTE


class FileArrayMixin:
    """Mixin class for an array stored in a file.

    .. versionadded:: 3.14.0

    """

    @property
    def _dask_meta(self):
        """The metadata for the containing dask array.

        This is the kind of array that will result from slicing the
        file array.

        .. versionadded:: 3.14.0

        .. seealso:: `dask.array.from_array`

        """
        return np.array((), dtype=self.dtype)

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return f"<CF {self.__class__.__name__}{self.shape}: {self}>"

    def __str__(self):
        """x.__str__() <==> str(x)"""
        return f"{self.get_filename()}, {self.get_address()}"

    @property
    def dtype(self):
        """Data-type of the array."""
        return self._get_component("dtype")

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
    def shape(self):
        """Shape of the array."""
        return self._get_component("shape")
