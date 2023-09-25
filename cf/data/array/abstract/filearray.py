from ....functions import _DEPRECATION_ERROR_ATTRIBUTE
from ..mixin import FileArrayMixin
from .array import Array


class FileArray(FileArrayMixin, Array):
    """Abstract base class for an array stored in a file."""

    def __getitem__(self, indices):
        """Return a subspace of the array.

        x.__getitem__(indices) <==> x[indices]

        Returns a subspace of the array as an independent numpy array.

        """
        raise NotImplementedError(
            f"Must implement {self.__class__.__name__}.__getitem__"
        )  # pragma: no cover

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

    def close(self):
        """Close the dataset containing the data."""
        raise NotImplementedError(
            f"Must implement {self.__class__.__name__}.close"
        )  # pragma: no cover

    def get_address(self):
        """The address in the file of the variable.

        .. versionadded:: 3.14.0

        :Returns:

            `str` or `None`
                The address, or `None` if there isn't one.

        """
        raise NotImplementedError(
            f"Must implement {self.__class__.__name__}.get_address "
            "in subclasses"
        )  # pragma: no cover

    def open(self):
        """Returns an open dataset containing the data array."""
        raise NotImplementedError(
            f"Must implement {self.__class__.__name__}.open"
        )  # pragma: no cover
