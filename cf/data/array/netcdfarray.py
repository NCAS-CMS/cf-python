import cfdm
from dask.utils import SerializableLock

from ...mixin_container import Container
from .mixin import ArrayMixin, FileArrayMixin

# Global lock for netCDF file access
_lock = SerializableLock()


class NetCDFArray(FileArrayMixin, ArrayMixin, Container, cfdm.NetCDFArray):
    """An array stored in a netCDF file."""

    def __dask_tokenize__(self):
        """Return a value fully representative of the object.

        .. versionadded:: 3.15.0

        """
        return super().__dask_tokenize__() + (self.get_mask(),)

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    @property
    def _lock(self):
        """Set the lock for use in `dask.array.from_array`.

        Returns a lock object because concurrent reads are not
        currently supported by the netCDF-C library. The lock object
        will be the same for all `NetCDFArray` instances, regardless
        of the dataset they access, which means that access to all
        netCDF files coordinates around the same lock.

        .. versionadded:: 3.14.0

        """
        return _lock
