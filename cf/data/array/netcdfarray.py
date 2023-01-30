import cfdm
from dask.utils import SerializableLock

from ...mixin_container import Container
from .mixin import FileArrayMixin

# Global lock for netCDF file access
_lock = SerializableLock()


class NetCDFArray(FileArrayMixin, Container, cfdm.NetCDFArray):
    """An array stored in a netCDF file."""

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    @property
    def _dask_lock(self):
        """Set the lock for use in `dask.array.from_array`.

        Returns a lock object (unless no file name has been set, in
        which case `False` is returned) because concurrent reads are
        not currently supported by the netCDF-C library. The lock
        object will be the same for all `NetCDFArray` instances,
        regardless of the dataset they access, which means that all
        files access coordinates around the same lock.

        .. versionadded:: 3.14.0

        """
        filename = self.get_filename(None)
        if filename is None:
            return False

        return _lock
