import cfdm
from dask.utils import SerializableLock

from ...mixin_container import Container
from .mixin import ActiveStorageMixin, ArrayMixin, FileArrayMixin

# Global lock for netCDF file access
_lock = SerializableLock()


class HDFArray(ActiveStorageMixin, FileArrayMixin, ArrayMixin, Container, cfdm.HDFArray):
    """An array stored in a netCDF file.]

        .. versionadded:: HDFVER

    """

    def __dask_tokenize__(self):
        """Return a value fully representative of the object.

        .. versionadded:: HDFVER

        """
        return super().__dask_tokenize__() + (self.get_mask(),)

    @property
    def _lock(self):
        """Set the lock for use in `dask.array.from_array`.

        Returns a lock object because concurrent reads are not
        currently supported by the netCDF-C library. The lock object
        will be the same for all `NetCDFArray` instances, regardless
        of the dataset they access, which means that access to all
        netCDF files coordinates around the same lock.

        .. versionadded:: HDFVER

        """
        return _lock
