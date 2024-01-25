import cfdm
from dask.utils import SerializableLock

from ...mixin_container import Container
from .locks import _lock
from .mixin import ActiveStorageMixin, ArrayMixin, FileArrayMixin


class NetCDFArray(
    ActiveStorageMixin, FileArrayMixin, ArrayMixin, Container, cfdm.NetCDFArray
):
    """An array stored in a netCDF file.

    TODOACTIVEDOCS

    """

    def __dask_tokenize__(self):
        """Return a value fully representative of the object.

        .. versionadded:: 3.15.0

        """
        return super().__dask_tokenize__() + (self.get_mask(),)

    @property
    def _lock(self):
        """Set the lock for use in `dask.array.from_array`.

        Returns a lock object because concurrent reads are not
        currently supported by the netCDF and HDF libraries. The lock
        object will be the same for all `NetCDFArray` and `HDFArray`
        instances, regardless of the dataset they access, which means
        that access to all netCDF and HDF files coordinates around the
        same lock.

        .. versionadded:: 3.14.0

        """
        return _lock
