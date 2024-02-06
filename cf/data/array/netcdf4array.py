import cfdm

from ...mixin_container import Container
from .locks import _lock
from .mixin import ActiveStorageMixin, ArrayMixin, FileArrayMixin


class NetCDF4Array(
    ActiveStorageMixin,
    FileArrayMixin,
    ArrayMixin,
    Container,
    cfdm.NetCDF4Array,
):
    """A netCDF array accessed with `netCDF4`.

    **Active storage reductions**

    Active storage reduction may be enabled with the `actify`
    method. See `cf.data.collapse.Collapse` for details.

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
        object will be the same for all `NetCDF4Array` and
        `H5netcdfArray` instances, regardless of the dataset they
        access, which means that access to all netCDF and HDF files
        coordinates around the same lock.

        .. versionadded:: 3.14.0

        """
        return _lock
