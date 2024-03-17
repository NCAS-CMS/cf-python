import cfdm
from dask.utils import SerializableLock

from ...mixin_container import Container
from .mixin import ArrayMixin, FileArrayMixin, IndexMixin

# Global lock for netCDF file access
_lock = SerializableLock()


class NetCDFArray(
    IndexMixin, FileArrayMixin, ArrayMixin, Container, cfdm.NetCDFArray
):
    """An array stored in a netCDF file."""

    def __dask_tokenize__(self):
        """Return a value fully representative of the object.

        .. versionadded:: 3.15.0

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

        .. versionadded:: 3.14.0

        """
        return _lock

    def _get_array(self):
        """Returns a subspace of the dataset variable.

        The subspace is defined by the indices stored in the `index`
        attribute.

        .. versionadded:: NEXTVERSION

        .. seealso:: `__array__`, `index`

        :Returns:

            `numpy.ndarray`
                The subspace.

        """
        # Note: It's cfdm.NetCDFArray.__getitem__ that we want to
        #       call, but we use 'Container' in super because that
        #       comes immediately before cfdm.NetCDFArray in the
        #       method resolution order.
        return super(Container, self).__getitem__(self.index)
