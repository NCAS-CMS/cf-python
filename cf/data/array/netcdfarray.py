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

    def _get_array(self, index=None):
        """Returns a subspace of the dataset variable.

        The subspace is defined by the indices stored in the `index`
        attribute.

        .. versionadded:: NEXTVERSION

        .. seealso:: `__array__`, `index`

        :Parameters:

            index: `tuple` or `None`, optional
               Provide the indices that define the subspace. If `None`
               then the `index` attribute is used.

        :Returns:

            `numpy.ndarray`
                The subspace.

        """
        if index is None:
            index = self.index

        # Note: We need to use the lock because the netCDF file is
        #       going to be read.
        self._lock.acquire()

        # Note: It's cfdm.NetCDFArray.__getitem__ that we want to call
        #       here, but we use 'Container' in super because that
        #       comes immediately before cfdm.NetCDFArray in the
        #       method resolution order.
        array = super(Container, self).__getitem__(index)

        self._lock.release()
        return array
