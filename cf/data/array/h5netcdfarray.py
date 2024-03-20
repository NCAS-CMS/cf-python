import cfdm

from ...mixin_container import Container
from .locks import netcdf_lock
from .mixin import ActiveStorageMixin, ArrayMixin, FileArrayMixin, IndexMixin


class H5netcdfArray(
    ActiveStorageMixin,
    IndexMixin,
    FileArrayMixin,
    ArrayMixin,
    Container,
    cfdm.H5netcdfArray,
):
    """A netCDF array accessed with `h5netcdf`.

    **Active storage reductions**

    Active storage reduction may be enabled with the `actify`
    method. See `cf.data.collapse.Collapse` for details.

    .. versionadded:: NEXTVERSION

    """

    def __dask_tokenize__(self):
        """Return a value fully representative of the object.

        .. versionadded:: NEXTVERSION

        """
        return super().__dask_tokenize__() + (self.get_mask(),)

    @property
    def _lock(self):
        """Set the lock for use in `dask.array.from_array`.

        Returns a lock object because concurrent reads are not
        currently supported by the HDF5 library. The lock object will
        be the same for all `NetCDF4Array` and `H5netcdfArray`
        instances, regardless of the dataset they access, which means
        that access to all netCDF and HDF files coordinates around the
        same lock.

        .. versionadded:: NEXTVERSION

        """
        return netcdf_lock

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

        # Note: We need to lock because the netCDF file is about to be
        #       accessed.
        self._lock.acquire()

        # Note: It's cfdm.H5netcdfArray.__getitem__ that we want to
        #       call here, but we use 'Container' in super because
        #       that comes immediately before cfdm.H5netcdfArray in
        #       the method resolution order.
        array = super(Container, self).__getitem__(index)

        self._lock.release()
        return array
