import cfdm
from dask.utils import SerializableLock

from ...mixin_container import Container
from .mixin import ArrayMixin, FileArrayMixin, IndexMixin

# Global lock for netCDF file access
_lock = SerializableLock()


class NetCDFArray(IndexMixin, FileArrayMixin, ArrayMixin, Container, cfdm.NetCDFArray):
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
        """TODO"""
        print ('cf.NetCDFArray._get_array', self.index)
#        return super(cfdm.NetCDFArray, self).__getitem__(self.index)
#        return super(cfdm.NetCDFArray, self).__getitem__(self.index)

        netcdf, address = self.open()
        dataset = netcdf

        groups, address = self.get_groups(address)
        if groups:
            # Traverse the group structure, if there is one (CF>=1.8).
            netcdf = self._group(netcdf, groups)

        if isinstance(address, str):
            # Get the variable by netCDF name
            variable = netcdf.variables[address]
        else:
            # Get the variable by netCDF integer ID
            for variable in netcdf.variables.values():
                if variable._varid == address:
                    break

        # Get the data, applying masking and scaling as required.
#        array = cfdm.netcdf_indexer(
#            variable,
#            mask=self.get_mask(),
#            unpack=self.get_unpack(),
#            always_mask=False,
#        )
        array = variable[self.index]

        # Set the units, if they haven't been set already.
#        self._set_attributes(variable)

        # Set the units, if they haven't been set already.
        self._set_units(variable)

        self.close(dataset)
        del netcdf, dataset

        if not self.ndim:
            # Hmm netCDF4 has a thing for making scalar size 1, 1d
            array = array.squeeze()

        return array
