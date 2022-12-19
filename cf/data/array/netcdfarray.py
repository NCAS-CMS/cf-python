import cfdm

from .mixin import FileArrayMixin


class NetCDFArray(FileArrayMixin, cfdm.NetCDFArray):
    """An array stored in a netCDF file."""

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    @property
    def _dask_lock(self):
        """Set the lock for use in `dask.array.from_array`.

        Returns `True` because concurrent reads are not currently
        supported by the netCDF-C library.

        .. versionadded:: TODODASKVER

        """
        return True
