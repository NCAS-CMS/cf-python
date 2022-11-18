import cfdm

from .mixin import ActiveStorageMixin, FileArrayMixin


class NetCDFArray(ActiveStorageMixin, FileArrayMixin, cfdm.NetCDFArray):
    """An array stored in a netCDF file.

    TODOACTIVEDOCS

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)
