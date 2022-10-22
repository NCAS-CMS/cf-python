import cfdm

from .abstract import FileArray


class NetCDFArray(cfdm.NetCDFArray, FileArray):
    """An array stored in a netCDF file."""

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        out = super().__repr__()
        return out[:-1] + f", {self.get_ncvar()}>"
