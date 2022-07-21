import cfdm

from .abstract import FileArray


class NetCDFArray(cfdm.NetCDFArray, FileArray):
    """An array stored in a netCDF file."""
