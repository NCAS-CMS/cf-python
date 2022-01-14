import cfdm

from . import abstract


class NetCDFArray(cfdm.NetCDFArray, abstract.FileArray):
    """An array stored in a netCDF file."""
