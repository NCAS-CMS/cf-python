import cfdm

from ...mixin_container import Container


class Netcdf_fileArray(
    Container,
    cfdm.Netcdf_fileArray,
):
    """A netCDF-3 array accessed with `scipy.io.netcdf_file`.

    .. versionadded:: NEXTVERSION

    """
