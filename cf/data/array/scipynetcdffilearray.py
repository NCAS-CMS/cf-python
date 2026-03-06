import cfdm

from ...mixin_container import Container


class ScipyNetcdfFileArray(
    Container,
    cfdm.ScipyNetcdfFileArray,
):
    """A netCDF-3 array accessed with `scipy.io.netcdf_file`.

    .. versionadded:: NEXTVERSION

    """
