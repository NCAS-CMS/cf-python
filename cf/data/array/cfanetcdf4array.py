# REVIEW: h5: `CFAnetCDF4Array`: New class for accessing CFA with `netCDF4`
from .mixin import CFAMixin
from .netcdf4array import NetCDF4Array


class CFANetCDF4Array(CFAMixin, NetCDF4Array):
    """A CFA-netCDF array accessed with `netCDF4`.

    .. versionadded:: NEXTVERSION

    """
