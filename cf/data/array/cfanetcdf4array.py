from .mixin import CFAMixin
from .netcdf4array import NetCDF4Array


# REVIEW: h5
class CFANetCDF4Array(CFAMixin, NetCDF4Array):
    """A CFA-netCDF array accessed with `netCDF4`.

    .. versionadded:: NEXTVERSION

    """
