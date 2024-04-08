from .h5netcdfarray import H5netcdfArray
from .mixin import CFAMixin


# REVIEW: h5
class CFAH5netcdfArray(CFAMixin, H5netcdfArray):
    """A CFA-netCDF array accessed with `h5netcdf`

    .. versionadded:: NEXTVERSION

    """
