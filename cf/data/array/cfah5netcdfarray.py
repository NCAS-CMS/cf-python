# REVIEW: h5: `CFAH5netcdfArray`: New class for accessing CFA with `h5netcdf`
from .h5netcdfarray import H5netcdfArray
from .mixin import CFAMixin


class CFAH5netcdfArray(CFAMixin, H5netcdfArray):
    """A CFA-netCDF array accessed with `h5netcdf`

    .. versionadded:: NEXTVERSION

    """
