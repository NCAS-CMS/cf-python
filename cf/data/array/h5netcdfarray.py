import cfdm

from ...mixin_container import Container
from .mixin import ActiveStorageMixin


class H5netcdfArray(
    ActiveStorageMixin,
    Container,
    cfdm.H5netcdfArray,
):
    """A netCDF array accessed with `h5netcdf` using the `h5py` backend.

    .. versionadded:: 3.16.3

    """
