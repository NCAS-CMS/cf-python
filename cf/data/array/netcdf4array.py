import cfdm

from ...mixin_container import Container
from .mixin import ActiveStorageMixin


class NetCDF4Array(
    ActiveStorageMixin,
    Container,
    cfdm.NetCDF4Array,
):
    """A netCDF array accessed with `netCDF4`."""
