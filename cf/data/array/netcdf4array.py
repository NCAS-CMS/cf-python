import cfdm

from ...mixin_container import Container
from .mixin import ActiveStorageMixin, ArrayMixin, FileArrayMixin


class NetCDF4Array(
    ActiveStorageMixin,
    FileArrayMixin,
    ArrayMixin,
    Container,
    cfdm.NetCDF4Array,
):
    """A netCDF array accessed with `netCDF4`.

    **Active storage reductions**

    An active storage reduction may be enabled with the `actify`
    method. See `cf.data.collapse.Collapse` for details.

    """
