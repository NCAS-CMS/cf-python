import cfdm

from ...mixin_container import Container
from .mixin import ActiveStorageMixin, ArrayMixin, FileArrayMixin


class H5netcdfArray(
    ActiveStorageMixin,
    FileArrayMixin,
    ArrayMixin,
    Container,
    cfdm.H5netcdfArray,
):
    """A netCDF array accessed with `h5netcdf`.

    **Active storage reductions**

    An active storage reduction may be enabled with the `actify`
    method. See `cf.data.collapse.Collapse` for details.

    .. versionadded:: 1.11.2.0

    """
