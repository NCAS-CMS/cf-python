import cfdm

from ...mixin_container import Container
from .mixin import ActiveStorageMixin


class PyfiveArray(
    ActiveStorageMixin,
    Container,
    cfdm.PyfiveArray,
):
    """A netCDF array accessed with `pyfive`.

    .. versionadded:: 3.20.0

    """
