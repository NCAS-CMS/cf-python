import cfdm

from ...mixin_container import Container
#from .mixin import ActiveStorageMixin


class ZarrArray(
#    ActiveStorageMixin,
    Container,
    cfdm.ZarrArray,
):
    """A Zarr array accessed with `zarr`."""
