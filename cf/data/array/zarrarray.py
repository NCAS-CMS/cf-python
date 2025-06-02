import cfdm

from ...mixin_container import Container

# Uncomment when we can use active storage on Zarr datasets:
# from .mixin import ActiveStorageMixin


class ZarrArray(
    # Uncomment when we can use active storage on Zarr datasets:
    # ActiveStorageMixin,
    Container,
    cfdm.ZarrArray,
):
    """A Zarr array accessed with `zarr`."""
