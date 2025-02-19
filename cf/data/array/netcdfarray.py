class NetCDFArray:
    """A netCDF array accessed with `netCDF4`.

    Deprecated at version 3.16.3 and is no longer available. Use
    `cf.NetCDF4Array` instead.

    """

    def __init__(self, *args, **kwargs):
        """**Initialisation**"""
        from ...functions import DeprecationError

        raise DeprecationError(
            f"{self.__class__.__name__} was deprecated at version 3.16.3 "
            "and is no longer available. Use cf.NetCDF4Array instead."
        )
