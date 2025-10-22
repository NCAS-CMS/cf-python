from cfunits import Units as cfUnits


class Units:
    """Store, combine and compare physical units and convert numeric
    values to different units.

    This is a convenience class that creates a `cfunits.Units`
    instance.

    The full documentation is available via a `cf.Units` instance,
    e.g. ``help(cf.Units())``.

    """

    def __new__(cls, *args, **kwargs):
        """Return a new Units instance."""
        return cfUnits(*args, **kwargs)

    @staticmethod
    def conform(*args, **kwargs):
        """Conform values to equivalent values in a compatible unit."""
        return cfUnits.conform(*args, **kwargs)
