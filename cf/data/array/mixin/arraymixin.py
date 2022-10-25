from ....units import Units


class ArrayMixin:
    """Mixin class for a container of an array.

    .. versionadded:: TODODASKVER

    """

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented

    @property
    def Units(self):
        """TODODASKDOCS."""
        return Units(self.get_units(), self.get_calendar(None))
