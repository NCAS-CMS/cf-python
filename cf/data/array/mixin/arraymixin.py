from ....units import Units


class ArrayMixin:
    """Mixin class for a container of an array.

    .. versionadded:: TODODASKVER

    """

    def __array_function__(self, func, types, args, kwargs):
        """Implement the `numpy` ``__array_function__`` protocol."""
        return NotImplemented

    @property
    def Units(self):
        """TODODASKDOCS."""
        return Units(self.get_units(), self.get_calendar(None))
