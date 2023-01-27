from ....units import Units


class ArrayMixin:
    """Mixin class for a container of an array.

    .. versionadded:: TODODASKVER

    """

    def __array_function__(self, func, types, args, kwargs):
        """Implement the `numpy` ``__array_function__`` protocol.

        .. versionadded:: TODODASKVER

        """
        return NotImplemented

    @property
    def Units(self):
        """The `cf.Units` object containing the units of the array.

        .. versionadded:: TODODASKVER

        """
        return Units(self.get_units(), self.get_calendar(None))
