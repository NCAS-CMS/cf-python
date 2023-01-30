from ....units import Units


class ArrayMixin:
    """Mixin class for a container of an array.

    .. versionadded:: 3.14.0

    """

    def __array_function__(self, func, types, args, kwargs):
        """Implement the `numpy` ``__array_function__`` protocol.

        .. versionadded:: 3.14.0

        """
        return NotImplemented

    @property
    def Units(self):
        """The `cf.Units` object containing the units of the array.

        .. versionadded:: 3.14.0

        """
        return Units(self.get_units(), self.get_calendar(None))
