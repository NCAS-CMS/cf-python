from ...units import Units


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

    def get_calendar(self, default=ValueError()):
        """The calendar of the array.

        If the calendar is `None` then the CF default calendar is
        assumed, if applicable.

        .. versionadded:: TODODASKVER

        :Parameters:

            TODODASKDOCS

        :Returns:

            `str` or `None`

        """
        try:
            return self._get_component("calendar")
        except AttributeError:
            if default is None:
                return

            return self._default(
                default,
                f"{self.__class__.__name__} 'calendar' has not been set",
            )

    def get_units(self, default=ValueError()):
        """The units of the array.

        If the units are `None` then the array has no defined units.

        .. versionadded:: TODODASKVER

        :Parameters:

            TODODASKDOCS

        :Returns:

            `str` or `None`

        """
        try:
            return self._get_component("units")
        except AttributeError:
            if default is None:
                return

            return self._default(
                default,
                f"{self.__class__.__name__} 'units' have not been set",
            )
