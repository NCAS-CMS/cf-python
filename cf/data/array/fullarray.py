import cfdm

from ...mixin_container import Container


class FullArray(Container, cfdm.FullArray):
    """A array filled with a given value.

    The array may be empty or all missing values.

    .. versionadded:: 3.14.0

    """
