class ArrayMixin:
    """Mixin class for a container of an array.

    .. versionadded:: TODODASKVER

    """

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented
