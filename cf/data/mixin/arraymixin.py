class ArrayMixin:
    """Mixin class for a container of an array.

    .. versionadded:: TODODASK

    """

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented
