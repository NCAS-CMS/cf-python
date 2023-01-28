import cfdm

from ..mixin import ArrayMixin


class Array(ArrayMixin, cfdm.Array):
    """Abstract base class for a container of an underlying array.

    The form of the array is defined by the initialisation parameters
    of a subclass.

    .. versionadded:: 3.0.0

    """
