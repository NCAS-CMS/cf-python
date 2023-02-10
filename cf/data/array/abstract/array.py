import cfdm

from ....mixin_container import Container
from ..mixin import ArrayMixin


class Array(ArrayMixin, Container, cfdm.Array):
    """Abstract base class for a container of an underlying array.

    The form of the array is defined by the initialisation parameters
    of a subclass.

    .. versionadded:: 3.0.0

    """
