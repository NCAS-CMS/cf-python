import cfdm

from ...mixin_container import Container
from .mixin import ArrayMixin, FileArrayMixin


class AggregatedArray(
    FileArrayMixin,
    ArrayMixin,
    Container,
    cfdm.AggregatedArray
):
    """An array stored in a CF aggregation variable.

    .. versionadded:: NEXTVERSION

    """
