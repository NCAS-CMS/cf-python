import cfdm

from ...mixin_container import Container
from .mixin import ArrayMixin, CompressedArrayMixin


class PointTopologyArray(
    CompressedArrayMixin,
    ArrayMixin,
    Container,
    cfdm.PointTopologyArray,
):
    """A point cell domain topology array derived from a UGRID variable.

    A point cell domain topology array derived from an underlying
    UGRID "edge_node_connectivity" or UGRID "face_node_connectivity"
    array.

    .. versionadded:: 3.16.0

    """
