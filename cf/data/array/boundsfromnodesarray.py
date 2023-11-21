import cfdm

from ...mixin_container import Container
from .mixin import ArrayMixin, CompressedArrayMixin


class BoundsFromNodesArray(
    CompressedArrayMixin,
    ArrayMixin,
    Container,
    cfdm.BoundsFromNodesArray,
):
    """An array of cell bounds defined by UGRID node coordinates.

    The UGRID node coordinates contain the locations of the nodes of
    the domain topology. In UGRID, the bounds of edge, face and volume
    cells may be defined by these locations in conjunction with a
    mapping from each cell boundary vertex to its corresponding
    coordinate value.

    .. versionadded:: 3.16.0

    """
