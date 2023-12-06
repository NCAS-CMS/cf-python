import cfdm

from ...mixin_container import Container
from .mixin import ArrayMixin, CompressedArrayMixin


class CellConnectivityArray(
    CompressedArrayMixin,
    ArrayMixin,
    Container,
    cfdm.CellConnectivityArray,
):
    """A connectivity array derived from a UGRID connectivity variable.

    A UGRID connectivity variable contains indices which map each cell
    to its neighbours, as found in a UGRID "face_face_connectivity" or
    "volume_volume_connectivity" variable.

    The connectivity array has one more column than the corresponding
    UGRID variable. The extra column, in the first position, contains
    the identifier for each cell.

    .. versionadded:: 3.16.0

    """
