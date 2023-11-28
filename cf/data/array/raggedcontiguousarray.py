import cfdm

from ...mixin_container import Container
from .mixin import ArrayMixin, CompressedArrayMixin


class RaggedContiguousArray(
    CompressedArrayMixin, ArrayMixin, Container, cfdm.RaggedContiguousArray
):
    """An underlying contiguous ragged array.

    A collection of features stored using a contiguous ragged array
    combines all features along a single dimension (the "sample
    dimension") such that each feature in the collection occupies a
    contiguous block.

    The information needed to uncompress the data is stored in a
    "count variable" that gives the size of each block.

    It is assumed that the compressed dimension is the left-most
    dimension in the compressed array.

    See CF section 9 "Discrete Sampling Geometries".

    .. versionadded:: 3.0.0

    """
