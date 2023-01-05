import cfdm

from ...mixin_container import Container
from .mixin import ArrayMixin, RaggedArrayMixin


class RaggedIndexedContiguousArray(
    RaggedArrayMixin, ArrayMixin, Container, cfdm.RaggedIndexedContiguousArray
):
    """An underlying indexed contiguous ragged array.

    A collection of features, each of which is sequence of (vertical)
    profiles, stored using an indexed contiguous ragged array combines
    all feature elements along a single dimension (the "sample
    dimension") such that a contiguous ragged array representation is
    used for each profile and the indexed ragged array representation
    to organise the profiles into timeseries.

    The information needed to uncompress the data is stored in a
    "count variable" that gives the size of each profile; and in a
    "index variable" that specifies the feature that each profile
    belongs to.

    It is assumed that the compressed dimensions are the two left-most
    dimensions in the compressed array.

    See CF section 9 "Discrete Sampling Geometries".

    .. versionadded:: 3.0.0

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        .. versionadded:: 3.0.0

        """
        return super().__repr__().replace("<", "<CF ", 1)
