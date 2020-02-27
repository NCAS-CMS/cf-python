import cfdm

from . import abstract


class RaggedContiguousArray(abstract.CompressedArray,
                            cfdm.RaggedContiguousArray):
    '''An underlying contiguous ragged array.

    A collection of features stored using a contiguous ragged array
    combines all features along a single dimension (the "sample
    dimension") such that each feature in the collection occupies a
    contiguous block.

    The information needed to uncompress the data is stored in a
    "count variable" that gives the size of each block.

    .. versionadded:: 3.0.0

    '''
# --- End: class
