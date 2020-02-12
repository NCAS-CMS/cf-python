import cfdm

from . import abstract


class RaggedIndexedArray(abstract.CompressedArray,
                         cfdm.RaggedIndexedArray):
    '''An underlying indexed ragged array.

    A collection of features stored using an indexed ragged array
    combines all features along a single dimension (the "sample
    dimension") such that the values of each feature in the collection
    are interleaved.

    The information needed to uncompress the data is stored in an
    "index variable" that specifies the feature that each element of
    the sample dimension belongs to.

    .. versionadded:: 3.0.0

    '''
# --- End: class
