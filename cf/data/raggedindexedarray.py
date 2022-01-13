import cfdm


class RaggedIndexedArray(cfdm.RaggedIndexedArray):
    """An underlying indexed ragged array.

    A collection of features stored using an indexed ragged array
    combines all features along a single dimension (the "sample
    dimension") such that the values of each feature in the collection
    are interleaved.

    The information needed to uncompress the data is stored in an
    "index variable" that specifies the feature that each element of
    the sample dimension belongs to.

    It is assumed that the compressed dimension is the left-most
    dimension in the compressed array.

    See CF section 9 "Discrete Sampling Geometries".

    .. versionadded:: 3.0.0

    """

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented


#    @property
#    def dask_asarray(self):
#        return False
