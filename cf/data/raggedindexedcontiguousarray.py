import cfdm


class RaggedIndexedContiguousArray(cfdm.RaggedIndexedContiguousArray):
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

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented


#    @property
#    def dask_asarray(self):
#        return False
