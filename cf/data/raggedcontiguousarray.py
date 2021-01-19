import cfdm


class RaggedContiguousArray(cfdm.RaggedContiguousArray):
    """An underlying contiguous ragged array.

    A collection of features stored using a contiguous ragged array
    combines all features along a single dimension (the "sample
    dimension") such that each feature in the collection occupies a
    contiguous block.

    The information needed to uncompress the data is stored in a
    "count variable" that gives the size of each block.

    .. versionadded:: 3.0.0

    """

    def __getitem__(self, indices):
        '''x.__getitem__(indices) <==> x[indices]

    Returns an subspace of the uncompressed data an independent numpy
    array.

    The indices that define the subspace are relative to the
    uncompressed data and must be either `Ellipsis` or a sequence that
    contains an index for each dimension. In the latter case, each
    dimension's index must either be a `slice` object or a sequence of
    two or more integers.

    Indexing is similar to numpy indexing. The only difference to
    numpy indexing (given the restrictions on the type of indices
    allowed) is:

    * When two or more dimension's indices are sequences of integers
      then these indices work independently along each dimension
      (similar to the way vector subscripts work in Fortran).

    .. versionadded:: (cfdm) 1.7.0

        '''
        # ------------------------------------------------------------
        # Method: Uncompress the entire array and then subspace it
        # ------------------------------------------------------------

        compressed_array = self._get_compressed_Array()

        # Find subspace shape
        subspace_shape = None
        
        # Initialise the un-sliced uncompressed array
        uarray = numpy.ma.masked_all(subspace_shape, dtype=self.dtype)

        # --------------------------------------------------------
        # Compression by contiguous ragged array
        #
        # The uncompressed array has dimensions (instance
        # dimension, element dimension).
        # --------------------------------------------------------

        count_array = self.get_count().data.array

        start = 0
        for i, (n, c) in enumerate(count_array, count_array.cumsum()):
            if i "not in" subspace:
                continue
            
            n = int(n)
            if dim1 index is a slice called s:
                if s.stop <= n: # also modofied to reversed slices
                    sample_indices = s

            u_indices = (i,
                         slice(0, sample_indices.stop - sample_indices.start))

            uarray[u_indices] = compressed_array[(sample_indices,)]
            start += n

        return self.get_subspace(uarray, indices, copy=True)

# --- End: class
