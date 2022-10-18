from itertools import accumulate, product
from numbers import Number

from .fragment import NetCDFFragmentArray


class AggregatedData(Something):
    """An array stored in a netCDF file."""

    def __new__(cls, *args, **kwargs):
        """Store fragment array classes.

        .. versionadded:: TODODASKVER

        """
        instance = super().__new__(cls)
        instance._FragmentArray = {"nc": NetCDFFragmentArray,
                                   'um': None}
        return instance

    def __init__(self, aggregated_data, fragment_shape, shape, dtype,
                 units, fmt=None):

        """
        aggregated_data = {(0, 0, 0, 0): {'shape': (6, 1, 73, 144), 
                                          'file': "sdas",
                                          'address: "temp",
                                          'format': "nc", # optional
                           }     
        fragement_shape = [2, 1, 1, 1]   
        """
        pass
    
    def get_FragmentArray(self, fmt=None):
        """Return the Fragment class.

        .. versionadded:: (cfdm) 1.10.0.0

        :Returns:

            `FragmentArray`
                The class for representing fragment arrays.

        """
        if fmt is None:
            fmt = self.get_fmt(None)

        if fmt is None:
            raise IndexError(
                "TODOCan't get Subarray class when no interpolation_name "
                "TODOhas been set"
            )

        FragmentArray = self._FragmentArray.get(fmt)
        if FragmentArray is None:
            raise ValueError(
                "Can't get FrragmentArray class for unknown "
                f"fragment dataset format: {fmt!r}"
            )

        return FragmentArray
    
    def get_fragmented_dimensions(self):
        return [i for i, size in enumerate(self.fragment_shape) if size > 1]
    
    def subarray_shapes(self, shapes):
        """Create the subarray shapes along each uncompressed dimension.

        .. versionadded:: (cfdm) 1.10.0.0

        .. seealso:: `subarray`

        :Parameters:

            {{subarray_shapes chunks: `int`, sequence, `dict`, or `str`, optional}}

        :Returns:

            `list`
                The subarray sizes along each uncompressed dimension.

        >>> a.shape
        (4, 20, 30)
        >>> a.compressed_dimensions()
        {1: (1,), 2: (2,)}
        >>> a.subarray_shapes(-1)
        [(4,), None, None]
        >>> a.subarray_shapes("auto")
        ["auto", None, None]
        >>> a.subarray_shapes(2)
        [2, None, None]
        >>> a.subarray_shapes("60B")
        ["60B", None, None]
        >>> a.subarray_shapes((2, None, None))
        [2, None, None]
        >>> a.subarray_shapes(((1, 3), None, None))
        [(1, 3), None, None]
        >>> a.subarray_shapes(("auto", None, None))
        ["auto", None, None]
        >>> a.subarray_shapes(("60B", None, None))
        ["60B", None, None]

        """
        # Indices of fragmented dimensions
        f_dims =  self.get_fragmented_dimensions()

        aggregated_data = self.aggregated_data
    
        # Create the dask chunks
        shapes2 = []
        for dim, (n_fragments, size) in enumerate(
                zip(self.fragment_shape, self.shape)
        ):
            if n_fragments == 1:
                # Aggregated dimension 'dim' is spanned by exactly one
                # fragment
                shapes2.append(None)
                continue

            
            # Still here? Then aggregated dimension 'dim' is spanned by
            # more than one fragment.            
            c = []
            location = [0] * ndim
            for j in range(n_fragments):
                location[n] = j
                chunk_size = aggregated_data[tuple(location)]["shape"][n]
                c.append(chunk_size)
                
            shapes2.append(tuple(c))

        if shapes == -1:
            shape = self.shape
            return [shapes2[i] if i in f_dims else shape[i]
                    for i in range(self.ndim)]
            
        if isinstance(shapes, (str, Number)):
            return [shapes2[i] if i in f_dims else shapes
                    for i in range(self.ndim)]

        if isinstance(shapes, dict):
            return [
                shapes2[i] if i in f_dim else shapes[i]
                for i in range(self.ndim)
            ]

        if len(shapes) != self.ndim:
            # chunks is a sequence
            raise ValueError(
                f"Wrong number of 'shapes' elements in {shapes}: "
                f"Got {len(shapes)}, expected {self.ndim}"
            )

        # chunks is a sequence
        return [shapes2[i] if i in f_dims else shapes[i]
                for i in range(self.ndim)]

    def subarrays(self, shapes=-1):
        """Return descriptors for every subarray.

        These descriptors are used during subarray decompression.

        .. versionadded:: (cfdm) 1.10.0.0
        """
        f_dims =  self.get_fragmented_dimensions()

        # The indices of the uncompressed array that correspond to
        # each subarray, the shape of each uncompressed subarray, and
        # the location of each subarray
        locations = []
        u_shapes = []
        u_indices = []
        f_locations = []
        for dim, c in enumerate(self.subarray_shapes(shapes)):
            nc = len(c)
            locations.append(tuple(range(nc)))
            u_shapes.append(c)
            
            if dim in f_dims:
                # no fragmentation along this dimension
                f_locations.append(tuple(range(nc)))
            else:
                f_locations.append((0,) * nc)
               
                
            c = tuple(accumulate((0,) + c))
            u_indices.append([slice(i, j) for i, j in zip(c[:-1], c[1:])])
    
        # The indices of each fragment that correspond to each
        # subarray
        f_indices = []
        for dim, u in enumerate(u_indices):
            if dim in f_dims:
                f_indices.append(u)
            else:
                f_indices.append((slice(None),) * len(u))
                 
        return (
            product(*u_indices),
            product(*u_shapes),
            product(*f_indices),
            product(*locations),
            product(*f_locations), # produces keys in self.aggregated-data
        )

