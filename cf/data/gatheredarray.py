import cfdm

from .mixin import ArrayMixin


class GatheredArray(ArrayMixin, cfdm.GatheredArray):
    """An underlying gathered array.

    Compression by gathering combines axes of a multidimensional array
    into a new, discrete axis whilst omitting the missing values and
    thus reducing the number of values that need to be stored.

    The information needed to uncompress the data is stored in a "list
    variable" that gives the indices of the required points.

    See CF section 8.2. "Lossless Compression by Gathering".

    .. versionadded:: 3.0.0

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        .. versionadded:: 3.0.0

        """
        return super().__repr__().replace("<", "<CF ", 1)

    def to_dask_array(self, chunks="auto"):
        """Create a dask array TODODASKDOCS

        .. versionadded:: TODODASKVER
    
        :Parameters:
    
            chunks: `int`, `tuple`, `dict` or `str`, optional
                Specify the chunking of the returned dask array.
    
                Any value accepted by the *chunks* parameter of the
                `dask.array.from_array` function is allowed.
    
                The chunk sizes implied by *chunks* for a dimension that
                has been fragemented are ignored and replaced with values
                that are implied by that dimensions fragment sizes.
    
        :Returns:
    
            `dask.array.Array`
    
        """
        name = (f"{self.__class__.__name__}-{tokenize(self)}",)

        dtype = self.dtype
        
        context = partial(config.set, scheduler="synchronous")
        
        compressed_dimensions = self.compressed_dimensions()
        conformed_data = self.conformed_data()
        compressed_data = conformed_data["data"]
        uncompressed_indices = conformed_data["uncompressed_indices"]

        units = self.get_units()
        calendar = self.get_calendar()
        aggregated_data = self.get_aggregated_data(copy=False)
    
        # Set the chunk sizes for the dask array
        chunks = self.subarray_shapes(chunks)
        chunks = normalize_chunks(
            array.subarray_shapes(chunks),
            shape=self.shape,
            dtype=dtype,
        )

        dsk = {}
        for u_indices, u_shape, c_indices, chunk_location in zip(
            *self.subarrays(chunks)
        ):
            subarray = Subarray(
                data=compressed_data,
                indices=c_indices,
                shape=u_shape,
                compressed_dimensions=compressed_dimensions,
                uncompressed_indices=uncompressed_indices,
                context_manager=context,
            )

            dsk[name + chunk_location] = (
                getter,
                subarray,
                Ellipsis,
                False,
                False,
            )

        # Return the dask array
        return da.Array(dsk, name[0], chunks=chunks, dtype=dtype)
