import dask.array as da


class CompressedArrayMixin:
    """Mixin class for compressed arrays.

    .. versionadded:: 3.14.0

    """

    def _lock_file_read(self, array):
        """Try to return a dask array that does not support concurrent
        reads.

        .. versionadded:: 3.14.0

        :Parameters:

            array: array_like
                The array to process.

        :Returns"

            `dask.array.Array` or array_like
                The new `dask` array, or the orginal array if it
                couldn't be ascertained how to form the `dask` array.

        """
        try:
            return array.to_dask_array()
        except AttributeError:
            pass

        try:
            chunks = array.chunks
        except AttributeError:
            chunks = "auto"

        try:
            array = array.source()
        except (ValueError, AttributeError):
            pass

        try:
            array.get_filenames()
        except AttributeError:
            pass
        else:
            array = da.from_array(array, chunks=chunks, lock=True)

        return array

    def to_dask_array(self, chunks="auto"):
        """Convert the data to a `dask` array.

        .. versionadded:: 3.14.0

        :Parameters:

            chunks: `int`, `tuple`, `dict` or `str`, optional
                Specify the chunking of the returned dask array.

                Any value accepted by the *chunks* parameter of the
                `dask.array.from_array` function is allowed.

                The chunk sizes implied by *chunks* for a dimension that
                has been fragmented are ignored and replaced with values
                that are implied by that dimensions fragment sizes.

        :Returns:

            `dask.array.Array`
                The `dask` array representation.

        """
        from functools import partial

        import dask.array as da
        from dask import config
        from dask.array.core import getter
        from dask.base import tokenize

        from ...utils import normalize_chunks

        name = (f"{self.__class__.__name__}-{tokenize(self)}",)

        dtype = self.dtype

        context = partial(config.set, scheduler="synchronous")

        # If possible, convert the compressed data to a dask array
        # that doesn't support concurrent reads. This prevents
        # "compute called by compute" failures problems at compute
        # time.
        #
        # TODO: This won't be necessary if this is refactored so that
        #       the compressed data is part of the same dask graph as
        #       the compressed subarrays.
        conformed_data = self.conformed_data()
        conformed_data = {
            k: self._lock_file_read(v) for k, v in conformed_data.items()
        }
        subarray_kwargs = {**conformed_data, **self.subarray_parameters()}

        # Get the (cfdm) subarray class
        Subarray = self.get_Subarray()
        subarray_name = Subarray().__class__.__name__

        # Set the chunk sizes for the dask array
        chunks = normalize_chunks(
            self.subarray_shapes(chunks),
            shape=self.shape,
            dtype=dtype,
        )

        dsk = {}
        for u_indices, u_shape, c_indices, chunk_location in zip(
            *self.subarrays(chunks)
        ):
            subarray = Subarray(
                indices=c_indices,
                shape=u_shape,
                context_manager=context,
                **subarray_kwargs,
            )

            key = f"{subarray_name}-{tokenize(subarray)}"
            dsk[key] = subarray
            dsk[name + chunk_location] = (getter, key, Ellipsis, False, False)

        # Return the dask array
        return da.Array(dsk, name[0], chunks=chunks, dtype=dtype)
