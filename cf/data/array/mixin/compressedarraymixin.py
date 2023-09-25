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
