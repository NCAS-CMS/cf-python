import numpy as np


class FileArrayMixin:
    """Mixin class for an array stored in a file.

    .. versionadded:: TODODASKVER

    """

    @property
    def _dask_meta(self):
        """The metadata for the containing dask array.

        This is the kind of array that will result from slicing the
        file array.

        .. versionadded:: TODODASKVER

        .. seealso:: `dask.array.from_array`

        """
        return np.array((), dtype=self.dtype)
