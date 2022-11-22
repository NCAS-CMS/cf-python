import numpy as np


class FileArrayMixin:
    """Mixin class for an array stored in a file.

    .. versionadded:: TODODASKVER

    """

    @property
    def _dask_lock(self):
        """TODODASKDOCS.

        Concurrent reads are assumed to be not supported.

        .. versionadded:: TODODASKVER

        """
        return True

    @property
    def _dask_meta(self):
        """TODODASKDOCS.

        .. versionadded:: TODODASKVER

        """
        return np.array((), dtype=self.dtype)
