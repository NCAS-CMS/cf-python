class CompressedArray:
    """TODODASK."""

    @property
    def dask_lock(self):
        return getattr(self._get_compressed_Array(), "dask_lock", False)

    @property
    def dask_asarray(self):
        return False


# -- End; class
