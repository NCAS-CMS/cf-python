class CompressedArray:
    '''TODO

    '''
    _dask_asarray = False

    @property
    def _dask_lock(self):
        return getattr(self._get_compressed_Array(), "_dask_lock", False)

# -- End; class
