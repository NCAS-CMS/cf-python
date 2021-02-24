import cfdm

from . import abstract

from .functions import _open_netcdf_file, _close_netcdf_file


class NetCDFArray(cfdm.NetCDFArray, abstract.FileArray):
    """A sub-array stored in a netCDF file."""

    def __init__(
        self,
        filename=None,
        ncvar=None,
        varid=None,
        group=None,
        dtype=None,
        ndim=None,
        shape=None,
        size=None,
        mask=True,
    ):
        """**Initialization**

        :Parameters:

            filename: `str`
                The name of the netCDF file containing the array.

            ncvar: `str`, optional

                The name of the netCDF variable containing the
                array. Required unless *varid* is set.

            varid: `int`, optional
                The UNIDATA netCDF interface ID of the variable
                containing the array. Required if *ncvar* is not set,
                ignored if *ncvar* is set.

            group: `None` or sequence of `str`, optional
                Specify the netCDF4 group to which the netCDF variable
                belongs. By default, or if *group* is `None` or an
                empty sequence, it assumed to be in the root
                group. The last element in the sequence is the name of
                the group in which the variable lies, with other
                elements naming any parent groups (excluding the root
                group).

                :Parameter example:
                  To specify that a variable is in the root group:
                  ``group=()`` or ``group=None``

                :Parameter example:
                  To specify that a variable is in the group
                  '/forecasts': ``group=['forecasts']``

                :Parameter example:
                  To specify that a variable is in the group
                  '/forecasts/model2': ``group=['forecasts',
                  'model2']``

                .. versionadded:: 3.6.0

            dtype: `numpy.dtype`
                The data type of the array in the netCDF file. May be
                `None` if the numpy data-type is not known (which can
                be the case for netCDF string types, for example).

            shape: `tuple`
                The array dimension sizes in the netCDF file.

            size: `int`
                Number of elements in the array in the netCDF file.

            ndim: `int`
                The number of array dimensions in the netCDF file.

            mask: `bool`, optional
                If False then do not mask by convention when reading
                data from disk. By default data is masked by
                convention.

                A netCDF array is masked depending on the values of
                any of the netCDF variable attributes ``valid_min``,
                ``valid_max``, ``valid_range``, ``_FillValue`` and
                ``missing_value``.

                .. versionadded:: 3.4.0

        **Examples:**

        >>> import netCDF4
        >>> nc = netCDF4.Dataset('file.nc', 'r')
        >>> v = nc.variable['tas']
        >>> a = NetCDFFileArray(filename='file.nc', ncvar='tas',
        ...                     group=['forecast'], dtype=v.dtype,
        ...                     ndim=v.ndim, shape=v.shape, size=v.size)

        """
        super().__init__(
            filename=filename,
            ncvar=ncvar,
            varid=varid,
            group=group,
            dtype=dtype,
            ndim=ndim,
            shape=shape,
            size=size,
            mask=mask,
        )

    #        # By default, keep the netCDF file open after data array
    #        # access
    #        self._set_component('close', False, copy=False)
    #        self._set_component('close', True, copy=False)

    @property
    def dask_lock(self):
        """TODODASK.

        Concurrent reads are supported, because __getitem__ opens its
        own independent netCDF4.Dataset instance.

        """
        return False

    @property
    def file_pointer(self):
        """The file pointer starting at the position of the netCDF
        variable."""
        offset = getattr(self, "ncvar", None)
        if offset is None:
            offset = self.varid

        return (self.get_filename(), offset)


#    def close(self):
#        """Close the file containing the data array.
#
#    If the file is not open then no action is taken.
#
#    :Returns:
#
#        `None`
#
#    **Examples:**
#
#    >>> f.close()
#
#        """
#        _close_netcdf_file(self.get_filename())
#
#    def open(self):
#        """Return a `netCDF4.Dataset` object for the file containing the data
#    array.
#
#    :Returns:
#
#        `netCDF4.Dataset`
#
#    **Examples:**
#
#    >>> f.open()
#    <netCDF4.Dataset at 0x115a4d0>
#
#        """
#        return _open_netcdf_file(self.get_filename(), 'r')

# --- End: class

# abstract.Array.register(NetCDFArray)
