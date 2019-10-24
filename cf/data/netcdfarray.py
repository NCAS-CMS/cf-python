import cfdm

from . import abstract

from .functions import _open_netcdf_file, _close_netcdf_file


class NetCDFArray(cfdm.NetCDFArray,
                  abstract.FileArray):
    '''A sub-array stored in a netCDF file.
    '''
    def __init__(self, filename=None, ncvar=None, varid=None,
                 dtype=None, ndim=None, shape=None, size=None):
        '''**Initialization**

    :Parameters:
    
        filename: `str`
            The name of the netCDF file containing the array.
    
        ncvar: `str`, optional
            The name of the netCDF variable containing the array. Required
            unless *varid* is set.
    
        varid: `int`, optional
            The UNIDATA netCDF interface ID of the variable containing the
            array. Required if *ncvar* is not set, ignored if *ncvar* is
            set.
    
        dtype: `numpy.dtype`
            The data type of the array in the netCDF file.
    
        shape: `tuple`
            The array dimension sizes in the netCDF file.
    
        size: `int`
            Number of elements in the array in the netCDF file.
    
        ndim: `int`
            The number of array dimensions in the netCDF file.
    
    **Examples:**
    
    >>> import netCDF4
    >>> nc = netCDF4.Dataset('file.nc', 'r')
    >>> v = nc.variable['tas']
    >>> a = NetCDFFileArray(filename='file.nc', ncvar='tas', dtype=v.dtype, 
    ...                     ndim=v.ndim, shape=v.shape, size=v.size)
    
    '''
        super().__init__(filename=filename, ncvar=ncvar, varid=varid,
                         dtype=dtype, ndim=ndim, shape=shape,
                         size=size)

        # By default, keep the netCDF file open after data array
        # access
        self._set_component('close', False, copy=False)

    
    @property
    def file_pointer(self):
        '''TODO

        '''
        offset = getattr(self, 'ncvar', None)
        if offset is None:
            offset = self.varid

        return (self.get_filename(), offset)


    def close(self):
        '''Close the file containing the data array.

    If the file is not open then no action is taken.
    
    :Returns:
    
        `None`
    
    **Examples:**
    
    >>> f.close()
    
        '''
        _close_netcdf_file(self.get_filename())


    def open(self):
        '''Return a `netCDF4.Dataset` object for the file containing the data
    array.
    
    :Returns:
    
        `netCDF4.Dataset`
    
    **Examples:**
    
    >>> f.open()
    <netCDF4.Dataset at 0x115a4d0>

        '''
        return _open_netcdf_file(self.get_filename(), 'r')


#--- End: class

#abstract.Array.register(NetCDFArray)
