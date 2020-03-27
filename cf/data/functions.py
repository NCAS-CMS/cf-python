from os.path import isfile

from netCDF4 import Dataset as netCDF4_Dataset

from ..constants import _file_to_fh
from ..functions import open_files_threshold_exceeded, close_one_file

# from ..read_write.umread_lib.umfile import File #, UMFileException
from ..umread_lib.umfile import File  # , UMFileException

# if 'netCDF' not in _file_to_fh:
#     _file_to_fh['netCDF'] = {}

_file_to_UM = _file_to_fh.setdefault('UM', {})
_file_to_Dataset = _file_to_fh.setdefault('netCDF', {})


def _open_netcdf_file(filename, mode, fmt='NETCDF4'):  # set_auto_mask=True):
    '''Open a netCDF file and read it into a netCDF4.Dataset object.

    If the file is already open then the existing netCDF4.Dataset
    object will be returned.

    :Parameters:

        filename: `str`
            The netCDF file to be opened.

    #    set_auto_mask: `bool`, optional
    #        Turn on or off automatic conversion of variable data to and
    #        from masked arrays.

    :Returns:

        `netCDF4.Dataset`
            A netCDF4.Dataset instance for the netCDF file.

    **Examples:**

    >>> nc1 = _open_netcdf_file('file.nc')
    >>> nc1
    <netCDF4.Dataset at 0x1a7dad0>
    >>> nc2 = _open_netcdf_file('file.nc')
    >>> nc2 is nc1
    True

    '''
    if filename in _file_to_Dataset and mode == 'r':
        # File is already open
        return _file_to_Dataset[filename]
    elif open_files_threshold_exceeded():
        # Close a random data file to make way for this one
        close_one_file()

    if mode in ('a', 'r+'):
        if not isfile(filename):
            nc = netCDF4_Dataset(filename, 'w', format=fmt)
            nc.close()
        elif filename in _file_to_Dataset:
            _close_netcdf_file(filename)

    try:
        nc = netCDF4_Dataset(filename, mode, format=fmt)
    except RuntimeError as runtime_error:
        raise RuntimeError("{0}: {1}".format(runtime_error, filename))

    if mode == 'r':
        # Update the _file_to_Dataset dictionary
        _file_to_Dataset[filename] = nc

    return nc


def _close_netcdf_file(filename):
    '''Close a netCDF file

    Does nothing if the file is already closed.

    :Parameters:

        filename: `str`
            The netCDF file to be closed.

    :Returns:

        `None`

    '''
    nc = _file_to_Dataset.pop(filename, None)
    if nc is not None:
        nc.close()


def _open_um_file(filename, aggregate=True, fmt=None, word_size=None,
                  byte_ordering=None):
    '''Open a UM fields file or PP file and read it into a `umfile.File`
    object.

    If there is already a `umfile.File` object for the file then it is
    returned with an open file descriptor.

    :Parameters:

        filename: `str`
            The file to be opened.

    :Returns:

        `umfile.File`
            The opened file with an open file descriptor.

    **Examples:**

    '''
#    filename = abspath(filename)

    f = _file_to_UM.get(filename)

    if f is not None:
        if f.fd is None:
            if open_files_threshold_exceeded():
                # Close a random data array file to make way for this
                # one
                close_one_file()

            f.open_fd()
        # --- End: if
        return f

    if open_files_threshold_exceeded():
        # Close a random data array file to make way for this one
        close_one_file()

    try:
        f = File(filename, byte_ordering=byte_ordering,
                 word_size=word_size, format=fmt)
    except Exception as error:
        try:
            f.close_fd()
        except:
            pass
        raise Exception(error)

    # Add a close method to the file object
    f.close = f.close_fd

    # Update the _file_to_UM dictionary
    _file_to_UM[filename] = f

    return f


def _close_um_file(filename):
    '''Close a PP or UM fields file.

    Does nothing if the file is already closed.

    :Parameters:

        filename : str
            The file to be closed.

    :Returns:

        None

    **Examples:**

    '''
    f = _file_to_UM.pop(filename, None)
    if f is not None:
        f.close_fd()
