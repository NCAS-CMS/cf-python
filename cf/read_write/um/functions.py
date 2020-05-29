from ..constants import _file_to_fh
from ..functions import (open_files_threshold_exceeded,  # abspath
                         close_one_file)

from .umread.umfile import File, UMFileException

_file_to_UM = _file_to_fh.setdefault('UM', {})


def _open_um_file(filename, aggregate=True, fmt=None, word_size=None,
                  byte_ordering=None):
    '''Open a UM fields file or PP file and read it into a
    `umread.umfile.File` object.

    If there is already a `umread.umfile.File` object for the file
    then it is returned with an open file descriptor.

    :Parameters:

        filename: `str`
            The file to be opened.

    :Returns:

        `umread.umfile.File`
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
        f = File(filename,
                 byte_ordering=byte_ordering,
                 word_size=word_size,
                 format=fmt)
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

        filename: `str`
            The file to be closed.

    :Returns:

        `None`

    **Examples:**

    '''
    f = _file_to_UM.pop(filename, None)
    if f is not None:
        f.close_fd()
