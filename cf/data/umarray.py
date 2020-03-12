import numpy

from ..constants import _file_to_fh
from ..functions import (parse_indices,
                         get_subspace)
from .functions import _open_um_file, _close_um_file

from ..umread_lib.umfile import Rec

from . import abstract


class UMArray(abstract.FileArray):
    '''A sub-array stored in a PP or UM fields file.

    '''
    def __init__(self, filename=None, dtype=None, ndim=None,
                 shape=None, size=None, header_offset=None,
                 data_offset=None, disk_length=None, fmt=None,
                 word_size=None, byte_ordering=None):
        '''**Initialization**

    :Parameters:

        filename: `str`
            The file name in normalized, absolute form.

        dtype: `numpy.dtype`
            The data type of the data array on disk.

        ndim: `int`
            The number of dimensions in the unpacked data array.

        shape: `tuple`
            The shape of the unpacked data array.

        size: `int`
            The number of elements in the unpacked data array.

        header_offset: `int`
            The start position in the file of the header.

        data_offset: `int`
            The start position in the file of the data array.

        disk_length: `int`
            The number of words on disk for the data array, usually
            LBLREC-LBEXT. If set to 0 then `!size` is used.

        fmt: `str`, optional

        word_size: `int`, optional

        byte_ordering: `str`, optional

    **Examples:**

    >>> a = UMFileArray(file='file.pp', header_offset=3156, data_offset=3420,
    ...                 dtype=numpy.dtype('float32'), shape=(30, 24),
    ...                 size=720, ndim=2, disk_length=0)

    >>> a = UMFileArray(
    ...         file='packed_file.pp', header_offset=3156, data_offset=3420,
    ...         dtype=numpy.dtype('float32'), shape=(30, 24),
    ...         size=720, ndim=2, disk_length=423
    ...     )

    '''
        super().__init__(filename=filename, dtype=dtype, ndim=ndim,
                         shape=shape, size=size,
                         header_offset=header_offset,
                         data_offset=data_offset,
                         disk_length=disk_length, fmt=fmt,
                         word_size=word_size,
                         byte_ordering=byte_ordering)

        # By default, do not close the UM file after data array access
        self._close = False

    def __getitem__(self, indices):
        '''Implement indexing

    x.__getitem__(indices) <==> x[indices]

    Returns a numpy array.

        '''
        f = self.open()

        rec = Rec.from_file_and_offsets(
            f, self.header_offset, self.data_offset, self.disk_length)

        int_hdr = rec.int_hdr
        real_hdr = rec.real_hdr

        array = rec.get_data().reshape(int_hdr.item(17,), int_hdr.item(18,))

        if indices is not Ellipsis:
            indices = parse_indices(array.shape, indices)
            array = get_subspace(array, indices)

        LBUSER2 = int_hdr.item(38,)

        if LBUSER2 == 3:
            # Return the numpy array now if it is a boolean array
            return array.astype(bool)

        integer_array = LBUSER2 == 2

        # ------------------------------------------------------------
        # Convert to a masked array
        # ------------------------------------------------------------
        # Set the fill_value from BMDI
        fill_value = real_hdr.item(17,)
        if fill_value != -1.0e30:
            # -1.0e30 is the flag for no missing data
            if integer_array:
                # The fill_value must be of the same type as the data
                # values
                fill_value = int(fill_value)

            # Mask any missing values
            mask = (array == fill_value)
            if mask.any():
                array = numpy.ma.masked_where(mask, array, copy=False)
        # --- End: if

        # ------------------------------------------------------------
        # Unpack the array using the scale_factor and add_offset, if
        # either is available
        # ------------------------------------------------------------
        # Treat BMKS as a scale_factor if it is neither 0 nor 1
        scale_factor = real_hdr.item(18,)
        if scale_factor != 1.0 and scale_factor != 0.0:
            if integer_array:
                scale_factor = int(scale_factor)
            array *= scale_factor

        # Treat BDATUM as an add_offset if it is not 0
        add_offset = real_hdr.item(4,)
        if add_offset != 0.0:
            if integer_array:
                add_offset = int(add_offset)
            array += add_offset

        # Return the numpy array
        return array

    def __str__(self):
        '''x.__str__() <==> str(x)

        '''
        return "%s%s in %s" % (self.header_offset, self.shape, self.filename)

    @property
    def file_pointer(self):
        '''TODO

        '''
        return (self.filename, self.header_offset)

    @property
    def header_offset(self):
        '''TODO

        '''
        return self._get_component('header_offset')

    @property
    def data_offset(self):
        '''TODO

        '''
        return self._get_component('data_offset')

    @property
    def disk_length(self):
        '''TODO

        '''
        return self._get_component('disk_length')

    @property
    def fmt(self):
        '''TODO

        '''
        return self._get_component('fmt')

    @property
    def byte_ordering(self):
        '''TODO

        '''
        return self._get_component('byte_ordering')

    @property
    def word_size(self):
        '''TODO

        '''
        return self._get_component('word_size')

    def close(self):
        '''Close the file containing the data array.

    If the file is not open then no action is taken.

    :Returns:

        `None`

    **Examples:**

    >>> f.close()

        '''
        _close_um_file(self.filename)

    def open(self):
        '''Open the file containing the data array.

    :Returns:

        `umfile_lib.File`

    **Examples:**

    >>> f.open()

        '''
        return _open_um_file(self.filename,
                             fmt=self.fmt,
                             word_size=self.word_size,
                             byte_ordering=self.byte_ordering)


# --- End: class
