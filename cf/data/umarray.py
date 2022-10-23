import numpy as np

from ..functions import get_subspace, parse_indices
from ..umread_lib.umfile import File, Rec
from .abstract import FileArray


class UMArray(FileArray):
    """A sub-array stored in a PP or UM fields file."""

    def __init__(
        self,
        filename=None,
        dtype=None,
        ndim=None,
        shape=None,
        size=None,
        header_offset=None,
        data_offset=None,
        disk_length=None,
        fmt=None,
        word_size=None,
        byte_ordering=None,
        units=False,
        calendar=None,
        source=None,
        copy=True,
    ):
        """**Initialization**

        :Parameters:

            filename: `str`
                The file name in normalized, absolute form.

            dtype: `numpy.dtype`
                The data type of the data array on disk.

            shape: `tuple`
                The shape of the unpacked data array. Note that this
                is the shape as required by the object containing the
                `UMArray` object, and so may contain extra size one
                dimensions. When read, the data on disk is reshaped to
                *shape*.

            header_offset: `int`
                The start position in the file of the header.

            data_offset: `int`
                The start position in the file of the data array.

            disk_length: `int`
                The number of words on disk for the data array,
                usually LBLREC-LBEXT. If set to 0 then `!size` is
                used.

            fmt: `str`, optional

            word_size: `int`, optional

            byte_ordering: `str`, optional

            size: `int`
                Deprecated at version TODODASKVER. If set will be
                ignored.

                Number of elements in the uncompressed array.

            ndim: `int`
                Deprecated at version TODODASKVER. If set will be
                ignored.

                The number of uncompressed array dimensions.

        **Examples**

        >>> a = UMFileArray(file='file.pp', header_offset=3156,
        ...                 data_offset=3420,
        ...                 dtype=numpy.dtype('float32'),
        ...                 shape=(1, 1, 30, 24),
        ...                 disk_length=0)

        >>> a = UMFileArray(
        ...         file='packed_file.pp', header_offset=3156,
        ...         data_offset=3420,
        ...         dtype=numpy.dtype('float32'), shape=(30, 24),
        ...         disk_length=423
        ... )

        """
        super().__init__(source=source, copy=copy)

        if source is not None:
            try:
                shape = source._get_component("shape", None)
            except AttributeError:
                shape = None

            try:
                filename = source._get_component("filename", None)
            except AttributeError:
                filename = None

            try:
                fmt = source._get_component("fmt", None)
            except AttributeError:
                fmt = None

            try:
                disk_length = source._get_component("disk_length", None)
            except AttributeError:
                disk_length = None

            try:
                header_offset = source._get_component("header_offset", None)
            except AttributeError:
                header_offset = None

            try:
                data_offset = source._get_component("data_offset", None)
            except AttributeError:
                data_offset = None

            try:
                dtype = source._get_component("dtype", None)
            except AttributeError:
                dtype = None

            try:
                word_size = source._get_component("word_size", None)
            except AttributeError:
                word_size = None

            try:
                byte_ordering = source._get_component("byte_ordering", None)
            except AttributeError:
                byte_ordering = None

        self._set_component("shape", shape, copy=False)
        self._set_component("filename", filename, copy=False)
        self._set_component("dtype", dtype, copy=False)
        self._set_component("header_offset", header_offset, copy=False)
        self._set_component("data_offset", data_offset, copy=False)
        self._set_component("disk_length", disk_length, copy=False)

        if fmt is not None:
            self._set_component("fmt", fmt, copy=False)

        if byte_ordering is not None:
            self._set_component("byte_ordering", byte_ordering, copy=False)

        if word_size is not None:
            self._set_component("word_size", word_size, copy=False)

        # By default, close the UM file after data array access
        self._set_component("close", True, copy=False)

    def __getitem__(self, indices):
        """Return a subspace of the array.

        x.__getitem__(indices) <==> x[indices]

        Returns a subspace of the array as an independent numpy array.

        """
        f = self.open()

        rec = self._get_rec(f)

        int_hdr = rec.int_hdr
        real_hdr = rec.real_hdr

        self.close(f)
        del f

        array = rec.get_data().reshape(self.shape)

        if indices is not Ellipsis:
            indices = parse_indices(array.shape, indices)
            array = get_subspace(array, indices)

        LBUSER2 = int_hdr.item(38)

        if LBUSER2 == 3:
            # Return the numpy array now if it is a boolean array
            return array.astype(bool)

        integer_array = LBUSER2 == 2

        # ------------------------------------------------------------
        # Convert to a masked array
        # ------------------------------------------------------------
        # Set the fill_value from BMDI
        fill_value = real_hdr.item(17)
        if fill_value != -1.0e30:
            # -1.0e30 is the flag for no missing data
            if integer_array:
                # The fill_value must be of the same type as the data
                # values
                fill_value = int(fill_value)

            # Mask any missing values
            mask = array == fill_value
            if mask.any():
                array = np.ma.masked_where(mask, array, copy=False)

        # ------------------------------------------------------------
        # Unpack the array using the scale_factor and add_offset, if
        # either is available
        # ------------------------------------------------------------
        # Treat BMKS as a scale_factor if it is neither 0 nor 1
        scale_factor = real_hdr.item(18)
        if scale_factor != 1.0 and scale_factor != 0.0:
            if integer_array:
                scale_factor = int(scale_factor)

            array *= scale_factor

        # Treat BDATUM as an add_offset if it is not 0
        add_offset = real_hdr.item(4)
        if add_offset != 0.0:
            if integer_array:
                add_offset = int(add_offset)

            array += add_offset

        # Return the numpy array
        return array

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        out = super().__repr__()
        return out[:-1] + f", {self.header_offset}>"

    def _get_rec(self, f):
        """TODODASKDOCS.

        .. versionadded:: TODODASKVER

        :Parameters:

            f: `umread_lib.umfile.File`
                TODODASKDOCS

        :Returns:

            `umread_lib.umfile.Rec`
                TODODASKDOCS

        """
        header_offset = self.header_offset
        data_offset = self.data_offset
        disk_length = self.disk_length
        if data_offset is None or disk_length is None:
            # This method doesn't require data_offset and disk_length,
            # so plays nicely with CFA. Is it fast enough that we can
            # use this method always?
            for v in f.vars:
                for r in v.recs:
                    if r.hdr_offset == header_offset:
                        return r
        else:
            return Rec.from_file_and_offsets(
                f, header_offset, data_offset, disk_length
            )

    @property
    def file_address(self):
        """The file name and address.

        .. versionadded:: ???

        :Returns:

            `tuple`
                The file name and file address.

        **Examples**

        >>> a.file_address()
        ('file.pp', 234835)

        """
        return (self.filename, self.header_offset)

    @property
    def header_offset(self):
        """The start position in the file of the header.

        :Returns:

            `int`

        """
        return self._get_component("header_offset")

    @property
    def data_offset(self):
        """The start position in the file of the data array.

        :Returns:

            `int`

        """
        return self._get_component("data_offset")

    @property
    def disk_length(self):
        """The number of words on disk for the data array.

        :Returns:

            `int`

        """
        return self._get_component("disk_length")

    @property
    def fmt(self):
        """The file format of the UM file containing the array.

        :Returns:

            `str`
                'FF' or 'PP'

        """
        return self._get_component("fmt")

    @property
    def byte_ordering(self):
        """The endianness of the data.

        :Returns:

            `str`
                'little_endian' or 'big_endian'

        """
        return self._get_component("byte_ordering")

    @property
    def word_size(self):
        """Word size in bytes.

        :Returns:

            `int`
                4 or 8

        """
        return self._get_component("word_size")

    def close(self, f):
        """Close the dataset containing the data.

        :Parameters:

            f: `umfile_lib.File`
                The UM or PP dataset to be be closed.

                .. versionadded:: TODODASKVER

        :Returns:

            `None`

        """
        if self._get_component("close"):
            f.close_fd()

    def get_address(self):
        """TODODASKDOCS

        :Returns:

            `str` or `None`
                TODODASKDOCS, or `None` if there isn't one.

        """
        return self.header_offset

    def open(self):
        """Returns an open dataset containing the data array.

        :Returns:

            `umfile_lib.File`

        **Examples**

        >>> f.open()

        """
        try:
            f = File(
                path=self.filename,
                byte_ordering=self.byte_ordering,
                word_size=self.word_size,
                fmt=self.fmt,
            )
        except Exception as error:
            try:
                f.close_fd()
            except Exception:
                pass

            raise Exception(error)
        else:
            return f
