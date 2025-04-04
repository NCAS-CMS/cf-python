import cfdm

from ...constants import _stash2standard_name
from ...functions import _DEPRECATION_ERROR_ATTRIBUTE, load_stash2standard_name
from ...umread_lib.umfile import File, Rec
from .abstract import Array


class UMArray(
    cfdm.data.mixin.IndexMixin,
    cfdm.data.abstract.FileArray,
    Array,
):
    """A sub-array stored in a PP or UM fields file."""

    def __init__(
        self,
        filename=None,
        address=None,
        dtype=None,
        shape=None,
        fmt=None,
        word_size=None,
        byte_ordering=None,
        mask=True,
        unpack=True,
        attributes=None,
        storage_options=None,
        source=None,
        copy=True,
    ):
        """**Initialisation**

        :Parameters:

            filename: (sequence of) `str`, optional
                The file name(s).

            address: (sequence of) `int`, optional
                The start position in the file(s) of the header(s).

                .. versionadded:: 3.15.0

            dtype: `numpy.dtype`
                The data type of the data array on disk.

            shape: `tuple`
                The shape of the unpacked data array. Note that this
                is the shape as required by the object containing the
                `UMArray` object, and so may contain extra size one
                dimensions. When read, the data on disk is reshaped to
                *shape*.

            fmt: `str`, optional
                ``'PP'`` or ``'FF'``

            word_size: `int`, optional
                ``4`` or ``8``

            byte_ordering: `str`, optional
                ``'little_endian'`` or ``'big_endian'``

            {{init attributes: `dict` or `None`, optional}}

                During the first `__getitem__` call, any of the
                ``_FillValue``, ``add_offset``, ``scale_factor``,
                ``units``, and ``calendar`` attributes which haven't
                already been set will be inferred from the lookup
                header and cached for future use.

                .. versionadded:: 3.16.3

            {{init source: optional}}

            {{init copy: `bool`, optional}}

            size: `int`
                Deprecated at version 3.14.0.

            ndim: `int`
                Deprecated at version 3.14.0.

            header_offset: `int`
                Deprecated at version 3.15.0. Use the *address*
                parameter instead.

            data_offset: `int`, optional
                Deprecated at version 3.15.0.

            disk_length: `int`, optional
                Deprecated at version 3.15.0.

            units: `str` or `None`, optional
                Deprecated at version 3.16.3. Use the
                *attributes* parameter instead.

            calendar: `str` or `None`, optional
                Deprecated at version 3.16.3. Use the
                *attributes* parameter instead.

        """
        super().__init__(
            filename=filename,
            address=address,
            dtype=dtype,
            shape=shape,
            mask=mask,
            unpack=unpack,
            attributes=attributes,
            storage_options=storage_options,
            source=source,
            copy=copy,
        )

        if source is not None:
            try:
                fmt = source._get_component("fmt", None)
            except AttributeError:
                fmt = None

            try:
                word_size = source._get_component("word_size", None)
            except AttributeError:
                word_size = None

            try:
                byte_ordering = source._get_component("byte_ordering", None)
            except AttributeError:
                byte_ordering = None

        if fmt is not None:
            self._set_component("fmt", fmt, copy=False)

        if byte_ordering is not None:
            self._set_component("byte_ordering", byte_ordering, copy=False)

        if word_size is not None:
            self._set_component("word_size", word_size, copy=False)

        # By default, close the UM file after data array access
        self._set_component("close", True, copy=False)

    def _get_array(self, index=None):
        """Returns a subspace of the dataset variable.

        .. versionadded:: 3.16.3

        .. seealso:: `__array__`, `index`

        :Parameters:

            {{index: `tuple` or `None`, optional}}

        :Returns:

            `numpy.ndarray`
                The subspace.

        """
        # Note: No need to lock the UM file - concurrent reads are OK.

        if index is None:
            index = self.index()

        f, header_offset = self.open()
        rec = self._get_rec(f, header_offset)

        int_hdr = rec.int_hdr
        real_hdr = rec.real_hdr
        array = rec.get_data().reshape(self.original_shape)

        self.close(f)
        del f, rec

        # Set the netCDF attributes for the data
        attributes = self.get_attributes({})
        self._set_units(int_hdr, attributes)
        self._set_FillValue(int_hdr, real_hdr, attributes)
        self._set_unpack(int_hdr, real_hdr, attributes)
        self._set_component("attributes", attributes, copy=False)

        # Get the data subspace, applying any masking and unpacking
        array = cfdm.netcdf_indexer(
            array,
            mask=self.get_mask(),
            unpack=self.get_unpack(),
            always_masked_array=False,
            orthogonal_indexing=True,
            attributes=attributes,
            copy=False,
        )
        array = array[index]

        if int_hdr.item(38) == 3:
            # Convert the data to a boolean array
            array = array.astype(bool)

        # Set the data type
        self._set_component("dtype", array.dtype, copy=False)

        # Return the numpy array
        return array

    def _get_rec(self, f, header_offset):
        """Get a container for a record.

        This includes the lookup header and file offsets.

        .. versionadded:: 3.14.0

        .. seealso:: `close`, `open`

        :Parameters:

            f: `umread_lib.umfile.File`
                The open PP or FF file.

            header_offset: `int`

        :Returns:

            `umread_lib.umfile.Rec`
                The record container.

        """
        return Rec.from_file_and_offsets(f, header_offset)

        # ------------------------------------------------------------
        # Leave the following commented code here for debugging
        # purposes. Replacing the above line with this code moves the
        # calculation of the data offset and disk length from pure
        # Python to the C library, at the expense of completely
        # parsing the file. Note: If you do replace the above line
        # with the commented code, then you *must* also set
        # 'parse=True' in the `open` method.
        # ------------------------------------------------------------

        # for v in f.vars:
        #     for r in v.recs:
        #         if r.hdr_offset == header_offset:
        #             return r

    def _set_FillValue(self, int_hdr, real_hdr, attributes):
        """Set the ``_FillValue`` attribute.

        .. versionadded:: 3.16.3

        :Parameters:

            int_hdr: `numpy.ndarray`
                The integer header of the data.

            real_header: `numpy.ndarray`
                The real header of the data.

            attributes: `dict`
                The dictionary in which to store the new
                attributes. If a new attribute exists then
                *attributes* is updated in-place.

        :Returns:

            `None

        """
        if "FillValue" in attributes:
            return

        # Set the fill_value from BMDI
        _FillValue = real_hdr.item(17)
        if _FillValue != -1.0e30:
            # -1.0e30 is the flag for no missing data
            if int_hdr.item(38) == 2:
                # Must have an integer _FillValue for integer data
                _FillValue = int(_FillValue)

            attributes["_FillValue"] = _FillValue

    def _set_units(self, int_hdr, attributes):
        """Set the ``units`` attribute.

        .. versionadded:: 3.14.0

        :Parameters:

            int_hdr: `numpy.ndarray`
                The integer header of the data.

            real_header: `numpy.ndarray`
                The real header of the data.

            attributes: `dict`
                The dictionary in which to store the new
                attributes. If a new attribute exists then
                *attributes* is updated in-place.

        :Returns:

            `None`

        """
        if "units" in attributes:
            return

        units = None
        if not _stash2standard_name:
            load_stash2standard_name()

        submodel = int_hdr.item(44)
        stash = int_hdr.item(41)
        records = _stash2standard_name.get((submodel, stash))
        if records:
            LBSRCE = int_hdr.item(37)
            version, source = divmod(LBSRCE, 10000)
            if version <= 0:
                version = 405.0

            for (
                long_name,
                units0,
                valid_from,
                valid_to,
                standard_name,
                cf_info,
                condition,
            ) in records:
                if not self._test_version(
                    valid_from, valid_to, version
                ) or not self._test_condition(condition, int_hdr):
                    continue

                units = units0
                break

        attributes["units"] = units

    def _set_unpack(self, int_hdr, real_hdr, attributes):
        """Set the ``add_offset`` and ``scale_factor`` attributes.

        .. versionadded:: 3.16.3

        :Parameters:

            int_hdr: `numpy.ndarray`
                The integer header of the data.

            real_header: `numpy.ndarray`
                The real header of the data.

            attributes: `dict`
                The dictionary in which to store the new
                attributes. If any new attributes exist then
                *attributes* is updated in-place.

        :Returns:

            `None

        """
        if "scale_factor" not in attributes:
            # Treat BMKS as a scale_factor if it is neither 0 nor 1
            scale_factor = real_hdr.item(18)
            if scale_factor != 1.0 and scale_factor != 0.0:
                if int_hdr.item(38) == 2:
                    # Must have an integer scale_factor for integer data
                    scale_factor = int(scale_factor)

                attributes["scale_factor"] = scale_factor

        if "add_offset" not in attributes:
            # Treat BDATUM as an add_offset if it is not 0
            add_offset = real_hdr.item(4)
            if add_offset != 0.0:
                if int_hdr.item(38) == 2:
                    # Must have an integer add_offset for integer data
                    add_offset = int(add_offset)

                attributes["add_offset"] = add_offset

    def _test_condition(self, condition, int_hdr):
        """Return `True` if a field satisfies a condition for a STASH
        code to standard name conversion.

        .. versionadded:: 3.14.0

        :Parameters:

            condition: `str`
                The condition. If False then the condition is always
                passed, otherwise the condition is specified as
                ``'true_latitude_longitude'`` or
                ``'rotated_latitude_longitude'``.

            int_hdr: `numpy.ndarray`
                The integer lookup header used to evaluate the
                condition.

        :Returns:

            `bool`
                `True` if the data satisfies the condition specified,
                `False` otherwise.

        """
        if not condition:
            return True

        if condition == "true_latitude_longitude":
            LBCODE = int_hdr.item(15)
            # LBCODE 1: Unrotated regular lat/long grid
            # LBCODE 2 = Regular lat/lon grid boxes (grid points are
            #            box centres)
            if LBCODE in (1, 2):
                return True
        elif condition == "rotated_latitude_longitude":
            LBCODE = int_hdr.item(15)
            # LBCODE 101: Rotated regular lat/long grid
            # LBCODE 102: Rotated regular lat/lon grid boxes (grid
            #             points are box centres)
            # LBCODE 111: ?
            if LBCODE in (101, 102, 111):
                return True
        else:
            return False

    def _test_version(self, valid_from, valid_to, version):
        """Return `True` if the UM version applicable to this field is
        within the given range.

        If possible, the UM version is derived from the PP header and
        stored in the metadata object. Otherwise it is taken from the
        *version* parameter.

        .. versionadded:: 3.14.0

        :Parameters:

            valid_from: number or `None`
                The lower bound of the version range, e.g. ``4.5``,
                ``606.1``, etc.

            valid_to: number or `None`
                The upper bound of the version range, e.g. ``4.5``,
                ``606.1``, etc.

            version: number
                The version of field, e.g. ``4.5``, ``606.1``, etc.

        :Returns:

            `bool`
                `True` if the UM version applicable to this data is
                within the given range, `False` otherwise.

        """
        if valid_to is None:
            if valid_from is None:
                return True

            if valid_from <= version:
                return True
        elif valid_from is None:
            if version <= valid_to:
                return True
        elif valid_from <= version <= valid_to:
            return True

        return False

    @property
    def file_address(self):
        """The file name and address.

        Deprecated at version 3.14.0. Use methods `get_filename`
        and `get_address` instead.

        :Returns:

            `tuple`
                The file name and file address.

        **Examples**

        >>> a.file_address()
        ('file.pp', 234835)

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "file_address",
            "Use methods 'get_filename' and 'get_address' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def header_offset(self):
        """The start position in the file of the header.

        :Returns:

            `int` or `None`
                The address, or `None` if there isn't one.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "header_offset",
            "Use method 'get_address' instead.",
            version="3.15.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def data_offset(self):
        """The start position in the file of the data array.

        :Returns:

            `int`

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "data_offset",
            version="3.15.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def disk_length(self):
        """The number of words on disk for the data array.

        :Returns:

            `int`

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "disk_length",
            version="3.15.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def fmt(self):
        """The file format of the UM file containing the array.

        Deprecated at version 3.14.0. Use method `get_fmt`
        instead.

        :Returns:

            `str`
                'FF' or 'PP'

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "fmt",
            "Use method 'get_fmt' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def byte_ordering(self):
        """The endianness of the data.

        Deprecated at version 3.14.0. Use method
        `get_byte_ordering` instead.

        :Returns:

            `str`
                'little_endian' or 'big_endian'

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "byte_ordering",
            "Use method 'get_byte_ordering' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def word_size(self):
        """Word size in bytes.

        Deprecated at version 3.14.0. Use method `get_word_size`
        instead.

        :Returns:

            `int`
                4 or 8

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "word_size",
            "Use method 'get_word_size' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def close(self, f):
        """Close the dataset containing the data.

        :Parameters:

            f: `umfile_lib.File`
                The UM or PP dataset to be be closed.

                .. versionadded:: 3.14.0

        :Returns:

            `None`

        """
        if self._get_component("close"):
            f.close_fd()

    def get_byte_ordering(self):
        """The endianness of the data.

        .. versionadded:: 3.14.0

        .. seealso:: `open`

        :Returns:

            `str` or `None`
                ``'little_endian'`` or ``'big_endian'``. If the byte
                ordering has not been set then `None` is returned, in
                which case byte ordering will be detected
                automatically (if possible) when the file is opened
                with `open`.

        """
        return self._get_component("byte_ordering", None)

    def get_fmt(self):
        """The file format of the UM file containing the array.

        .. versionadded:: 3.14.0

        .. seealso:: `open`

        :Returns:

            `str` or `None`
                ``'FF'`` or ``'PP'``. If the word size has not been
                set then `None` is returned, in which case file format
                will be detected automatically (if possible) when the
                file is opened with `open`.

        """
        return self._get_component("fmt", None)

    def get_format(self):
        """The format of the files.

        .. versionadded:: 3.15.0

        .. seealso:: `get_address`, `get_filename`, `get_formats`

        :Returns:

            `str`
                The file format. Always ``'um'``, signifying PP/UM.

        **Examples**

        >>> a.get_format()
        'um'

        """
        return "um"

    def get_word_size(self):
        """Word size in bytes.

        .. versionadded:: 3.14.0

        .. seealso:: `open`

        :Returns:

            `int` or `None`
                ``4`` or ``8``. If the word size has not been set then
                `None` is returned, in which case word size will be
                detected automatically (if possible) when the file is
                opened with `open`.

        """
        return self._get_component("word_size", None)

    def open(self):
        """Returns an open dataset and the address of the data.

        :Returns:

            `umfile_lib.umfile.File`, `int`
                The open file object, and the start address in bytes
                of the lookup header.

        **Examples**

        >>> f.open()
        (<cf.umread_lib.umfile.File object at 0x7fdc25056340>, 4)

        """
        return super().open(
            File,
            byte_ordering=self.get_byte_ordering(),
            word_size=self.get_word_size(),
            fmt=self.get_fmt(),
            parse=False,
        )
