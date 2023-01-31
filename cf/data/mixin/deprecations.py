from ...functions import (
    _DEPRECATION_ERROR_ATTRIBUTE,
    _DEPRECATION_ERROR_METHOD,
    DeprecationError,
)


class DataClassDeprecationsMixin:
    """Deprecated attributes and methods for the Data class."""

    def __hash__(self):
        """The built-in function `hash`.

        Deprecated at version 3.14.0. Consider using the
        `cf.hash_array` function instead.

        Generating the hash temporarily realises the entire array in
        memory, which may not be possible for large arrays.

        The hash value is dependent on the data-type and shape of the data
        array. If the array is a masked array then the hash value is
        independent of the fill value and of data array values underlying
        any masked elements.

        The hash value may be different if regenerated after the data
        array has been changed in place.

        The hash value is not guaranteed to be portable across versions of
        Python, numpy and cf.

        :Returns:

            `int`
                The hash value.

        **Examples**

        >>> print(d.array)
        [[0 1 2 3]]
        >>> d.hash()
        -8125230271916303273
        >>> d[1, 0] = numpy.ma.masked
        >>> print(d.array)
        [[0 -- 2 3]]
        >>> hash(d)
        791917586613573563
        >>> d.hardmask = False
        >>> d[0, 1] = 999
        >>> d[0, 1] = numpy.ma.masked
        >>> d.hash()
        791917586613573563
        >>> d.squeeze()
        >>> print(d.array)
        [0 -- 2 3]
        >>> hash(d)
        -7007538450787927902
        >>> d.dtype = float
        >>> print(d.array)
        [0.0 -- 2.0 3.0]
        >>> hash(d)
        -4816859207969696442

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "__hash__",
            message="Consider using 'cf.hash_array' function array instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )

    @property
    def _HDF_chunks(self):
        """The HDF chunksizes.

        Deprecated at version 3.14.0.

        DO NOT CHANGE IN PLACE.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "_HDF_chunks", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover

    @_HDF_chunks.setter
    def _HDF_chunks(self, value):
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "_HDF_chunks", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover

    @_HDF_chunks.deleter
    def _HDF_chunks(self):
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "_HDF_chunks", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover

    @property
    def Data(self):
        """Deprecated at version 3.0.0, use attribute `data` instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "Data",
            "Use attribute 'data' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def dtvarray(self):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "dtvarray", version="3.0.0", removed_at="4.0.0"
        )  # pragma: no cover

    @property
    def in_memory(self):
        """True if the array is retained in memory.

        Deprecated at version 3.14.0.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "in_memory", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover

    @property
    def ismasked(self):
        """True if the data array has any masked values.

        Deprecated at version 3.14.0. Use the `is_masked` attribute
        instead.

        **Examples**

        >>> d = cf.Data([[1, 2, 3], [4, 5, 6]])
        >>> print(d.ismasked)
        False
        >>> d[0, ...] = cf.masked
        >>> d.ismasked
        True

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "ismasked",
            message="Use the 'is_masked' attribute instead",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def isscalar(self):
        """True if the data is a 0-d scalar array.

        Deprecated at version 3.14.0. Use `d.ndim == 0`` instead.

        **Examples**

        >>> d = cf.Data(9, 'm')
        >>> d.isscalar
        True
        >>> d = cf.Data([9], 'm')
        >>> d.isscalar
        False
        >>> d = cf.Data([9, 10], 'm')
        >>> d.isscalar
        False

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "isscalar",
            message="Use 'd.ndim == 0' instead",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @property
    def ispartitioned(self):
        """True if the data array is partitioned.

        Deprecated at version 3.14.0 and is no longer available.

        **Examples**

        >>> d._pmsize
        1
        >>> d.ispartitioned
        False

        >>> d._pmsize
        2
        >>> d.ispartitioned
        False

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "ispartitioned", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover

    @property
    def unsafe_array(self):
        """Deprecated at version 3.0.0.

        Use the `array` attribute instead.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "unsafe_array",
            message="Use the 'array' attribute instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def varray(self):
        """A numpy array view of the data array.

        Deprecated at version 3.14.0. Data are now stored as `dask`
        arrays for which, in general, a numpy array view is not
        robust.

        Note that making changes to elements of the returned view changes
        the underlying data.

        .. seealso:: `array`, `to_dask_array`, `datetime_array`

        **Examples**

        >>> a = d.varray
        >>> type(a)
        <type 'numpy.ndarray'>
        >>> a
        array([0, 1, 2, 3, 4])
        >>> a[0] = 999
        >>> d.varray
        array([999, 1, 2, 3, 4])

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "varray",
            message="Data are now stored as `dask` arrays for which, "
            "in general, a numpy array view is not robust.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def expand_dims(self, position=0, i=False):
        """Deprecated at version 3.0.0, use method `insert_dimension`
        instead.

        May get re-instated at a later version.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "expand_dims",
            "Use method 'insert_dimension' instead.",
            version="3.0.0",
        )  # pragma: no cover

    def files(self):
        """Deprecated at version 3.4.0, consider using method
        `get_filenames` instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "files",
            "Use method `get_filenames` instead.",
            version="3.4.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def fits_in_one_chunk_in_memory(self, itemsize):
        """Return True if the master array is small enough to be
        retained in memory.

        Deprecated at version 3.14.0.

        :Parameters:

            itemsize: `int`
                The number of bytes per word of the master data array.

        :Returns:

            `bool`

        **Examples**

        >>> print(d.fits_one_chunk_in_memory(8))
        False

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "fits_in_one_chunk_in_memory",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def close(self):
        """Close all files referenced by the data array.

        Deprecated at version 3.14.0. All files are now automatically
        closed when not being accessed.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "close",
            "All files are now automatically closed when not being accessed.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def chunk(self, chunksize=None, total=None, omit_axes=None, pmshape=None):
        """Partition the data array.

        Deprecated at version 3.14.0. Use the `rechunk` method
        instead.

        :Parameters:

            chunksize: `int`, optional
                The

            total: sequence of `int`, optional

            omit_axes: sequence of `int`, optional

            pmshape: sequence of `int`, optional

        :Returns:

            `None`

        **Examples**

        >>> d.chunk()
        >>> d.chunk(100000)
        >>> d.chunk(100000, )
        >>> d.chunk(100000, total=[2])
        >>> d.chunk(100000, omit_axes=[3, 4])

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "chunk",
            message="Use the 'rechunk' method instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def dumpd(self):
        """Return a serialisation of the data array.

        Deprecated at version 3.14.0. Consider inspecting the dask
        array returned by `to_dask_array` instead.

        .. seealso:: `loadd`, `loads`

        :Returns:

            `dict`
                The serialisation.

        **Examples**

        >>> d = cf.Data([[1, 2, 3]], 'm')
        >>> d.dumpd()
        {'Partitions': [{'location': [(0, 1), (0, 3)],
                         'subarray': array([[1, 2, 3]])}],
         'units': 'm',
         '_axes': ['dim0', 'dim1'],
         '_pmshape': (),
         'dtype': dtype('int64'),
         'shape': (1, 3)}

        >>> d.flip(1)
        >>> d.transpose()
        >>> d.Units *= 1000
        >>> d.dumpd()
        {'Partitions': [{'units': 'm',
                         'axes': ['dim0', 'dim1'],
                         'location': [(0, 3), (0, 1)],
                         'subarray': array([[1, 2, 3]])}],
        ` 'units': '1000 m',
         '_axes': ['dim1', 'dim0'],
         '_flip': ['dim1'],
         '_pmshape': (),
         'dtype': dtype('int64'),
         'shape': (3, 1)}

        >>> d.dumpd()
        {'Partitions': [{'units': 'm',
                         'location': [(0, 1), (0, 3)],
                         'subarray': array([[1, 2, 3]])}],
         'units': '10000 m',
         '_axes': ['dim0', 'dim1'],
         '_flip': ['dim1'],
         '_pmshape': (),
         'dtype': dtype('int64'),
         'shape': (1, 3)}

        >>> e = cf.Data(loadd=d.dumpd())
        >>> e.equals(d)
        True

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "dumpd",
            message="Consider inspecting the dask array returned "
            "by 'to_dask_array' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def dumps(self):
        """Return a JSON string serialisation of the data array.

        Deprecated at version 3.14.0. Consider inspecting the dask array
        returned by `to_dask_array` instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "dumps",
            message="Consider inspecting the dask array returned "
            "by 'to_dask_array' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def HDF_chunks(self, *chunks):
        """Get or set HDF chunk sizes.

        The HDF chunk sizes may be used by external code that allows
        `Data` objects to be written to netCDF files.

        Deprecated at version 3.14.0 and is no longer available. Use
        the methods `nc_clear_hdf5_chunksizes`, `nc_hdf5_chunksizes`,
        and `nc_set_hdf5_chunksizes` instead.

        .. seealso:: `nc_clear_hdf5_chunksizes`, `nc_hdf5_chunksizes`,
                     `nc_set_hdf5_chunksizes`

        :Parameters:

            chunks: `dict` or `None`, *optional*
                Specify HDF chunk sizes.

                When no positional argument is provided, the HDF chunk
                sizes are unchanged.

                If `None` then the HDF chunk sizes for each dimension
                are cleared, so that the HDF default chunk size value
                will be used when writing data to disk.

                If a `dict` then it defines for a subset of the
                dimensions, defined by their integer positions, the
                corresponding HDF chunk sizes. The HDF chunk sizes are
                set as a number of elements along the dimension.

        :Returns:

            `dict`
                The HDF chunks for each dimension prior to the change,
                or the current HDF chunks if no new values are
                specified. A value of `None` is an indication that the
                default chunk size should be used for that dimension.

        **Examples**

        >>> d = cf.Data(np.arange(30).reshape(5, 6))
        >>> d.HDF_chunks()
        {0: None, 1: None}
        >>> d.HDF_chunks({1: 2})
        {0: None, 1: None}
        >>> d.HDF_chunks()
        {0: None, 1: 2}
        >>> d.HDF_chunks({1:None})
        {0: None, 1: 2}
        >>> d.HDF_chunks()
        {0: None, 1: None}
        >>> d.HDF_chunks({0: 3, 1: 6})
        {0: None, 1: None}
        >>> d.HDF_chunks()
        {0: 3, 1: 6}
        >>> d.HDF_chunks({1: 4})
        {0: 3, 1: 6}
        >>> d.HDF_chunks()
        {0: 3, 1: 4}
        >>> d.HDF_chunks({1: 999})
        {0: 3, 1: 4}
        >>> d.HDF_chunks()
        {0: 3, 1: 999}
        >>> d.HDF_chunks(None)
        {0: 3, 1: 999}
        >>> d.HDF_chunks()
        {0: None, 1: None}

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "HDF_chunks",
            message="Use the methods 'nc_clear_hdf5_chunksizes', "
            "'nc_hdf5_chunksizes', and 'nc_set_hdf5_chunksizes' "
            "instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def loadd(self, d, chunk=True):
        """Reset the data in place from a dictionary serialisation.

        Deprecated at version 3.14.0. Consider inspecting the dask
        array returned by `to_dask_array` instead.

        .. seealso:: `dumpd`, `loads`

        :Parameters:

            d: `dict`
                A dictionary serialisation of a `cf.Data` object, such as
                one as returned by the `dumpd` method.

            chunk: `bool`, optional
                If True (the default) then the reset data array will be
                re-partitioned according the current chunk size, as
                defined by the `cf.chunksize` function.

        :Returns:

            `None`

        **Examples**

        >>> d = Data([[1, 2, 3]], 'm')
        >>> e = Data([6, 7, 8, 9], 's')
        >>> e.loadd(d.dumpd())
        >>> e.equals(d)
        True
        >>> e is d
        False

        >>> e = Data(loadd=d.dumpd())
        >>> e.equals(d)
        True

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "loadd",
            message="Consider inspecting the dask array returned "
            "by 'to_dask_array' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def loads(self, j, chunk=True):
        """Reset the data in place from a string serialisation.

        Deprecated at version 3.14.0. Consider inspecting the dask
        array returned by `to_dask_array` instead.

        .. seealso:: `dumpd`, `loadd`

        :Parameters:

            j: `str`
                A JSON document string serialisation of a `cf.Data` object.

            chunk: `bool`, optional
                If True (the default) then the reset data array will be
                re-partitioned according the current chunk size, as defined
                by the `cf.chunksize` function.

        :Returns:

            `None`

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "loads",
            message="Consider inspecting the dask array returned "
            "by 'to_dask_array' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def add_partitions(self, extra_boundaries, pdim):
        """Add partition boundaries.

        Deprecated at version 3.14.0. Use the `rechunk` method
        instead.

        :Parameters:

            extra_boundaries: `list` of `int`
                The boundaries of the new partitions.

            pdim: `str`
                The name of the axis to have the new partitions.

        :Returns:

            `None`

        **Examples**

        >>> d.add_partitions(    )

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "add_partitions",
            message="Use the 'rechunk' method instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @staticmethod
    def mask_fpe(*arg):
        """Masking of floating-point errors in the results of arithmetic
        operations.

        Deprecated at version 3.14.0. It is currently not possible
        to control how floating-point errors are handled, due to the
        use of `dask` for handling all array manipulations. This may
        change in the future (see
        https://github.com/dask/dask/issues/3245 for more details).

        If masking is allowed then only floating-point errors which would
        otherwise be raised as `FloatingPointError` exceptions are
        masked. Whether `FloatingPointError` exceptions may be raised is
        determined by `cf.Data.seterr`.

        If called without an argument then the current behaviour is
        returned.

        Note that if the raising of `FloatingPointError` exceptions has
        been suppressed then invalid values in the results of arithmetic
        operations may be subsequently converted to masked values with the
        `mask_invalid` method.

        .. seealso:: `cf.Data.seterr`, `mask_invalid`

        :Parameters:

            arg: `bool`, optional
                The new behaviour. True means that `FloatingPointError`
                exceptions are suppressed and replaced with masked
                values. False means that `FloatingPointError` exceptions
                are raised. The default is not to change the current
                behaviour.

        :Returns:

            `bool`
                The behaviour prior to the change, or the current
                behaviour if no new value was specified.

        **Examples:**

        >>> d = cf.Data([0., 1])
        >>> e = cf.Data([1., 2])

        >>> old = cf.Data.mask_fpe(False)
        >>> old = cf.Data.seterr('raise')
        >>> e/d
        FloatingPointError: divide by zero encountered in divide
        >>> e**123456
        FloatingPointError: overflow encountered in power

        >>> old = cf.Data.mask_fpe(True)
        >>> old = cf.Data.seterr('raise')
        >>> e/d
        <CF Data: [--, 2.0] >
        >>> e**123456
        <CF Data: [1.0, --] >

        >>> old = cf.Data.mask_fpe(True)
        >>> old = cf.Data.seterr('ignore')
        >>> e/d
        <CF Data: [inf, 2.0] >
        >>> e**123456
        <CF Data: [1.0, inf] >

        """
        raise DeprecationError(
            "Data method 'mask_fpe' has been deprecated at version 3.14.0 "
            "and is not available.\n\n"
            "It is currently not possible to control how floating-point errors "
            "are handled, due to the use of `dask` for handling all array "
            "manipulations. This may change in the future (see "
            "https://github.com/dask/dask/issues/3245 for more details)."
        )

    def mask_invalid(self, *args, **kwargs):
        """Mask the array where invalid values occur (NaN or inf).

        Deprecated at version 3.14.0. Use the method
        `masked_invalid` instead.

        .. seealso:: `where`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`
                The masked data, or `None` if the operation was
                in-place.

        **Examples**

        >>> d = cf.Data([0, 1, 2])
        >>> e = cf.Data([0, 2, 0])
        >>> f = d / e
        >>> f
        <CF Data(3): [nan, 0.5, inf]>
        >>> f.mask_invalid()
        <CF Data(3): [--, 0.5, --]>

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "mask_invalid",
            message="Use the method 'masked_invalid' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def partition_boundaries(self):
        """Return the partition boundaries for each partition matrix
        dimension.

        Deprecated at version 3.14.0. Consider using the `chunks`
        attribute instead.

        :Returns:

            `dict`

        **Examples**

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "partition_boundaries",
            message="Consider using the 'chunks' attribute instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def save_to_disk(self, itemsize=None):
        """Deprecated."""
        _DEPRECATION_ERROR_METHOD(
            self, "save_to_disk", removed_at="4.0.0"
        )  # pragma: no cover

    def to_disk(self):
        """Store the data array on disk.

        Deprecated at version 3.14.0.

        There is no change to partitions whose sub-arrays are already
        on disk.

        :Returns:

            `None`

        **Examples**

        >>> d.to_disk()

        """
        _DEPRECATION_ERROR_METHOD(
            self, "to_disk", version="3.14.0", removed_at="5.0.0"
        )  # pragma: no cover

    @staticmethod
    def seterr(all=None, divide=None, over=None, under=None, invalid=None):
        """Set how floating-point errors in the results of arithmetic
        operations are handled.

        Deprecated at version 3.14.0. It is currently not possible
        to control how floating-point errors are handled, due to the
        use of `dask` for handling all array manipulations. This may
        change in the future (see
        https://github.com/dask/dask/issues/3245 for more details).

        The options for handling floating-point errors are:

        ============  ========================================================
        Treatment     Action
        ============  ========================================================
        ``'ignore'``  Take no action. Allows invalid values to occur in the
                      result data array.

        ``'warn'``    Print a `RuntimeWarning` (via the Python `warnings`
                      module). Allows invalid values to occur in the result
                      data array.

        ``'raise'``   Raise a `FloatingPointError` exception.
        ============  ========================================================

        The different types of floating-point errors are:

        =================  =================================  =================
        Error              Description                        Default treatment
        =================  =================================  =================
        Division by zero   Infinite result obtained from      ``'warn'``
                           finite numbers.

        Overflow           Result too large to be expressed.  ``'warn'``

        Invalid operation  Result is not an expressible       ``'warn'``
                           number, typically indicates that
                           a NaN was produced.

        Underflow          Result so close to zero that some  ``'ignore'``
                           precision was lost.
        =================  =================================  =================

        Note that operations on integer scalar types (such as int16) are
        handled like floating point, and are affected by these settings.

        If called without any arguments then the current behaviour is
        returned.

        .. seealso:: `cf.Data.mask_fpe`, `mask_invalid`

        :Parameters:

            all: `str`, optional
                Set the treatment for all types of floating-point errors
                at once. The default is not to change the current
                behaviour.

            divide: `str`, optional
                Set the treatment for division by zero. The default is not
                to change the current behaviour.

            over: `str`, optional
                Set the treatment for floating-point overflow. The default
                is not to change the current behaviour.

            under: `str`, optional
                Set the treatment for floating-point underflow. The
                default is not to change the current behaviour.

            invalid: `str`, optional
                Set the treatment for invalid floating-point
                operation. The default is not to change the current
                behaviour.

        :Returns:

            `dict`
                The behaviour prior to the change, or the current
                behaviour if no new values are specified.

        **Examples:**

        Set treatment for all types of floating-point errors to
        ``'raise'`` and then reset to the previous behaviours:

        >>> cf.Data.seterr()
        {'divide': 'warn', 'invalid': 'warn', 'over': 'warn', 'under': 'ignore'}
        >>> old = cf.Data.seterr('raise')
        >>> cf.Data.seterr(**old)
        {'divide': 'raise', 'invalid': 'raise', 'over': 'raise', 'under': 'raise'}
        >>> cf.Data.seterr()
        {'divide': 'warn', 'invalid': 'warn', 'over': 'warn', 'under': 'ignore'}

        Set the treatment of division by zero to ``'ignore'`` and overflow
        to ``'warn'`` without changing the treatment of underflow and
        invalid operation:

        >>> cf.Data.seterr(divide='ignore', over='warn')
        {'divide': 'warn', 'invalid': 'warn', 'over': 'warn', 'under': 'ignore'}
        >>> cf.Data.seterr()
        {'divide': 'ignore', 'invalid': 'warn', 'over': 'ignore', 'under': 'ignore'}

        Some examples with data arrays:

        >>> d = cf.Data([0., 1])
        >>> e = cf.Data([1., 2])

        >>> old = cf.Data.seterr('ignore')
        >>> e/d
        <CF Data: [inf, 2.0] >
        >>> e**12345
        <CF Data: [1.0, inf] >

        >>> cf.Data.seterr(divide='warn')
        {'divide': 'ignore', 'invalid': 'ignore', 'over': 'ignore', 'under': 'ignore'}
        >>> e/d
        RuntimeWarning: divide by zero encountered in divide
        <CF Data: [inf, 2.0] >
        >>> e**12345
        <CF Data: [1.0, inf] >

        >>> old = cf.Data.mask_fpe(False)
        >>> cf.Data.seterr(over='raise')
        {'divide': 'warn', 'invalid': 'ignore', 'over': 'ignore', 'under': 'ignore'}
        >>> e/d
        RuntimeWarning: divide by zero encountered in divide
        <CF Data: [inf, 2.0] >
        >>> e**12345
        FloatingPointError: overflow encountered in power

        >>> cf.Data.mask_fpe(True)
        False
        >>> cf.Data.seterr(divide='ignore')
        {'divide': 'warn', 'invalid': 'ignore', 'over': 'raise', 'under': 'ignore'}
        >>> e/d
        <CF Data: [inf, 2.0] >
        >>> e**12345
        <CF Data: [1.0, --] >

        """
        raise DeprecationError(
            "Data method 'seterr' has been deprecated at version 3.14.0 "
            "and is not available.\n\n"
            "It is currently not possible to control how floating-point errors "
            "are handled, due to the use of `dask` for handling all array "
            "manipulations. This may change in the future (see "
            "https://github.com/dask/dask/issues/3245 for more details)."
        )

    @classmethod
    def reconstruct_sectioned_data(cls, sections, cyclic=(), hardmask=None):
        """Expects a dictionary of Data objects with ordering
        information as keys, as output by the section method when called
        with a Data object. Returns a reconstructed cf.Data object with
        the sections in the original order.

        Deprecated at version 3.14.0 and is no longer available.

        :Parameters:

            sections: `dict`
                The dictionary of `Data` objects with ordering information
                as keys.

        :Returns:

            `Data`
                The resulting reconstructed Data object.

        **Examples**

        >>> d = cf.Data(numpy.arange(120).reshape(2, 3, 4, 5))
        >>> x = d.section([1, 3])
        >>> len(x)
        8
        >>> e = cf.Data.reconstruct_sectioned_data(x)
        >>> e.equals(d)
        True

        """
        raise DeprecationError(
            "Data method 'reconstruct_sectioned_data' has been deprecated "
            "at version 3.14.0 and is no longer available"
        )

    @classmethod
    def concatenate_data(cls, data_list, axis):
        """Concatenates a list of Data objects along the specified axis.

        See cf.Data.concatenate for details.

        In the case that the list contains only one element, that element
        is simply returned.

        :Parameters:

            data_list: `list`
                The list of data objects to concatenate.

            axis: `int`
                The axis along which to perform the concatenation.

        :Returns:

            `Data`
                The resulting single `Data` object.

        """
        raise DeprecationError(
            "Data method 'concatenate_data' has been deprecated at "
            "version 3.14.0 and is no longer available. Use "
            "'concatenate' instead."
        )
