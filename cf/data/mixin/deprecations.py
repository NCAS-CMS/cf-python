from ...functions import (
    _DEPRECATION_ERROR_ATTRIBUTE,
    _DEPRECATION_ERROR_METHOD,
)


class DataClassDeprecationsMixin:
    """Deprecated attributes and methods for the Data class."""

    @property
    def Data(self):
        """Deprecated at version 3.0.0, use attribute `data` instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "Data", "Use attribute 'data' instead."
        )  # pragma: no cover

    @property
    def dtvarray(self):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_ATTRIBUTE(self, "dtvarray")  # pragma: no cover

    def files(self):
        """Deprecated at version 3.4.0, use method `get_` instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "files",
            "Use method `get_filenames` instead.",
            version="3.4.0",
        )  # pragma: no cover

    @property
    def unsafe_array(self):
        """Deprecated at version 3.0.0, use `array` attribute
        instead."""
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, "unsafe_array", "Use 'array' attribute instead."
        )  # pragma: no cover

    def expand_dims(self, position=0, i=False):
        """Deprecated at version 3.0.0, use method `insert_dimension`
        instead."""
        _DEPRECATION_ERROR_METHOD(
            self,
            "expand_dims",
            "Use method 'insert_dimension' instead.",
            version="3.0.0",
        )  # pragma: no cover

    @property
    def ispartitioned(self):
        """True if the data array is partitioned.

        **Examples:**

        >>> d._pmsize
        1
        >>> d.ispartitioned
        False

        >>> d._pmsize
        2
        >>> d.ispartitioned
        False

        """
        _DEPRECATION_ERROR_METHOD("TODODASK")

    def chunk(self, chunksize=None, total=None, omit_axes=None, pmshape=None):
        """Partition the data array.

        :Parameters:

            chunksize: `int`, optional
                The

            total: sequence of `int`, optional

            omit_axes: sequence of `int`, optional

            pmshape: sequence of `int`, optional

        :Returns:

            `None`

        **Examples:**

        >>> d.chunk()
        >>> d.chunk(100000)
        >>> d.chunk(100000, )
        >>> d.chunk(100000, total=[2])
        >>> d.chunk(100000, omit_axes=[3, 4])

        """
        _DEPRECATION_ERROR_METHOD("TODODASK. Use 'rechunk' instead")

    @property
    def ismasked(self):
        """True if the data array has any masked values.

        TODODASK

        **Examples:**

        >>> d = cf.Data([[1, 2, 3], [4, 5, 6]])
        >>> print(d.ismasked)
        False
        >>> d[0, ...] = cf.masked
        >>> d.ismasked
        True

        """
        _DEPRECATION_ERROR_METHOD("TODODASK use is_masked instead")

    @property
    def varray(self):
        """A numpy array view the data array.

        Note that making changes to elements of the returned view changes
        the underlying data.

        .. seealso:: `array`, `datetime_array`

        **Examples:**

        >>> a = d.varray
        >>> type(a)
        <type 'numpy.ndarray'>
        >>> a
        array([0, 1, 2, 3, 4])
        >>> a[0] = 999
        >>> d.varray
        array([999, 1, 2, 3, 4])

        """
        _DEPRECATION_ERROR_METHOD("TODODASK")

    def add_partitions(self, extra_boundaries, pdim):
        """Add partition boundaries.

        :Parameters:

            extra_boundaries: `list` of `int`
                The boundaries of the new partitions.

            pdim: `str`
                The name of the axis to have the new partitions.

        :Returns:

            `None`

        **Examples:**

        >>> d.add_partitions(    )

        """
        _DEPRECATION_ERROR_METHOD("TODODASK Consider using rechunk instead")

    def partition_boundaries(self):
        """Return the partition boundaries for each partition matrix
        dimension.

        :Returns:

            `dict`

        **Examples:**

        """
        _DEPRECATION_ERROR_METHOD("TODODASK - consider using 'chunks' instead")
