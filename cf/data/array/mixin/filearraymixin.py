from os.path import dirname

import numpy as np

from ....functions import _DEPRECATION_ERROR_ATTRIBUTE, abspath


class FileArrayMixin:
    """Mixin class for an array stored in a file.

    .. versionadded:: 3.14.0

    """

    @property
    def _dask_meta(self):
        """The metadata for the containing dask array.

        This is the kind of array that will result from slicing the
        file array.

        .. versionadded:: 3.14.0

        .. seealso:: `dask.array.from_array`

        """
        return np.array((), dtype=self.dtype)

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return f"<CF {self.__class__.__name__}{self.shape}: {self}>"

    def __str__(self):
        """x.__str__() <==> str(x)"""
        return f"{self.get_filename()}, {self.get_address()}"

    @property
    def filename(self):
        """The name of the file containing the array.

        Deprecated at version 3.14.0. Use method `get_filename` instead.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "filename",
            message="Use method 'get_filename' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    def del_file_location(self, location):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        :Parameters:

            location: `str`
                TODOCFADOCS

        :Returns:

            `{{class}}`
                TODOCFADOCS

        **Examples**

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file2')
        >>> a.get_addresses()
        ('tas1', 'tas2')
        >>> b = a.del_file_location('/data1')
        >>> b = get_filenames()
        ('/data2/file2',)
        >>> b.get_addresses()
        ('tas2',)

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file1', '/data2/file2')
        >>> a.get_addresses()
        ('tas1', 'tas1', 'tas2')
        >>> b = a.del_file_location('/data2')
        >>> b.get_filenames()
        ('/data1/file1',)
        >>> b.get_addresses()
        ('tas1',)

        """
        from os import sep

        location = abspath(location).rstrip(sep)

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a file URI.
        new_filenames = []
        new_addresses = []
        for filename, address in zip(
            self.get_filenames(), self.get_addresses()
        ):
            if dirname(filename) != location:
                new_filenames.append(filename)
                new_addresses.append(address)

        if not new_filenames:
            raise ValueError("TODOCFADOCS")

        a = self.copy()
        a._set_component("filename", tuple(new_filenames), copy=False)
        a._set_component("address", tuple(new_addresses), copy=False)
        return a

    def file_locations(self):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        :Returns:

            `tuple`
                TODOCFADOCS

        **Examples**

        >>> a.get_filenames()
        ('/data1/file1',)
        >>> a.file_locations()
        ('/data1,)

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file2')
        >>> a.file_locations()
        ('/data1', '/data2')

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file2', '/data1/file2')
        >>> a.file_locations()
        ('/data1', '/data2', '/data1')

        """
        return tuple(map(dirname, self.get_filenames()))

    def set_file_location(self, location):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        :Parameters:

            location: `str`
                TODOCFADOCS

        :Returns:

            `{{class}}`
                TODOCFADOCS

        **Examples**

        >>> a.get_filenames()
        ('/data1/file1',)
        >>> a.get_addresses()
        ('tas',)
        >>> b = a.set_file_location('/home/user')
        >>> b.get_filenames()
        ('/data1/file1', '/home/user/file1')
        >>> b.get_addresses()
        ('tas', 'tas')

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file2',)
        >>> a.get_addresses()
        ('tas', 'tas')
        >>> b = a.set_file_location('/home/user')
        >>> b = get_filenames()
        ('/data1/file1', '/data2/file2', '/home/user/file1', '/home/user/file2')
        >>> b.get_addresses()
        ('tas', 'tas', 'tas', 'tas')

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file1',)
        >>> a.get_addresses()
        ('tas1', 'tas2')
        >>> b = a.set_file_location('/home/user')
        >>> b.get_filenames()
        ('/data1/file1', '/data2/file1', '/home/user/file1')
        >>> b.get_addresses()
        ('tas1', 'tas2', 'tas1')

        >>> a.get_filenames()
        ('/data1/file1', '/data2/file1',)
        >>> a.get_addresses()
        ('tas1', 'tas2')
        >>> b = a.set_file_location('/data1')
        >>> b.get_filenames()
        ('/data1/file1', '/data2/file1')
        >>> b.get_addresses()
        ('tas1', 'tas2')

        """
        from os import sep
        from os.path import basename, join

        location = abspath(location).rstrip(sep)

        filenames = self.get_filenames()
        addresses = self.get_addresses()

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a fully qualified URI.
        new_filenames = list(filenames)
        new_addresses = list(addresses)
        for filename, address in zip(filenames, addresses):
            new_filename = join(location, basename(filename))
            if new_filename not in new_filenames:
                new_filenames.append(new_filename)
                new_addresses.append(address)

        a = self.copy()
        a._set_component("filename", tuple(new_filenames), copy=False)
        a._set_component(
            "address",
            tuple(new_addresses),
            copy=False,
        )
        return a
