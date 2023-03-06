import numpy as np

from ....functions import _DEPRECATION_ERROR_ATTRIBUTE

# import cfdm



class FileArrayMixin:  # (cfdm.data.mixin.FileArrayMixin):
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

    #    @property
    #    def dtype(self):
    #        """Data-type of the array."""
    #        return self._get_component("dtype")

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

    #    @property
    #    def shape(self):
    #        """Shape of the array."""
    #        return self._get_component("shape")

    def del_file_location(self, location):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        :Parameters:

            location: `str`
                TODOCFADOCS

        :Returns:

            `{{class}}`
                TODOCFADOCS

        """
        from os import sep
        from os.path import dirname

        location = location.rstrip(sep)

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

        #        if not new_filenames:
        #            raise ValueError(
        #                f"Can't remove location {location} when doing so "
        #                "results in there being no defined files"
        #            )

        a = self.copy()
        a._set_component("filename", tuple(new_filenames), copy=False)
        a._set_component("address", tuple(new_addresses), copy=False)
        return a

    def file_locations(self):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        :Returns:

            `set`
                TODOCFADOCS

        """
        from os.path import dirname

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a file URI.
        return set([dirname(f) for f in self.get_filenames()])

    #    def get_addresses(self):
    #        """TODOCFADOCS Return the names of any files containing the data array.
    #
    #        .. versionadded:: TODOCFAVER
    #
    #        :Returns:
    #
    #            `tuple`
    #                TODOCFADOCS
    #
    #        """
    #        out = self._get_component("address", None)
    #        if not out:
    #            return ()
    #
    #        return (out,)
    #
    #    def get_formats(self):
    #        """Return the format of the file.
    #
    #        .. versionadded:: TODOCFAVER
    #
    #        .. seealso:: `get_format`, `get_filenames`, `get_addresses`
    #
    #        :Returns:
    #
    #            `tuple`
    #                The fragment file formats.
    #
    #        """
    #        return (self.get_format(),)
    #
    #    def open(self):
    #        """Returns an open dataset containing the data array.
    #
    #        When multiple fragment files have been provided an attempt is
    #        made to open each one, in arbitrary order, and the
    #        `netCDF4.Dataset` is returned from the first success.
    #
    #        .. versionadded:: TODOCFAVER
    #
    #        :Returns:
    #
    #            `netCDF4.Dataset`
    #
    #        """
    #        # Loop round the files, returning as soon as we find one that
    #        # works.
    #        filenames = self.get_filenames()
    #        for filename, address in zip(filenames, self.get_addresses()):
    #            url = urlparse(filename)
    #            if url.scheme == "file":
    #                # Convert a file URI into an absolute path
    #                filename = url.path
    #
    #            try:
    #                nc = netCDF4.Dataset(filename, "r")
    #            except FileNotFoundError:
    #                continue
    #            except RuntimeError as error:
    #                raise RuntimeError(f"{error}: {filename}")
    #
    #            if isisntance(address, str):
    #                self._set_component("ncvar", address, copy=False)
    #            else:
    #                self._set_component("varid", address, copy=False)
    #
    #            return nc
    #
    #        if len(filenames) == 1:
    #            raise FileNotFoundError(f"No such netCDF file: {filenames[0]}")
    #
    #        raise FileNotFoundError(f"No such netCDF files: {filenames}")

    def set_file_location(self, location):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        :Parameters:

            location: `str`
                TODOCFADOCS

        :Returns:

            `{{class}}`
                TODOCFADOCS

        """
        from os import sep
        from os.path import basename, dirname, join

        location = location.rstrip(sep)

        filenames = self.get_filenames()
        addresses = self.get_addresses()

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a fully qualified URI.
        new_filenames = []
        new_addresses = []
        for filename, address in zip(filenames, addresses):
            if dirname(filename) == location:
                continue

            new_filename = join(location, basename(filename))
            if new_filename in new_filenames:
                continue

            new_filenames.append(new_filename)
            new_addresses.append(address)

        a = self.copy()
        a._set_component(
            "filename", filenames + tuple(new_filenames), copy=False
        )
        a._set_component(
            "address",
            addresses + tuple(new_addresses),
            copy=False,
        )
        return a
