class FragmentFileArrayMixin:
    """Mixin class for a fragment array stored in a file.

    .. versionadded:: TODOCFAVER

    """

    def del_fragment_location(self, location):
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

        a = self.copy()
        location = location.rstrip(sep)

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a file URI.
        new_filenames = []
        new_addresses = []
        for filename, address in zip(a.get_filenames(), a.get_addresses()):
            if dirname(filename) != location:
                new_filenames.append(filename)
                new_addresses.append(address)

        a._set_component("filenames", new_filenames, copy=False)
        a._set_component("addresses", new_addresses, copy=False)

        return a

    def fragment_locations(self):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        :Parameters:

            location: `str`
                TODOCFADOCS

        :Returns:

            `{{class}}`
                TODOCFADOCS

        """
        from os.path import dirname

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a file URI.
        return set([dirname(f) for f in self.get_filenames()])

    def get_addresses(self, default=AttributeError()):
        """TODOCFADOCS Return the names of any files containing the data array.

        .. versionadded:: TODOCFAVER

        :Returns:

            `tuple`
                TODOCFADOCS

        """
        return self._get_component("addresses", default)

    def get_filenames(self, default=AttributeError()):
        """TODOCFADOCS Return the names of any files containing the data array.

        .. versionadded:: TODOCFAVER

        :Returns:

            `tuple`
                The fragment file names.

        """
        filenames = self._get_component("filenames", None)
        if filenames is None:
            if default is None:
                return

            return self._default(
                default, f"{self.__class__.__name__} has no fragement files"
            )

        return filenames

    def get_formats(self, default=AttributeError()):
        """Return the format of each fragment file.

        .. versionadded:: TODOCFAVER

        .. seealso:: `get_filenames`, `get_addresses`

        :Returns:

            `tuple`
                The fragment file formats.

        """
        return (self.get_format(),) * len(self.get_filenames(default))

    def set_fragment_location(self, location):
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

        a = self.copy()
        location = location.rstrip(sep)

        filenames = a.get_filenames()
        addresses = a.get_addresses()

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a fully qualified URI.
        new_filenames = []
        new_addresses = []
        basenames = []
        for filename, address in zip(filenames, addresses):
            if dirname(filename) == location:
                continue

            basename = basename(filename)
            if basename in basenames:
                continue

            basenames.append(filename)
            new_filenames.append(join(location, basename))
            new_addresses.append(address)

        a._set_component(
            "filenames", filenames + tuple(new_filenames), copy=False
        )
        a._set_component(
            "addresses",
            addresses + tuple(new_addresses),
            copy=False,
        )

        return a
