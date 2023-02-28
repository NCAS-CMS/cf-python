class FragmentFileArrayMixin:
    """Mixin class for a fragment array stored in a file.

    .. versionadded:: TODOCFAVER

    """

    def add_fragment_location(self, location):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        :Parameters:

            location: `str`
                TODOCFADOCS

        :Returns:

            `{{class}}`
                TODOCFADOCS

        """
        from os.path import basename, dirname, join

        a = self.copy()

        # Note: It is assumed that each existing file name is either
        #       an absolute path or a file URI.
        filenames = a.get_filenames()
        addresses = a.get_addresses()

        new_filenames = tuple(
            [
                join(location, basename(f))
                for f in filenames
                if dirname(f) != location
            ]
        )

        a._set_component("filenames", filenames + new_filenames, copy=False)
        a._set_component(
            "addresses",
            addresses + addresses[-1] * len(new_filenames),
            copy=False,
        )

        return a

    def get_addresses(self, default=AttributeError()):
        """TODOCFADOCS Return the names of any files containing the data array.

        .. versionadded:: TODOCFAVER

        :Returns:

            `tuple`
                The file names in normalised, absolute
                form. TODOCFADOCS then an empty `set` is returned.

        """
        return self._get_component("addresses", default)

    def get_filenames(self, default=AttributeError()):
        """TODOCFADOCS Return the names of any files containing the data array.

        .. versionadded:: TODOCFAVER

        :Returns:

            `tuple`
                The file names in normalised, absolute
                form. TODOCFADOCS then an empty `set` is returned.

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
        """TODOCFADOCS Return the names of any files containing the data array.

        .. versionadded:: TODOCFAVER

        .. seealso:: `get_filenames`, `get_addresses`

        :Returns:

            `tuple`
                The file names in normalised, absolute
                form. TODOCFADOCS then an empty `set` is returned.

        """
        return (self.get_format(),) * len(self.get_filenames(default))
