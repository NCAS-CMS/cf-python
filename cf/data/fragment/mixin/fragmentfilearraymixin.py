from ....decorators import (
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
)


class FragmentFileArrayMixin:
    """Mixin class for a fragment array stored in a file.

    .. versionadded:: TODOCFAVER

    """

    @_inplace_enabled(default=False)
    def add_fragment_location(location, inplace=False):
        """TODOCFADOCS"""
        from os.path import  basename, dirname, join
        
        a = _inplace_enabled_define_and_cleanup(self)

        # Note - it is assumed that all filenames are absolute paths
        filenames = a.get_filenames()
        addresses = a.get_addresses()
        
        new_filenames = tuple([join(location, basename(f))
                               for f in filenames
                               if dirname(f) != location])

        a._set_component('filename', filenames + new_filenames, copy=False)
        a._set_component(
            'address',
            addresses + addresses[-1] * len(new_filenames),
            copy=False
        )
        
        return a

    def get_addresses(self):
        """TODOCFADOCS Return the names of any files containing the data array.

        .. versionadded:: TODOCFAVER

        :Returns:

            `tuple`
                The file names in normalised, absolute
                form. TODOCFADOCS then an empty `set` is returned.

        """
        try:
            return self._get_component("address")
        except ValueError:
            return ()
        
    def get_filenames(self):
        """TODOCFADOCS Return the names of any files containing the data array.

        .. versionadded:: TODOCFAVER

        :Returns:

            `tuple`
                The file names in normalised, absolute
                form. TODOCFADOCS then an empty `set` is returned.

        """
        try:
            return self._get_component("filename")
        except ValueError:
            return ()
        
    def get_formats(self):
        """TODOCFADOCS Return the names of any files containing the data array.

        .. versionadded:: TODOCFAVER

        :Returns:

            `set`
                The file names in normalised, absolute
                form. TODOCFADOCS then an empty `set` is returned.

        """
        raise NotImplementedError
