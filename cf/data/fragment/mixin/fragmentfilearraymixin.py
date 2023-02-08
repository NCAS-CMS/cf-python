class FragmentFileArrayMixin:
    """Mixin class for a fragment array stored in a file.

    .. versionadded:: TODOCFAVER

    """

    def get_address(self):
        """The address of the fragment in the file.

        .. versionadded:: TODOCFAVER

        :Returns:

                The file address of the fragment, or `None` if there
                isn't one.

        """
        try:
            return self.get_array().get_address()
        except AttributeError:
            return

    def get_filename(self):
        """Return the name of the file containing the array.

        .. versionadded:: TODOCFAVER

        :Returns:

            `str` or `None`
                The filename, or `None` if there isn't one.

        **Examples**

        >>> a.get_filename()
        'file.nc'

        """
        try:
            return self.get_array().get_filename()
        except AttributeError:
            return
