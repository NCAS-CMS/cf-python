# REVIEW: active: `ActiveStorageMixin`: new mixin class `ActiveStorageMixin`


class ActiveStorageMixin:
    """Mixin class for enabling active storage operations.

    .. versionadded:: NEXTVERSION

    """

    @property
    def active_storage(self):
        """Whether active storage operations are allowed.

        Currently, active storage operations are allowed unless the
        data are numerically packed.

        .. versionadded:: NEXTVERSION

        :Returns:

            `bool`
                `True` if active storage operations are allowed,
                otherwise `False`.

        """
        attributes = self.get_attributes({})
        if "add_offset" in attributes or "scale_factor" in attributes:
            return False

        return True


#        return self.get_filename(None) is not None

#    @property
#    def actified(self):
#        """Whether active storage operations are possible.
#
#        .. versionadded:: NEXTVERSION
#
#        .. seealso:: `actify`, `get_active_storage_url`
#
#        :Returns:
#
#            `bool`
#                `True` if active stoage operations are possible,
#                otherwise `False`.
#
#        """
#        return self.get_active_storage_url() is not None
#
#    def actify(self, active_storage_url):
#        """Return a new actified `{{class}}` instance.
#
#        The new instance is a deep copy of the original, with the
#        additional setting of the active storage URL.
#
#        .. versionadded:: NEXTVERSION
#
#        .. seealso:: `actified`, `get_active_storage_url`
#
#        :Parameters:
#
#            active_storage_url: `str` or `None`, optional
#                The URL of the active storage server. If `None` then
#                `actified` will be `False`
#
#        :Returns:
#
#            `{{class}}`
#                The new `{{class}}`` instance that ues an active
#                storage operation.
#
#        """
#        # Don't actify when the data are packed. Note: There may come
#        # a time when activestorage.Active can cope with packed data,
#        # in which case we can remove this test.
#        attributes = self.get_attributes({})
#        if "add_offset" in attributes or "scale_factor" in attributes:
#            raise AttributeError(
#                "Can't actify {self.__class__.__name__} when "
#                "the data have been numerically packed"
#            )
#
#        if Active is None:
#            raise AttributeError(
#                "Can't actify {self.__class__.__name__} when "
#                "activestorage.Active is not available"
#            )
#
#        a = self.copy()
#        a._custom["active_storage_url"] = active_storage_url
#        return a
#
#    def get_active_storage_url(self):
#        """Return the active storage reduction URL.
#
#        An active storage reduction URL is set with `actify`.
#
#        .. versionadded:: NEXTVERSION
#
#        .. seealso:: `actified`, `actify`
#
#        :Returns:
#
#            `str` or `None`
#                The active storage URL, or `None` if no active storage
#                reduction is possible.
#
#        **Examples**
#
#        >>> a.get_active_storage()
#        'https://183.175.143.286:8080'
#
#        """
#        return self._custom.get("active_storage_url")
