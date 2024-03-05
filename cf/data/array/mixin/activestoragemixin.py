try:
    from activestorage import Active
except ModuleNotFoundError:
    Active = None


class ActiveStorageMixin:
    """Mixin class for enabling active storage reductions.

    .. versionadded:: NEXTVERSION

    """

    def __getitem__(self, indices):
        """Returns a subspace of the array as a numpy array.

        x.__getitem__(indices) <==> x[indices]

        The indices that define the subspace must be either `Ellipsis` or
        a sequence that contains an index for each dimension. In the
        latter case, each dimension's index must either be a `slice`
        object or a sequence of two or more integers.

        Indexing is similar to numpy indexing. The only difference to
        numpy indexing (given the restrictions on the type of indices
        allowed) is:

          * When two or more dimension's indices are sequences of integers
            then these indices work independently along each dimension
            (similar to the way vector subscripts work in Fortran).

        .. versionadded:: NEXTVERSION

        """
        method = self.get_active_method()
        if Active is None or method is None:
            # The instance has not been actified so do a normal read,
            # returning an un-reduced numpy array.
            return super().__getitem__(indices)

        # Still here? Then do an active storage reduction. Returns a
        # dictionary of reduced values.
        filename = self.get_filename()

        active = Active(
            filename,
            self.get_address(),
            storage_options=self.get_storage_options(),
            active_storage_url=self.get_active_storage_url(),
            storage_type="s3",  # Temporary requirement!
        )
        active.method = method
        active.components = True

        # Provide a file lock
        try:
            active.lock = self._lock
        except AttributeError:
            pass

        return active[indices]

    def actify(self, method, axis=None, active_storage_url=None):
        """Return a new actified `{{class}}` instance.

        The new instance is a deep copy of the original, with the
        additional setting of the active storage method and axis.

        When the instance is indexed, the result of applying the
        active storage method to the subspace will be returned.

        .. versionadded:: NEXTVERSION

        .. seealso:: `get_active_axis`, `get_active_method`

        :Parameters:

            method: `str`
                The name of the reduction method.

                *Parameter example:*
                  ``'min'``

            axis: `None` or (sequence of) `int`, optional
                Axis or axes along which to operate. By default, or if
                `None`, flattened input is used.

            active_storage_url: `str` or `None`, optional
                The URL of the active storage server.

        :Returns:

            `{{class}}`
                The new `{{class}}`` instance that ues an active
                storage operation.

        """
        # Don't actify when the data are packed. Note: There may come
        # a time when activestorage.Active can cope with packed data,
        # in which case we can remove this test.
        attributes = self.get_attributes({})
        if "add_offset" in attributes or "scale_factor" in attributes:
            raise AttributeError(
                "Can't actify {self.__class__.__name__} when "
                "the data has been numerically packed"
            )

        if Active is None:
            # Note: We don't really expect to be here because if
            #       activestorage.Active is not available then we
            #       wouldn't even attempt to actify the instance
            #       during a reduction (see
            #       `cf.data.collapse.active_storage`). However, it's
            #       worth checking in case `actify` is called by the
            #       user.
            raise AttributeError(
                "Can't actify {self.__class__.__name__} when "
                "activestorage.Active is not available"
            )

        a = self.copy()
        a._custom["active_method"] = method
        a._custom["active_axis"] = axis
        a._custom["active_storage_url"] = active_storage_url
        return a

    def get_active_axis(self):
        """Return the active storage reduction axes.

        Active storage reduction axes are set with `actify`.

        .. versionadded:: NEXTVERSION

        .. seealso:: `actify`, `get_active_method`,
                     `get_active_storage_url`

        :Returns:

            `None` or (sequence of) `int`
                The active storage reduction axes, or `None` if there
                is no active storage reduction.

        """
        return self._custom.get("active_axis")

    def get_active_method(self):
        """Return the name of the active storage reduction method.

        An active storage reduction method is set with `actify`.

        .. versionadded:: NEXTVERSION

        .. seealso:: `actify`, `get_active_axis`,
                     `get_active_storage_url`

        :Returns:

            `str` or `None`
                The name of the active storage reduction method, or
                `None` if there is no active storage reduction.

        """
        return self._custom.get("active_method")

    def get_active_storage_url(self):
        """Return the active storage reduction URL.

        An active storage reduction URL is set with `actify`.

        .. versionadded:: NEXTVERSION

        .. seealso:: `actify`, `get_active_axis`, `get_active_method`

        :Returns:

            `str` or `None`
                The active storage URL, or `None` if there is no
                active storage reduction.

        """
        return self._custom.get("active_storage_url")
