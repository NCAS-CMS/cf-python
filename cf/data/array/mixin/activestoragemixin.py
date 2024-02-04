try:
    from activestorage import Active
except ModuleNotFoundError:
    Active = None


class ActiveStorageMixin:
    """Mixin class for enabling active storage reductions.

    .. versionadded:: ACTIVEVERSION

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

        .. versionadded:: ACTIVEVERSION

        """
        method = self.get_active_method()
        if method is None:
            # Normal read by local client. Returns a numpy array.
            return super().__getitem__(indices)

        # Active storage reduction. Returns a dictionary.
        try:
            missing_values = self.get_missing_values()
        except AttributeError:
            missing_values = {}
        else:
            if missing_values is None:
                missing_values = {}

        try:
            storage_options = self.get_storage_options()
        except AttributeError:
            storage_options = {}

        active = Active(
            self.get_filename(),
            self.get_address(),
            #            storage_options=storage_options,
            **missing_values,
        )
        active.method = method
        active.components = True

        # Provide a file lock
        try:
            active.lock = self._lock
        except AttributeError:
            pass

        print(active.__dict__)
        out = active[indices]
        print(repr(out))
        return out

    def actify(self, method, axis=None):
        """Return a new actified `{{class}}` instance.

        The new instance is a deep copy of the original, with the
        additional setting of the active storage method and axis.

        .. versionadded:: ACTIVEVERSION

        .. seealso:: `set_active_axis`, `set_active_method`

        :Parameters:

            method: `str`
                The name of the reduction method.

                *Parameter example:*
                  ``'min'``

            axis: `None` or (sequence of) `int`, optional
                Axis or axes along which to operate. By default, or if
                `None`, flattened input is used.

        :Returns:

            `{{class}}`
                The new `{{class}}`` instance that ues an active
                storage operation.

        """
        if Active is None:
            # The active storage import dependency is not met, so
            # using active storage is not possible.
            raise AttributeError(
                f"Can't actify {self.__class__.__name__} when "
                "activestorage.Active is not available"
            )

        a = self.copy()
        a.set_active_method(method)
        a.set_active_axis(axis)
        return a

    def get_active_axis(self):
        """Return the active storage reduction axes.

        .. versionadded:: ACTIVEVERSION

        .. seealso:: `set_active_axis`

        :Returns:

            `None` or (sequence of) `int
                The active storage reduction axes. `None` signifies
                that all axes will be reduced.

        """
        return self._custom.get("active_axis")

    def get_active_method(self):
        """Return the name of the active storage reduction method.

        .. versionadded:: ACTIVEVERSION

        .. seealso:: `set_active_method`

        :Returns:

            `str` or `None`
                The name of the active storage reduction method, or
                `None` if one hasn't been set.

        """
        return self._custom.get("active_method")

    def set_active_axis(self, value):
        """Set the active storage reduction axes.

        .. versionadded:: ACTIVEVERSION

        .. seealso:: `get_active_axis`

        :Parameters:

            value: `None` or (sequence of) `int`
                The active storage reduction axes. If `None` then all
                axes will be reduced.

        :Returns:

            `None`

        """
        self._custom["active_axis"] = value

    def set_active_method(self, value):
        """Set the name of the active storage reduction method.

        .. versionadded:: ACTIVEVERSION

        .. seealso:: `get_active_method`

        :Parameters:

            value: `str`
                The active storage reduction method.

        :Returns:

            `None`

        """
        self._custom["active_method"] = value
