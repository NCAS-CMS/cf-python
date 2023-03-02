from ...utils import netcdf_lock


class ActiveStorageMixin:
    """TODOACTIVEDOCS.

    .. versionadded:: TODOACTIVEVER

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

        .. versionadded:: TODOACTIVEVER

        """
        method = self.get_active_method()
        if method is None:
            # Normal read by local client. Returns a numpy array.
            return super().__getitem__(indices)

        # Active storage read and reduction. Returns a dictionary.
        active = Active(self.get_filename(), self.get_ncvar())
        active.method = method
        active.components = True
        active.lock = netcdf_lock
        return active[indices]

    def actify(self, method, axis=None):
        """Return a new actified `{{class}}` instance.

        The new instance is a deep copy of the original, with the
        additional setting of the active storage method and axis.

        .. versionadded:: TODOACTIVEVER

        .. seealso:: `set_active_axis`, `set_active_method`

        :Parameters:

            method: `str`
                TODOACTIVEDOCS

            axis: `None` or (sequence of) `int`, optional
                TODOACTIVEDOCS

        :Returns:

            `{{class}}`
                TODOACTIVEDOCS

        """
        a = self.copy()
        a.set_active_method(method)
        a.set_active_axis(axis)
        return a

    def get_active_axis(self):
        """TODOACTIVEDOC.

        .. versionadded:: TODOACTIVEVER

        .. seealso:: `set_active_axis`

        :Returns:

            TODOACTIVEDOC

        """
        return self._custom.get("active_axis")

    def get_active_method(self):
        """TODOACTIVEDOC.

        .. versionadded:: TODOACTIVEVER

        .. seealso:: `set_active_method`

        :Returns:

            `str` or `None`
                The name of the active reduction method, or `None` if
                one hasn't been set.

        """
        return self._custom.get("active_method")

    def set_active_axis(self, value):
        """TODOACTIVEDOC.

        .. versionadded:: TODOACTIVEVER

        .. seealso:: `get_active_axis`

        :Parameters:

            TODOACTIVEDOCS

        :Returns:

            `None`

        """
        self._custom["active_axis"] = value

    def set_active_method(self, value):
        """TODOACTIVEDOC.

        .. versionadded:: TODOACTIVEVER

        .. seealso:: `get_active_method`

        :Parameters:

            TODOACTIVEDOCS

        :Returns:

            `None`

        """
        self._custom["active_method"] = value
