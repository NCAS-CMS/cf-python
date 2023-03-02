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

        # Active storage read. Returns a dictionary.
        active = Active(self.filename, self.ncvar)
        active.method = method
        active.components = True
        active.lock = netcdf_lock
        return active[indices]

    #    def _active_chunk_functions(self):
    #        """Mapping of method names to active chunk functions.
    #
    #        .. versionadded:: TODOACTIVEVER
    #
    #        :Returns:
    #
    #            `dict`
    #                The mapping.
    #
    #        """
    #        return {
    #            "min": self.active_min,
    #            "max": self.active_max,
    #            "mean": self.active_mean,
    #            "sum": self.active_sum,
    #        }

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
        #        if method not in self._active_chunk_functions():
        #            raise ValueError(f"Invalid active storage operation: {method!r}")

        a = self.copy()
        a.set_active_method(method)
        a.set_active_axis(axis)
        return a

    #    @staticmethod
    #    def active_min(a, **kwargs):
    #        """Chunk calculations for the minimum.
    #
    #        Assumes that the calculations have already been done,
    #        i.e. that *a* is already the minimum.
    #
    #        This function is intended to be passed to
    #        `dask.array.reduction` as the ``chunk`` parameter. Its return
    #        signature must be the same as the non-active chunk function
    #        that it is replacing.
    #
    #        .. versionadded:: TODOACTIVEVER
    #
    #        :Parameters:
    #
    #            a: `dict`
    #                TODOACTIVEDOCS
    #
    #        :Returns:
    #
    #            `dict`
    #                Dictionary with the keys:
    #
    #                * N: The sample size.
    #                * min: The minimum of `a``.
    #
    #        """
    #        return {"N": a["n"], "min": a["min"]}
    #
    #    @staticmethod
    #    def active_max(a, **kwargs):
    #        """Chunk calculations for the maximum.
    #
    #        Assumes that the calculations have already been done,
    #        i.e. that *a* is already the maximum.
    #
    #        This function is intended to be passed to
    #        `dask.array.reduction` as the ``chunk`` parameter. Its return
    #        signature must be the same as the non-active chunk function
    #        that it is replacing.
    #
    #        .. versionadded:: TODOACTIVEVER
    #
    #        :Parameters:
    #
    #            a: `dict`
    #                TODOACTIVEDOCS
    #
    #        :Returns:
    #
    #            `dict`
    #                Dictionary with the keys:
    #
    #                * N: The sample size.
    #                * max: The maximum of `a``.
    #
    #        """
    #        return {"N": a["n"], "max": a["max"]}
    #
    #    @staticmethod
    #    def active_mean(a, **kwargs):
    #        """Chunk calculations for the unweighted mean.
    #
    #        Assumes that the calculations have already been done,
    #        i.e. that *a* is already the uweighted mean.
    #
    #        This function is intended to be passed to
    #        `dask.array.reduction` as the ``chunk`` parameter. Its return
    #        signature must be the same as the non-active chunk function
    #        that it is replacing.
    #
    #        .. versionadded:: TODOACTIVEVER
    #
    #        :Parameters:
    #
    #            a: `dict`
    #                TODOACTIVEDOCS
    #
    #        :Returns:
    #
    #            `dict`
    #                Dictionary with the keys:
    #
    #                * N: The sample size.
    #                * V1: The sum of ``weights``. Equal to ``N`` because
    #                      weights have not been set.
    #                * sum: The weighted sum of ``a``.
    #                * weighted: True if weights have been set. Always
    #                            False.
    #
    #        """
    #        return {"N": a["n"], "V1": a["n"], "sum": a["sum"], "weighted": False}
    #
    #    @staticmethod
    #    def active_sum(a, **kwargs):
    #        """Chunk calculations for the unweighted sum.
    #
    #        Assumes that the calculations have already been done,
    #        i.e. that *a* is already the uweighted sum.
    #
    #        This function is intended to be passed to
    #        `dask.array.reduction` as the ``chunk`` parameter. Its return
    #        signature must be the same as the non-active chunk function
    #        that it is replacing.
    #
    #        .. versionadded:: TODOACTIVEVER
    #
    #        :Parameters:
    #
    #            a: `dict`
    #                TODOACTIVEDOCS
    #
    #        :Returns:
    #
    #            `dict`
    #                Dictionary with the keys:
    #
    #                * N: The sample size.
    #                * sum: The weighted sum of ``a``
    #
    #        """
    #        return {"N": a["n"], "sum": a["sum"]}

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

    #    def get_active_chunk_function(self):
    #        """TODOACTIVEDOC.
    #
    #        .. versionadded:: TODOACTIVEVER
    #
    #        :Returns:
    #
    #            TODOACTIVEDOC
    #
    #        """
    #        try:
    #            return self._active_chunk_functions()[self.get_active_method()]
    #        except KeyError:
    #            raise ValueError("no active storage operation has been set")

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
