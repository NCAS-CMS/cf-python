import cfdm

from .abstract import FileArray


class NetCDFArray(cfdm.NetCDFArray, FileArray):
    """An array stored in a netCDF file."""

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

        .. versionadded:: TODODASKVER

        """
        if self.active_storage_op:
            # Active storage read. Returns a dictionary.
            active = Active(self.filename, self.ncvar)
            active.method = self.active_storage_op
            active.components = True

            return active[indices]

        # Normal read by local client. Returns a numpy array.
        #
        # In production code groups, masks, string types, etc. will
        # need to be accounted for here.
        return super().__getitme__(indices)

    def _active_chunk_functions(self):
        return {
            "min": self.active_min,
            "max": self.active_max,
            "mean": self.active_mean,
        }

    @property
    def active_storage_op(self):
        return self._custom.get("active_storage_op")

    @active_storage_op.setter
    def active_storage_op(self, value):
        self._custom["active_storage_op"] = value

    @property
    def op_axis(self):
        return self._custom.get("op_axis")

    @op_axis.setter
    def op_axis(self, value):
        self._custom["op_axis"] = value

    @staticmethod
    def active_min(a, **kwargs):
        """Chunk calculations for the minimum.

        Assumes that the calculations have already been done,
        i.e. that *a* is already the minimum.

        This function is intended to be passed in to
        `dask.array.reduction()` as the ``chunk`` parameter. Its
        return signature must be the same as the non-active chunks
        function that it is replacing.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dict`

        :Returns:

            `dict`
                Dictionary with the keys:

                * N: The sample size.
                * min: The minimum of `a``.

        """
        return {"N": a["n"], "min": a["min"]}

    @staticmethod
    def active_max(a, **kwargs):
        """Chunk calculations for the maximum.

        Assumes that the calculations have already been done,
        i.e. that *a* is already the maximum.

        This function is intended to be passed in to
        `dask.array.reduction()` as the ``chunk`` parameter. Its
        return signature must be consistent with that expected by the
        functions of the ``aggregate`` and ``combine`` parameters.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dict`

        :Returns:

            `dict`
                Dictionary with the keys:

                * N: The sample size.
                * max: The maximum of `a``.

        """
        return {"N": a["n"], "max": a["max"]}

    @staticmethod
    def active_mean(a, **kwargs):
        """Chunk calculations for the mean.

        Assumes that the calculations have already been done,
        i.e. that *a* is already the mean.

        This function is intended to be passed in to
        `dask.array.reduction()` as the ``chunk`` parameter. Its
        return signature must be the same as the non-active chunks
        function that it is replacing.

        .. versionadded:: TODODASKVER

        :Parameters:

            a: `dict`

        :Returns:

            `dict`
                Dictionary with the keys:

                * N: The sample size.
                * V1: The sum of ``weights``. Equal to ``N`` because
                      weights have not been set.
                * sum: The weighted sum of ``x``.
                * weighted: True if weights have been set. Always
                            False.

        """
        return {"N": a["n"], "V1": a["n"], "sum": a["sum"], "weighted": False}

    def get_active_chunk_function(self):
        try:
            return self._active_chunk_functions()[self.active_storage_op]
        except KeyError:
            raise ValueError("no active storage operation has been set")

    def set_active_storage_op(self, op, axis=None):
        if op not in self._active_chunk_functions():
            raise ValueError(f"Invalid active storage operation: {op!r}")

        a = self.copy()
        a.active_storage_op = op
        a.op_axis = axis
        return a
