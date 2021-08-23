import cfdm


class Array(cfdm.Array):
    """Abstract base class for a container of an underlying array.

    The form of the array is defined by the initialization parameters
    of a subclass.

    .. versionadded:: 3.0.0

    """

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        .. versionadded:: 3.0.0

        """
        return super().__repr__().replace("<", "<CF ", 1)

    @property
    def dask_asarray(self):
        """TODODASK."""
        return False


# --- End: class
