import cfdm


class FragmentFileArray(cfdm.data.fragment.FragmentFileArray):
    """Fragment of aggregated data in a file.

    .. versionadded:: NEXTVERSION

    """

    def __new__(cls, *args, **kwargs):
        """Store fragment classes.

        .. versionadded:: NEXTVERSION

        """
        # Import fragment classes. Do this here (as opposed to outside
        # the class) to aid subclassing.
        from .fragmentumarray import FragmentUMArray

        instance = super().__new__(cls)
        instance._FragmentArrays = instance._FragmentArrays + (
            FragmentUMArray,
        )
        return instance
