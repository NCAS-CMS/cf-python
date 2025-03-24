import cfdm

from ...mixin_container import Container
from ..array.mixin import ActiveStorageMixin


class FragmentFileArray(
    ActiveStorageMixin, Container, cfdm.data.fragment.FragmentFileArray
):
    """Fragment of aggregated data in a file.

    .. versionadded:: 3.17.0

    """

    def __new__(cls, *args, **kwargs):
        """Store fragment classes.

        .. versionadded:: 3.17.0

        """
        # Import fragment classes. Do this here (as opposed to outside
        # the class) to aid subclassing.
        from .fragmentumarray import FragmentUMArray

        instance = super().__new__(cls)
        instance._FragmentArrays = instance._FragmentArrays + (
            FragmentUMArray,
        )
        return instance
