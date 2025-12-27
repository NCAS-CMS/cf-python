import cfdm

from ...mixin_container import Container
from ..fragment import FragmentFileArray


class AggregatedArray(Container, cfdm.AggregatedArray):
    """An array stored in a CF aggregation variable.

    .. versionadded:: 3.17.0

    """

    def __new__(cls, *args, **kwargs):
        """Store fragment array classes.

        .. versionadded:: 3.17.0

        """
        # Override the inherited FragmentFileArray class
        instance = super().__new__(cls)
        instance._FragmentArray["uri"] = FragmentFileArray
        return instance
