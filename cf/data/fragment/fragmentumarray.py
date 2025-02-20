import cfdm

from ..array.umarray import UMArray


class FragmentUMArray(
    cfdm.data.fragment.mixin.FragmentFileArrayMixin, UMArray
):
    """A fragment of aggregated data in a PP or UM file.

    .. versionadded:: 3.14.0

    """
