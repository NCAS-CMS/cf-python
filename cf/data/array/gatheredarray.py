import cfdm

from ...mixin_container import Container
from .mixin import ArrayMixin, CompressedArrayMixin


class GatheredArray(
    CompressedArrayMixin, ArrayMixin, Container, cfdm.GatheredArray
):
    """An underlying gathered array.

    Compression by gathering combines axes of a multidimensional array
    into a new, discrete axis whilst omitting the missing values and
    thus reducing the number of values that need to be stored.

    The information needed to uncompress the data is stored in a "list
    variable" that gives the indices of the required points.

    See CF section 8.2. "Lossless Compression by Gathering".

    .. versionadded:: 3.0.0

    """
