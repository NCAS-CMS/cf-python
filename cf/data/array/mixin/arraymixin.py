import numpy as np

from ....units import Units


class ArrayMixin:
    """Mixin class for a container of an array.

    .. versionadded:: 3.14.0

    """


#    def __array_function__(self, func, types, args, kwargs):
#        """Implement the `numpy` ``__array_function__`` protocol.
#
#        .. versionadded:: 3.14.0
#
#        """
#        return NotImplemented
#
#    @property
#    def _meta(self):
#        """Normalise the array to an appropriate Dask meta object.
#
#        The Dask meta can be thought of as a suggestion to Dask. Dask
#        uses this meta to generate the task graph until it can infer
#        the actual metadata from the values. It does not force the
#        output to have the structure or dtype of the specified meta.
#
#        .. versionadded:: NEXTVERSION
#
#        .. seealso:: `dask.utils.meta_from_array`
#
#        """
#   #    return np.array((), dtype=self.dtype)
#
#   @property
#   def Units(self):
#       """The `cf.Units` object containing the units of the array.
#
#       .. versionadded:: 3.14.0
#
#       """
#       return Units(self.get_units(None), self.get_calendar(None))
