import numpy as np

from ...functions import indices_shape, parse_indices
from .abstract import Array
from .mixin import IndexMixin

_FULLARRAY_HANDLED_FUNCTIONS = {}


class FullArray(IndexMixin, Array):
    """A array filled with a given value.

    The array may be empty or all missing values.

    .. versionadded:: 3.14.0

    """

    # REVIEW: h5: `__init__`: replace units/calendar API with attributes
    def __init__(
        self,
        fill_value=None,
        dtype=None,
        shape=None,
        attributes=None,
        source=None,
        copy=True,
    ):
        """**Initialisation**

        :Parameters:

            fill_value : scalar, optional
                The fill value for the array. May be set to
                `cf.masked` or `np.ma.masked`.

            dtype: `numpy.dtype`
                The data type of the array.

            shape: `tuple`
                The array dimension sizes.

            {{init attributes: `dict` or `None`, optional}}

                .. versionadded:: NEXTVERSION

            {{init source: optional}}

            {{init copy: `bool`, optional}}

            units: `str` or `None`, optional
                Deprecated at version NEXTVERSION. Use the
                *attributes* parameter instead.

            calendar: `str` or `None`, optional
                Deprecated at version NEXTVERSION. Use the
                *attributes* parameter instead.

        """
        super().__init__(source=source, copy=copy)

        if source is not None:
            try:
                fill_value = source._get_component("full_value", None)
            except AttributeError:
                fill_value = None

            try:
                dtype = source._get_component("dtype", None)
            except AttributeError:
                dtype = None

            try:
                shape = source._get_component("shape", None)
            except AttributeError:
                shape = None

            try:
                attributes = source._get_component("attributes", False)
            except AttributeError:
                attributes = None

        self._set_component("full_value", fill_value, copy=False)
        self._set_component("dtype", dtype, copy=False)
        self._set_component("shape", shape, copy=False)
        self._set_component("attributes", attributes, copy=False)

    def __array_function__(self, func, types, args, kwargs):
        """The `numpy` `__array_function__` protocol.

        .. versionadded:: 3.15.0

        """
        if func not in _FULLARRAY_HANDLED_FUNCTIONS:
            return NotImplemented

        # Note: This allows subclasses that don't override
        #       __array_function__ to handle FullArray objects
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        return _FULLARRAY_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return f"<CF {self.__class__.__name__}{self.shape}: {self}>"

    def __str__(self):
        """Called by the `str` built-in function.

        x.__str__() <==> str(x)

        """
        fill_value = self.get_full_value()
        if fill_value is None:
            return "Uninitialised"

        return f"Filled with {fill_value!r}"

    # REVIEW: getitem: `_get_array`: new method to convert subspace to numpy array
    def _get_array(self, index=None):
        """Returns the full array.

        .. versionadded:: NEXTVERSION

        .. seealso:: `__array__`, `index`

        :Parameters:

            {{index: `tuple` or `None`, optional}}

        :Returns:

            `numpy.ndarray`
                The subspace.

        """
        if index is None:
            shape = self.shape
        else:
            original_shape = self.original_shape
            index = parse_indices(original_shape, index, keepdims=False)
            shape = indices_shape(index, original_shape, keepdims=False)

        fill_value = self.get_full_value()
        if fill_value is np.ma.masked:
            array = np.ma.masked_all(shape, dtype=self.dtype)
        elif fill_value is not None:
            array = np.full(shape, fill_value=fill_value, dtype=self.dtype)
        else:
            array = np.empty(shape, dtype=self.dtype)

        return array

    # REVIEW: getitem: `array`: New property to convert subspace to numpy array
    @property
    def array(self):
        """Return an independent numpy array containing the data.

        .. versionadded:: NEXTVERSION

        :Returns:

            `numpy.ndarray`
                An independent numpy array of the data.
        """
        return np.asanyarray(self)

    @property
    def dtype(self):
        """Data-type of the data elements."""
        return self._get_component("dtype")

    @property
    def shape(self):
        """Tuple of array dimension sizes."""
        return self._get_component("shape")

    def get_full_value(self, default=AttributeError()):
        """Return the data array fill value.

        .. versionadded:: 3.14.0

        .. seealso:: `set_full_value`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                fill value has not been set. If set to an `Exception`
                instance then it will be raised instead.

        :Returns:

                The fill value.

        """
        return self._get_component("full_value", default=default)

    def set_full_value(self, fill_value):
        """Set the data array fill value.

        .. versionadded:: 3.14.0

        .. seealso:: `get_full_value`

        :Parameters:

            fill_value : scalar, optional
                The fill value for the array. May be set to
                `cf.masked` or `np.ma.masked`.

        :Returns:

            `None`

        """
        self._set_component("full_value", fill_value, copy=False)


def fullarray_implements(numpy_function):
    """Register an __array_function__ implementation for FullArray objects.

    .. versionadded:: 3.15.0

    """

    def decorator(func):
        _FULLARRAY_HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@fullarray_implements(np.unique)
def unique(
    a, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    """Version of `np.unique` that is optimised for `FullArray` objects.

    .. versionadded:: 3.15.0

    """
    if return_index or return_inverse or return_counts or axis is not None:
        # Fall back to the slow unique. (I'm sure we could probably do
        # something more clever here, but there is no use case at
        # present.)
        return np.unique(
            a[...],
            return_index=return_index,
            return_inverse=return_inverse,
            return_counts=return_counts,
            axis=axis,
        )

    # Fast unique based on the full value
    x = a.get_full_value()
    if x is np.ma.masked:
        return np.ma.masked_all((1,), dtype=a.dtype)

    return np.array([x], dtype=a.dtype)
