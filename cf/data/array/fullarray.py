import numpy as np

from .abstract import Array

_FULLARRAY_HANDLED_FUNCTIONS = {}


class FullArray(Array):
    """A array filled with a given value.

    The array may be empty or all missing values.

    .. versionadded:: 3.14.0

    """

    def __init__(
        self,
        fill_value=None,
        dtype=None,
        shape=None,
        units=False,
        calendar=False,
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

            units: `str` or `None`, optional
                The units of the netCDF variable. Set to `None` to
                indicate that there are no units. If unset then the
                units will be set to `None` during the first
                `__getitem__` call.

            calendar: `str` or `None`, optional
                The calendar of the netCDF variable. By default, or if
                set to `None`, then the CF default calendar is
                assumed, if applicable. If unset then the calendar
                will be set to `None` during the first `__getitem__`
                call.

            {{init source: optional}}

            {{init copy: `bool`, optional}}

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
                units = source._get_component("units", False)
            except AttributeError:
                units = False

            try:
                calendar = source._get_component("calendar", None)
            except AttributeError:
                calendar = None

        self._set_component("full_value", fill_value, copy=False)
        self._set_component("dtype", dtype, copy=False)
        self._set_component("shape", shape, copy=False)
        self._set_component("units", units, copy=False)
        self._set_component("calendar", calendar, copy=False)

    def __array_function__(self, func, types, args, kwargs):
        """TODOCFADOCS"""
        if func not in _FULLARRAY_HANDLED_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        return _FULLARRAY_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __getitem__(self, indices):
        """x.__getitem__(indices) <==> x[indices]

        Returns a numpy array.

        """
        # If 'apply_indices' is False then we can make a filled numpy
        # array that already has the correct shape. Otherwise we need
        # to make one with shape 'self.shape' and then apply the
        # indices to that.
        apply_indices = False

        if indices is Ellipsis:
            array_shape = self.shape
        else:
            array_shape = []
            for i, size in zip(indices, self.shape):
                if not isinstance(i, slice):
                    continue

                start, stop, step = i.indices(size)
                a, b = divmod(stop - start, step)
                if b:
                    a += 1

                array_shape.append(a)

            if len(array_shape) != self.ndim:
                apply_indices = True
                array_shape = self.shape

        fill_value = self.get_full_value()
        if fill_value is np.ma.masked:
            array = np.ma.masked_all(array_shape, dtype=self.dtype)
        elif fill_value is not None:
            array = np.full(
                array_shape, fill_value=fill_value, dtype=self.dtype
            )
        else:
            array = np.empty(array_shape, dtype=self.dtype)

        if apply_indices:
            array = self.get_subspace(array, indices)

        self._set_units()

        return array

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

    def _set_units(self):
        """The units and calendar properties.

        These are the values set during initialisation, defaulting to
        `None` if either was not set at that time.

        .. versionadded:: 3.14.0

        :Returns:

            `tuple`
                The units and calendar values, either of which may be
                `None`.

        """
        # TODOCFA: Consider moving _set_units to cfdm.Array, or some
        #          other common ancestor so that this, and other,
        #          subclasses can access it.
        units = self.get_units(False)
        if units is False:
            units = None
            self._set_component("units", units, copy=False)

        calendar = self.get_calendar(False)
        if calendar is False:
            calendar = None
            self._set_component("calendar", calendar, copy=False)

        return units, calendar

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

    .. versionadded:: TODOCFAVER

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

    .. versionadded:: TODOCFAVER

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

    x = a.get_full_value()
    if x is np.ma.masked:
        return np.ma.masked_all((1,), dtype=a.dtype)

    return np.array([x], dtype=a.dtype)
