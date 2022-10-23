import numpy as np

from .abstract import Array


class FullArray(Array):
    """A array filled with a given value.
    
    The array may be empty or all missing values.    

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
        """**Initialization**

            :Parameters:

                dtype : numpy.dtype
                    The numpy data type of the data array.

            shape : tuple
                The data array's dimension sizes.

                size : int
                    Number of elements indexin the data array.

                fill_value : scalar, optional

        #        masked_all: `bool`

        """
        super().__init__(source=source, copy=copy)

        if source is not None:
            try:
                fill_value = source._get_component("fill_value", None)
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

        self._set_component("fill_value", fill_value, copy=False)
        self._set_component("dtype", dtype, copy=False)
        self._set_component("shape", shape, copy=False)
        self._set_component("units", units, copy=False)
        self._set_component("calendar", calendar, copy=False)

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
            for i, size, i in zip(indices, self.shape):
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

        fill_value = self.get_fill_value()
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

        x.__repr__() <==> repr(x)"""
        return f"<CF {self.__class__.__name__}{self.shape}: {self}>"

    def __str__(self):
        """Called by the `str` built-in function.

        x.__str__() <==> str(x)

        """
        fill_value = self.get_fill_value()
        if fill_value is None:
            return "Empty"

        return f"Filled with {fill_value!r}"

    def _set_units(self):
        """TODO.

        .. versionadded:: TODODASKVER

        To cfdm.Array ?

        """
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

    def get_fill_value(self):
        """Return the data array fill value.

        .. versionadded:: TODODASKVER

        :Returns:

            asdasda TODO

        """
        return self._get_component("fill_value", None)
