from numbers import Integral

from ....units import Units
from ...abstract import Array


class FragmentArray(Array):
    """A CFA fragment array.

    .. versionadded:: TODODASKVER

    """

    def __init__(
        self,
        filename=None,
        address=None,
        dtype=None,
        shape=None,
        aggregated_units=False,
        aggregated_calendar=None,
        array=None,
        source=None,
        copy=True,
    ):
        """**Initialisation**

        :Parameters:

            filename: `str`
                The name of the netCDF fragment file containing the
                array.

            address: `str`, optional
                The name of the netCDF variable containing the
                fragment array. Required unless *varid* is set.

            dtype: `numpy.dtype`
                The data type of the aggregated array. May be `None`
                if the numpy data-type is not known (which can be the
                case for netCDF string types, for example). This may
                differ from the data type of the netCDF fragment
                variable.

            shape: `tuple`
                The shape of the fragment within the aggregated
                array. This may differ from the shape of the netCDF
                fragment variable in that the latter may have fewer
                size 1 dimensions.

            aggregated_units: `str` or `None`, optional
                The units of the aggregated array. Set to `None` to
                indicate that there are no units.

            aggregated_calendar: `str` or `None`, optional
                The calendar of the aggregated array. By default, or
                if set to `None`, then the CF default calendar is
                assumed, if applicable.

            source: optional
                Initialise the array from the given object.

                {{init source}}

            {{deep copy}}

        """
        super().__init__(source=source, copy=copy)

        if source is not None:
            try:
                filename = source._get_component("filename", None)
            except AttributeError:
                filename = None

            try:
                address = source._get_component("address", None)
            except AttributeError:
                address = None

            try:
                dtype = source._get_component("dtype", None)
            except AttributeError:
                dtype = None

            try:
                shape = source._get_component("shape", None)
            except AttributeError:
                shape = None

            try:
                aggregated_units = source._get_component(
                    "aggregated_units", False
                )
            except AttributeError:
                aggregated_units = False

            try:
                aggregated_calendar = source._get_component(
                    "aggregated_calendar", False
                )
            except AttributeError:
                aggregated_calendar = False

            try:
                array = source._get_component("array")
            except AttributeError:
                array = None

        self._set_component("filename", filename, copy=False)
        self._set_component("address", address, copy=False)
        self._set_component("dtype", dtype, copy=False)
        self._set_component("shape", shape, copy=False)
        self._set_component("array", array, copy=copy)
        self._set_component("aggregated_units", aggregated_units, copy=False)
        self._set_component(
            "aggregated_calendar", aggregated_calendar, copy=False
        )

    def __getitem__(self, indices):
        """Returns a subspace of the fragment as a numpy array.

        x.__getitem__(indices) <==> x[indices]

        Indexing is similar to numpy indexing, with the following
        differences:

          * A dimension's index can't be rank-reducing, i.e. it can't
            be an integer, nor a scalar `numpy` or `dask` array.

          * When two or more dimension's indices are sequences of
            integers then these indices work independently along each
            dimension (similar to the way vector subscripts work in
            Fortran).

        .. versionadded:: TODODASKVER

        """
        array = self.get_array()
        indices = self._conform_indices(indices)
        array = array[indices]
        array = self._conform_units(array)
        return array

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return (
            f"<CF {self.__class__.__name__}{self.shape}: {self.get_array()}>"
        )

    def _conform_indices(self, indices):
        """Conform the indices that retrieve the fragment data.

        The indicies may need modification in order to ensure that

        .. versionadded:: TODODASKVER

        :Parameters:

            indices: `tuple` or `Ellipsis`
                TODODASKDOCS

        :Returns:

            `tuple`
                TODODASKDOCS

        """
        ndim = self.ndim
        if indices is Ellipsis:
            return (slice(None),) * ndim

        # Check indices
        has_ellipsis = False
        for i in indices:
            if isinstance(i, slice):
                continue

            if i is Ellipsis:
                has_ellipsis = True
                continue

            if isinstance(i, Integral) or not getattr(i, "ndim", True):
                # TODODASK: what about [] or np.array([])?

                # 'i' is an integer or a scalar numpy/dask array
                raise ValueError(
                    f"Can't subspace {self.__class__.__name__} with a "
                    f"rank-reducing index: {i!r}"
                )

        if has_ellipsis:
            # Replace Ellipsis with one or more slice(None)
            indices2 = []
            length = len(indices)
            n = ndim
            for i in indices:
                if i is Ellipsis:
                    m = n - length + 1
                    indices2.extend([slice(None)] * m)
                    n -= m
                else:
                    indices2.append(i)
                    n -= 1

                length -= 1

            indices = tuple(indices2)

        return indices

    def _conform_units(self, array):
        """Conform the array to have the aggregated units.

        .. versionadded:: TODODASKVER

        :Parameters:

            TODODASKDOCS

        :Returns:

            `numpy.ndarray`
                TODODASKDOCS

        """
        units = self.Units
        if units:
            aggregated_units = self.aggregated_Units
            if units != aggregated_units:
                array = Units.conform(
                    array, units, aggregated_units, inplace=True
                )

        return array

    @property
    def aggregated_Units(self):
        """TODODASKDOCS.

        .. versionadded:: TODODASKVER

        :Returns:

            `Units`
                TODODASKDOCS

        """
        return Units(
            self.get_aggregated_units(), self.get_aggregated_calendar(None)
        )

    @property
    def dtype(self):
        """Data-type of the data elements."""
        return self._get_component("dtype")

    @property
    def shape(self):
        """Tuple of array dimension sizes."""
        return self._get_component("shape")

    def get_address(self):
        """TODODASKDOCS.

        .. versionadded:: TODODASKVER

        :Returns:

                The file address of the fragment. If no address has
                been set then `None` is returned.

        """
        return self._get_component("address", None)

    def get_aggregated_calendar(self, default=ValueError()):
        """The calendar of the aggregated array.

        If the calendar is `None` then the CF default calendar is
        assumed, if applicable.

        .. versionadded:: TODODASKVER

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                calendar has not been set. If set to an `Exception`
                instance then it will be raised instead.

        :Returns:

            `str` or `None`
                The calendar value.

        """
        calendar = self._get_component("aggregated_calendar", False)
        if calendar is False:
            if default is None:
                return

            return self._default(
                default,
                f"{self.__class__.__name__} 'aggregated_calendar' has not "
                "been set",
            )

        return calendar

    def get_aggregated_units(self, default=ValueError()):
        """The units of the aggregated array.

        If the units are `None` then the aggregated array has no
        defined units.

        .. versionadded:: TODODASKVER

        .. seealso:: `get_aggregated_calendar`

        :Parameters:

            TODODASKDOCS

        :Returns:

            `str` or `None`

        """
        units = self._get_component("aggregated_units", False)
        if units is False:
            if default is None:
                return

            return self._default(
                default,
                f"{self.__class__.__name__} 'aggregated_units' have not "
                "been set",
            )

        return units

    def get_array(self):
        """TODODASKDOCS.

        .. versionadded:: TODODASKVER

        :Returns:

            `Array` or `None`

        """
        return self._get_component("array", None)

    def get_units(self, default=ValueError()):
        """The units of the netCDF variable.

        .. versionadded:: (cfdm) 1.10.0.1

        .. seealso:: `get_calendar`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                units have not been set. If set to an `Exception`
                instance then it will be raised instead.

        :Returns:

            `str` or `None`
                The units value.

        """
        return self.get_array().get_units(default)
