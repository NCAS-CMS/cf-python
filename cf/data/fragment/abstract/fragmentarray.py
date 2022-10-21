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
                    "aggregated_calendar", None
                )
            except AttributeError:
                aggregated_calendar = None

            try:
                array = source._get_component("array")
            except AttributeError:
                array = None

        self._set_component("address", address, copy=False)
        self._set_component("dtype", dtype, copy=False)
        self._set_component("filename", filename, copy=False)
        self._set_component("shape", shape, copy=False)
        self._set_component("aggregated_units", aggregated_units, copy=False)
        self._set_component(
            "aggregated_calendar", aggregated_calendar, copy=False
        )
        self._set_component("array", array, copy=copy)

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
        indices = self._parse_indices(indices)
        array = self.get_array()
        array = array[indices]
        return self._conform_units(array)

    def __repr__(self):
        return (
            f"<CF {self.__class__.__name__}{self.shape}: "
            f"{self.get_filename()}, {self.get_address()}>"
        )

    def _conform_units(self, array):
        # Convert the data to have the aggregated array units. (A CFA
        # conventions requirement.)
        units = Units(self.get_units(), self.get_calendar())
        if units:
            aggregated_units = Units(
                self.get_aggregated_units(), self.get_aggregated_calendar()
            )
            if units != aggregated_units:
                array = Units.conform(
                    array, units, aggregated_units, inplace=True
                )

        return array

    def _parse_indices(self, indices):
        ndim = self.ndim
        if indices is Ellipsis:
            return (slice(None),) * ndim

        # Check indices
        for i in indices:
            if isinstance(i, slice) or i is Ellipsis:
                continue

            if isinstance(i, Integral) or not getattr(i, "ndim", True):
                # 'i' is an integer or a scalar numpy/dask array
                raise ValueError(f"Can't provide rank-reducing index: {i!r}")

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

        return indices2

    @property
    def shape(self):
        """Tuple of array dimension sizes."""
        return self._get_component("shape")

    @property
    def dtype(self):
        """Data-type of the data elements."""
        return self._get_component("dtype")

    def get_address(self):
        """The calendar of the aggregated array.

        If the calendar is `None` then the CF default calendar is
        assumed, if applicable.

        .. versionadded:: TODODASKVER

        :Returns:

            `str` or `None`

        """
        return self._get_component("address", None)

    def get_aggregated_calendar(self):
        """The calendar of the aggregated array.

        If the calendar is `None` then the CF default calendar is
        assumed, if applicable.

        .. versionadded:: TODODASKVER

        :Returns:

            `str` or `None`

        """
        return self._get_component("aggregated_calendar", None)

    def get_aggregated_units(self):
        """The units of the aggregated array.

        If the units are `None` then the aggregated array has no
        defined units.

        .. versionadded:: TODODASKVER

        :Returns:

            `str` or `None`

        """
        units = self._get_component("aggregated_units", False)
        if units is False:
            raise ValueError(
                f"{self.__class__.__name__} 'aggregated_units' have not "
                "been set"
            )

        return units

    def get_array(self):
        """TODO.

        .. versionadded:: TODODASKVER

        :Returns:

            `Array`

        """
        return self._get_component("array")

    def get_calendar(self):
        """The calendar of the fragment data.

        If the calendar is `None` then the CF default calendar is
        assumed, if applicable.

        .. versionadded:: TODODASKVER

        :Returns:

            `str` or `None`

        """
        return self.get_array().get_calendar()

    def get_filename(self):
        """The name of the TODO netCDF file containing the array.

        .. versionadded:: TODODASKVER

        :Returns:

            `str` or `None`
                The filename, or `None` if there isn't one.

        **Examples**

        >>> a.get_filename()
        'file.nc'

        """
        return self._get_component("filename", None)

    def get_units(self):
        """The units of the fragment data.

        If the units are `None` then it is assumed that the fragment
        has the same units as the aggregated array.

        .. versionadded:: TODODASKVER

        :Returns:

            `str` or `None`

        """
        units = self.get_array().get_units()
        if units is False:
            units = None

        return units
