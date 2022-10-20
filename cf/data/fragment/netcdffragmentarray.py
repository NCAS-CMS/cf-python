from numbers import Integral

from ...units import Units
from ..netcdfarray import NetCDFArray


class NetCDFFragmentArray(NetCDFArray):
    """A CFA fragment array stored in a netCDF file.

    .. versionadded:: TODODASKVER

    """

    def __init__(
        self,
        filename=None,
        address=None,
        varid=None,
        group=None,
        dtype=None,
        shape=None,
        mask=True,
        units=False,
        calendar=None,
        aggregated_units=False,
        aggregated_calendar=None,
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

            varid: `int`, optional
                The UNIDATA netCDF interface ID of the variable
                containing the fragment array. Required if *address*
                is not set, ignored if *address* is set.

            group: `None` or sequence of `str`, optional
                Specify the netCDF4 group to which the netCDF fragment
                variable belongs. By default, or if *group* is `None`
                or an empty sequence, it assumed to be in the root
                group. The last element in the sequence is the name of
                the group in which the variable lies, with other
                elements naming any parent groups (excluding the root
                group).

                *Parameter example:*
                  To specify that a fragment variable is in the root
                  group: ``group=()`` or ``group=None``

                *Parameter example:*
                  To specify that a fragment variable is in the group
                  '/forecasts': ``group=['forecasts']``

                *Parameter example:*
                  To specify that a fragment variable is in the group
                  '/forecasts/model2': ``group=['forecasts',
                  'model2']``

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

            mask: `bool`
                If True (the default) then mask by convention when
                reading data from disk.

                A netCDF array is masked depending on the values of
                any of the netCDF variable attributes ``valid_min``,
                ``valid_max``, ``valid_range``, ``_FillValue`` and
                ``missing_value``.

            units: `str`, optional
                The units of the netCDF fragment variable. Set to
                `None` to indicate that there are no units.

            calendar: `str`, optional
                The calendar of the netCDF fragment variable. By
                default, or if set to `None`, then the CF default
                calendar is assumed, if applicable.

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
        super().__init__(
            filename=filename,
            ncvar=address,
            varid=varid,
            group=group,
            dtype=dtype,
            shape=shape,
            mask=mask,
            units=units,
            calendar=calendar,
            source=source,
            copy=copy,
        )

        if source is not None:
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

        **Size 1 dimensions**

        If the netCDF fragment variable has fewer dimensions than
        defined by `ndim` then any missing size 1 dimensions will be
        automatically inserted into the requested subspace.

        **Performance**

        If the netCDF fragment variable has fewer dimensions than
        defined by `ndim` then the entire array is read into memory
        before the subspace is returned.

        .. versionadded:: TODODASKVER

        """
        # Check indices
        for i in indices:
            if isinstance(i, slice) or i is Ellipsis:
                continue

            if isinstance(i, Integral) or not getattr(i, "ndim", True):
                # 'i' is an integer or a scalar numpy/dask array
                raise ValueError(f"Can't provide rank-reducing index: {i!r}")

        # Re-cast indices so that it has at least ndim elements
        ndim = self.ndim
        indices += (slice(None),) * (ndim - len(indices))

        try:
            array = super().__getitem__(indices)
        except ValueError:
            # A value error is raised if indices has at least ndim
            # elements but the netCDF fragment variable has fewer than
            # ndim dimensions. In this case we get the entire fragment
            # array, insert the missing size 1 dimensions, and then
            # apply the requested slice. (A CFA conventions
            # requirement.)
            array = super().__getitem__(Ellipsis)
            if array.ndim < ndim:
                array = array.reshape(self.shape)

            array = array[indices]

        # Convert the data to have the aggregated array units. (A CFA
        # conventions requirement.)
        units = Units(self.get_units(), self.get_calendar())
        if units:
            aggregated_units = Units(
                self.get_aggregated_units(), self.get_aggregated_calendar()
            )
            if aggregated_units and aggregated_units != units:
                array = Units.conform(
                    array, units, aggregated_units, inplace=True
                )

        return array

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

        If the units are `None` then it is assumed that the fragment
        data has no same units.

        .. versionadded:: TODODASKVER

        :Returns:

            `str` or `None`

        """
        units = self._get_component("aggregated", False)
        if units is False:
            raise ValueError(
                f"{self.__class__.__name__} must have 'aggregated_units'"
            )

        return units

    def get_units(self):
        """The units of the fragment data.

        If the units are `None` then it is assumed that the fragment
        data has the same units as the aggregated array.

        .. versionadded:: TODODASKVER

        :Returns:

            `str` or `None`

        """
        return self._get_component("units", None)
