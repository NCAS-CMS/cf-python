from ..units import Units
from .netcdfarray import NetCDFArray


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
        calendar=False,
        aggregated_units=False,
        aggregated_calendar=False,
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
                The calendar of the netCDF fragment variable.  Set to
                `None` to indicate that there is no calendar, or the
                CF default calendar if applicable.

            aggregated_units: `str` or `None`, optional
                The units of the aggregated array. By default, or if
                set to `None` then the aggregated array is assumed to
                have no units.

            aggregated_calendar: `str` or `None`, optional
                The calendar of the aggregated array. By default, or
                if set to `None` then the aggregated array is assumed
                to have no calendar, or the CF default calendar if
                applicable.

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
                    "aggregated_calendar", False
                )
            except AttributeError:
                aggregated_calendar = False

        self._set_component("aggregated_units", aggregated_units, copy=False)
        self._set_component(
            "aggregated_calendar", aggregated_calendar, copy=False
        )

    def __getitem__(self, indices):
        """Returns a subspace of the fragment as a numpy array.

        x.__getitem__(indices) <==> x[indices]

        The indices that define the subspace must be either `Ellipsis` or
        a sequence that contains an index for each dimension. In the
        latter case, each dimension's index must either be a `slice`
        object or a sequence of two or more integers.

        Indexing is similar to numpy indexing. The only difference to
        numpy indexing (given the restrictions on the type of indices
        allowed) is:

          * When two or more dimension's indices are sequences of integers
            then these indices work independently along each dimension
            (similar to the way vector subscripts work in Fortran).

        **Size 1 dimensions**

        If the netCDF fragment variable has fewer dimensions than
        given by `ndim` then an attempt will be made to insert any
        missing size 1 dimensions into the returned array. This will
        happen unless *indices* contains an element that reduces the
        rank of the result (such as a bare `int` index), in which case
        that dimension will not appear in the returned array, as
        usual. Given that the purpose of indexing this object is to
        provide the data to fill a subspace of a CFA aggregated data
        array, it is not recommended to provide dimension-reducing
        indices.

        **Performance**

        If a part of the fragment array is required, and the netCDF
        fragment variable has fewer dimensions than given by `ndim`,
        then the entire array is read xfrom disk prior having its
        missing size 1 dimensions added and then having the *indices*
        applied.

        .. versionadded:: TODODASKVER

        """
        # Re-cast indices so that it has at least ndim elements
        full_slice = False
        if all(i in (Ellipsis, slice(None, None, None)) for i in indices):
            full_slice = True
            indices = (slice(None),) * self.ndim
        else:
            diff = self.ndim - len(indices)
            if diff > 0:
                indices += (slice(None),) * diff

        try:
            array = super().__getitem__(indices)
        except ValueError:
            # A value error is raised if indices has ndim elements but
            # the netCDF fragment variable has fewer dimensions. In
            # this case we get the entire fragment array, insert the
            # missing size 1 dimensions, and then apply the requested
            # slice.
            array = super().__getitem__(Ellipsis)
            array = self._reshape(array)
            if not full_slice:
                array = array[indices]

        # Get the fragment's units (which are guaranteed to exist
        # after a `super().__getitem__` call).
        units = self._get_Units()
        if units:
            # Convert array to have parent units
            aggregated_units = self._get_aggregated_Units()
            if aggregated_units and aggregated_units != units:
                array = Units.conform(
                    array, units, aggregated_units, inplace=True
                )

        return array

    def _get_aggregated_Units(self):
        """Get the aggregated array units as a `Units` instance.

        .. versionadded:: TODODASKVER

        :Returns:

            `Units`

        """
        units = self._get_component("aggregated_units", False)
        if units is False:
            raise ValueError(
                f"{self.__class__.__name__} must have 'aggregate_units'"
            )

        calendar = self._get_component("aggregated_calendar", False)
        if calendar is False:
            raise ValueError(
                f"{self.__class__.__name__} must have 'aggregate_calendar'"
            )

        return Units(units, calendar)

    def _get_Units(self):
        """Get the fragment variable units as a `Units` instance.

        .. note:: The fragment's units are guaranteed to exist after
                  `super().__getitem__` call has been called.

        .. versionadded:: TODODASKVER

        :Returns:

            `Units`

        """
        return Units(
            self._get_component("units", None),
            self._get_component("calendar", None),
        )

    def _reshape(self, array):
        """Add missing size 1 dimensions to an array.

        Adds any size 1 dimension required to give *array* the same
        number of dimension as `ndim`.

        .. versionadded:: TODODASKVER

        """
        if array.ndim < self.ndim and array.size == self.size:
            array = array.reshape(self.shape)

        return array
