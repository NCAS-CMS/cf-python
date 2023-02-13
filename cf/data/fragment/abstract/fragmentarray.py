# from numbers import Integral
#
# from ....units import Units
#
#
# class FragmentArrayMixin:
#    """Mixin for a CFA fragment array.
#
#    .. versionadded:: 3.14.0
#
#    """
#
#
#        def __init__(
#            self,
#            dtype=None,
#            shape=None,
#            aggregated_units=False,
#            aggregated_calendar=None,
#            array=None,
#            source=None,
#            copy=True,
#        ):
#            """**Initialisation**
#
#            :Parameters:
#
#                dtype: `numpy.dtype`
#                    The data type of the aggregated array. May be `None`
#                    if the numpy data-type is not known (which can be the
#                    case for netCDF string types, for example). This may
#                    differ from the data type of the netCDF fragment
#                    variable.
#
#                shape: `tuple`
#                    The shape of the fragment within the aggregated
#                    array. This may differ from the shape of the netCDF
#                    fragment variable in that the latter may have fewer
#                    size 1 dimensions.
#
#                {{aggregated_units: `str` or `None`, optional}}
#
#                {{aggregated_calendar: `str` or `None`, optional}}
#
#                array: `Array`
#                    The fragment array stored in a file.
#
#                source: optional
#                    Initialise the array from the given object.
#
#                    {{init source}}
#
#                {{deep copy}}
#
#            """
#            super().__init__(source=source, copy=copy)
#
#            if source is not None:
#                try:
#                    dtype = source._get_component("dtype", None)
#                except AttributeError:
#                    dtype = None
#
#                try:
#                    shape = source._get_component("shape", None)
#                except AttributeError:
#                    shape = None
#
#                try:
#                    aggregated_units = source._get_component(
#                        "aggregated_units", False
#                    )
#                except AttributeError:
#                    aggregated_units = False
#
#                try:
#                    aggregated_calendar = source._get_component(
#                        "aggregated_calendar", False
#                    )
#                except AttributeError:
#                    aggregated_calendar = False
#
#                try:
#                    array = source._get_component("array", None)
#                except AttributeError:
#                    array = None
#
#            self._set_component("dtype", dtype, copy=False)
#            self._set_component("shape", shape, copy=False)
#            self._set_component("aggregated_units", aggregated_units, copy=False)
#            self._set_component(
#                "aggregated_calendar", aggregated_calendar, copy=False
#            )
#
#            if array is not None:
#                self._set_component("array", array, copy=copy)
#
#    def __getitem__(self, indices):
#        """Returns a subspace of the fragment as a numpy array.
#
#        x.__getitem__(indices) <==> x[indices]
#
#        Indexing is similar to numpy indexing, with the following
#        differences:
#
#          * A dimension's index can't be rank-reducing, i.e. it can't
#            be an integer, a scalar `numpy` array, nor a scalar `dask`
#            array.
#
#          * When two or more dimension's indices are sequences of
#            integers then these indices work independently along each
#            dimension (similar to the way vector subscripts work in
#            Fortran).
#
#        .. versionadded:: TODOCFAVER
#
#        """
#        indices = self._parse_indices(indices)
#
#        try:
#            array = super().__getitem__(indices)
#        except ValueError:
#            # A ValueError is expected to be raised when the fragment
#            # variable has fewer than 'self.ndim' dimensions (given
#            # that 'indices' now has 'self.ndim' elements).
#            axis = self._missing_size_1_axis(indices)
#            if axis is not None:
#                # There is a unique size 1 index, that must correspond
#                # to the missing dimension => Remove it from the
#                # indices, get the fragment array with the new
#                # indices; and then insert the missing size one
#                # dimension.
#                indices = list(indices)
#                indices.pop(axis)
#                array = super().__getitem__(tuple(indices))
#                array = np.expand_dims(array, axis)
#            else:
#                # There are multiple size 1 indices, so we don't know
#                # how many missing dimensions there are nor their
#                # positions => Get the full fragment array; and then
#                # reshape it to the shape of the storage chunk.
#                array = super().__getitem__(Ellipsis)
#                if array.size != self.size:
#                    raise ValueError(
#                        "Can't get CFA fragment data from "
#                        f"{self.get_filename()} ({self.get_address()}) when "
#                        "the fragment has two or more missing size 1 "
#                        "dimensions whilst also spanning two or more "
#                        "storage chunks"
#                        "\n\n"
#                        "Consider recreating the data with exactly one"
#                        "storage chunk per fragment."
#                    )
#
#                array = array.reshape(self.shape)
#
#        array = self._conform_units(array)
#        return array
#
#    def _missing_size_1_axis(self, indices):
#        """Find the position of a unique size 1 index.
#
#        .. versionadded:: TODOCFAVER
#
#        .. seealso:: `_parse_indices`
#
#        :Parameters:
#
#            indices: `tuple`
#                The array indices to be parsed, as returned by
#                `_parse_indices`.
#
#        :Returns:
#
#            `int` or `None`
#                The position of the unique size 1 index, or `None` if
#                there isn't one.
#
#        **Examples**
#
#        >>> a._missing_size_1_axis(([2, 4, 5], slice(0, 1), slice(0, 73)))
#        1
#        >>> a._missing_size_1_axis(([2, 4, 5], slice(3, 4), slice(0, 73)))
#        1
#        >>> a._missing_size_1_axis(([2, 4, 5], [0], slice(0, 73)))
#        1
#        >>> a._missing_size_1_axis(([2, 4, 5], slice(0, 144), slice(0, 73)))
#        None
#        >>> a._missing_size_1_axis(([2, 4, 5], slice(3, 7), [0, 1]))
#        None
#        >>> a._missing_size_1_axis(([2, 4, 5], slice(0, 1), [0]))
#        None
#
#        """
#        axis = None  # Position of unique size 1 index
#
#        n = 0  # Number of size 1 indices
#        for i, index in enumerate(indices):
#            try:
#                if index.stop - index.start == 1:
#                    # Index is a size 1 slice
#                    n += 1
#                    axis = i
#            except AttributeError:
#                try:
#                    if index.size == 1:
#                        # Index is a size 1 numpy or dask array
#                        n += 1
#                        axis = i
#                except AttributeError:
#                    if len(index) == 1:
#                        # Index is a size 1 list
#                        n += 1
#                        axis = i
#
#        if n > 1:
#            # There are two or more size 1 indices
#            axis = None
#
#        return axis
#
#    def _parse_indices(self, indices):
#        """Parse the indices that retrieve the fragment data.
#
#        Ellipses are replaced with the approriate number `slice`
#        instances, and rank-reducing indices (such as an integer or
#        scalar array) are disallowed.
#
#        .. versionadded:: 3.14.0
#
#        :Parameters:
#
#            indices: `tuple` or `Ellipsis`
#                The array indices to be parsed.
#
#        :Returns:
#
#            `tuple`
#                The parsed indices.
#
#        **Examples**
#
#        >>> a.shape
#        (12, 1, 73, 144)
#        >>> a._parse_indices(([2, 4, 5], Ellipsis, slice(45, 67))
#        ([2, 4, 5], slice(0, 1), slice(0, 73), slice(45, 67))
#
#        """
#        shape = self.shape
#        if indices is Ellipsis:
#            return tuple([slice(0, n) for n in shape])
#
#        # Check indices
#        has_ellipsis = False
#        indices = list(indices)
#        for i, (index, n) in enumerate(zip(indices, shape)):
#            if isinstance(index, slice):
#                if index == slice(None):
#                    indices[i] = slice(0, n)
#
#                continue
#
#            if index is Ellipsis:
#                has_ellipsis = True
#                continue
#
#            if isinstance(index, Integral) or not getattr(index, "ndim", True):
#                # TODOCFA: what about [] or np.array([])?
#
#                # 'index' is an integer or a scalar numpy/dask array
#                raise ValueError(
#                    f"Can't subspace {self.__class__.__name__} with a "
#                    f"rank-reducing index: {index!r}"
#                )
#
#        if has_ellipsis:
#            # Replace Ellipsis with one or more slices
#            indices2 = []
#            length = len(indices)
#            n = self.ndim
#            for index in indices:
#                if index is Ellipsis:
#                    m = n - length + 1
#                    indices2.extend([slice(None)] * m)
#                    n -= m
#                else:
#                    indices2.append(i)
#                    n -= 1
#
#                length -= 1
#
#            indices = indices2
#
#            for i, (index, n) in enumerate(zip(indices, shape)):
#                if index == slice(None):
#                    indices[i] = slice(0, n)
#
#        return tuple(indices)
#
#    def _conform_units(self, array):
#        """Conform the array to have the aggregated units.
#
#        .. versionadded:: 3.14.0
#
#        :Parameters:
#
#            array: `numpy.ndarray` or `dict`
#                The array to be conformed. If *array* is a `dict` with
#                `numpy` array values then each value is conformed.
#
#        :Returns:
#
#            `numpy.ndarray` or `dict`
#                The conformed array. The returned array may or may not
#                be the input array updated in-place, depending on its
#                data type and the nature of its units and the
#                aggregated units.
#
#                If *array* is a `dict` then a dictionary of conformed
#                arrays is returned.
#
#        """
#        units = self.Units
#        if units:
#            aggregated_units = self.aggregated_Units
#            if not units.equivalent(aggregated_units):
#                raise ValueError(
#                    f"Can't convert fragment data with units {units!r} to "
#                    f"have aggregated units {aggregated_units!r}"
#                )
#
#            if units != aggregated_units:
#                if isinstance(array, dict):
#                    # 'array' is a dictionary
#                    for key, value in array.items():
#                        if key in _active_chunk_methds:
#                            array[key] = Units.conform(
#                                value, units, aggregated_units, inplace=True
#                            )
#                else:
#                    # 'array' is a numpy array
#                    array = Units.conform(
#                        array, units, aggregated_units, inplace=True
#                    )
#
#        return array
#
#    @property
#    def aggregated_Units(self):
#        """The units of the aggregated data.
#
#        .. versionadded:: 3.14.0
#
#        :Returns:
#
#            `Units`
#                The units of the aggregated data.
#
#        """
#        return Units(
#            self.get_aggregated_units(), self.get_aggregated_calendar(None)
#        )
#
#    def get_aggregated_calendar(self, default=ValueError()):
#        """The calendar of the aggregated array.
#
#        If the calendar is `None` then the CF default calendar is
#        assumed, if applicable.
#
#        .. versionadded:: 3.14.0
#
#        :Parameters:
#
#            default: optional
#                Return the value of the *default* parameter if the
#                calendar has not been set. If set to an `Exception`
#                instance then it will be raised instead.
#
#        :Returns:
#
#            `str` or `None`
#                The calendar value.
#
#        """
#        calendar = self._get_component("aggregated_calendar", False)
#        if calendar is False:
#            if default is None:
#                return
#
#            return self._default(
#                default,
#                f"{self.__class__.__name__} 'aggregated_calendar' has not "
#                "been set",
#            )
#
#        return calendar
#
#    def get_aggregated_units(self, default=ValueError()):
#        """The units of the aggregated array.
#
#        If the units are `None` then the aggregated array has no
#        defined units.
#
#        .. versionadded:: 3.14.0
#
#        .. seealso:: `get_aggregated_calendar`
#
#        :Parameters:
#
#            default: optional
#                Return the value of the *default* parameter if the
#                units have not been set. If set to an `Exception`
#                instance then it will be raised instead.
#
#        :Returns:
#
#            `str` or `None`
#                The units value.
#
#        """
#        units = self._get_component("aggregated_units", False)
#        if units is False:
#            if default is None:
#                return
#
#            return self._default(
#                default,
#                f"{self.__class__.__name__} 'aggregated_units' have not "
#                "been set",
#            )
#
#        return units
#
#
#    def get_array(self):
#        """The fragment array.
#
#        .. versionadded:: 3.14.0
#
#        :Returns:
#
#            Subclass of `Array`
#                The object defining the fragment array.
#
#        """
#        return self._get_component("array")
#
#    def get_units(self, default=ValueError()):
#        """The units of the netCDF variable.
#
#        .. versionadded:: (cfdm) 1.10.0.1
#
#        .. seealso:: `get_calendar`
#
#        :Parameters:
#
#            default: optional
#                Return the value of the *default* parameter if the
#                units have not been set. If set to an `Exception`
#                instance then it will be raised instead.
#
#        :Returns:
#
#            `str` or `None`
#                The units value.
#
#        """
#        return self.get_array().get_units(default)
