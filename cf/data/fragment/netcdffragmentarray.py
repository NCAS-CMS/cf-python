from ..array.netcdfarray import NetCDFArray
from .abstract import FragmentArray


class NetCDFFragmentArray(FragmentArray):
    """A CFA fragment array stored in a netCDF file.

    .. versionadded:: 3.14.0

    """

    def __init__(
        self,
        filename=None,
        address=None,
        dtype=None,
        shape=None,
        aggregated_units=False,
        aggregated_calendar=False,
        units=False,
        calendar=None,
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

            units: `str` or `None`, optional
                The units of the fragment data. Set to `None` to
                indicate that there are no units. If unset then the
                units will be set during the first `__getitem__` call.

            calendar: `str` or `None`, optional
                The calendar of the fragment data. Set to `None` to
                indicate the CF default calendar, if applicable. If
                unset then the calendar will be set during the first
                `__getitem__` call.

            {{aggregated_units: `str` or `None`, optional}}

            {{aggregated_calendar: `str` or `None`, optional}}

            {{init source: optional}}

            {{init copy: `bool`, optional}}

        """
        if source is not None:
            super().__init__(source=source, copy=copy)
            return

        if isinstance(address, int):
            ncvar = None
            varid = address
        else:
            ncvar = address
            varid = None

        # TODO set groups from ncvar
        group = None

        array = NetCDFArray(
            filename=filename,
            ncvar=ncvar,
            varid=varid,
            group=group,
            dtype=dtype,
            shape=shape,
            mask=True,
            units=units,
            calendar=calendar,
            copy=False,
        )

        super().__init__(
            filename=filename,
            address=address,
            dtype=dtype,
            shape=shape,
            aggregated_units=aggregated_units,
            aggregated_calendar=aggregated_calendar,
            array=array,
            source=source,
            copy=False,
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

        **Performance**

        If the netCDF fragment variable has fewer than `ndim`
        dimensions then the entire array is read into memory before
        the requested subspace of it is returned.

        .. versionadded:: 3.14.0

        """
        indices = self._parse_indices(indices)
        array = self.get_array()

        try:
            array = array[indices]
        except ValueError:
            # A value error is raised if indices has at least ndim
            # elements but the netCDF fragment variable has fewer than
            # ndim dimensions. In this case we get the entire fragment
            # array, insert the missing size 1 dimensions, and then
            # apply the requested slice.
            array = array[Ellipsis]
            if array.ndim < self.ndim:
                array = array.reshape(self.shape)

            array = array[indices]

        array = self._conform_units(array)
        return array
