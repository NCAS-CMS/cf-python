from ..umarray import UMArray
from .abstract import FragmentArray


class UMFragmentArray(FragmentArray):
    """A CFA fragment array stored in a UM or PP file.

    .. versionadded:: TODODASKVER

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
        calendar=False,
        source=None,
        copy=True,
    ):
        """**Initialisation**

        :Parameters:

            filename: `str`
                The name of the UM or PP file containing the fragment.

            address: `int`, optional
                The start word in the file of the header.

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

            units: `str`, optional
                The units of the fragment variable. Set to `None` to
                indicate that there are no units. By default, or if TODO

            calendar: `str`, optional
                The calendar of the fragment variable. By default, or
                if set to `None`, then the CF default calendar is
                assumed, if applicable.

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
        if source is not None:
            super().__init__(source=source, copy=copy)
            return

        array = UMArray(
            filename=filename,
            header_offset=address,
            dtype=dtype,
            shape=shape,
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
