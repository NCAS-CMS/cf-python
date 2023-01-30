from ..array.umarray import UMArray
from .abstract import FragmentArray


class UMFragmentArray(FragmentArray):
    """A CFA fragment array stored in a UM or PP file.

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
