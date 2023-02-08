from ..array.fullarray import FullArray
from .abstract import FragmentArray


class FullFragmentArray(FragmentArray):
    """A CFA fragment array that is filled with a value.

    .. versionadded:: TODOCFAVER

    """

    def __init__(
        self,
        fill_value=None,
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

            fill_value: scalar
                The fill value.

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
                units will be set to `None` during the first
                `__getitem__` call.

            calendar: `str` or `None`, optional
                The calendar of the fragment data. Set to `None` to
                indicate the CF default calendar, if applicable. If
                unset then the calendar will be set to `None` during
                the first `__getitem__` call.

            {{aggregated_units: `str` or `None`, optional}}

            {{aggregated_calendar: `str` or `None`, optional}}

            {{init source: optional}}

            {{init copy: `bool`, optional}}

        """
        if source is not None:
            super().__init__(source=source, copy=copy)
            return

        array = FullArray(
            fill_value=fill_value,
            dtype=dtype,
            shape=shape,
            units=units,
            calendar=calendar,
            copy=False,
        )

        super().__init__(
            dtype=dtype,
            shape=shape,
            aggregated_units=aggregated_units,
            aggregated_calendar=aggregated_calendar,
            array=array,
            copy=False,
        )

    def get_full_value(self, default=AttributeError()):
        """The fragment array fill value.

        .. versionadded:: TODOCFAVER

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                fill value has not been set. If set to an `Exception`
                instance then it will be raised instead.

        :Returns:

                The fill value.

        """
        return self.get_array().get_full_value(default=default)