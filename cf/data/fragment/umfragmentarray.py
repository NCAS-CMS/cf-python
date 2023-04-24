from ..array.umarray import UMArray
from .mixin import FragmentArrayMixin


class UMFragmentArray(FragmentArrayMixin, UMArray):
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

            filename: (sequence of `str`), optional
                The names of the UM or PP files containing the fragment.

            address: (sequence of `str`), optional
                The start words in the files of the header.

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
        super().__init__(
            filename=filename,
            address=address,
            dtype=dtype,
            shape=shape,
            units=units,
            calendar=calendar,
            source=source,
            copy=False,
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
