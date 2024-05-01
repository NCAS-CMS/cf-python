from ..array.fullarray import FullArray
from .mixin import FragmentArrayMixin


class FullFragmentArray(FragmentArrayMixin, FullArray):
    """A CFA fragment array that is filled with a value.

    .. versionadded:: 3.15.0

    """

    # REVIEW: h5: `__init__`: replace units/calendar API with attributes
    def __init__(
        self,
        fill_value=None,
        dtype=None,
        shape=None,
        aggregated_units=False,
        aggregated_calendar=False,
        attributes=None,
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

            {{init attributes: `dict` or `None`, optional}}

                .. versionadded:: NEXTVERSION

            {{aggregated_units: `str` or `None`, optional}}

            {{aggregated_calendar: `str` or `None`, optional}}

            {{init source: optional}}

            {{init copy: `bool`, optional}}

            units: `str` or `None`, optional
                Deprecated at version NEXTVERSION. Use the
                *attributes* parameter instead.

            calendar: `str` or `None`, optional
                Deprecated at version NEXTVERSION. Use the
                *attributes* parameter instead.

        """
        super().__init__(
            fill_value=fill_value,
            dtype=dtype,
            shape=shape,
            attributes=attributes,
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
