from ..array.umarray import UMArray
from .mixin import FragmentArrayMixin


class UMFragmentArray(FragmentArrayMixin, UMArray):
    """A CFA fragment array stored in a UM or PP file.

    .. versionadded:: 3.14.0

    """

    # REVIEW: h5: `__init__`: replace units/calendar API with attributes
    # REVIEW: h5: `__init__`: new keyword 'storage_options'
    def __init__(
        self,
        filename=None,
        address=None,
        dtype=None,
        shape=None,
        aggregated_units=False,
        aggregated_calendar=False,
        attributes=None,
        storage_options=None,
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

            {{init attributes: `dict` or `None`, optional}}

                During the first `__getitem__` call, any of the
                ``_FillValue``, ``add_offset``, ``scale_factor``,
                ``units``, and ``calendar`` attributes which haven't
                already been set will be inferred from the lookup
                header and cached for future use.

                .. versionadded:: NEXTVERSION

            {{aggregated_units: `str` or `None`, optional}}

            {{aggregated_calendar: `str` or `None`, optional}}

            {{init storage_options: `dict` or `None`, optional}}

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
            filename=filename,
            address=address,
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
