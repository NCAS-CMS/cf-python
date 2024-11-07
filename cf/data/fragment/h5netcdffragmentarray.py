# from ..array.h5netcdfarray import H5netcdfArray
# from .mixin import FragmentArrayMixin
#
#
# class H5netcdfFragmentArray(FragmentArrayMixin, H5netcdfArray):
#    """A netCDF fragment array accessed with `h5netcdf`.
#
#    .. versionadded:: NEXTVERSION
#
#    """
#
#    def __init__(
#        self,
#        filename=None,
#        address=None,
#        dtype=None,
#        shape=None,
#        aggregated_units=False,
#        aggregated_calendar=False,
#        attributes=None,
#        storage_options=None,
#        source=None,
#        copy=True,
#    ):
#        """**Initialisation**
#
#        :Parameters:
#
#            filename: (sequence of `str`), optional
#                The names of the netCDF fragment files containing the
#                array.
#
#            address: (sequence of `str`), optional
#                The name of the netCDF variable containing the
#                fragment array. Required unless *varid* is set.
#
#            dtype: `numpy.dtype`, optional
#                The data type of the aggregated array. May be `None`
#                if the numpy data-type is not known (which can be the
#                case for netCDF string types, for example). This may
#                differ from the data type of the netCDF fragment
#                variable.
#
#            shape: `tuple`, optional
#                The shape of the fragment within the aggregated
#                array. This may differ from the shape of the netCDF
#                fragment variable in that the latter may have fewer
#                size 1 dimensions.
#
#            {{init attributes: `dict` or `None`, optional}}
#
#                If *attributes* is `None`, the default, then the
#                attributes will be set from the netCDF variable during
#                the first `__getitem__` call.
#
#            {{aggregated_units: `str` or `None`, optional}}
#
#            {{aggregated_calendar: `str` or `None`, optional}}
#
#            {{init storage_options: `dict` or `None`, optional}}
#
#            {{init source: optional}}
#
#            {{init copy: `bool`, optional}}
#
#        """
#        super().__init__(
#            filename=filename,
#            address=address,
#            dtype=dtype,
#            shape=shape,
#            mask=True,
#            attributes=attributes,
#            storage_options=storage_options,
#            source=source,
#            copy=copy,
#        )
#
#        if source is not None:
#            try:
#                aggregated_units = source._get_component(
#                    "aggregated_units", False
#                )
#            except AttributeError:
#                aggregated_units = False
#
#            try:
#                aggregated_calendar = source._get_component(
#                    "aggregated_calendar", False
#                )
#            except AttributeError:
#                aggregated_calendar = False
#
#        self._set_component("aggregated_units", aggregated_units, copy=False)
#        self._set_component(
#            "aggregated_calendar", aggregated_calendar, copy=False
#        )
