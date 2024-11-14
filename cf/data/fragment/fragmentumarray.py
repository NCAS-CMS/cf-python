import cfdm

from ..array.umarray import UMArray


class FragmentUMArray(
    cfdm.data.fragment.mixin.FragmentFileArrayMixin, UMArray
):
    """A fragment of aggregated data in a PP or UM file.

    .. versionadded:: 3.14.0

    """


#
#    def __init__(
#        self,
#        filename=None,
#        address=None,
#        dtype=None,
#        shape=None,
#        storage_options=None,
#        min_file_versions=None,
#        unpack_aggregated_data=True,
#        aggregated_attributes=None,
#        source=None,
#        copy=True,
#    ):
#        """**Initialisation**
#
#        :Parameters:
#
#            filename: (sequence of `str`), optional
#                The names of the UM or PP files containing the fragment.
#
#            address: (sequence of `str`), optional
#                The start words in the files of the header.
#
#            dtype: `numpy.dtype`
#                The data type of the aggregated array. May be `None`
#                if the numpy data-type is not known (which can be the
#                case for netCDF string types, for example). This may
#                differ from the data type of the netCDF fragment
#                variable.
#
#            shape: `tuple`
#                The shape of the fragment within the aggregated
#                array. This may differ from the shape of the netCDF
#                fragment variable in that the latter may have fewer
#                size 1 dimensions.
#
#            {{init attributes: `dict` or `None`, optional}}
#
#                During the first `__getitem__` call, any of the
#                ``_FillValue``, ``add_offset``, ``scale_factor``,
#                ``units``, and ``calendar`` attributes which haven't
#                already been set will be inferred from the lookup
#                header and cached for future use.
#
#                .. versionadded:: NEXTVERSION
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
#            units: `str` or `None`, optional
#                Deprecated at version NEXTVERSION. Use the
#                *attributes* parameter instead.
#
#            calendar: `str` or `None`, optional
#                Deprecated at version NEXTVERSION. Use the
#                *attributes* parameter instead.
#
#        """
#        super().__init__(
#            filename=filename,
#            address=address,
#            dtype=dtype,
#            shape=shape,
#            mask=True,
#            unpack=True,
#            attributes=None,
#            storage_options=storage_options,
#            min_file_versions=min_file_versions,
#            source=source,
#            copy=copy
#        )
#
#        if source is not None:
#            try:
#                aggregated_attributes = source._get_component(
#                    "aggregated_attributes", None
#                )
#            except AttributeError:
#                aggregated_attributes = None
#
#            try:
#                unpack_aggregated_data = source._get_component(
#                    "unpack_aggregated_data", True
#                )
#            except AttributeError:
#                unpack_aggregated_data = True
#
#        self._set_component(
#            "unpack_aggregated_data",
#            unpack_aggregated_data,
#            copy=False,
#        )
#        self._set_component(
#            "aggregated_attributes", aggregated_attributes, copy=False
#        )
