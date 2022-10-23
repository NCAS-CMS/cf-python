import numpy as np

from ..fullarray import FullArray
from .abstract import FragmentArray


class MissingFragmentArray(FragmentArray):
    """A CFA fragment array stored in a netCDF file TODODASKDOCS.

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
        calendar=None,
        source=None,
        copy=True,
    ):
        """**Initialisation**

        :Parameters:

            filename: `str` or `None`
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

            mask: `bool`
                If True (the default) then mask by convention when
                reading data from disk.

                A netCDF array is masked depending on the values of
                any of the netCDF variable attributes ``valid_min``,
                ``valid_max``, ``valid_range``, ``_FillValue`` and
                ``missing_value``.

            units: `str`, optional
                The units of the netCDF fragment variable. Set to
                `None` to indicate that there are no units. If units is False then units will be grappded on the fly TODO

            calendar: `str`, optional
                The calendar of the netCDF fragment variable. By
                default, or if set to `None`, then the CF default
                calendar is assumed, if applicable.

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

        array = FullArray(
            fill_value=np.ma.masked,
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
