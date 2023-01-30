import numpy as np

from ..array.fullarray import FullArray
from .abstract import FragmentArray


class MissingFragmentArray(FragmentArray):
    """A CFA fragment array that is wholly missing data.

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

            units: `str` or `None`, optional
                The units of the fragment data. Ignored, as the data
                are all missing values.

            calendar: `str` or `None`, optional
                The calendar of the fragment data. Ignored, as the data
                are all missing values.

            {{aggregated_units: `str` or `None`, optional}}"

            {{aggregated_calendar: `str` or `None`, optional}}

            {{init source: optional}}

            {{init copy: `bool`, optional}}

        """
        if source is not None:
            super().__init__(source=source, copy=copy)
            return

        array = FullArray(
            fill_value=np.ma.masked,
            dtype=dtype,
            shape=shape,
            units=None,
            calendar=None,
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
