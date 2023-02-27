from urllib.parse import urlparse

import netCDF4

from ..array.netcdfarray import NetCDFArray
from .mixin import FragmentArrayMixin


class NetCDFFragmentArray(FragmentArrayMixin, NetCDFArray):
    """A CFA fragment array stored in a netCDF file.

    .. versionadded:: 3.14.0

    """

    def __init__(
        self,
        filenames=None,
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

            filenames: `tuple`
                The names of the netCDF fragment files containing the
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
        group = None  # TODO ???

        super().__init__(
            filename=filenames,
            ncvar=address,
            group=group,
            dtype=dtype,
            shape=shape,
            mask=True,
            units=units,
            calendar=calendar,
            source=source,
            copy=False,
        )

        if source is not None:
            try:
                address = source._get_component("address", False)
            except AttributeError:
                address = None

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

        if address is not None:
            self._set_component("address", address, copy=False)

        self._set_component("aggregated_units", aggregated_units, copy=False)
        self._set_component(
            "aggregated_calendar", aggregated_calendar, copy=False
        )

    def open(self):
        """Returns an open dataset containing the data array.

        When multiple fragment files have been provided an attempt is
        made to open each one, in arbitrary order, and the
        `netCDF4.Dataset` is returned from the first success.

        .. versionadded:: TODOCFAVER

        :Returns:

            `netCDF4.Dataset`

        """
        filenames = self.get_filenames()
        for filename, address in zip(filenames, self.get_addresses()):
            url = urlparse(filename)
            if url.scheme == "file":
                # Convert file URI into an absolute path
                filename = url.path

            try:
                nc = netCDF4.Dataset(filename, "r")
            except FileNotFoundError:
                continue
            except RuntimeError as error:
                raise RuntimeError(f"{error}: {filename}")

            self._set_component("ncvar", address, copy=False)
            return nc

        raise FileNotFoundError(
            f"No such netCDF fragment files: {tuple(filenames)}"
        )
