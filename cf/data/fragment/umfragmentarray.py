from ..array.umarray import UMArray
from .mixin import FragmentArrayMixin, FragmentFileArrayMixin


class UMFragmentArray(FragmentFileArrayMixin, FragmentArrayMixin, UMArray):
    """A CFA fragment array stored in a UM or PP file.

    .. versionadded:: 3.14.0

    """

    def __init__(
        self,
        filenames=None,
        addresses=None,
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

            filenames: sequence of `str`, optional
                The names of the UM or PP file containing the fragment.

            addresses: sequence of `str`, optional
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
            dtype=dtype,
            shape=shape,
            units=units,
            calendar=calendar,
            source=source,
            copy=False,
        )

        if source is not None:
            try:
                filenames = source._get_component("filenames", None)
            except AttributeError:
                filenames = None

            try:
                addresses = source._get_component("addresses ", None)
            except AttributeError:
                addresses = None

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

        if filenames:
            self._set_component("filenames", tuple(filenames), copy=False)

        if addresses:
            self._set_component("addresses ", tuple(addresses), copy=False)

        self._set_component("aggregated_units", aggregated_units, copy=False)
        self._set_component(
            "aggregated_calendar", aggregated_calendar, copy=False
        )

    def get_formats(self):
        """TODOCFADOCS

        .. versionadded:: TODOCFAVER

        .. seealso:: `get_filenames`, `get_addresses`

        :Returns:

            `tuple`

        """
        return ("um",) * len(self.get_filenames())

    def open(self):
        """Returns an open dataset containing the data array.

        When multiple fragment files have been provided an attempt is
        made to open each one, in arbitrary order, and the
        `umfile_lib.File` is returned from the first success.

        .. versionadded:: TODOCFAVER

        :Returns:

            `umfile_lib.File`

        """
        # Loop round the files, returning as soon as we find one that
        # works.
        filenames = self.get_filenames()
        for filename, address in zip(filenames, self.get_addresses()):
            url = urlparse(filename)
            if url.scheme == "file":
                # Convert file URI into an absolute path
                filename = url.path

                try:
                    f = File(
                        path=filename,
                        byte_ordering=None,
                        word_size=None,
                        fmt=None,
                    )
                except FileNotFoundError:
                    continue
                except Exception as error:
                    try:
                        f.close_fd()
                    except Exception:
                        pass

                    raise Exception(f"{error}: {filename}")

                self._set_component("header_offset", address, copy=False)
                return f

        raise FileNotFoundError(
            f"No such PP or UM fragment files: {filenames}"
        )
