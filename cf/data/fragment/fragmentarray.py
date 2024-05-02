import cfdm

from ..array.abstract import Array
from ..array.mixin import FileArrayMixin, IndexMixin
from .h5netcdffragmentarray import H5netcdfFragmentArray
from .mixin import FragmentArrayMixin
from .netcdf4fragmentarray import NetCDF4FragmentArray
from .umfragmentarray import UMFragmentArray


_fragment = {'netCDF4': NetCDF4FragmentArray
             'h5netcdf': H5netcdfFragmentArray,
             'um': UMFragmentArray}

# REVIEW: TODO getitem: `NetCDFFragmentArray`: new inheritance to allow for different netCDF backends
class FragmentArray(
    FragmentArrayMixin,
    IndexMixin,
    FileArrayMixin,
    cfdm.data.mixin.FileArrayMixin,
    Array,
):
    """A netCDF fragment array.

    Access will be with either `netCDF4` or `h5netcdf`.

    .. versionadded:: NEXTVERSION

    """

    # REVIEW: h5: `__init__`: replace units/calendar API with attributes
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
                The locations fragment datasets containing the array.

            address: (sequence of `str`), optional
                How to find the fragments in the fragment datasets.

            dtype: `numpy.dtype`, optional
                The data type of the aggregated array. May be `None`
                if is not known. This may differ from the data type of
                the fragment's data.

            shape: `tuple`, optional
                The shape of the fragment in its canonical form.

            {{init attributes: `dict` or `None`, optional}}

                If *attributes* is `None`, the default, then the
                attributes will be set from the fragment dataset
                during the first `__getitem__` call.

            {{aggregated_units: `str` or `None`, optional}}

            {{aggregated_calendar: `str` or `None`, optional}}

            {{init storage_options: `dict` or `None`, optional}}

            {{init source: optional}}

            {{init copy: `bool`, optional}}

        """
        super().__init__(
            source=source,
            copy=copy,
        )

        if source is not None:
            try:
                shape = source._get_component("shape", None)
            except AttributeError:
                shape = None

            try:
                filename = source._get_component("filename", None)
            except AttributeError:
                filename = None

            try:
                address = source._get_component("address", None)
            except AttributeError:
                address = None

            try:
                dtype = source._get_component("dtype", None)
            except AttributeError:
                dtype = None

            try:
                attributes = source._get_component("attributes", None)
            except AttributeError:
                attributes = None

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

            try:
                storage_options = source._get_component(
                    "storage_options", None
                )
            except AttributeError:
                storage_options = None

        if filename is not None:
            if isinstance(filename, str):
                filename = (filename,)
            else:
                filename = tuple(filename)

            self._set_component("filename", filename, copy=False)

        if address is not None:
            if isinstance(address, int):
                address = (address,)
            else:
                address = tuple(address)

            self._set_component("address", address, copy=False)

        if storage_options is not None:
            self._set_component("storage_options", storage_options, copy=False)

        self._set_component("shape", shape, copy=False)
        self._set_component("dtype", dtype, copy=False)
        self._set_component("attributes", attributes, copy=False)
        self._set_component("mask", True, copy=False)

        self._set_component("aggregated_units", aggregated_units, copy=False)
        self._set_component(
            "aggregated_calendar", aggregated_calendar, copy=False
        )

        # By default, close the file after data array access
        self._set_component("close", True, copy=False)

    # REVIEW: getitem: `_get_array`: new method to convert subspace to numpy array
    def _get_array(self, index=None):
        """Returns a subspace of the dataset variable.

        The method acts as a factory for either a
        `NetCDF4FragmentArray`, `H5netcdfFragmentArray`, or
        `UMFragmentArray` class, and it is the result of calling
        `!_get_array` on the newly created instance that is returned.

        `H5netcdfFragmentArray` will only be used if
        `NetCDF4FragmentArray` returns a `FileNotFoundError`
        exception; and `UMFragmentArray` will only be used
        if `H5netcdfFragmentArray` returns an `Exception`.

        .. versionadded:: NEXTVERSION

        .. seealso:: `__array__`, `index`

        :Parameters:

            {{index: `tuple` or `None`, optional}}

               When a `tuple`, there must be a distinct entry for each
               fragment dimension.

        :Returns:

            `numpy.ndarray`
                The subspace.

        """
        kwargs = {
            "dtype": self.dtype,
            "shape": self.shape,
            "aggregated_units": self.get_aggregated_units(None),
            "aggregated_calendar": self.get_aggregated_calendar(None),
            "attributes": self.get_attributes(None),
            "copy": False,
        }

        # Loop round the files, returning as soon as we find one that
        # is accessible.
        filenames = self.get_filenames()
        for filename, address in zip(filenames, self.get_addresses()):
            kwargs["filename"] = filename
            kwargs["address"] = address
            kwargs["storage_options"] = self.get_storage_options(
                create_endpoint_url=False
            )
            
            for backend in dataset_backends:
                try:
                    return _fragment[backend](**kwargs)._get_array(index)
                except FileNotFoundError:
                    pass
                except KeyError:
                    raise ValueError("unknown backend: T sadasds TODO")
                
        # Still here?
        if len(filenames) == 1:
            raise FileNotFoundError(f"No such fragment file: {filenames[0]}")

        raise FileNotFoundError(f"No such fragment files: {filenames}")
