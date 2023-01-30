import random
from string import hexdigits

import cfdm
import dask.array as da
import numpy as np

from ... import Bounds, Coordinate, DomainAncillary
from .netcdfread import NetCDFRead

_cfa_message = (
    "Writing CFA files has been temporarily disabled, "
    "and will return at version 4.0.0. "
    "CFA-0.4 functionality is still available at version 3.13.x."
)


class NetCDFWrite(cfdm.read_write.netcdf.NetCDFWrite):
    """A container for writing Fields to a netCDF dataset."""

    def __new__(cls, *args, **kwargs):
        """Store the NetCDFRead class.

        .. note:: If a child class requires a different NetCDFRead class
        than the one defined here, then it must be redefined in the
        child class.

        """
        instance = super().__new__(cls)
        instance._NetCDFRead = NetCDFRead
        return instance

    def _write_as_cfa(self, cfvar):
        """True if the variable should be written as a CFA variable.

        .. versionadded:: 3.0.0

        """
        if not self.write_vars["cfa"]:
            return False

        data = self.implementation.get_data(cfvar, None)
        if data is None:
            return False

        if data.size == 1:
            return False

        if isinstance(cfvar, (Coordinate, DomainAncillary)):
            return cfvar.ndim > 1

        if isinstance(cfvar, Bounds):
            return cfvar.ndim > 2

        return True

    def _customize_createVariable(self, cfvar, kwargs):
        """Customise keyword arguments for
        `netCDF4.Dataset.createVariable`.

        .. versionadded:: 3.0.0

        :Parameters:

            cfvar: cf instance that contains data

            kwargs: `dict`

        :Returns:

            `dict`
                Dictionary of keyword arguments to be passed to
                `netCDF4.Dataset.createVariable`.

        """
        kwargs = super()._customize_createVariable(cfvar, kwargs)

        if self._write_as_cfa(cfvar):
            raise ValueError(_cfa_message)

            kwargs["dimensions"] = ()
            kwargs["chunksizes"] = None

        return kwargs

    def _write_data(
        self,
        data,
        cfvar,
        ncvar,
        ncdimensions,
        unset_values=(),
        compressed=False,
        attributes={},
    ):
        """Write a Data object.

        .. versionadded:: 3.0.0

        :Parameters:

            data: `Data`

            cfvar: cf instance

            ncvar: `str`

            ncdimensions: `tuple` of `str`

            unset_values: sequence of numbers

        """
        g = self.write_vars

        if self._write_as_cfa(cfvar):
            raise ValueError(_cfa_message)

            self._write_cfa_data(ncvar, ncdimensions, data, cfvar)
            return

        # Still here?
        if compressed:
            # --------------------------------------------------------
            # Write data in its compressed form
            # --------------------------------------------------------
            data = data.source().source()

        # Get the dask array
        dx = da.asanyarray(data)

        # Convert the data type
        new_dtype = g["datatype"].get(dx.dtype)
        if new_dtype is not None:
            dx = dx.astype(new_dtype)

        # VLEN variables can not be assigned to by masked arrays
        # (https://github.com/Unidata/netcdf4-python/pull/465), so
        # fill missing data in string (as opposed to char) data types.
        if g["fmt"] == "NETCDF4" and dx.dtype.kind in "SU":
            dx = dx.map_blocks(
                self._filled_string_array,
                fill_value="",
                meta=np.array((), dx.dtype),
            )

        # Check for out-of-range values
        if g["warn_valid"]:
            dx = dx.map_blocks(
                self._check_valid,
                cfvar=cfvar,
                attributes=attributes,
                meta=np.array((), dx.dtype),
            )

        da.store(dx, g["nc"][ncvar], compute=True, return_stored=False)

    def _write_dimension_coordinate(
        self, f, key, coord, ncdim=None, coordinates=None
    ):
        """Write a coordinate variable and its bound variable to the
        file.

        This also writes a new netCDF dimension to the file and, if
        required, a new netCDF dimension for the bounds.

        .. versionadded:: 3.0.0

        :Parameters:

            f: Field construct

            key: `str`

            coord: Dimension coordinate construct

            ncdim: `str` or `None`
                The name of the netCDF dimension for this dimension
                coordinate construct, including any groups structure. Note
                that the group structure may be different to the
                corodinate variable, and the basename.

                .. versionadded:: 3.6.0

            coordinates: `list`
               This list may get updated in-place.

                .. versionadded:: 3.7.0

        :Returns:

            `str`
                The netCDF name of the dimension coordinate.

        """
        coord = self._change_reference_datetime(coord)

        return super()._write_dimension_coordinate(
            f, key, coord, ncdim=ncdim, coordinates=coordinates
        )

    def _write_scalar_coordinate(
        self, f, key, coord_1d, axis, coordinates, extra=None
    ):
        """Write a scalar coordinate and its bounds to the netCDF file.

        It is assumed that the input coordinate has size 1, but this is
        not checked.

        If an equal scalar coordinate has already been written to the file
        then the input coordinate is not written.

        .. versionadded:: 3.0.0

        :Parameters:

            f: Field construct

            key: `str`
                The coordinate construct key

            coord_1d: Coordinate construct

            axis: `str`
                The field's axis identifier for the scalar coordinate.

            coordinates: `list`

        :Returns:

            coordinates: `list`
                The updated list of netCDF auxiliary coordinate names.

        """
        # Unsafe to set mutable '{}' as default in the func signature.
        if extra is None:  # distinguish from falsy '{}'
            extra = {}

        coord_1d = self._change_reference_datetime(coord_1d)

        return super()._write_scalar_coordinate(
            f, key, coord_1d, axis, coordinates, extra=extra
        )

    def _write_auxiliary_coordinate(self, f, key, coord, coordinates):
        """Write auxiliary coordinates and bounds to the netCDF file.

        If an equal auxiliary coordinate has already been written to the
        file then the input coordinate is not written.

        .. versionadded:: 3.0.0

        :Parameters:

            f: Field construct

            key: `str`

            coord: Coordinate construct

            coordinates: `list`

        :Returns:

            coordinates: `list`
                The list of netCDF auxiliary coordinate names updated in
                place.

        **Examples:**

        >>> coordinates = _write_auxiliary_coordinate(f, 'aux2', coordinates)

        """
        coord = self._change_reference_datetime(coord)

        return super()._write_auxiliary_coordinate(f, key, coord, coordinates)

    def _change_reference_datetime(self, coord):
        """Change the units of a reference date-time value.

        .. versionadded:: 3.0.0

        :Parameters:

            coord: Coordinate instance

        :Returns:

                The coordinate construct with changed units.

        """
        reference_datetime = self.write_vars.get("reference_datetime")
        if not reference_datetime or not coord.Units.isreftime:
            return coord

        coord2 = coord.copy()
        try:
            coord2.reference_datetime = reference_datetime
        except ValueError:
            raise ValueError(
                "Can't override coordinate reference date-time "
                f"{coord.reference_datetime!r} with {reference_datetime!r}"
            )
        else:
            return coord2

    def _write_cfa_data(self, ncvar, ncdimensions, data, cfvar):
        """Write a CFA variable to the netCDF file.

        Any CFA private variables required will be autmatically created
        and written to the file.

        .. versionadded:: 3.0.0

        :Parameters:

            ncvar: `str`
                The netCDF name for the variable.

            ncdimensions: sequence of `str`

            netcdf_attrs: `dict`

            data: `Data`

        :Returns:

            `None`

        """
        raise ValueError(_cfa_message)

    def _random_hex_string(self, size=10):
        """Return a random hexadecimal string with the given number of
        characters.

        .. versionadded:: 3.0.0

        :Parameters:

            size: `int`, optional
                The number of characters in the generated string.

        :Returns:

            `str`
                The hexadecimal string.

        **Examples:**

        >>> _random_hex_string()
        'C3eECbBBcf'
        >>> _random_hex_string(6)
        '7a4acc'

        """
        return "".join(random.choice(hexdigits) for i in range(size))

    def _convert_to_builtin_type(self, x):
        """Convert a non-JSON-encodable object to a JSON-encodable
        built-in type.

        Possible conversions are:

        ==============  =============  ======================================
        Input object    Output object  numpy data types covered
        ==============  =============  ======================================
        numpy.bool_     bool           bool
        numpy.integer   int            int, int8, int16, int32, int64, uint8,
                                       uint16, uint32, uint64
        numpy.floating  float          float, float16, float32, float64
        ==============  =============  ======================================

        .. versionadded:: 3.0.0

        :Parameters:

            x:

        :Returns:

            'int' or `float` or `bool`

        **Examples:**

        >>> type(_convert_to_builtin_type(numpy.bool_(True)))
        bool
        >>> type(_convert_to_builtin_type(numpy.array([1.0])[0]))
        double
        >>> type(_convert_to_builtin_type(numpy.array([2])[0]))
        int

        """
        if isinstance(x, np.bool_):
            return bool(x)

        if isinstance(x, np.integer):
            return int(x)

        if isinstance(x, np.floating):
            return float(x)

        raise TypeError(
            f"{type(x)!r} object can't be converted to a JSON serializable "
            f"type: {x!r}"
        )

    def _check_valid(self, array, cfvar=None, attributes=None):
        """Checks for array values outside of the valid range.

        Specifically, checks array for out-of-range values, as
        defined by the valid_[min|max|range] attributes.

        .. versionadded:: 3.14.0

        :Parameters:

            array: `numpy.ndarray`
                The array to be checked.

            cfvar: construct
                The CF construct containing the array.

            attributes: `dict`
                The variable's CF properties.

        :Returns:

            `numpy.ndarray`
                The input array, unchanged.

        """
        super()._check_valid(cfvar, array, attributes)
        return array

    def _filled_string_array(self, array, fill_value=""):
        """Fill a string array.

        .. versionadded:: 3.14.0

        :Parameters:

            array: `numpy.ndarray`
                The `numpy` array with string (byte or unicode) data
                type.

        :Returns:

            `numpy.ndarray`
                The string array array with any missing data replaced
                by the fill value.

        """
        if np.ma.isMA(array):
            return array.filled(fill_value)

        return array


#    def _convert_dtype(self, array, new_dtype=None):
#        """Convert the data type of a numpy array.
#
#        .. versionadded:: 3.14.0
#
#        :Parameters:
#
#            array: `numpy.ndarray`
#                The `numpy` array
#
#            new_dtype: data-type
#                The new data type.
#
#        :Returns:
#
#            `numpy.ndarray`
#                The array with converted data type.
#
#        """
#        return array.astype(new_dtype)
#
