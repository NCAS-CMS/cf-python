import cfdm


class NetCDFWrite(cfdm.read_write.netcdf.NetCDFWrite):
    """A container for writing Fields to a netCDF dataset."""

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
                coordinate construct, including any groups
                structure. Note that the group structure may be
                different to the coordinate variable, and the
                basename.

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
        if extra is None:
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
