import cfdm
import netCDF4
import numpy as np

_cfa_message = (
    "Reading CFA files has been temporarily disabled, "
    "but will return at a version later soon after TODODASKVER."
    "CFA-0.4 functionality is still available at versions<=3.13.1"
)


class NetCDFRead(cfdm.read_write.netcdf.NetCDFRead):
    """A container for instantiating Fields from a netCDF dataset.

    .. versionadded:: 3.0.0

    """

    def _ncdimensions(self, ncvar, ncdimensions=None, parent_ncvar=None):
        """Return a list of the netCDF dimensions corresponding to a
        netCDF variable.

        If the variable has been compressed then the *implied
        uncompressed* dimensions are returned.

        For a CFA variable, the netCDF dimensions are taken from the
        'cfa_dimensions' netCDF attribute.

        .. versionadded:: 3.0.0

        :Parameters:

            ncvar: `str`
                The netCDF variable name.

            ncdimensions: sequence of `str`, optional
                Use these netCDF dimensions, rather than retrieving them
                from the netCDF variable itself. This allows the
                dimensions of a domain variable to be parsed. Note that
                this only parameter only needs to be used once because the
                parsed domain dimensions are automatically stored in
                `self.read_var['domain_ncdimensions'][ncvar]`.

                .. versionadded:: 3.11.0

            parent_ncvar: `str`, optional
                TODO

                .. versionadded:: TODO

        :Returns:

            `list`
                The netCDF dimension names spanned by the netCDF variable.

        **Examples**

        >>> n._ncdimensions('humidity')
        ['time', 'lat', 'lon']

        For a variable compressed by gathering:

           dimensions:
             lat=73;
             lon=96;
             landpoint=2381;
             depth=4;
           variables:
             int landpoint(landpoint);
               landpoint:compress="lat lon";
             float landsoilt(depth,landpoint);
               landsoilt:long_name="soil temperature";
               landsoilt:units="K";

        we would have

        >>> n._ncdimensions('landsoilt')
        ['depth', 'lat', 'lon']

        """
        g = self.read_vars

        cfa = (
            g.get("cfa")
            and ncvar not in g["external_variables"]
            and g["variable_attributes"][ncvar].get("cf_role")
            == "cfa_variable"
        )

        if not cfa:
            return super()._ncdimensions(
                ncvar, ncdimensions=ncdimensions, parent_ncvar=parent_ncvar
            )

        # Still here? Then we have a CFA variable.
        raise ValueError(_cfa_message)

        # Leave the following CFA code here for now, as it may be
        # useful at version>TODODASKVER
        ncdimensions = (
            g["variable_attributes"][ncvar].get("cfa_dimensions", "").split()
        )

        return list(map(str, ncdimensions))

    def _get_domain_axes(self, ncvar, allow_external=False, parent_ncvar=None):
        """Return the domain axis identifiers that correspond to a
        netCDF variable's netCDF dimensions.

        For a CFA variable, the netCDF dimensions are taken from the
        'cfa_dimensions' netCDF attribute.

        :Parameter:

            ncvar: `str`
                The netCDF variable name.

            allow_external: `bool`
                If `True` and *ncvar* is an external variable then return an
                empty list.

            parent_ncvar: `str`, optional
                TODO

                .. versionadded:: TODO

        :Returns:

            `list`

        **Examples**

        >>> r._get_domain_axes('areacello')
        ['domainaxis0', 'domainaxis1']

        >>> r._get_domain_axes('areacello', allow_external=True)
        []

        """
        g = self.read_vars

        cfa = (
            g.get("cfa")
            and ncvar not in g["external_variables"]
            and g["variable_attributes"][ncvar].get("cf_role")
            == "cfa_variable"
        )

        if not cfa:
            return super()._get_domain_axes(
                ncvar=ncvar,
                allow_external=allow_external,
                parent_ncvar=parent_ncvar,
            )

        # Still here?
        raise ValueError(_cfa_message)

        # Leave the following CFA code here, as it may be useful at
        # v4.0.0.
        cfa_dimensions = (
            g["variable_attributes"][ncvar].get("cfa_dimensions", "").split()
        )

        ncdim_to_axis = g["ncdim_to_axis"]
        axes = [
            ncdim_to_axis[ncdim]
            for ncdim in cfa_dimensions
            if ncdim in ncdim_to_axis
        ]

        return axes

    def _create_data(
        self,
        ncvar,
        construct=None,
        unpacked_dtype=False,
        uncompress_override=None,
        parent_ncvar=None,
        coord_ncvar=None,
    ):
        """Create data for a netCDF or CFA-netCDF variable.

        .. versionadded:: 3.0.0

        :Parameters:

            ncvar: `str`
                The name of the netCDF variable that contains the data.

            construct: optional

            unpacked_dtype: `False` or `numpy.dtype`, optional

            uncompress_override: `bool`, optional

            parent_ncvar: `str`, optional

            coord_ncvar: `str`, optional

                .. versionadded:: TODO

        :Returns:

            `Data`

        """
        g = self.read_vars

        is_cfa_variable = (
            g.get("cfa")
            and construct.get_property("cf_role", None) == "cfa_variable"
        )

        if not is_cfa_variable:
            # --------------------------------------------------------
            # Create data for a normal netCDF variable
            # --------------------------------------------------------
            return super()._create_data(
                ncvar=ncvar,
                construct=construct,
                unpacked_dtype=unpacked_dtype,
                uncompress_override=uncompress_override,
                parent_ncvar=parent_ncvar,
                coord_ncvar=coord_ncvar,
            )

        # ------------------------------------------------------------
        # Still here? Then create data for a CFA netCDF variable
        # ------------------------------------------------------------
        raise ValueError(_cfa_message)

    def _create_Data(
        self,
        array=None,
        units=None,
        calendar=None,
        ncvar=None,
        **kwargs,
    ):
        """Create a Data object.

        .. versionadded:: 3.0.0

        :Parameters:

            ncvar: `str`
                The netCDF variable from which to get units and calendar.

        """
        if array.dtype is None:
            # The array is based on a netCDF VLEN variable, and
            # therefore has unknown data type. To find the correct
            # data type (e.g. "|S7"), we need to read the data into
            # memory.
            array = self._array_from_variable(ncvar)

        chunks = self.read_vars.get("chunks", "auto")

        return super()._create_Data(
            array=array,
            units=units,
            calendar=calendar,
            ncvar=ncvar,
            chunks=chunks,
            **kwargs,
        )

    def _customize_read_vars(self):
        """Customize the read parameters.

        .. versionadded:: 3.0.0

        """
        super()._customize_read_vars()

        g = self.read_vars

        # ------------------------------------------------------------
        # Find out if this is a CFA file
        # ------------------------------------------------------------
        g["cfa"] = "CFA" in g["global_attributes"].get(
            "Conventions", ()
        )  # TODO

        # ------------------------------------------------------------
        # Do not create fields from CFA private variables
        # ------------------------------------------------------------
        if g["cfa"]:
            raise ValueError(_cfa_message)

            # Leave the following CFA code here for now, as it may be
            # useful at version>TODODASKVER
            for ncvar in g["variables"]:
                if (
                    g["variable_attributes"][ncvar].get("cf_role", None)
                    == "cfa_private"
                ):
                    g["do_not_create_field"].add(ncvar)

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        if g["cfa"]:
            raise ValueError(_cfa_message)

            # Leave the following CFA code here for now, as it may be
            # useful at version>TODODASKVER
            for ncvar, ncdims in tuple(g["variable_dimensions"].items()):
                if ncdims != ():
                    continue

                if not (
                    ncvar not in g["external_variables"]
                    and g["variable_attributes"][ncvar].get("cf_role")
                    == "cfa_variable"
                ):
                    continue

                ncdimensions = (
                    g["variable_attributes"][ncvar]
                    .get("cfa_dimensions", "")
                    .split()
                )
                if ncdimensions:
                    g["variable_dimensions"][ncvar] = tuple(
                        map(str, ncdimensions)
                    )

    def _array_from_variable(self, ncvar):
        """Convert a netCDF variable to a `numpy` array.

        For char and string netCDF types, the array is processed into
        a CF-friendly format.

        .. versionadded:: TODODASKVER

        .. note:: This code is copied from
                  `cfdm.NetCDFArray.__getitem__`.

        :Parmaeters:

            ncvar: `str`
                The netCDF variable name of the variable to be read.

        :Returns:

            `numpy.ndarray`
                The array containing the variable's data.

        """
        variable = self.read_vars["variable_dataset"][ncvar][ncvar]
        array = variable[...]

        string_type = isinstance(array, str)
        if string_type:
            # --------------------------------------------------------
            # A netCDF string type scalar variable comes out as Python
            # str object, so convert it to a numpy array.
            # --------------------------------------------------------
            array = np.array(array, dtype=f"S{len(array)}")

        if not variable.ndim:
            # Hmm netCDF4 has a thing for making scalar size 1 , 1d
            array = array.squeeze()

        kind = array.dtype.kind
        if not string_type and kind in "SU":
            # --------------------------------------------------------
            # Collapse (by concatenation) the outermost (fastest
            # varying) dimension of char array into
            # memory. E.g. [['a','b','c']] becomes ['abc']
            # --------------------------------------------------------
            if kind == "U":
                array = array.astype("S")

            array = netCDF4.chartostring(array)
            shape = array.shape
            array = np.array([x.rstrip() for x in array.flat], dtype="S")
            array = np.reshape(array, shape)
            array = np.ma.masked_where(array == b"", array)

        elif not string_type and kind == "O":
            # --------------------------------------------------------
            # A netCDF string type N-d (N>=1) variable comes out as a
            # numpy object array, so convert it to numpy string array.
            # --------------------------------------------------------
            array = array.astype("S")  # , copy=False)

            # --------------------------------------------------------
            # netCDF4 does not auto-mask VLEN variable, so do it here.
            # --------------------------------------------------------
            array = np.ma.where(array == b"", np.ma.masked, array)

        elif not string_type and kind == "O":
            # --------------------------------------------------------
            # A netCDF string type N-d (N>=1) variable comes out as a
            # numpy object array, so convert it to numpy string array.
            # --------------------------------------------------------
            array = array.astype("S")

        return array
