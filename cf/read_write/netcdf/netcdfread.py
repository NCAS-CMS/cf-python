import cfdm
import netCDF4
import numpy as np

"""
TODOCFA: remove aggregation_* properties from constructs

TODOCFA: Create auxiliary coordinates from non-standardized terms

TODOCFA: Reference instruction variables (and/or set as
         "do_not_create_field")

TODOCFA: Create auxiliary coordinates from non-standardized terms

TODOCFA: Consider scanning for cfa variables to the top (e.g. where
         scanning for geometry varables is). This will probably need a
         change in cfdm so that a customizable hook can be overlaoded
         (like `_customize_read_vars` does).

TODOCFA: What about groups/netcdf_flattener?

"""


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
        'aggregated_dimensions' netCDF attribute.

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

        if not self._is_cfa_variable(ncvar):
            return super()._ncdimensions(
                ncvar, ncdimensions=ncdimensions, parent_ncvar=parent_ncvar
            )

        # Still here? Then we have a CFA variable.
        ncdimensions = self.read_vars["variable_attributes"][ncvar][
            "aggregated_dimensions"
        ].split()

        return list(map(str, ncdimensions))

    def _get_domain_axes(self, ncvar, allow_external=False, parent_ncvar=None):
        """Return the domain axis identifiers that correspond to a
        netCDF variable's netCDF dimensions.

        For a CFA variable, the netCDF dimensions are taken from the
        'aggregated_dimensions' netCDF attribute.

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
        if not self._is_cfa_variable(ncvar):
            return super()._get_domain_axes(
                ncvar=ncvar,
                allow_external=allow_external,
                parent_ncvar=parent_ncvar,
            )

        # Still here? Then we have a CFA variable.
        g = self.read_vars

        ncdimensions = g["variable_attributes"][ncvar][
            "aggregated_dimensions"
        ].split()

        ncdim_to_axis = g["ncdim_to_axis"]
        axes = [
            ncdim_to_axis[ncdim]
            for ncdim in ncdimensions
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
        if not self._is_cfa_variable(ncvar):
            # Create data for a normal netCDF variable
            return super()._create_data(
                ncvar=ncvar,
                construct=construct,
                unpacked_dtype=unpacked_dtype,
                uncompress_override=uncompress_override,
                parent_ncvar=parent_ncvar,
                coord_ncvar=coord_ncvar,
            )

        # ------------------------------------------------------------
        # Still here? Then create data for a CFA-netCDF variable
        # ------------------------------------------------------------
        cfa_array, kwargs = self._create_cfanetcdfarray(
            ncvar,
            unpacked_dtype=unpacked_dtype,
            coord_ncvar=coord_ncvar,
        )

        # Return the data
        return self._create_Data(
            cfa_array,
            units=kwargs["units"],
            calendar=kwargs["calendar"],
            ncvar=ncvar,
        )

    def _is_cfa_variable(self, ncvar):
        """Return True if *ncvar* is a CFA variable.

        .. versionadded:: TODODASKVER

        :Parameters:

            ncvar: `str`
                The name of the netCDF variable.

        :Returns:

            `bool`
                Whether or not *ncvar* is a CFA variable.

        """
        g = self.read_vars

        if not g["cfa"] or ncvar in g["external_variables"]:
            return False

        attributes = g["variable_attributes"][ncvar]

        # TODOCFA: test on the version of CFA given by g["cfa"]. See
        #          also `_customize_read_vars`.
        cfa = "aggregated_dimensions" in attributes
        if cfa:
            # TODOCFA: Modify this message for v4.0.0
#            raise ValueError(
#                "The reading of CFA files has been temporarily disabled, "
#                "but will return for CFA-0.6 files at version 4.0.0. "
#                "CFA-0.4 functionality is still available at version 3.13.1."
#            )

            # TODOCFA: The 'return' remains when the exception is
            #          removed at v4.0.0.
            return True

        cfa_04 = attributes.get("cf_role") == "cfa_variable"
        if cfa_04:
            # TODOCFA: Modify this message for v4.0.0.
            raise ValueError(
                "The reading of CFA-0.4 files was permanently disabled at "
                "version TODODASKVER. However, CFA-0.4 functionality is "
                "still available at version 3.13.1. "
                "The reading and writing of CFA-0.6 files will become "
                "available at version 4.0.0."
            )

        return False

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

        data = super()._create_Data(
            array=array,
            units=units,
            calendar=calendar,
            ncvar=ncvar,
            chunks=chunks,
            **kwargs,
        )
        self._cache_data_elements(data, ncvar)

        return data

    def _customize_read_vars(self):
        """Customize the read parameters.

        .. versionadded:: 3.0.0

        """
        super()._customize_read_vars()

        g = self.read_vars

        # ------------------------------------------------------------
        # Find out if this is a CFA file
        # ------------------------------------------------------------
        g["cfa"] = "CFA" in g["global_attributes"].get("Conventions", ())

        if g["cfa"]:
            attributes = g["variable_attributes"]
            dimensions = g["variable_dimensions"]

            # Do not create fields from CFA private
            # variables. TODOCFA: get private variables from
            # CFANetCDFArray instances
            for ncvar in g["variables"]:
                if attributes[ncvar].get("cf_role", None) == "cfa_private":
                    g["do_not_create_field"].add(ncvar)

            for ncvar, ncdims in tuple(dimensions.items()):
                if ncdims != ():
                    continue

                if not (
                    ncvar not in g["external_variables"]
                    and "aggregated_dimensions" in attributes[ncvar]
                ):
                    continue

                ncdimensions = attributes[ncvar][
                    "aggregated_dimensions"
                ].split()
                if ncdimensions:
                    dimensions[ncvar] = tuple(map(str, ncdimensions))

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

    def _cache_data_elements(self, data, ncvar):
        """Cache selected element values.

        Updates *data* in-place to store its first, second and last
        element values inside its ``custom`` dictionary.

        Doing this here is cheap because only the individual elements
        are read from the already-open file, as opposed to being
        retrieved from *data* (which would require a whole dask chunk
        to be read to get each single value).

        .. versionadded:: TODODASKVER

        :Parameters:

            data: `Data`
                The data to be updated with its cached values.

            ncvar: `str`
                The name of the netCDF variable that contains the
                data.

        :Returns:

            `None`

        """
        g = self.read_vars

        # Get the netCDF4.Variable for the data
        if g["has_groups"]:
            group, name = self._netCDF4_group(
                g["variable_grouped_dataset"][ncvar], ncvar
            )
            variable = group.variables.get(name)
        else:
            variable = g["variables"].get(ncvar)

        # Get the required element values
        size = variable.size
        if size == 1:
            value = variable[(slice(0, 1, 1),) * variable.ndim]
            values = (value, None, value)
        elif size == 3:
            values = variable[...].flatten()
        else:
            ndim = variable.ndim
            values = (
                variable[(slice(0, 1, 1),) * ndim],
                None,
                variable[(slice(-1, None, 1),) * ndim],
            )

        # Create a dictionary of the element values
        elements = {}
        for element, value in zip(
            ("first_element", "second_element", "last_element"),
            values,
        ):
            if value is None:
                continue

            if np.ma.is_masked(value):
                value = np.ma.masked
            else:
                try:
                    value = value.item()
                except AttributeError:
                    # A netCDF string type scalar variable comes out
                    # as Python str object, which has no 'item'
                    # method.
                    pass

            elements[element] = value

        # Store the elements in the data object
        data._set_cached_elements(elements)

    def _create_cfanetcdfarray(
        self,
        ncvar,
        unpacked_dtype=False,
        coord_ncvar=None,
    ):
        """Create a CFA-netCDF variable array.

        .. versionadded:: (cfdm) 1.10.0.1

        :Parameters:

            ncvar: `str`

            unpacked_dtype: `False` or `numpy.dtype`, optional

            coord_ncvar: `str`, optional

        :Returns:

            (`CFANetCDFArray`, `dict`)
                The new `NetCDFArray` instance and dictionary of the
                kwargs used to create it.

        """
        # Get the kwargs needed to instantiate a general NetCDFArray
        # instance
        kwargs = self._create_netcdfarray(
            ncvar,
            unpacked_dtype=unpacked_dtype,
            coord_ncvar=coord_ncvar,
            return_kwargs_only=True,
        )

        # Get rid of the incorrect shape
        kwargs.pop("shape", None)

        # Add the aggregated_data attribute (that can be used by
        # dask.base.tokenize).
        kwargs["instructions"] = self.read_vars["variable_attributes"][
            ncvar
        ].get("aggregated_data")

        # Use the kwargs to create a specialised CFANetCDFArray
        # instance
        array = self.implementation.initialise_CFANetCDFArray(**kwargs)

        return array, kwargs
