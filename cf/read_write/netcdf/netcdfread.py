import cfdm
import numpy as np
from packaging.version import Version

"""
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

        # ------------------------------------------------------------
        # Still here? Then we have a CFA-netCDF variable.
        # ------------------------------------------------------------
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
        cfa_term=None,
    ):
        """Create data for a netCDF or CFA-netCDF variable.

        .. versionadded:: 3.0.0

        :Parameters:

            ncvar: `str`
                The name of the netCDF variable that contains the
                data. See the *cfa_term* parameter.

            construct: optional

            unpacked_dtype: `False` or `numpy.dtype`, optional

            uncompress_override: `bool`, optional

            coord_ncvar: `str`, optional

            cfa_term: `dict`, optional
                The name of a non-standard aggregation instruction
                term from which to create the data. If set then
                *ncvar* must be the value of the term in the
                ``aggregation_data`` attribute.

                .. versionadded:: TODOCFAVER

        :Returns:

            `Data`

        """
        if cfa_term is None and not self._is_cfa_variable(ncvar):
            # Create data for a normal netCDF variable
            data = super()._create_data(
                ncvar=ncvar,
                construct=construct,
                unpacked_dtype=unpacked_dtype,
                uncompress_override=uncompress_override,
                parent_ncvar=parent_ncvar,
                coord_ncvar=coord_ncvar,
            )

            # Set the CFA write status to True when there is exactly
            # one dask chunk
            if data.npartitions == 1:
                data._set_cfa_write(True)

            self._cache_data_elements(data, ncvar)

            return data

        # ------------------------------------------------------------
        # Still here? Create data for a CFA variable
        # ------------------------------------------------------------
        if construct is not None:
            # Remove the aggregation attributes from the construct
            self.implementation.del_property(
                construct, "aggregation_dimensions", None
            )
            aggregation_data = self.implementation.del_property(
                construct, "aggregation_data", None
            )
        else:
            aggregation_data = None

        cfa_array, kwargs = self._create_cfanetcdfarray(
            ncvar,
            unpacked_dtype=unpacked_dtype,
            coord_ncvar=coord_ncvar,
            non_standard_term=cfa_term,
        )

        data = self._create_Data(
            cfa_array,
            ncvar,
            units=kwargs["units"],
            calendar=kwargs["calendar"],
        )

        # Set the CFA write status to True iff each non-aggregated
        # axis has exactly one dask storage chunk
        if cfa_term is None:
            cfa_write = True
            for n, numblocks in zip(
                cfa_array.get_fragment_shape(), data.numblocks
            ):
                if n == 1 and numblocks > 1:
                    # Note: 'n == 1' is True for non-aggregated axes
                    cfa_write = False
                    break

            data._set_cfa_write(cfa_write)

            # Store the 'aggregation_data' attribute
            if aggregation_data:
                data.cfa_set_aggregation_data(aggregation_data)

            # Store the file substitutions
            data.cfa_set_file_substitutions(kwargs.get("substitutions"))

        # Note: We don't cache elements from aggregated data

        return data

    def _is_cfa_variable(self, ncvar):
        """Return True if *ncvar* is a CFA aggregated variable.

        .. versionadded:: 3.14.0

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

        return "aggregated_dimensions" in g["variable_attributes"][ncvar]

    def _create_Data(
        self,
        array,
        ncvar,
        units=None,
        calendar=None,
        ncdimensions=(),
        **kwargs,
    ):
        """Create a Data object from a netCDF variable.

        .. versionadded:: 3.0.0

        :Parameters:

            array: `Array`
                The file array.

            ncvar: `str`
                The netCDF variable containing the array.

            ncdimensions: sequence of `str`, optional
                The netCDF dimensions spanned by the array.

                .. versionadded:: 3.14.fill_value:
                The units of *array*. By default, or if `None`, it is
                assumed that there are no units.

            calendar: `str`, optional
                The calendar of *array*. By default, or if `None`, it is
                assumed that there is no calendar.

            kwargs: optional
                Extra parameters to pass to the initialisation of the
                returned `Data` object.

        :Returns:

            `Data`

        """
        if array.dtype is None:
            # The array is based on a netCDF VLEN variable, and
            # therefore has unknown data type. To find the correct
            # data type (e.g. "|S7"), we need to read the data into
            # memory.
            array = array[...]

        # Parse dask chunks
        chunks = self._parse_chunks(ncvar)

        data = super()._create_Data(
            array,
            ncvar,
            units=units,
            calendar=calendar,
            chunks=chunks,
            **kwargs,
        )

        return data

    def _customize_read_vars(self):
        """Customize the read parameters.

        Take the opportunity to apply CFA updates to
        `read_vars['variable_dimensions']` and
        `read_vars['do_not_create_field']`.

        .. versionadded:: 3.0.0

        """
        from re import split

        super()._customize_read_vars()

        g = self.read_vars

        # Check the 'Conventions' for CFA
        Conventions = g["global_attributes"].get("Conventions", "")

        # If the string contains any commas, it is assumed to be a
        # comma-separated list.
        all_conventions = split(",\s*", Conventions)
        if all_conventions[0] == Conventions:
            all_conventions = Conventions.split()

        CFA_version = None
        for c in all_conventions:
            if c.startswith("CFA-"):
                CFA_version = c.replace("CFA-", "", 1)
                break

            if c == "CFA":
                # Versions <= 3.13.1 wrote CFA-0.4 files with a plain
                # 'CFA' in the Conventions string
                CFA_version = "0.4"
                break

        g["cfa"] = CFA_version is not None
        if g["cfa"]:
            # --------------------------------------------------------
            # This is a CFA-netCDF file
            # --------------------------------------------------------

            # Check the CFA version
            g["CFA_version"] = Version(CFA_version)
            if g["CFA_version"] < Version("0.6.2"):
                raise ValueError(
                    f"Can't read file {g['filename']} that uses obselete "
                    f"CFA conventions version CFA-{CFA_version}. "
                    "(Note that version 3.13.1 can be used to read and "
                    "write CFA-0.4 files.)"
                )

            # Get the pdirectory path of the CFA-netCDF file being
            # read
            from os.path import abspath
            from pathlib import PurePath

            g["cfa_dir"] = PurePath(abspath(g["filename"])).parent

            # Process the aggregation instruction variables, and the
            # aggregated dimensions.
            dimensions = g["variable_dimensions"]
            attributes = g["variable_attributes"]
            for ncvar, attributes in attributes.items():
                if "aggregate_dimensions" not in attributes:
                    # This is not an aggregated variable
                    continue

                # Set the aggregated variable's dimensions as its
                # aggregated dimensions
                ncdimensions = attributes["aggregated_dimensions"].split()
                dimensions[ncvar] = tuple(map(str, ncdimensions))

                # Do not create fields/domains from aggregation
                # instruction variables
                parsed_aggregated_data = self._parse_aggregated_data(
                    ncvar, attributes.get("aggregated_data")
                )
                for x in parsed_aggregated_data:
                    variable = tuple(x.items())[0][1]
                    g["do_not_create_field"].add(variable)

    def _cache_data_elements(self, data, ncvar):
        """Cache selected element values.

        Updates *data* in-place to store its first, second and last
        element values inside its ``custom`` dictionary.

        Doing this here is cheap because only the individual elements
        are read from the already-open file, as opposed to being
        retrieved from *data* (which would require a whole dask chunk
        to be read to get each single value).

        .. versionadded:: 3.14.0

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
                except (AttributeError, ValueError):
                    # AttributeError: A netCDF string type scalar
                    # variable comes out as Python str object, which
                    # has no 'item' method.
                    #
                    # ValueError: A size-0 array can't be converted to
                    # a Python scalar.
                    pass

            elements[element] = value

        # Store the elements in the data object
        data._set_cached_elements(elements)

    def _create_cfanetcdfarray(
        self,
        ncvar,
        unpacked_dtype=False,
        coord_ncvar=None,
        non_standard_term=None,
    ):
        """Create a CFA-netCDF variable array.

        .. versionadded:: 3.14.0

        :Parameters:

            ncvar: `str`
                The name of the CFA-netCDF aggregated variable. See
                the *term* parameter.

            unpacked_dtype: `False` or `numpy.dtype`, optional

            coord_ncvar: `str`, optional

            non_standard_term: `str`, optional
                The name of a non-standard aggregation instruction
                term from which to create the array. If set then
                *ncvar* must be the value of the non-standard term in
                the ``aggregation_data`` attribute.

                .. versionadded:: TODOCFAVER

        :Returns:

            (`CFANetCDFArray`, `dict`)
                The new `NetCDFArray` instance and dictionary of the
                kwargs used to create it.

        """
        g = self.read_vars

        # Get the kwargs needed to instantiate a general NetCDFArray
        # instance
        kwargs = self._create_netcdfarray(
            ncvar,
            unpacked_dtype=unpacked_dtype,
            coord_ncvar=coord_ncvar,
            return_kwargs_only=True,
        )

        # Specify a non-standardised term from which to create the
        # data
        if non_standard_term is not None:
            kwargs["term"] = non_standard_term

        # Get rid of the incorrect shape - this will get set by the
        # CFAnetCDFArray instance.
        kwargs.pop("shape", None)

        # Add the aggregated_data attribute (that can be used by
        # dask.base.tokenize)
        aggregated_data = self.read_vars["variable_attributes"][ncvar].get(
            "aggregated_data"
        )
        kwargs["instructions"] = aggregated_data

        # Find URI substitutions that may be stored in the CFA file
        # instruction variable's "substitutions" attribute
        subs = {}
        for x in self._parse_aggregated_data(ncvar, aggregated_data):
            term, term_ncvar = tuple(x.items())[0]
            if term != "file":
                continue

            subs = g["variable_attributes"][term_ncvar].get("substitutions")
            if subs is None:
                subs = {}
            else:
                # Convert the string "${base}: value" to the
                # dictionary {"${base}": "value"}
                subs = self.parse_x(term_ncvar, subs)
                subs = {
                    key: value[0] for d in subs for key, value in d.items()
                }

            break

        # Apply user-defined substitutions, which take precedence over
        # those defined in the file.
        subs = subs.update(g["cfa_substitutions"])
        if subs:
            kwargs["substitutions"] = subs

        # Use the kwargs to create a CFANetCDFArray instance
        array = self.implementation.initialise_CFANetCDFArray(**kwargs)

        return array, kwargs

    def _parse_chunks(self, ncvar):
        """Parse the dask chunks.

        .. versionadded:: 3.14.0

        :Parameters:

            ncvar: `str`
                The name of the netCDF variable containing the array.

        :Returns:

            `str`, `int` or `dict`
                The parsed chunks that are suitable for passing to a
                `Data` object containing the variable's array.

        """
        g = self.read_vars

        default_chunks = "auto"
        chunks = g.get("chunks", default_chunks)

        if chunks is None:
            return -1

        if isinstance(chunks, dict):
            if not chunks:
                return default_chunks

            # For ncdimensions = ('time', 'lat'):
            #
            # chunks={} -> ["auto", "auto"]
            # chunks={'ncdim%time': 12} -> [12, "auto"]
            # chunks={'ncdim%time': 12, 'ncdim%lat': 10000} -> [12, 10000]
            # chunks={'ncdim%time': 12, 'ncdim%lat': "20MB"} -> [12, "20MB"]
            # chunks={'ncdim%time': 12, 'latitude': -1} -> [12, -1]
            # chunks={'ncdim%time': 12, 'Y': None} -> [12, None]
            # chunks={'ncdim%time': 12, 'ncdim%lat': (30, 90)} -> [12, (30, 90)]
            # chunks={'ncdim%time': 12, 'ncdim%lat': None, 'X': 5} -> [12, None]
            attributes = g["variable_attributes"]
            chunks2 = []
            for ncdim in g["variable_dimensions"][ncvar]:
                key = f"ncdim%{ncdim}"
                if key in chunks:
                    chunks2.append(chunks[key])
                    continue

                found_coord_attr = False
                dim_coord_attrs = attributes.get(ncdim)
                if dim_coord_attrs is not None:
                    for attr in ("standard_name", "axis"):
                        key = dim_coord_attrs.get(attr)
                        if key in chunks:
                            found_coord_attr = True
                            chunks2.append(chunks[key])
                            break

                if not found_coord_attr:
                    # Use default chunks for this dimension
                    chunks2.append(default_chunks)

            chunks = chunks2

        return chunks

    def _parse_aggregated_data(self, ncvar, aggregated_data):
        """Parse a CFA-netCDF aggregated_data attribute.

        .. versionadded:: TODOCFAVER

        :Parameters:

            ncvar: `str`
                The netCDF variable name.

            aggregated_data: `str` or `None`
                The CFA-netCDF ``aggregated_data`` attribute.

        :Returns:

            `list`

        """
        if not aggregated_data:
            return []

        return self._parse_x(
            ncvar,
            aggregated_data,
            keys_are_variables=True,
        )

    def _customize_field_ancillaries(self, parent_ncvar, f):
        """Create field ancillary constructs from CFA terms.

        This method is primarily aimed at providing a customisation
        entry point for subclasses.

        This method currently creates:

        * Field ancillary constructs derived from non-standardised
          terms in CFA aggregation instructions. Each construct spans
          the same domain axes as the parent field construct.
          Constructs are never created for `Domain` instances.

        .. versionadded:: TODODASKCFA

        :Parameters:

            parent_ncvar: `str`
                The netCDF variable name of the parent variable.

            f: `Field`
                The parent field construct.

        :Returns:

            `dict`
                A mapping of netCDF variable names to newly-created
                construct identifiers.

        **Examples**

        >>> n._customize_field_ancillaries('tas', f)
        {}

        >>> n._customize_field_ancillaries('pr', f)
        {'tracking_id': 'fieldancillary1'}

        """
        if not self._is_cfa_variable(parent_ncvar):
            return {}

        # ------------------------------------------------------------
        # Still here? Then we have a CFA-netCDF variable: Loop round
        # the aggregation instruction terms and convert each
        # non-standard term into a field ancillary construct that
        # spans the same domain axes as the parent field.
        # ------------------------------------------------------------
        g = self.read_vars

        out = {}

        attributes = g["variable_attributes"][parent_ncvar]
        parsed_aggregated_data = self._parse_aggregated_data(
            parent_ncvar, attributes.get("aggregated_data")
        )
        standardised_terms = ("location", "file", "address", "format")
        for x in parsed_aggregated_data:
            term, ncvar = tuple(x.items())[0]
            if term in standardised_terms:
                continue

            # Still here? Then we've got a non-standard aggregation
            #             term from which we can create a field
            #             ancillary construct.
            anc = self.implementation.initialise_FieldAncillary()

            self.implementation.set_properties(
                anc, g["variable_attributes"][ncvar]
            )
            anc.set_property("long_name", term)

            # Store the term name as the 'id' attribute. This will be
            # used as the term name if the field field ancillary is
            # written to disk as a non-standard CFA term.
            anc.id = term

            data = self._create_data(parent_ncvar, anc, cfa_term=term)
            self.implementation.set_data(anc, data, copy=False)

            self.implementation.nc_set_variable(anc, ncvar)

            # Set the CFA term status
            anc._custom["cfa_term"] = True

            key = self.implementation.set_field_ancillary(
                f,
                anc,
                axes=self.implementation.get_field_data_axes(f),
                copy=False,
            )
            out[ncvar] = key

        return out

    def _cfa(self, ncvar, f):
        """TODOCFADOCS.

        .. versionadded:: TODOCFAVER

        :Parameters:

            ncvar: `str`
                The netCDF variable name.

            f: `Field` or `Domain`
                TODOCFADOCS.

        :Returns:

            TODOCFADOCS.

        """
        pass


#        x = self._parse_x(ncvar, aggregated_data, keys_are_variables=True)
