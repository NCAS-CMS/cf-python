import cfdm
import netCDF4
import numpy as np
from packaging.version import Version


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
        compression_index=False,
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

                .. versionadded:: 3.15.0

           compression_index: `bool`, optional
                True if the data being created are compression
                indices.

                .. versionadded:: 3.15.2

        :Returns:

            `Data`

        """
        if not cfa_term and not self._is_cfa_variable(ncvar):
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
                data._cfa_set_write(True)
            if (
                not compression_index
                and self.implementation.get_construct_type(construct)
                != "field"
            ):
                # Only cache values from non-field data and
                # non-compression-index data, on the assumptions that:
                #
                # a) Field data is, in general, so large that finding
                #    the cached values takes too long.
                #
                # b) Cached values are never really required for
                #    compression index data.
                self._cache_data_elements(data, ncvar)

            return data

        # ------------------------------------------------------------
        # Still here? Create data for a CFA variable
        # ------------------------------------------------------------
        if construct is not None:
            # Remove the aggregation attributes from the construct
            self.implementation.del_property(
                construct, "aggregated_dimensions", None
            )
            aggregated_data = self.implementation.del_property(
                construct, "aggregated_data", None
            )
        else:
            aggregated_data = None

        if cfa_term:
            term, term_ncvar = tuple(cfa_term.items())[0]
            cfa_array, kwargs = self._create_cfanetcdfarray_term(
                ncvar, term, term_ncvar
            )
        else:
            cfa_array, kwargs = self._create_cfanetcdfarray(
                ncvar,
                unpacked_dtype=unpacked_dtype,
                coord_ncvar=coord_ncvar,
            )

        data = self._create_Data(
            cfa_array,
            ncvar,
            units=kwargs["units"],
            calendar=kwargs["calendar"],
        )

        # Note: We don't cache elements from CFA variables, because
        #       the data are in fragment files which have not been
        #       opened; and may not not even be openable, such as
        #       could be the case if a fragement was on tape storage.

        # Set the CFA write status to True iff each non-aggregated
        # axis has exactly one dask storage chunk
        if cfa_term:
            data._cfa_set_term(True)
        else:
            cfa_write = True
            for n, numblocks in zip(
                cfa_array.get_fragment_shape(), data.numblocks
            ):
                if n == 1 and numblocks > 1:
                    # Note: 'n == 1' is True for non-aggregated axes
                    cfa_write = False
                    break

            data._cfa_set_write(cfa_write)

            # Store the 'aggregated_data' attribute
            if aggregated_data:
                data.cfa_set_aggregated_data(aggregated_data)

            # Store the file substitutions
            data.cfa_update_file_substitutions(kwargs.get("substitutions"))

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
        return (
            g["cfa"]
            and ncvar in g["cfa_aggregated_data"]
            and ncvar not in g["external_variables"]
        )

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

            units: `str`, optional
                The units of *array*. By default, or if `None`, it is
                assumed that there are no units.

            calendar: `str`, optional
                The calendar of *array*. By default, or if `None`, it is
                assumed that there is no calendar.

            ncdimensions: sequence of `str`, optional
                The netCDF dimensions spanned by the array.

                .. versionadded:: 3.14.0

            kwargs: optional
                Extra parameters to pass to the initialisation of the
                returned `Data` object.

        :Returns:

            `Data`

        """
        if array.dtype is None:
            # The array is based on a netCDF VLEN variable, and
            # therefore has unknown data type. To find the correct
            # data type (e.g. "<U7"), we need to read the entire array
            # from its netCDF variable into memory to find the longest
            # string.
            g = self.read_vars
            if g["has_groups"]:
                group, name = self._netCDF4_group(
                    g["variable_grouped_dataset"][ncvar], ncvar
                )
                variable = group.variables.get(name)
            else:
                variable = g["variables"].get(ncvar)

            array = variable[...]

            string_type = isinstance(array, str)
            if string_type:
                # A netCDF string type scalar variable comes out as Python
                # str object, so convert it to a numpy array.
                array = np.array(array, dtype=f"U{len(array)}")

            if not variable.ndim:
                # NetCDF4 has a thing for making scalar size 1
                # variables into 1d arrays
                array = array.squeeze()

            if not string_type:
                # A N-d (N>=1) netCDF string type variable comes out
                # as a numpy object array, so convert it to numpy
                # string array.
                array = array.astype("U", copy=False)
                # NetCDF4 doesn't auto-mask VLEN variables
                array = np.ma.where(array == "", np.ma.masked, array)

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

    def _customise_read_vars(self):
        """Customise the read parameters.

        Take the opportunity to apply CFA updates to
        `read_vars['variable_dimensions']` and
        `read_vars['do_not_create_field']`.

        .. versionadded:: 3.0.0

        """
        super()._customise_read_vars()
        g = self.read_vars

        if not g["cfa"]:
            return

        g["cfa_aggregated_data"] = {}
        g["cfa_aggregation_instructions"] = {}
        g["cfa_file_substitutions"] = {}

        # ------------------------------------------------------------
        # Still here? Then this is a CFA-netCDF file
        # ------------------------------------------------------------
        if g["CFA_version"] < Version("0.6.2"):
            raise ValueError(
                f"Can't read file {g['filename']} that uses obsolete "
                f"CFA conventions version CFA-{g['CFA_version']}. "
                "(Note that cf version 3.13.1 can be used to read and "
                "write CFA-0.4 files.)"
            )

        # Get the directory of the CFA-netCDF file being read
        from os.path import abspath
        from pathlib import PurePath

        g["cfa_dir"] = PurePath(abspath(g["filename"])).parent

        # Process the aggregation instruction variables, and the
        # aggregated dimensions.
        dimensions = g["variable_dimensions"]
        attributes = g["variable_attributes"]

        for ncvar, attributes in attributes.items():
            if "aggregated_dimensions" not in attributes:
                # This is not an aggregated variable
                continue

            # Set the aggregated variable's dimensions as its
            # aggregated dimensions
            ncdimensions = attributes["aggregated_dimensions"].split()
            dimensions[ncvar] = tuple(map(str, ncdimensions))

            # Do not create fields/domains from aggregation
            # instruction variables
            parsed_aggregated_data = self._cfa_parse_aggregated_data(
                ncvar, attributes.get("aggregated_data")
            )
            for term_ncvar in parsed_aggregated_data.values():
                g["do_not_create_field"].add(term_ncvar)

    def _cache_data_elements(self, data, ncvar):
        """Cache selected element values.

        Updates *data* in-place to store its first, second,
        penultimate, and last element values (as appropriate).

        These values are used by, amongst other things,
        `cf.Data.equals`, `cf.aggregate` and for inspection.

        Doing this here is quite cheap because only the individual
        elements are read from the already-open file, as opposed to
        being retrieved from *data* (which would require a whole dask
        chunk to be read to get each single value).

        However, empirical evidence shows that using netCDF4 to access
        the first and last elements of a large array on disk
        (e.g. shape (1, 75, 1207, 1442)) is slow (e.g. ~2 seconds) and
        doesn't scale well with array size (i.e. it takes
        disproportionally longer for larger arrays). Such arrays are
        usually in field constructs, for which `cf.aggregate` does not
        need to know any array values, so this method should be used
        with caution, if at all, on field construct data.

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
        if data.data.get_compression_type():
            # Don't get cached elements from arrays compressed by
            # convention, as they'll likely be wrong.
            return

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
        size = data.size
        ndim = data.ndim

        char = False
        if variable.ndim == ndim + 1:
            dtype = variable.dtype
            if dtype is not str and dtype.kind in "SU":
                # This variable is a netCDF classic style char array
                # with a trailing dimension that needs to be collapsed
                char = True

        if ndim == 1:
            # Also cache the second element for 1-d data, on the
            # assumption that they may well be dimension coordinate
            # data.
            if size == 1:
                indices = (0, -1)
                value = variable[...]
                values = (value, value)
            elif size == 2:
                indices = (0, 1, -1)
                value = variable[-1:]
                values = (variable[:1], value, value)
            else:
                indices = (0, 1, -1)
                values = (variable[:1], variable[1:2], variable[-1:])
        elif ndim == 2 and data.shape[-1] == 2:
            # Assume that 2-d data with a last dimension of size 2
            # contains coordinate bounds, for which it is useful to
            # cache the upper and lower bounds of the the first and
            # last cells.
            indices = (0, 1, -2, -1)
            ndim1 = ndim - 1
            values = (
                variable[(slice(0, 1),) * ndim1 + (slice(0, 1),)],
                variable[(slice(0, 1),) * ndim1 + (slice(1, 2),)],
            )
            if data.size == 2:
                values = values + values
            else:
                values += (
                    variable[(slice(-1, None, 1),) * ndim1 + (slice(0, 1),)],
                    variable[(slice(-1, None, 1),) * ndim1 + (slice(1, 2),)],
                )
        elif size == 1:
            indices = (0, -1)
            value = variable[...]
            values = (value, value)
        elif size == 3:
            indices = (0, 1, -1)
            if char:
                values = variable[...].reshape(3, variable.shape[-1])
            else:
                values = variable[...].flatten()
        else:
            indices = (0, -1)
            values = (
                variable[(slice(0, 1),) * ndim],
                variable[(slice(-1, None, 1),) * ndim],
            )

        # Create a dictionary of the element values
        elements = {}
        for index, value in zip(indices, values):
            if char:
                # Variable is a netCDF classic style char array, so
                # collapse (by concatenation) the outermost (fastest
                # varying) dimension. E.g. [['a','b','c']] becomes
                # ['abc']
                if value.dtype.kind == "U":
                    value = value.astype("S")

                a = netCDF4.chartostring(value)
                shape = a.shape
                a = np.array([x.rstrip() for x in a.flat])
                a = np.reshape(a, shape)
                value = np.ma.masked_where(a == "", a)

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

            elements[index] = value

        # Store the elements in the data object
        data._set_cached_elements(elements)

    def _create_cfanetcdfarray(
        self,
        ncvar,
        unpacked_dtype=False,
        coord_ncvar=None,
        term=None,
    ):
        """Create a CFA-netCDF variable array.

        .. versionadded:: 3.14.0

        :Parameters:

            ncvar: `str`
                The name of the CFA-netCDF aggregated variable. See
                the *term* parameter.

            unpacked_dtype: `False` or `numpy.dtype`, optional

            coord_ncvar: `str`, optional

            term: `str`, optional
                The name of a non-standard aggregation instruction
                term from which to create the array. If set then
                *ncvar* must be the value of the non-standard term in
                the ``aggregation_data`` attribute.

                .. versionadded:: 3.15.0

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

        # Get rid of the incorrect shape of (). This will end up
        # getting set correctly by the CFANetCDFArray instance.
        kwargs.pop("shape", None)

        aggregated_data = g["cfa_aggregated_data"][ncvar]

        standardised_terms = ("location", "file", "address", "format")

        instructions = []
        aggregation_instructions = {}
        for t, term_ncvar in aggregated_data.items():
            if t not in standardised_terms:
                continue

            aggregation_instructions[t] = g["cfa_aggregation_instructions"][
                term_ncvar
            ]
            instructions.append(f"{t}: {term_ncvar}")

            if t == "file":
                kwargs["substitutions"] = g["cfa_file_substitutions"].get(
                    term_ncvar
                )

        kwargs["x"] = aggregation_instructions
        kwargs["instructions"] = " ".join(sorted(instructions))

        # Use the kwargs to create a CFANetCDFArray instance
        array = self.implementation.initialise_CFANetCDFArray(**kwargs)

        return array, kwargs

    def _create_cfanetcdfarray_term(
        self,
        parent_ncvar,
        term,
        ncvar,
    ):
        """Create a CFA-netCDF variable array.

        .. versionadded:: 3.14.0

        :Parameters:

            parent_ncvar: `str`
                The name of the CFA-netCDF aggregated variable. See
                the *term* parameter.

            term: `str`, optional
                The name of a non-standard aggregation instruction
                term from which to create the array. If set then
                *ncvar* must be the value of the non-standard term in
                the ``aggregation_data`` attribute.

                .. versionadded:: 3.15.0

            ncvar: `str`
                The name of the CFA-netCDF aggregated variable. See
                the *term* parameter.

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
            return_kwargs_only=True,
        )

        instructions = []
        aggregation_instructions = {}
        for t, term_ncvar in g["cfa_aggregated_data"][parent_ncvar].items():
            if t in ("location", term):
                aggregation_instructions[t] = g[
                    "cfa_aggregation_instructions"
                ][term_ncvar]
                instructions.append(f"{t}: {ncvar}")

        kwargs["term"] = term
        kwargs["dtype"] = aggregation_instructions[term].dtype
        kwargs["x"] = aggregation_instructions
        kwargs["instructions"] = " ".join(sorted(instructions))

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

    def _customise_field_ancillaries(self, parent_ncvar, f):
        """Create customised field ancillary constructs.

        This method currently creates:

        * Field ancillary constructs derived from non-standardised
          terms in CFA aggregation instructions. Each construct spans
          the same domain axes as the parent field construct.
          Constructs are never created for `Domain` instances.

        .. versionadded:: 3.15.0

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

        >>> n._customise_field_ancillaries('tas', f)
        {}

        >>> n._customise_field_ancillaries('pr', f)
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

        standardised_terms = ("location", "file", "address", "format")

        out = {}
        for term, term_ncvar in g["cfa_aggregated_data"][parent_ncvar].items():
            if term in standardised_terms:
                continue

            if g["variables"][term_ncvar].ndim != f.ndim:
                # Can only create field ancillaries with the same rank
                # as the field
                continue

            # Still here? Then we've got a non-standard aggregation
            #             term from which we can create a field
            #             ancillary construct.
            anc = self.implementation.initialise_FieldAncillary()

            self.implementation.set_properties(
                anc, g["variable_attributes"][term_ncvar]
            )
            anc.set_property("long_name", term)

            # Store the term name as the 'id' attribute. This will be
            # used as the term name if the field field ancillary is
            # written to disk as a non-standard CFA term.
            anc.id = term

            data = self._create_data(
                parent_ncvar, anc, cfa_term={term: term_ncvar}
            )

            self.implementation.set_data(anc, data, copy=False)
            self.implementation.nc_set_variable(anc, term_ncvar)

            key = self.implementation.set_field_ancillary(
                f,
                anc,
                axes=self.implementation.get_field_data_axes(f),
                copy=False,
            )
            out[term_ncvar] = key

        return out

    def _cfa_parse_aggregated_data(self, ncvar, aggregated_data):
        """Parse a CFA-netCDF ``aggregated_data`` attribute.

        .. versionadded:: 3.15.0

        :Parameters:

            ncvar: `str`
                The netCDF variable name.

            aggregated_data: `str` or `None`
                The CFA-netCDF ``aggregated_data`` attribute.

        :Returns:

            `dict`
                The parsed attribute.

        """
        if not aggregated_data:
            return {}

        g = self.read_vars
        aggregation_instructions = g["cfa_aggregation_instructions"]
        variable_attributes = g["variable_attributes"]

        out = {}
        for x in self._parse_x(
            ncvar,
            aggregated_data,
            keys_are_variables=True,
        ):
            term, term_ncvar = tuple(x.items())[0]
            term_ncvar = term_ncvar[0]
            out[term] = term_ncvar

            if term_ncvar in aggregation_instructions:
                # Already processed this term
                continue

            array = g["variables"][term_ncvar][...]
            aggregation_instructions[term_ncvar] = self._cfa_conform_array(
                array
            )

            if term == "file":
                # Find URI substitutions that may be stored in the
                # CFA file instruction variable's "substitutions"
                # attribute
                subs = variable_attributes[term_ncvar].get(
                    "substitutions",
                )
                if subs:
                    # Convert the string "${base}: value" to the
                    # dictionary {"${base}": "value"}
                    s = subs.split()
                    subs = {
                        base[:-1]: sub for base, sub in zip(s[::2], s[1::2])
                    }

                    # Apply user-defined substitutions, which take
                    # precedence over those defined in the file.
                    subs.update(g["cfa_options"].get("substitutions", {}))
                    g["cfa_file_substitutions"][term_ncvar] = subs

        g["cfa_aggregated_data"][ncvar] = out
        return out

    def _cfa_conform_array(self, array):
        """Conform an array so that it is suitable for CFA processing.

        .. versionadded: 3.15.0

        :Parameters:

            array: `np.ndarray`
                The array to conform.

        :Returns:

            array: `np.ndarray`
                The conformed array.

        """
        if isinstance(array, str):
            # string
            return np.array(array, dtype=f"S{len(array)}").astype("U")

        kind = array.dtype.kind
        if kind == "O":
            # string
            return array.astype("U")

        if kind in "SU":
            # char
            if kind == "U":
                array = array.astype("S")

            array = netCDF4.chartostring(array)
            shape = array.shape
            array = np.array([x.rstrip() for x in array.flat], dtype="S")
            array = np.reshape(array, shape)
            array = np.ma.masked_where(array == b"", array)
            return array.astype("U")

        # number
        return array
