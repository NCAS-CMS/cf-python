from os import remove

import cfdm
import dask.array as da
import numpy as np

from .netcdfread import NetCDFRead


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

    def _unlimited(self, field, axis):
        """Whether an axis is unlimited.

        If a CFA-netCDF file is being written then no axis can be
        unlimited, i.e. `False` is always returned.

        .. versionadded:: 3.15.3

        :Parameters:

            field: `Field` or `Domain`

            axis: `str`
                Domain axis construct identifier,
                e.g. ``'domainaxis1'``.

        :Returns:

            `bool`

        """
        if self.write_vars["cfa"]:
            return False

        return super()._unlimited(field, axis)

    def _write_as_cfa(self, cfvar, construct_type, domain_axes):
        """Whether or not to write as a CFA variable.

        .. versionadded:: 3.0.0

        :Parameters:

            cfvar: cf instance that contains data

            construct_type: `str`
                The construct type of the *cfvar*, or its parent if
                *cfvar* is not a construct.

                .. versionadded:: 3.15.0

            domain_axes: `None`, or `tuple` of `str`
                The domain axis construct identifiers for *cfvar*.

                .. versionadded:: 3.15.0

        :Returns:

            `bool`
                True if the variable is to be written as a CFA
                variable.

        """
        if construct_type is None:
            # This prevents recursion whilst writing CFA-netCDF term
            # variables.
            return False

        g = self.write_vars
        if not g["cfa"]:
            return False

        data = self.implementation.get_data(cfvar, None)
        if data is None:
            return False

        cfa_options = g["cfa_options"]
        for ctype, ndim in cfa_options.get("constructs", {}).items():
            # Write as CFA if it has an appropriate construct type ...
            if ctype in ("all", construct_type):
                # ... and then only if it satisfies the
                # number-of-dimenions criterion and the data is
                # flagged as OK.
                if ndim is None or ndim == len(domain_axes):
                    cfa_get_write = data.cfa_get_write()
                    if not cfa_get_write and cfa_options["strict"]:
                        if g["mode"] == "w":
                            remove(g["filename"])

                        raise ValueError(
                            f"Can't write {cfvar!r} as a CFA-netCDF "
                            "aggregation variable. Consider setting "
                            "cfa={'strict': False}"
                        )

                    return cfa_get_write

                break

        return False

    def _customise_createVariable(
        self, cfvar, construct_type, domain_axes, kwargs
    ):
        """Customise keyword arguments for
        `netCDF4.Dataset.createVariable`.

        .. versionadded:: 3.0.0

        :Parameters:

            cfvar: cf instance that contains data

            construct_type: `str`
                The construct type of the *cfvar*, or its parent if
                *cfvar* is not a construct.

                .. versionadded:: 3.15.0

            domain_axes: `None`, or `tuple` of `str`
                The domain axis construct identifiers for *cfvar*.

                .. versionadded:: 3.15.0

            kwargs: `dict`

        :Returns:

            `dict`
                Dictionary of keyword arguments to be passed to
                `netCDF4.Dataset.createVariable`.

        """
        kwargs = super()._customise_createVariable(
            cfvar, construct_type, domain_axes, kwargs
        )

        if self._write_as_cfa(cfvar, construct_type, domain_axes):
            kwargs["dimensions"] = ()
            kwargs["chunksizes"] = None

        return kwargs

    def _write_data(
        self,
        data,
        cfvar,
        ncvar,
        ncdimensions,
        domain_axes=None,
        unset_values=(),
        compressed=False,
        attributes={},
        construct_type=None,
    ):
        """Write a Data object.

        .. versionadded:: 3.0.0

        :Parameters:

            data: `Data`

            cfvar: cf instance

            ncvar: `str`

            ncdimensions: `tuple` of `str`

            domain_axes: `None`, or `tuple` of `str`
                The domain axis construct identifiers for *cfvar*.

                .. versionadded:: 3.15.0

            unset_values: sequence of numbers

            attributes: `dict`, optional
                The netCDF attributes for the constructs that have been
                written to the file.

            construct_type: `str`, optional
                The construct type of the *cfvar*, or its parent if
                *cfvar* is not a construct.

                .. versionadded:: 3.15.0

        :Returns:

            `None`

        """
        g = self.write_vars

        if self._write_as_cfa(cfvar, construct_type, domain_axes):
            # --------------------------------------------------------
            # Write the data as CFA aggregated data
            # --------------------------------------------------------
            self._create_cfa_data(
                ncvar,
                ncdimensions,
                data,
                cfvar,
            )
            return

        # ------------------------------------------------------------
        # Still here? The write a normal (non-CFA) variable
        # ------------------------------------------------------------
        if compressed:
            # Write data in its compressed form
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
            if construct_type:
                var = cfvar
            else:
                var = None

            dx = dx.map_blocks(
                self._check_valid,
                cfvar=var,
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

    def _create_cfa_data(self, ncvar, ncdimensions, data, cfvar):
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
        g = self.write_vars

        ndim = data.ndim

        cfa = self._cfa_aggregation_instructions(data, cfvar)

        # ------------------------------------------------------------
        # Get the location netCDF dimensions. These always start with
        # "f_{size}_loc".
        # ------------------------------------------------------------
        location_ncdimensions = []
        for size in cfa["location"].shape:
            l_ncdim = f"f_{size}_loc"
            if l_ncdim not in g["dimensions"]:
                # Create a new location dimension
                self._write_dimension(l_ncdim, None, size=size)

            location_ncdimensions.append(l_ncdim)

        location_ncdimensions = tuple(location_ncdimensions)

        # ------------------------------------------------------------
        # Get the fragment netCDF dimensions. These always start with
        # "f_".
        # ------------------------------------------------------------
        aggregation_address = cfa["address"]
        fragment_ncdimensions = []
        for ncdim, size in zip(
            ncdimensions + ("extra",) * (aggregation_address.ndim - ndim),
            aggregation_address.shape,
        ):
            f_ncdim = f"f_{ncdim}"
            if f_ncdim not in g["dimensions"]:
                # Create a new fragement dimension
                self._write_dimension(f_ncdim, None, size=size)

            fragment_ncdimensions.append(f_ncdim)

        fragment_ncdimensions = tuple(fragment_ncdimensions)

        # ------------------------------------------------------------
        # Write the standardised aggregation instruction variables to
        # the CFA-netCDF file
        # ------------------------------------------------------------
        substitutions = data.cfa_file_substitutions()
        substitutions.update(g["cfa_options"].get("substitutions", {}))

        aggregated_data = data.cfa_get_aggregated_data()
        aggregated_data_attr = []

        # Location
        term = "location"
        data = cfa[term]
        self.implementation.nc_set_hdf5_chunksizes(data, data.shape)
        term_ncvar = self._cfa_write_term_variable(
            data,
            aggregated_data.get(term, f"cfa_{term}"),
            location_ncdimensions,
        )
        aggregated_data_attr.append(f"{term}: {term_ncvar}")

        # File
        term = "file"
        if substitutions:
            # Create the "substitutions" netCDF attribute
            subs = []
            for base, sub in substitutions.items():
                subs.append(f"{base}: {sub}")

            attributes = {"substitutions": " ".join(sorted(subs))}
        else:
            attributes = None

        data = cfa[term]
        self.implementation.nc_set_hdf5_chunksizes(data, data.shape)
        term_ncvar = self._cfa_write_term_variable(
            data,
            aggregated_data.get(term, f"cfa_{term}"),
            fragment_ncdimensions,
            attributes=attributes,
        )
        aggregated_data_attr.append(f"{term}: {term_ncvar}")

        # Address
        term = "address"

        # Attempt to reduce addresses to a common scalar value
        u = cfa[term].unique().compressed().persist()
        if u.size == 1:
            cfa[term] = u.squeeze()
            dimensions = ()
        else:
            dimensions = fragment_ncdimensions

        data = cfa[term]
        self.implementation.nc_set_hdf5_chunksizes(data, data.shape)
        term_ncvar = self._cfa_write_term_variable(
            data,
            aggregated_data.get(term, f"cfa_{term}"),
            dimensions,
        )
        aggregated_data_attr.append(f"{term}: {term_ncvar}")

        # Format
        term = "format"

        # Attempt to reduce addresses to a common scalar value
        u = cfa[term].unique().compressed().persist()
        if u.size == 1:
            cfa[term] = u.squeeze()
            dimensions = ()
        else:
            dimensions = fragment_ncdimensions

        data = cfa[term]
        self.implementation.nc_set_hdf5_chunksizes(data, data.shape)
        term_ncvar = self._cfa_write_term_variable(
            data,
            aggregated_data.get(term, f"cfa_{term}"),
            dimensions,
        )
        aggregated_data_attr.append(f"{term}: {term_ncvar}")

        # ------------------------------------------------------------
        # Look for non-standard CFA terms stored as field ancillaries
        # on a field and write them to the CFA-netCDF file
        # ------------------------------------------------------------
        if self.implementation.is_field(cfvar):
            non_standard_terms = self._cfa_write_non_standard_terms(
                cfvar, fragment_ncdimensions[:ndim], aggregated_data
            )
            aggregated_data_attr.extend(non_standard_terms)

        # ------------------------------------------------------------
        # Add the CFA aggregation variable attributes
        # ------------------------------------------------------------
        self._write_attributes(
            None,
            ncvar,
            extra={
                "aggregated_dimensions": " ".join(ncdimensions),
                "aggregated_data": " ".join(sorted(aggregated_data_attr)),
            },
        )

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

    def _write_field_ancillary(self, f, key, anc):
        """Write a field ancillary to the netCDF file.

        If an equal field ancillary has already been written to the file
        then it is not re-written.

        .. versionadded:: 3.15.0

        :Parameters:

            f: `Field`

            key: `str`

            anc: `FieldAncillary`

        :Returns:

            `str`
                The netCDF variable name of the field ancillary
                object. If no ancillary variable was written then an
                empty string is returned.

        """
        if anc.data.cfa_get_term():
            # This field ancillary construct is to be written as a
            # non-standard CFA term belonging to the parent field, or
            # else not at all.
            return ""

        return super()._write_field_ancillary(f, key, anc)

    def _cfa_write_term_variable(
        self, data, ncvar, ncdimensions, attributes=None
    ):
        """Write a CFA aggregation instruction term variable

        .. versionadded:: 3.15.0

        :Parameters:

            data `Data`
                The data to write.

            ncvar: `str`
                The netCDF variable name.

            ncdimensions: `tuple` of `str`
                The variable's netCDF dimensions.

            attributes: `dict`, optional
                Any attributes to attach to the variable.

        :Returns:

            `str`
                The netCDF variable name of the CFA term variable.

        """
        create = not self._already_in_file(data, ncdimensions)

        if create:
            # Create a new CFA term variable in the file
            ncvar = self._netcdf_name(ncvar)
            self._write_netcdf_variable(
                ncvar, ncdimensions, data, None, extra=attributes
            )
        else:
            # This CFA term variable has already been written to the
            # file
            ncvar = self.write_vars["seen"][id(data)]["ncvar"]

        return ncvar

    def _cfa_write_non_standard_terms(
        self, field, fragment_ncdimensions, aggregated_data
    ):
        """Write a non-standard CFA aggregation instruction term variable.

        Writes non-standard CFA terms stored as field ancillaries.

        .. versionadded:: 3.15.0

        :Parameters:

            field: `Field`

            fragment_ncdimensions: `list` of `str`

            aggregated_data: `dict`

        """
        aggregated_data_attr = []
        terms = ["location", "file", "address", "format"]
        for key, field_anc in self.implementation.get_field_ancillaries(
            field
        ).items():
            if not field_anc.data.cfa_get_term():
                continue

            data = self.implementation.get_data(field_anc, None)
            if data is None:
                continue

            # Check that the field ancillary has the same axes as its
            # parent field, and in the same order.
            if field.get_data_axes(key) != field.get_data_axes():
                continue

            # Still here? Then this field ancillary can be represented
            #             by a non-standard aggregation term.

            # Then transform the data so that it spans the fragment
            # dimensions, with one value per fragment. If a chunk has
            # more than one unique value then the fragment's value is
            # missing data.
            dx = data.to_dask_array()
            dx_ind = tuple(range(dx.ndim))
            out_ind = dx_ind
            dx = da.blockwise(
                self._cfa_unique,
                out_ind,
                dx,
                dx_ind,
                adjust_chunks={i: 1 for i in out_ind},
                dtype=dx.dtype,
            )

            # Get the non-standard term name from the field
            # ancillary's 'id' attribute
            term = getattr(field_anc, "id", "term")
            term = term.replace(" ", "_")
            name = term
            n = 0
            while term in terms:
                n += 1
                term = f"{name}_{n}"

            terms.append(term)

            # Create the new CFA term variable
            data = type(data)(dx)
            self.implementation.nc_set_hdf5_chunksizes(data, data.shape)
            term_ncvar = self._cfa_write_term_variable(
                data=data,
                ncvar=aggregated_data.get(term, f"cfa_{term}"),
                ncdimensions=fragment_ncdimensions,
            )

            aggregated_data_attr.append(f"{term}: {term_ncvar}")

        return aggregated_data_attr

    @classmethod
    def _cfa_unique(cls, a):
        """Return the unique value of an array.

        If there are multiple unique vales then missing data is
        returned.

        .. versionadded:: 3.15.0

        :Parameters:

             a: `numpy.ndarray`
                The array.

        :Returns:

            `numpy.ndarray`
                A size 1 array containing the unique value, or missing
                data if there is not a unique value.

        """
        out_shape = (1,) * a.ndim
        a = np.unique(a)
        if np.ma.isMA(a):
            # Remove a masked element
            a = a.compressed()

        if a.size == 1:
            return a.reshape(out_shape)

        return np.ma.masked_all(out_shape, dtype=a.dtype)

    def _cfa_aggregation_instructions(self, data, cfvar):
        """Convert data to standardised CFA aggregation instruction terms.

        .. versionadded:: 3.15.0

        :Parameters:

            data: `Data`
                The data to be converted to standardised CFA
                aggregation instruction terms.

            cfvar: construct
                The construct that contains the *data*.

        :Returns:

            `dict`
                A dictionary whose keys are the standardised CFA
                aggregation instruction terms, with values of `Data`
                instances containing the corresponding variables.

        **Examples**

        >>> n._cfa_aggregation_instructions(data, cfvar)
        {'location': <CF Data(2, 1): [[5, 8]]>,
         'file': <CF Data(1, 1): [[file:///home/file.nc]]>,
         'format': <CF Data(1, 1): [[nc]]>,
         'address': <CF Data(1, 1): [[q]]>}

        """
        from os.path import abspath, join, relpath
        from pathlib import PurePath
        from urllib.parse import urlparse

        g = self.write_vars

        # Define the CFA file susbstitutions, giving precedence over
        # those set on the Data object to those provided by the CFA
        # options.
        substitutions = data.cfa_file_substitutions()
        substitutions.update(g["cfa_options"].get("substitutions", {}))

        absolute_paths = g["cfa_options"].get("absolute_paths")
        cfa_dir = g["cfa_dir"]

        # Size of the trailing dimension
        n_trailing = 0

        aggregation_file = []
        aggregation_address = []
        aggregation_format = []
        for indices in data.chunk_indices():
            file_details = self._cfa_get_file_details(data[indices])

            if len(file_details) != 1:
                if file_details:
                    raise ValueError(
                        "Can't write CFA-netCDF aggregation variable from "
                        f"{cfvar!r} when the "
                        f"dask storage chunk defined by indices {indices} "
                        "spans two or more files"
                    )

                raise ValueError(
                    "Can't write CFA-netCDF aggregation variable from "
                    f"{cfvar!r} when the "
                    f"dask storage chunk defined by indices {indices} spans "
                    "zero files"
                )

            filenames, addresses, formats = file_details.pop()

            if len(filenames) > n_trailing:
                n_trailing = len(filenames)

            filenames2 = []
            for filename in filenames:
                uri = urlparse(filename)
                uri_scheme = uri.scheme
                if not uri_scheme:
                    filename = abspath(join(cfa_dir, filename))
                    if absolute_paths:
                        filename = PurePath(filename).as_uri()
                    else:
                        filename = relpath(filename, start=cfa_dir)
                elif not absolute_paths and uri_scheme == "file":
                    filename = relpath(uri.path, start=cfa_dir)

                if substitutions:
                    # Apply the CFA file susbstitutions
                    for base, sub in substitutions.items():
                        filename = filename.replace(sub, base)

                filenames2.append(filename)

            aggregation_file.append(tuple(filenames2))
            aggregation_address.append(addresses)
            aggregation_format.append(formats)

        # Pad each value of the aggregation instruction arrays so that
        # it has 'n_trailing' elements
        a_shape = data.numblocks
        pad = None
        if n_trailing > 1:
            a_shape += (n_trailing,)

            # Pad the ...
            for i, (filenames, addresses, formats) in enumerate(
                zip(aggregation_file, aggregation_address, aggregation_format)
            ):
                n = n_trailing - len(filenames)
                if n:
                    # This chunk has fewer fragment files than some
                    # others, so some padding is required.
                    pad = ("",) * n
                    aggregation_file[i] = filenames + pad
                    aggregation_format[i] = formats + pad
                    if isinstance(addresses[0], int):
                        pad = (-1,) * n

                    aggregation_address[i] = addresses + pad

        # Reshape the 1-d aggregation instruction arrays to span the
        # data dimensions, plus the extra trailing dimension if there
        # is one.
        aggregation_file = np.array(aggregation_file).reshape(a_shape)
        aggregation_address = np.array(aggregation_address).reshape(a_shape)
        aggregation_format = np.array(aggregation_format).reshape(a_shape)

        # Mask any padded elements
        if pad:
            aggregation_file = np.ma.where(
                aggregation_file == "", np.ma.masked, aggregation_file
            )
            mask = aggregation_file.mask
            aggregation_address = np.ma.array(aggregation_address, mask=mask)
            aggregation_format = np.ma.array(aggregation_format, mask=mask)

        # ------------------------------------------------------------
        # Create the location array
        # ------------------------------------------------------------
        dtype = np.dtype(np.int32)
        if max(data.to_dask_array().chunksize) > np.iinfo(dtype).max:
            dtype = np.dtype(np.int64)

        ndim = data.ndim
        aggregation_location = np.ma.masked_all(
            (ndim, max(a_shape[:ndim])), dtype=dtype
        )

        for i, chunks in enumerate(data.chunks):
            aggregation_location[i, : len(chunks)] = chunks

        # ------------------------------------------------------------
        # Return Data objects
        # ------------------------------------------------------------
        data = type(data)
        return {
            "location": data(aggregation_location),
            "file": data(aggregation_file),
            "format": data(aggregation_format),
            "address": data(aggregation_address),
        }

    def _customise_write_vars(self):
        """Customise the write parameters.

        .. versionadded:: 3.15.0

        """
        g = self.write_vars

        if g.get("cfa"):
            from os.path import abspath
            from pathlib import PurePath

            # Find the absolute directory path of the output
            # CFA-netCDF file URI
            g["cfa_dir"] = PurePath(abspath(g["filename"])).parent

    def _cfa_get_file_details(self, data):
        """Get the details of all files referenced by the data.

        .. versionadded:: 3.15.0

        :Parameters:

             data: `Data`
                The data

        :Returns:

            `set` of 3-tuples
                A set containing 3-tuples giving the file names,
                the addresses in the files, and the file formats. If
                no files are required to compute the data then
                an empty `set` is returned.

        **Examples**

        >>> n._cfa_get_file_details(data):
        {(('/home/file.nc',), ('tas',), ('nc',))}

        >>> n._cfa_get_file_details(data):
        {(('/home/file.pp',), (34556,), ('um',))}

        """
        out = []
        out_append = out.append
        for a in data.todict().values():
            try:
                out_append(
                    (a.get_filenames(), a.get_addresses(), a.get_formats())
                )
            except AttributeError:
                pass

        return set(out)
