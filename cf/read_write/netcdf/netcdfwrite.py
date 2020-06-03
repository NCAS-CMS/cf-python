import json
import random

from string import hexdigits

import numpy

import cfdm

from ... import DomainAncillary, Coordinate, Bounds


class NetCDFWrite(cfdm.read_write.netcdf.NetCDFWrite):
    '''TODO

    '''
    def _write_as_cfa(self, cfvar):
        '''TODO

    .. versionadded:: 3.0.0

        '''
        if not self.write_vars['cfa']:
            return False

        data = self.implementation.get_data(cfvar, None)
        if data is None:
            return False

        if data.in_memory:
            return False

        if data.size == 1:
            return False

        if isinstance(cfvar, (Coordinate, DomainAncillary)):
            return cfvar.ndim > 1

        if isinstance(cfvar, Bounds):
            return cfvar.ndim > 2

        return True

    def _customize_createVariable(self, cfvar, kwargs):
        '''TODO

    .. versionadded:: 3.0.0

    :Returns:

        `dict`
            Dictionary of keyword arguments to be passed to
            `netCDF4.Dataset.createVariable`.

        '''
        kwargs = super()._customize_createVariable(cfvar, kwargs)

        if self._write_as_cfa(cfvar):
            kwargs['dimensions'] = ()
            kwargs['chunksizes'] = None

        return kwargs

    def _write_data(self, data, cfvar, ncvar, ncdimensions,
                    unset_values=(), compressed=False, attributes={}):
        '''TODO

    :Parameters:

        data: `Data`

        cfvar: cf instance

        ncvar: `str`

        ncdimensions: `tuple` of `str`

        unset_values: sequence of numbers

        '''
        g = self.write_vars

        if self._write_as_cfa(cfvar):
            self._write_cfa_data(ncvar, ncdimensions, data, cfvar)
            return

        # Still here?
        if compressed:
            # --------------------------------------------------------
            # Write data in its compressed form
            # --------------------------------------------------------
            data = data.source().source()

        warned_valid = False

        config = data.partition_configuration(readonly=True)

        for partition in data.partitions.flat:
            partition.open(config)
            array = partition.array

            # Convert data type
            new_dtype = g['datatype'].get(array.dtype)
            if new_dtype is not None:
                array = array.astype(new_dtype)

            # Check that the array doesn't contain any elements
            # which are equal to any of the missing data values
            if unset_values:
                if partition.masked:
                    temp_array = array.compressed()
                else:
                    temp_array = array

                if numpy.intersect1d(unset_values, temp_array).size:
                    raise ValueError(
                        "ERROR: Can't write field when array has _FillValue or"
                        " missing_value at unmasked point: {!r}".format(ncvar)
                    )
            # --- End: if

            if (g['fmt'] == 'NETCDF4' and array.dtype.kind in 'SU' and
                    numpy.ma.isMA(array)):
                # VLEN variables can not be assigned to by masked arrays
                # https://github.com/Unidata/netcdf4-python/pull/465
                array = array.filled('')

            if not warned_valid and g['warn_valid']:
                # Check for out-of-range values
                warned_valid = self._check_valid(cfvar, array, attributes)

            # Copy the array into the netCDF variable
            g['nc'][ncvar][partition.indices] = array

            partition.close()

    def _write_dimension_coordinate(self, f, key, coord):
        '''Write a coordinate variable and its bound variable to the file.

    This also writes a new netCDF dimension to the file and, if
    required, a new netCDF dimension for the bounds.

    :Parameters:

        f: Field construct

        key: `str`

        coord: Dimension coordinate construct

    :Returns:

        `str`
            The netCDF name of the dimension coordinate.

        '''
        coord = self._change_reference_datetime(coord)

        return super()._write_dimension_coordinate(f, key, coord)

    def _write_scalar_coordinate(self, f, key, coord_1d, axis,
                                 coordinates, extra=None):
        '''Write a scalar coordinate and its bounds to the netCDF file.

It is assumed that the input coordinate has size 1, but this is not checked.

If an equal scalar coordinate has already been written to the file
then the input coordinate is not written.

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

        '''
        # Unsafe to set mutable '{}' as default in the func signature.
        if extra is None:  # distinguish from falsy '{}'
            extra = {}
        coord_1d = self._change_reference_datetime(coord_1d)

        return super()._write_scalar_coordinate(f, key, coord_1d,
                                                axis, coordinates,
                                                extra=extra)

    def _write_auxiliary_coordinate(self, f, key, coord, coordinates):
        '''Write auxiliary coordinates and bounds to the netCDF file.

    If an equal auxiliary coordinate has already been written to the
    file then the input coordinate is not written.

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

        '''
        coord = self._change_reference_datetime(coord)

        return super()._write_auxiliary_coordinate(f, key, coord, coordinates)

    def _change_reference_datetime(self, coord):
        '''TODO

    .. versionadded:: 3.0.0

    :Parameters:

        coord: Coordinate instance

    :Returns:

            The coordinate construct with changed units.

    '''
        reference_datetime = self.write_vars.get('reference_datetime')
        if not reference_datetime or not coord.Units.isreftime:
            return coord

        coord2 = coord.copy()
        try:
            coord2.reference_datetime = reference_datetime
        except ValueError:
            raise ValueError(
                "Can't override coordinate reference date-time {0!r} with "
                "{1!r}".format(coord.reference_datetime, reference_datetime)
            )
        else:
            return coord2

    def _write_cfa_data(self, ncvar, ncdimensions, data, cfvar):
        '''Write a CFA variable to the netCDF file.

    Any CFA private variables required will be autmatically created
    and written to the file.

    :Parameters:

        ncvar: `str`
            The netCDF name for the variable.

        ncdimensions: sequence of `str`

        netcdf_attrs: `dict`

        data: `Data`

    :Returns:

        `None`

        '''
        g = self.write_vars

        netcdf_attrs = {
            'cf_role': 'cfa_variable',
            'cfa_dimensions': ' '.join(ncdimensions)
        }

        # Create a dictionary representation of the data object
        data = data.copy()
        axis_map = {}
        for axis0, axis1 in zip(data._axes, ncdimensions):
            axis_map[axis0] = axis1

        data._change_axis_names(axis_map)
        data._move_flip_to_partitions()

        cfa_array = data.dumpd()

        # Modify the dictionary so that it is suitable for JSON
        # serialization
        del cfa_array['_axes']
        del cfa_array['shape']
        del cfa_array['Units']
        del cfa_array['dtype']
        cfa_array.pop('_cyclic', None)
        cfa_array.pop('_fill', None)
        cfa_array.pop('fill_value', None)

        pmshape = cfa_array.pop('_pmshape', None)
        if pmshape:
            cfa_array['pmshape'] = pmshape

        pmaxes = cfa_array.pop('_pmaxes', None)
        if pmaxes:
            cfa_array['pmdimensions'] = pmaxes

        config = data.partition_configuration(readonly=True)

        base = g['cfa_options'].get('base', None)
        if base is not None:
            cfa_array['base'] = base

        convert_dtype = g['datatype']

        for attrs in cfa_array['Partitions']:
            fmt = attrs.get('format', None)

            if fmt is None:
                # --------------------------------------------------------
                # This partition has an internal sub-array. This could be
                # a numpy array or a temporary FileArray object.
                # --------------------------------------------------------
                index = attrs.get('index', ())
                if len(index) == 1:
                    index = index[0]
                else:
                    index = tuple(index)

                partition = data.partitions.matrix.item(index)

                partition.open(config)
                array = partition.array

                # Convert data type
                new_dtype = convert_dtype.get(array.dtype, None)
                if new_dtype is not None:
                    array = array.astype(new_dtype)

                shape = array.shape
                ncdim_strlen = []
                if array.dtype.kind == 'S':
                    # This is an array of strings
                    strlen = array.dtype.itemsize
                    if strlen > 1:
                        # Convert to an array of characters
                        array = _character_array(array)
                        # Get the netCDF dimension for the string length
                        ncdim_strlen = [
                            _string_length_dimension(strlen, g=None)]
                # --- End: if

                # Create a name for the netCDF variable to contain the array
                p_ncvar = 'cfa_'+self._random_hex_string()
                while p_ncvar in g['ncvar_names']:
                    p_ncvar = 'cfa_'+self._random_hex_string()

                g['ncvar_names'].add(p_ncvar)

                # Get the private CFA netCDF dimensions for the array.
                cfa_dimensions = [self._netcdf_name('cfa{0}'.format(size),
                                                    dimsize=size,
                                                    role='cfa_private')
                                  for size in array.shape]

                for ncdim, size, in zip(cfa_dimensions, array.shape):
                    if ncdim not in g['ncdim_to_size']:
                        # This cfa private dimension needs creating
                        g['ncdim_to_size'][ncdim] = size
                        g['netcdf'].createDimension(ncdim, size)

                # Create the private CFA variable and write the array to it
#                v = g['netcdf'].createVariable(p_ncvar, self._datatype(array),
#                                               cfa_dimensions + ncdim_strlen,
# #                                              fill_value=fill_value,
#                                               fill_value=False,
#                                               least_significant_digit=None,
#                                               endian=g['endian'],
#                                               **g['netcdf_compression'])

                kwargs = {
                    'varname': p_ncvar,
                    'datatype': self._datatype(array),
                    'dimensions': cfa_dimensions + ncdim_strlen,
                    'fill_value': None,  # False,
                    'least_significant_digit': None,
                    'endian': g['endian']
                }
                kwargs.update(g['netcdf_compression'])

                self._createVariable(**kwargs)

                self._write_attributes(parent=None, ncvar=p_ncvar,
                                       extra={'cf_role': 'cfa_private'})

                g['nc'][p_ncvar][...] = array

                # Update the attrs dictionary.
                #
                # Note that we don't need to set 'part', 'dtype', 'units',
                # 'calendar', 'dimensions' and 'reverse' since the
                # partition's in-memory data array always matches up with
                # the master data array.
                attrs['subarray'] = {'shape': shape,
                                     'ncvar': p_ncvar}

            else:
                # --------------------------------------------------------
                # This partition has an external sub-array
                # --------------------------------------------------------
                # PUNITS, PCALENDAR: Change from Units object to netCDF
                #                    string(s)
                units = attrs.pop('Units', None)
                if units is not None:
                    attrs['punits'] = units.units
                    if hasattr(units, 'calendar'):
                        attrs['pcalendar'] = units.calendar

                # PDIMENSIONS:
                p_axes = attrs.pop('axes', None)
                if p_axes is not None:
                    attrs['pdimensions'] = p_axes

                # REVERSE
                p_flip = attrs.pop('flip', None)
                if p_flip:
                    attrs['reverse'] = p_flip

                # DTYPE: Change from numpy.dtype object to netCDF string
                dtype = attrs['subarray'].pop('dtype', None)
                if dtype is not None:
                    if dtype.kind != 'S':
                        attrs['subarray']['dtype'] = (
                            _convert_to_netCDF_datatype(dtype))

                # FORMAT:
                sfmt = attrs.pop('format', None)
                if sfmt is not None:
                    attrs['subarray']['format'] = sfmt
            # --- End: if

            # LOCATION: Change from python to CFA indexing (i.e. range
            #           includes the final index)
            attrs['location'] = [(x[0], x[1]-1) for x in attrs['location']]

            # PART: Change from python to to CFA indexing (i.e. slice
            #       range includes the final index)
            part = attrs.get('part', None)
            if part:
                p = []
                for x, size in zip(part, attrs['subarray']['shape']):
                    if isinstance(x, slice):
                        x = x.indices(size)
                        if x[2] > 0:
                            p.append([x[0], x[1]-1, x[2]])
                        elif x[1] == -1:
                            p.append([x[0], 0, x[2]])
                        else:
                            p.append([x[0], x[1]+1, x[2]])
                    else:
                        p.append(tuple(x))
                # --- End: for
                attrs['part'] = str(p)
            # --- End: if

            if 'base' in cfa_array and 'file' in attrs['subarray']:
                # Make the file name relative to base
                attrs['subarray']['file'] = relpath(attrs['subarray']['file'],
                                                    cfa_array['base'])
        # --- End: for

        # Add the description (as a JSON string) of the partition array to
        # the netcdf attributes.
        netcdf_attrs['cfa_array'] = json.dumps(
            cfa_array, default=self._convert_to_builtin_type)

        # Write the netCDF attributes to the file
        self._write_attributes(parent=None, ncvar=ncvar,
                               extra=netcdf_attrs)

    def _random_hex_string(self, size=10):
        '''Return a random hexadecimal string with the given number of
    characters.

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

        '''
        return ''.join(random.choice(hexdigits) for i in range(size))

    def _convert_to_builtin_type(self, x):
        '''Convert a non-JSON-encodable object to a JSON-encodable built-in
    type.

    Possible conversions are:

    ==============  =============  ======================================
    Input object    Output object  numpy data types covered
    ==============  =============  ======================================
    numpy.bool_     bool           bool
    numpy.integer   int            int, int8, int16, int32, int64, uint8,
                                   uint16, uint32, uint64
    numpy.floating  float          float, float16, float32, float64
    ==============  =============  ======================================

    :Parameters:

        x:

    :Returns:

        'int' or `float` or `bool`

    **Examples:**

    >>> type(_convert_to_netCDF_datatype(numpy.bool_(True)))
    bool
    >>> type(_convert_to_netCDF_datatype(numpy.array([1.0])[0]))
    double
    >>> type(_convert_to_netCDF_datatype(numpy.array([2])[0]))
    int

        '''
        if isinstance(x, numpy.bool_):
            return bool(x)

        if isinstance(x, numpy.integer):
            return int(x)

        if isinstance(x, numpy.floating):
            return float(x)

        raise TypeError(
            "{!r} object can't be converted to a JSON serializable type: "
            "{!r}".format(type(x), x)
        )


# --- End: class
