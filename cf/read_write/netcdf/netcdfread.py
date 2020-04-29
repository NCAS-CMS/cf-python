import json

from ast import literal_eval as ast_literal_eval

import cfdm

from ...constants import _file_to_fh


class NetCDFRead(cfdm.read_write.netcdf.NetCDFRead):
    '''TODO

    .. versionadded:: 3.0.0

    '''
    def _ncdimensions(self, ncvar):
        '''Return a list of the netCDF dimensions corresponding to a netCDF
    variable.

    If the variable has been compressed then the *implied
    uncompressed* dimensions are returned.

    For a CFA variable, the netCDF dimensions are taken from the
    'cfa_dimensions' netCDF attribute.

    .. versionadded:: 3.0.0

    :Parameters:

        ncvar: `str`
            The netCDF variable name.

    :Returns:

        `list`
            The netCDF dimension names spanned by the netCDF variable.

    **Examples:**

    >>> n._ncdimensions('humidity')
    ['time', 'lat', 'lon']

        '''
        g = self.read_vars

        cfa = (g['cfa'] and
               ncvar not in g['external_variables'] and
               g['variable_attributes'][ncvar].get('cf_role') ==
               'cfa_variable')

        if not cfa:
            return super()._ncdimensions(ncvar)

        # Still here?
        ncdimensions = g['variable_attributes'][ncvar].get(
            'cfa_dimensions', '').split()

        return list(map(str, ncdimensions))

    def _get_domain_axes(self, ncvar, allow_external=False):
        '''Return the domain axis identifiers that correspond to a netCDF
    variable's netCDF dimensions.

    For a CFA variable, the netCDF dimensions are taken from the
    'cfa_dimensions' netCDF attribute.

    :Parameter:

        ncvar: `str`
            The netCDF variable name.

        allow_external: `bool`
            If `True` and *ncvar* is an external variable then return an
            empty list.

    :Returns:

        `list`

    **Examples:**

    >>> r._get_domain_axes('areacello')
    ['domainaxis0', 'domainaxis1']

    >>> r._get_domain_axes('areacello', allow_external=True)
    []

        '''
        g = self.read_vars

        cfa = (g['cfa'] and
               ncvar not in g['external_variables'] and
               g['variable_attributes'][ncvar].get('cf_role') ==
               'cfa_variable')

        if not cfa:
            return super()._get_domain_axes(ncvar=ncvar,
                                            allow_external=allow_external)

        # Still here?
        cfa_dimensions = g['variable_attributes'][ncvar].get(
            'cfa_dimensions', '').split()

        ncdim_to_axis = g['ncdim_to_axis']
        axes = [ncdim_to_axis[ncdim] for ncdim in cfa_dimensions
                if ncdim in ncdim_to_axis]

        return axes

    def _create_data(self, ncvar, construct=None,
                     unpacked_dtype=False, uncompress_override=None,
                     parent_ncvar=None):
        '''TODO

    .. versionadded:: 3.0.0

    :Parameters:


    :Returns:

        `Data`

        '''
        g = self.read_vars

        is_cfa_variable = (
            g['cfa'] and
            construct.get_property('cf_role', None) == 'cfa_variable')

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
            )

        # ------------------------------------------------------------
        # Still here? Then create data for a CFA netCDF variable
        # ------------------------------------------------------------
#        print ('    Creating data from CFA variable', repr(ncvar),
#               repr(construct))
        try:
            cfa_data = json.loads(construct.get_property('cfa_array'))
        except ValueError as error:
            raise ValueError(
                "Error during JSON-decoding of netCDF attribute 'cfa_array': "
                "{}".format(error)
            )

        variable = g['variables'][ncvar]

        cfa_data['file'] = g['filename']
        cfa_data['Units'] = construct.Units
        cfa_data['fill_value'] = construct.fill_value()
        cfa_data['_pmshape'] = cfa_data.pop('pmshape', ())
        cfa_data['_pmaxes'] = cfa_data.pop('pmdimensions', ())

        base = cfa_data.get('base', None)
        if base is not None:
            cfa_data['base'] = pathjoin(dirname(g['filename']), base)

        ncdimensions = construct.get_property('cfa_dimensions', '').split()
        dtype = variable.dtype

        if dtype is str:
            # netCDF string types have a dtype of `str`, which needs
            # to be reset as a numpy.dtype, but we don't know what
            # without reading the data, so set it to None for now.
            dtype = None

        # UNICODE???? TODO
        if self._is_char(ncvar) and dtype.kind in 'SU' and ncdimensions:
            strlen = g['nc'].dimensions[ncdimensions[-1]].size
            if strlen > 1:
                ncdimensions.pop()
                dtype = numpy_dtype('S{0}'.format(strlen))
        # --- End: if

        cfa_data['dtype'] = dtype
        cfa_data['_axes'] = ncdimensions
        cfa_data['shape'] = [g['nc'].dimensions[ncdim].size
                             for ncdim in ncdimensions]

        for attrs in cfa_data['Partitions']:
            # FORMAT
            sformat = attrs.get('subarray', {}).pop('format', 'netCDF')
            if sformat is not None:
                attrs['format'] = sformat

            # DTYPE
            dtype = attrs.get('subarray', {}).pop('dtype', None)
            if dtype not in (None, 'char'):
                attrs['subarray']['dtype'] = numpy_dtype(dtype)

            # UNITS and CALENDAR
            units = attrs.pop('punits', None)
            calendar = attrs.pop('pcalendar', None)
            if units is not None or calendar is not None:
                attrs['Units'] = Units(units, calendar)

            # AXES
            pdimensions = attrs.pop('pdimensions', None)
            if pdimensions is not None:
                attrs['axes'] = pdimensions

            # REVERSE
            reverse = attrs.pop('reverse', None)
            if reverse is not None:
                attrs['reverse'] = reverse

            # LOCATION: Change to python indexing (i.e. range does not
            #           include the final index)
            for r in attrs['location']:
                r[1] += 1

            # PART: Change to python indexing (i.e. slice range does
            #       not include the final index)
            part = attrs.get('part', None)
            if part:
                p = []
                for x in ast_literal_eval(part):
                    if isinstance(x, list):
                        if x[2] > 0:
                            p.append(slice(x[0], x[1]+1, x[2]))
                        elif x[1] == 0:
                            p.append(slice(x[0], None, x[2]))
                        else:
                            p.append(slice(x[0], x[1]-1, x[2]))
                    else:
                        p.append(list(x))
                # --- End: for

                attrs['part'] = p
        # --- End: for

        construct.del_property('cf_role')
        construct.del_property('cfa_array')
        construct.del_property('cfa_dimensions', None)

        out = self._create_Data(loadd=cfa_data)

        return out

    def _create_Data(self, array=None, units=None, calendar=None,
                     ncvar=None, loadd=None, **kwargs):
        '''TODO

    .. versionadded:: 3.0.0

        '''
        try:
            compressed = array.get_compression_type()  # TODO
        except AttributeError:
            compressed = False

        if not compressed:
            # Do not chunk compressed data (for now ...)
            chunk = False
        else:
            chunk = self.read_vars.get('chunk', True)

        return super()._create_Data(array=array, units=units,
                                    calendar=calendar, ncvar=ncvar,
                                    loadd=loadd, **kwargs)

    def _customize_read_vars(self):
        '''TODO

    .. versionadded:: 3.0.0

        '''
        super()._customize_read_vars()

        g = self.read_vars

        # ------------------------------------------------------------
        # Find out if this is a CFA file
        # ------------------------------------------------------------
        g['cfa'] = 'CFA' in g['global_attributes'].get(
            'Conventions', ())  # TODO

        # ------------------------------------------------------------
        # Do not create fields from CFA private variables
        # ------------------------------------------------------------
        if g['cfa']:
            for ncvar in g['variables']:
                if (g['variable_attributes'][ncvar].get('cf_role', None) ==
                        'cfa_private'):
                    g['do_not_create_field'].add(ncvar)
        # --- End: if

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        if g['cfa']:
            for ncvar, ncdims in tuple(g['variable_dimensions'].items()):
                if ncdims != ():
                    continue

                if not (ncvar not in g['external_variables'] and
                        g['variable_attributes'][ncvar].get('cf_role') ==
                        'cfa_variable'):
                    continue

                ncdimensions = g['variable_attributes'][ncvar].get(
                    'cfa_dimensions', '').split()
                if ncdimensions:
                    g['variable_dimensions'][ncvar] = tuple(
                        map(str, ncdimensions))

    def file_open(self, filename):
        '''Open the netCDf file for reading.

    :Paramters:

        filename: `str`
            The netCDF file to be read.

    :Returns:

        `netCDF4.Dataset`
            The object for the file.

        '''
        out = super().file_open(filename)
        _file_to_fh['netCDF'].pop(filename, None)
        return out

# --- End: class
