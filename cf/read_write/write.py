from os.path import abspath

import numpy

from .netcdf import NetCDFWrite

from ..cfimplementation import implementation

from ..decorators import _manage_log_level_via_verbosity

from ..functions import flat
from ..functions import _DEPRECATION_ERROR_FUNCTION_KWARGS

# from . import mpi_on
mpi_on = False
if mpi_on:
    from . import mpi_comm
    from . import mpi_size
    from . import mpi_rank


netcdf = NetCDFWrite(implementation())


@_manage_log_level_via_verbosity
def write(fields, filename, fmt='NETCDF4', overwrite=True,
          global_attributes=None, file_descriptors=None,
          external=None, Conventions=None, datatype=None,
          least_significant_digit=None, endian='native', compress=0,
          fletcher32=False, shuffle=True, reference_datetime=None,
          verbose=None, cfa_options=None, mode='w', single=None,
          double=None, variable_attributes=None, string=True,
          warn_valid=True, HDF_chunksizes=None, no_shuffle=None,
          unlimited=None):
    '''Write field constructs to a netCDF file.

    **File format**

    See the *fmt* parameter for details on which output netCDF file
    formats are supported.

    **NetCDF variable and dimension names**

    These names are stored within constructs read a from dataset, or
    may be set manually. They are used when writing a field construct
    to the file. If a name has not been set then one will be
    constructed (usually based on the standard name if it exists). The
    names may be modified internally to prevent duplication in the
    file.

    Each construct, or construct component, that corresponds to a
    netCDF variable has the following methods to get, set and remove a
    netCDF variable name: `!nc_get_variable`, `!nc_set_variable` and
    `!nc_del_variable` method

    The domain axis construct has the following methods to get, set
    and remove a netCDF dimension name:
    `~cf.DomainAxis.nc_get_dimension`,
    `~cf.DomainAxis.nc_set_dimension` and
    `~cf.DomainAxis.nc_del_dimension`.

    **NetCDF attributes**

    Field construct properties may be written as netCDF global
    attributes and/or netCDF data variable attributes. See the
    *file_descriptors*, *global_attributes* and *variable_attributes*
    parameters for details.

    **External variables**

    Metadata constructs marked as external are omitted from the file
    and referred to via the netCDF "external_variables" global
    attribute. However, omitted constructs may be written to an
    external file (see the *external* parameter for details).

    **NetCDF unlimited dimensions**

    Domain axis constructs that correspond to NetCDF unlimited
    dimensions may be accessed with the
    `~cf.DomainAxis.nc_is_unlimited` and
    `~cf.DomainAxis.nc_set_unlimited` methods of a domain axis
    construct.

    **NetCDF4 HDF chunk sizes**

    HDF5 chunksizes may be set on contruct's data. See the
    `~cf.Data.nc_hdf5_chunksizes`,
    `~cf.Data.nc_clear_hdf5_chunksizes` and
    `~cf.Data.nc_set_hdf5_chunksizes` metods of a `Data` instance.

    .. seealso:: `cf.read`

    :Parameters:

        fields: (arbitrarily nested sequence of) `Field` or `FieldList`
            The field constructs to write to the file.


        filename: `str`
            The output netCDF file name. Various type of expansion are
            applied to the file names.

            Relative paths are allowed, and standard tilde and shell
            parameter expansions are applied to the string.

            *Parameter example:*
              The file file.nc in the user’s home directory could be
              described by any of the following: '``$HOME/file.nc'``,
              ``'${HOME}/file.nc'``, ``'~/file.nc'``,
              ``'~/tmp/../file.nc'``.

        fmt: `str`, optional
            The format of the output file. One of:

              ==========================  ================================
              *fmt*                       Output file type
              ==========================  ================================
              ``'NETCDF4'``               NetCDF4 format file. This is the
                                          default.

              ``'NETCDF4_CLASSIC'``       NetCDF4 classic format file (see
                                          below)

              ``'NETCDF3_CLASSIC'``       NetCDF3 classic format file
                                          (limited to file sizes less
                                          than 2GB).

              ``'NETCDF3_64BIT_OFFSET'``  NetCDF3 64-bit offset format
                                          file

              ``'NETCDF3_64BIT'``         An alias for
                                          ``'NETCDF3_64BIT_OFFSET'``

              ``'NETCDF3_64BIT_DATA'``    NetCDF3 64-bit offset format
                                          file with extensions (see below)

              ``'CFA'`` or ``'CFA4'``     CFA-netCDF4 format file

              ``'CFA3'``                  CFA-netCDF3 classic format file
              ==========================  ================================

            By default the format is ``'NETCDF4'``.

            All formats support large files (i.e. those greater than
            2GB) except ``'NETCDF3_CLASSIC'``.

            ``'NETCDF3_64BIT_DATA'`` is a format that requires version
            4.4.0 or newer of the C library (use `cf.environment` to
            see which version if the netCDF-C library is in use). It
            extends the ``'NETCDF3_64BIT_OFFSET'`` binary format to
            allow for unsigned/64 bit integer data types and 64-bit
            dimension sizes.

            ``'NETCDF4_CLASSIC'`` files use the version 4 disk format
            (HDF5), but omits features not found in the version 3
            API. They can be read by HDF5 clients. They can also be
            read by netCDF3 clients only if they have been re-linked
            against the netCDF4 library.

            ``'NETCDF4'`` files use the version 4 disk format (HDF5)
            and use the new features of the version 4 API.

        overwrite: `bool`, optional
            If False then raise an error if the output file
            pre-exists. By default a pre-existing output file is
            overwritten.

        Conventions: (sequence of) `str`, optional
             Specify conventions to be recorded by the netCDF global
             "Conventions" attribute. By default the current
             conventions are always included, but if an older CF
             conventions is defined then this is used instead.

             *Parameter example:*
               ``Conventions='UGRID-1.0'``

             *Parameter example:*
               ``Conventions=['UGRID-1.0']``

             *Parameter example:*
               ``Conventions=['CMIP-6.2', 'UGRID-1.0']``

             *Parameter example:*
               ``Conventions='CF-1.7'``

             *Parameter example:*
               ``Conventions=['CF-1.7', 'UGRID-1.0']``

             Note that if the "Conventions" property is set on a field
             construct then it is ignored.

        file_descriptors: `dict`, optional
             Create description of file contents netCDF global
             attributes from the specified attributes and their
             values.

             If any field construct has a property with the same name
             then it will be written as a netCDF data variable
             attribute, even if it has been specified by the
             *global_attributes* parameter, or has been flagged as
             global on any of the field constructs (see
             `cf.Field.nc_global_attributes` for details).

             Identification of the conventions being adhered to by the
             file are not specified as a file descriptor, but by the
             *Conventions* parameter instead.

             *Parameter example:*
               ``file_attributes={'title': 'my data'}``

             *Parameter example:*
               ``file_attributes={'history': 'created 2019-01-01', 'foo': 'bar'}``

        global_attributes: (sequence of) `str`, optional
             Create netCDF global attributes from the specified field
             construct properties, rather than netCDF data variable
             attributes.

             These attributes are in addition to the following field
             construct properties, which are created as netCDF global
             attributes by default:

             * the description of file contents properties (as defined
               by the CF conventions), and

             * properties flagged as global on any of the field
               constructs being written (see
               `cf.Field.nc_global_attributes` for details).

             Note that it is not possible to create a netCDF global
             attribute from a property that has different values for
             different field constructs being written. In this case
             the property will not be written as a netCDF global
             attribute, even if it has been specified by the
             *global_attributes* parameter or is one of the default
             properties, but will appear as an attribute on the netCDF
             data variable corresponding to each field construct that
             contains the property.

             Any global attributes that are also specified as file
             descriptors will not be written as netCDF global
             variables, but as netCDF data variable attributes
             instead.

             *Parameter example:*
               ``global_attributes='project'``

             *Parameter example:*
               ``global_attributes=['project']``

             *Parameter example:*
               ``global_attributes=['project', 'experiment']``

        variable_attributes: (sequence of) `str`, optional
             Create netCDF data variable attributes from the specified
             field construct properties.

             By default, all field construct properties that are not
             created as netCDF global properties are created as
             attributes netCDF data variables. See the
             *global_attributes* parameter for details.

             Any field construct property named by the
             *variable_attributes* parameter will always be created as
             a netCDF data variable attribute

             *Parameter example:*
               ``variable_attributes='project'``

             *Parameter example:*
               ``variable_attributes=['project']``

             *Parameter example:*
               ``variable_attributes=['project', 'doi']``

        external: `str`, optional
            Write metadata constructs that have data and are marked as
            external to the named external file. Ignored if there are
            no such constructs.

        mode: `str`, optional
            Specify the mode of write access for the output file. One of:

            =======  =================================================
            *mode*   Description
            =======  =================================================
            ``'w'``  Open a new file for writing to. If it exists and
                     *overwrite* is True then the file is deleted
                     prior to being recreated.
            =======  =================================================

            By default the file is opened with write access mode
            ``'w'``.

        cfa_options: `dict`, optional
            A dictionary giving parameters for configuring the output
            CFA-netCDF file:

            ==========  ===============================================
            Key         Value
            ==========  ===============================================
            ``'base'``  * If ``None`` (the default) then file names
                          within CFA-netCDF files are stored with
                          absolute paths.

                        * If set to an empty string then file names
                          within CFA-netCDF files are given relative to
                          the directory or URL base containing the
                          output CFA-netCDF file.

                        * If set to a string then file names within
                          CFA-netCDF files are given relative to the
                          directory or URL base described by the
                          value. For example: ``'../archive'``.
            ==========  ===============================================

            By default no parameters are specified.


        endian: `str`, optional
            The endian-ness of the output file. Valid values are
            ``'little'``, ``'big'`` or ``'native'``. By default the
            output is native endian. See the `netCDF4 package
            <http://unidata.github.io/netcdf4-python>`_ for more
            details.

            *Parameter example:*
              ``endian='big'``

        compress: `int`, optional
            Regulate the speed and efficiency of compression. Must be
            an integer between ``0`` and ``9``. ``0`` means no
            compression; ``1`` is the fastest, but has the lowest
            compression ratio; ``9`` is the slowest but best
            compression ratio. The default value is ``0``. An error is
            raised if compression is requested for a netCDF3 output
            file format. See the `netCDF4 package
            <http://unidata.github.io/netcdf4-python>`_ for more
            details.

            *Parameter example:*
              ``compress=4``

        least_significant_digit: `int`, optional
            Truncate the input field construct data arrays, but not
            the data arrays of metadata constructs. For a given
            positive integer, N the precision that is retained in the
            compressed data is 10 to the power -N. For example, a
            value of 2 will retain a precision of 0.01. In conjunction
            with the *compress* parameter this produces 'lossy', but
            significantly more efficient, compression. See the
            `netCDF4 package
            <http://unidata.github.io/netcdf4-python>`_ for more
            details.

            *Parameter example:*
              ``least_significant_digit=3``

        fletcher32: `bool`, optional
            If True then the Fletcher-32 HDF5 checksum algorithm is
            activated to detect compression errors. Ignored if
            *compress* is ``0``. See the `netCDF4 package
            <http://unidata.github.io/netcdf4-python>`_ for details.

        shuffle: `bool`, optional
            If True (the default) then the HDF5 shuffle filter (which
            de-interlaces a block of data before compression by
            reordering the bytes by storing the first byte of all of a
            variable's values in the chunk contiguously, followed by
            all the second bytes, and so on) is turned off. By default
            the filter is applied because if the data array values are
            not all wildly different, using the filter can make the
            data more easily compressible.  Ignored if the *compress*
            parameter is ``0`` (which is its default value). See the
            `netCDF4 package
            <http://unidata.github.io/netcdf4-python>`_ for more
            details.

            This parameter replaces the deprecated *no_shuffle*
            parameter.

        datatype: `dict`, optional
            Specify data type conversions to be applied prior to
            writing data to disk. This may be useful as a means of
            packing, or because the output format does not support a
            particular data type (for example, netCDF3 classic files
            do not support 64-bit integers). By default, input data
            types are preserved. Any data type conversion is only
            applied to the arrays on disk, and not to the input field
            constructs themselves.

            Data types conversions are defined by `numpy.dtype`
            objects in a dictionary whose keys are input data types
            with values of output data types.

            *Parameter example:*
              To convert 64-bit integers to 32-bit integers:
              ``datatype={numpy.dtype('int64'):
              numpy.dtype('int32')}``.

        single: `bool`, optional
            If True then write 64-bit floats as 32-bit floats and
            64-bit integers as 32-bit integers.

            If False then write 32-bit floats as 64-bit floats and
            32-bit integers as 64-bit integers.

            By default, input data types are preserved.

            .. note:: ``single=True`` is exactly equivalent to
                      ``double=False``, as well as
                      ``datatype={numpy.dtype(float):
                      numpy.dtype('float32'), numpy.dtype(int):
                      numpy.dtype('int32')}``.

                      ``single=False`` is exactly equivalent to
                      ``double=True``.

        double: `bool`, optional
            If True then write 32-bit floats as 64-bit floats and
            32-bit integers as 64-bit integers.

            If False then write 64-bit floats as 32-bit floats and
            64-bit integers as 32-bit integers.

            By default, input data types are preserved.

            .. note:: ``double=True`` is exactly equivalent to
                      ``single=False``, as well as
                      ``datatype={numpy.dtype('float32'):
                      numpy.dtype(float), numpy.dtype('int32'):
                      numpy.dtype(int)}``.

                      ``double=False`` is exactly equivalent to
                      ``single=True``.

        string: `bool`, optional
            By default string-valued construct data are written as
            netCDF arrays of type string if the output file format is
            ``'NETCDF4'``, or of type char with an extra dimension
            denoting the maximum string length for any other output
            file format (see the *fmt* parameter). If *string* is False
            then string-valued construct data are written as netCDF
            arrays of type char with an extra dimension denoting the
            maximum string length, regardless of the selected output
            file format.

        verbose: `int` or `None`, optional
            If an integer from ``0`` to ``3``, corresponding to increasing
            verbosity (else ``-1`` as a special case of maximal and extreme
            verbosity), set for the duration of the method call (only) as
            the minimum severity level cut-off of displayed log messages,
            regardless of the global configured `cf.LOG_LEVEL`.

            Else, if `None` (the default value), log messages will be
            filtered out, or otherwise, according to the value of the
            `cf.LOG_LEVEL` setting.

            Overall, the higher a non-negative integer that is set (up to
            a maximum of ``3``) the more description that is printed to
            convey how constructs map to output netCDF dimensions, variables
            and attributes.

        warn_valid: `bool`, optional
            If False then do not print a warning when writing
            "out-of-range" data, as indicated by the values, if
            present, of any of the ``valid_min``, ``valid_max`` or
            ``valid_range`` properties on field and metadata
            constructs that have data. By default a warning is printed
            if any such construct has any of these properties in
            combination with out-of-range data.

            The consequence of writing out-of-range data values is
            that, by default, these values will be masked when the
            file is subsequently read.

            *Parameter example:*
              If a construct has ``valid_max`` property with value
              ``100`` and data with maximum value ``999``, then the
              resulting warning may be suppressed by setting
              ``warn_valid=False``.

            .. versionadded:: 3.4.0

        HDF_chunksizes: deprecated at version 3.0.0
            HDF chunk sizes may be set for individual constructs prior
            to writing, instead. See `cf.Data.nc_set_hdf5_chunksizes`.

        no_shuffle: deprecated at version 3.0.0
            Use keyword *shuffle* instead.

        unlimited: deprecated at version 3.0.0
            Use method `DomainAxis.nc_set_unlimited` instead.

    :Returns:

        `None`

    **Examples:**

    There are further worked examples
    :ref:`in the tutorial <Writing-to-a-netCDF-dataset>`.

    >>> cf.write(f, 'file.nc')

    >>> cf.write(f, 'file.nc', fmt='NETCDF3_CLASSIC')

    >>> cf.write(f, 'file.nc', external='cell_measures.nc')

    >>> cf.write(f, 'file.nc', Conventions='CMIP-6.2')

    '''
    if unlimited is not None:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'cf.write', {'unlimited': unlimited},
            "Use method 'DomainAxis.nc_set_unlimited' instead."
        )  # pragma: no cover

    if no_shuffle is not None:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'cf.write', {'no_shuffle': no_shuffle},
            "Use keyword 'shuffle' instead."
        )  # pragma: no cover

    if HDF_chunksizes is not None:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'cf.write', {'HDF_chunksizes': HDF_chunksizes},
            "HDF chunk sizes may be set for individual field constructs "
            "prior to writing, instead."
        )  # pragma: no cover

    # Flatten the sequence of intput fields
    fields = tuple(flat(fields))

    mpi_on = False
    if mpi_on:
        path = abspath(filename)
        paths = mpi_comm.allgather(path)
        unique_paths = set(paths)
        n_unique_paths = len(unique_paths)
        if n_unique_paths == 1:
            write_only_on_pe0 = True
        elif n_unique_paths == mpi_size:
            write_only_on_pe0 = False
        else:
            raise RuntimeError(
                "write expects either one unique filename or as many as "
                "there are PEs"
            )

        if write_only_on_pe0 and not mpi_rank == 0:
            mpi_comm.Barrier()
            return
    # --- End: if

    if fields:
        # double and single
        if datatype:
            if single is not None:
                raise ValueError("Can't set datatype and single")
            if double is not None:
                raise ValueError("Can't set datatype and double")
        # --- End: if

        if single is not None and double is not None:
            raise ValueError("Can't set both the single and double "
                             "parameters")

        if single is not None and not single:
            double = True

        if double is not None and not double:
            single = True

        if single:
            datatype = {numpy.dtype(float): numpy.dtype('float32'),
                        numpy.dtype(int): numpy.dtype('int32')}

        if double:
            datatype = {numpy.dtype('float32'): numpy.dtype(float),
                        numpy.dtype('int32'): numpy.dtype(int)}

        extra_write_vars = {
            'cfa': False,
            'cfa_options': {},
            'reference_datetime': reference_datetime,
        }

        # CFA options
        if fmt in ('CFA', 'CFA4'):
            extra_write_vars['cfa'] = True
            fmt = 'NETCDF4'
            if cfa_options:
                extra_write_vars['cfa_options'] = cfa_options
        elif fmt == 'CFA3':
            extra_write_vars['cfa'] = True
            fmt = 'NETCDF3_CLASSIC'
            if cfa_options:
                extra_write_vars['cfa_options'] = cfa_options
        # --- End: if

        if extra_write_vars['cfa']:
            if Conventions:
                if isinstance(Conventions, str):
                    Conventions = (Conventions,)

                Conventions = tuple(Conventions) + ('CFA',)
            else:
                Conventions = 'CFA'
        # --- End: if

        netcdf.write(fields, filename, fmt=fmt, overwrite=overwrite,
                     global_attributes=global_attributes,
                     variable_attributes=variable_attributes,
                     file_descriptors=file_descriptors,
                     external=external, Conventions=Conventions,
                     datatype=datatype,
                     least_significant_digit=least_significant_digit,
                     endian=endian, compress=compress,
                     shuffle=shuffle, fletcher32=fletcher32,
                     verbose=verbose, string=string,
                     warn_valid=warn_valid,
                     extra_write_vars=extra_write_vars)
    # --- End: if

    if mpi_on and write_only_on_pe0:
        mpi_comm.Barrier()
