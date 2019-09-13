from os.path       import abspath

from . import implementation
from .netcdf import NetCDFWrite

from ..functions import flat
from ..functions import _DEPRECATION_ERROR_FUNCTION_KWARGS

#from . import mpi_on
mpi_on = False
if mpi_on:
    from . import mpi_comm
    from . import mpi_size
    from . import mpi_rank


netcdf = NetCDFWrite(implementation())

def write(fields, filename, fmt='NETCDF4', overwrite=True,
          global_attributes=None, file_descriptors=None,
          external=None, Conventions=None, datatype=None,
          least_significant_digit=None, endian='native', compress=0,
          fletcher32=False, shuffle=True, reference_datetime=None,
          verbose=False,
          
          cfa_options=None, mode='w',
          no_shuffle=False, 
          single=False, double=False, 
          variable_attributes=None, HDF_chunksizes=None,
          unlimited=None):
    '''Write fields to a netCDF file.
    
NetCDF dimension and variable names will be taken, if present, from
variables' `!ncvar` attributes and the domain axis `!ncdim`
attributes, otherwise they are inferred from standard names or set to
defaults. NetCDF names may be automatically given a numerical suffix
to avoid duplication.

Output netCDF file global properties are those which occur in the set
of CF global properties and non-standard data variable properties and
which have equal values across all input fields.

.. seealso:: `cf.read`

:Examples 1:

>>> cf.write(f, new_file.nc')

:Parameters:

    fields: (arbitrarily nested sequence of) `cf.Field` or `cf.FieldList`
        The field or fields to write to the file.

    filename: `str`
        The output netCDF file. Various type of expansion are applied
        to the file names:
        
          ====================  ======================================
          Expansion             Description
          ====================  ======================================
          Tilde                 An initial component of ``~`` or
                                ``~user`` is replaced by that user's
                                home directory.
           
          Environment variable  Substrings of the form ``$name`` or
                                ``${name}`` are replaced by the value
                                of environment variable ``name``.
          ====================  ======================================
    
        Where more than one type of expansion is used in the same
        string, they are applied in the order given in the above
        table.

          *Example:*
            If the environment variable ``MYSELF`` has been set to the
            ``'david'``, then ``'~$MYSELF/out.nc'`` is equivalent to
            ``'~david/out.nc'``.
  
    fmt: `str`, optional
        The format of the output file. One of:

           ==========================  ===============================
           *fmt*                       Output file type
           ==========================  ===============================
           ``'NETCDF4'``               NetCDF4 format file. This is
                                       the default.       
    
           ``'NETCDF4_CLASSIC'``       NetCDF4 classic format file
                                       (see below) 
    
           ``'NETCDF3_CLASSIC'``       NetCDF3 classic format file
                                       (limited to file sizes less
                                       than 2 Gb).

           ``'NETCDF3_64BIT_OFFSET'``  NetCDF3 64-bit offset format
                                       file

           ``'NETCDF3_64BIT_DATA'``    NetCDF3 64-bit offset format
                                       file with extensions (see
                                       below)

           ``'NETCDF3_64BIT'``         An alias for
                                       ``'NETCDF3_64BIT_OFFSET'``

           ``'CFA'`` or ``'CFA4'``     CFA-netCDF4 format file 

           ``'CFA3'``                  CFA-netCDF3 classic format file 
           ==========================  ===============================


        By default the format is ``'NETCDF4'``.

        Note that the netCDF3 formats may be considerably slower than
        any of the other options.

        All formats support large files (i.e. those greater than 2 Gb)
        except ``'NETCDF3_CLASSIC'``.

        ``'NETCDF3_64BIT_DATA'`` is a format that requires version
        4.4.0 or newer of the C library (use `cf.environment` to see
        which version if the netCDF-C library is in use). It extends
        the ``'NETCDF3_64BIT_OFFSET'`` binary format to allow for
        unsigned/64 bit integer data types and 64-bit dimension sizes.

        ``'NETCDF4_CLASSIC'`` files use the version 4 disk format
        (HDF5), but omits features not found in the version 3
        API. They can be read by HDF5 clients. They can also be read
        by netCDF 3 clients only if they have been relinked against
        the netCDF4 library.

        ``'NETCDF4'`` files use the version 4 disk format (HDF5) and
        use the new features of the version 4 API.

    overwrite: `bool`, optional
        If False then raise an exception if the output file
        pre-exists. By default a pre-existing output file is over
        written.

    verbose : `bool`, optional
        If True then print one-line summaries of each field written.

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
    
    mode: `str`, optional
        Specify the mode of write access for the output file. One of:

           =======  =====================================================
           mode     Description
           =======  =====================================================
           ``'w'``  Open a new file for writing to. If it exists and
                    *overwrite* is True then the file is deleted prior to
                    being recreated.
           =======  =====================================================
       
        By default the file is opened with write access mode ``'w'``.

    endian: `str`, optional
        The endian-ness of the output file. Valid values are
        ``'little'``, ``'big'`` or ``'native'``. By default the output
        is native endian.

    compress: `int`, optional
        Regulate the speed and efficiency of compression. Must be an
        integer between ``0`` and ``9``. ``0`` means no compression;
        ``1`` is the fastest, but has the lowest compression ratio;
        ``9`` is the slowest but best compression ratio. The default
        value is ``0``. An exception is raised if compression is
        requested for a netCDF3 output file format.
    
    least_significant_digit: `int`, optional
        Truncate the input field data arrays. For a positive integer,
        N the precision that is retained in the compressed data is '10
        to the power -N'. For example, a value of ``2`` will retain a
        precision of 0.01. In conjunction with the *compress*
        parameter this produces 'lossy', but significantly more
        efficient compression.
    
    fletcher32: `bool`, optional
        If True then the Fletcher-32 HDF5 checksum algorithm is
        activated to detect compression errors. Ignored if *compress*
        is ``0``.
 
    no_shuffle: `bool`, optional
        If True then the HDF5 shuffle filter (which de-interlaces a
        block of data before compression by reordering the bytes by
        storing the first byte of all of a variable's values in the
        chunk contiguously, followed by all the second bytes, and so
        on) is turned off. By default the filter is applied because if
        the data array values are not all wildly different, using the
        filter can make the data more easily compressible.  Ignored if
        *compress* is ``0``.

    datatype: `dict`, optional
        Specify data type conversions to be applied prior to writing
        data to disk. Arrays with data types which are not specified
        remain unchanged. By default, input data types are
        preserved. Data types defined by `numpy.dtype` objects in a
        dictionary whose are input data types with values of output
        data types. Note that data arrays of coordinates and other
        metadata may also be converted.

          *Example:*
            To convert 64-bit floats and 64-bit integers to their
            32-bit counterparts: ``datatype={numpy.dtype('float64'):
            numpy.dtype('float32'), numpy.dtype('int64'):
            numpy.dtype('int32')}``.
       
    single: `bool`, optional
        Write 64-bit floats as 32-bit floats and 64-bit integers as
        32-bit integers. By default, input data types are
        preserved. Note that ``single=True`` is exactly equivalent to
        ``datatype={numpy.dtype('float64'): numpy.dtype('float32'),
        numpy.dtype('int64'): numpy.dtype('int32')}``.
       
    double: `bool`, optional
        Write 32-bit floats as 64-bit floats and 32-bit integers as
        64-bit integers. By default, input data types are
        preserved. Note that ``double=True`` is exactly equivalent to
        ``datatype={numpy.dtype('float32'): numpy.dtype('float64'),
        numpy.dtype('int32'): numpy.dtype('int64')}``.

    HDF_chunksizes: `dict`, optional
        Manually specify HDF5 chunks for the field data arrays.

        Chunking refers to a storage layout where a data array is
        partitioned into fixed-size multi-dimensional chunks when
        written to a netCDF4 file on disk. Chunking is ignored if the
        field is written to a netCDF3 format file.

        A chunk has the same rank as the data array, but with fewer
        (or no more) elements along each axes. The chunk is defined by
        a dictionary whose keys identify axes with values of the
        chunks size for those axes.

        If a given chunk size for an axis is larger than the axis size
        for any field, then the size of the axis at the time of
        writing to disk will be used instead.

        If chunk sizes have been specified for some but not all axes,
        then the each unset chunk size is assumed to be the full size
        of the axis for each field.

        If no chunk sizes have been set for any axes then the netCDF
        default chunk is used
        (http://www.unidata.ucar.edu/software/netcdf/docs/netcdf_perf_chunking.html).

        If any chunk sizes have already been set on a field with the
        `cf.Field.HDF_chunks` method then these are used in instead.

        A detailed discussion of HDF chunking and I/O performance is
        available at
        https://www.hdfgroup.org/HDF5/doc/H5.user/Chunking.html and
        http://www.unidata.ucar.edu/software/netcdf/workshops/2011/nc4chunking.
        
    unlimited: sequence of `str`, optional
        Create unlimited dimensions (dimensions that can be appended
        to). A dimension is identified by either a standard name; one
        of T, Z, Y, X denoting time, height or horixontal axes
        repsectively (as defined by the CF conventions); or the value
        of an arbitrary CF property preceeded by the property name and
        a colon.

:Returns:

    `None`

:Examples 2:

>>> f
[<CF Field: air_pressure(30, 24)>,
 <CF Field: u_compnt_of_wind(19, 29, 24)>,
 <CF Field: v_compnt_of_wind(19, 29, 24)>,
 <CF Field: potential_temperature(19, 30, 24)>]
>>> write(f, 'file')

>>> cf.write([f, g], 'file.nc', verbose=True)
[<CF Field: air_pressure(30, 24)>,
 <CF Field: u_compnt_of_wind(19, 29, 24)>,
 <CF Field: v_compnt_of_wind(19, 29, 24)>,
 <CF Field: potential_temperature(19, 30, 24)>]

    '''
    if unlimited is not None:
        _DEPRECATION_ERROR_FUNCTION_KWARGS('cf.write', {'unlimited': unlimited},
                                           "Use method 'DomainAxis.nc_set_unlimited' instead.") # pragma: no cover
                                           
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
            raise RuntimeError('write expects either 1 unique filename or as many as there are PEs')
        #--- End: if

        if write_only_on_pe0 and not mpi_rank == 0:
            mpi_comm.Barrier()
            return
    #--- End: if
    
    if fields:
        extra_write_vars = {'cfa'               : False,
                            'cfa_options'       : {},
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
        #--- End: if
        
        if extra_write_vars['cfa']:
            if Conventions:
                if isinstance(Conventions, str):
                    Conventions = (Conventions,)
                    
                Conventions = tuple(Conventions) + ('CFA',)
            else:
                Conventions = 'CFA'
        #--- End: if

        netcdf.write(fields, filename, fmt=fmt, overwrite=overwrite,
                     global_attributes=global_attributes,
                     variable_attributes=variable_attributes,
                     file_descriptors=file_descriptors,
                     external=external, Conventions=Conventions,
                     datatype=datatype,
                     least_significant_digit=least_significant_digit,
                     endian=endian, compress=compress,
                     shuffle=shuffle, fletcher32=fletcher32,
                     verbose=verbose,
                     extra_write_vars=extra_write_vars)
    #--- End: if

    if mpi_on and write_only_on_pe0:
        mpi_comm.Barrier()
#--- End: def
