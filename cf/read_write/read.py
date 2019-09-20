import os

from glob    import glob
from os.path import isdir, isfile

import cfdm

from . import implementation

#from ..field     import FieldList
from ..fieldlist import FieldList
from ..functions import flat
from ..aggregate import aggregate as cf_aggregate

from .netcdf import NetCDFRead
from .um     import UMRead

from ..functions import _DEPRECATION_ERROR_FUNCTION_KWARGS

# --------------------------------------------------------------------
# Create an implementation container and initialize read objects
# --------------------------------------------------------------------
_implementation = implementation()
netcdf = NetCDFRead(_implementation)
UM     = UMRead(_implementation)


def read(files, external=None, verbose=False, warnings=False,
         ignore_read_error=False, aggregate=True, nfields=None,
         squeeze=False, unsqueeze=False, fmt=None, select=None,
         extra=None, height_at_top_of_model=None, recursive=False,
         followlinks=False, um=None, chunk=True, field=None,
         select_options=None, follow_symlinks=False):
    '''Read fields from netCDF, PP or UM fields files.

    Files may be on disk or on a OPeNDAP server.
    
    Any amount of any combination of CF-netCDF and CFA-netCDF files
    (or URLs if DAP access is enabled), Met Office (UK) PP files and
    Met Office (UK) fields files format files may be read.
    
    **PP and UM fields files**

    32-bit and 64-bit PP and UM fields files of any endian-ness can be
    read. In nearly all cases the file format is auto-detected from
    the first 64 bits in the file, but for the few occasions when this
    is not possible, the *um* keyword allows the format to be
    specified, as well as the UM version (if the latter is not
    inferrable from the PP or lookup header information).

    2-d "slices" within a single file are always combined, where
    possible, into field constructs with 3-d, 4-d or 5-d data. This is
    done prior to the field construct aggregation carried out by the
    `cf.read` function.

    When reading PP and UM fields files, the *relaxed_units* aggregate
    option is set to `True` by default, because units are not always
    available to field constructs derived from UM fields files or PP
    files.

    .. seealso:: `cf.write`, `cf.aggregate`,
                 `cf.load_stash2standard_name`
    
    :Parameters:
    
        files: (arbitrarily nested sequence of) `str`
            A string or arbitrarily nested sequence of strings giving
            the file names, directory names, or OPenDAP URLs from
            which to read fields. Various type of expansion are
            applied to the names:
            
              ====================  ======================================
              Expansion             Description
              ====================  ======================================
              Tilde                 An initial component of ``~`` or
                                    ``~user`` is replaced by that *user*'s
                                    home directory.
               
              Environment variable  Substrings of the form ``$name`` or
                                    ``${name}`` are replaced by the value
                                    of environment variable *name*.
    
              Pathname              A string containing UNIX file name
                                    metacharacters as understood by the
                                    :py:obj:`glob` module is replaced by
                                    the list of matching file names. This
                                    type of expansion is ignored for
                                    OPenDAP URLs.
              ====================  ======================================
        
            Where more than one type of expansion is used in the same
            string, they are applied in the order given in the above
            table.
    
            *Parameter example:*
              The file ``file.nc`` in the user's home directory could
              be described by any of the following:
              ``'$HOME/file.nc'``, ``'${HOME}/file.nc'``,
              ``'~/file.nc'``, ``'~/tmp/../file.nc'``.
    
            When a directory is specified, all files in that directory
            are read. Sub-directories are not read unless the
            *recursive* parameter is True. If any directories contain
            files that are not valid datasets then an exception will
            be raised, unless the *ignore_read_error* parameter is
            True.
    
         external: (sequence of) `str`, optional
            Read external variables (i.e. variables which are named by
            attributes, but are not present, in the parent file given
            by the *filename* parameter) from the given external
            files. Ignored if the parent file does not contain a
            global "external_variables" attribute. Multiple external
            files may be provided, which are searched in random order
            for the required external variables.
           
            If an external variable is not found in any external
            files, or is found in multiple external files, then the
            relevant metadata construct is still created, but without
            any metadata or data. In this case the construct's
            `!is_external` method will return `True`.
    
            *Parameter example:*
              ``external='cell_measure.nc'``
    
            *Parameter example:*
              ``external=['cell_measure.nc']``
    
            *Parameter example:*
              ``external=('cell_measure_A.nc', 'cell_measure_O.nc')``
    
        extra: (sequence of) `str`, optional
            Create extra, independent fields from netCDF variables
            that correspond to particular types metadata
            constructs. The *extra* parameter may be one, or a
            sequence, of:
    
              ==========================  ================================
              *extra*                     Metadata constructs
              ==========================  ================================
              ``'field_ancillary'``       Field ancillary constructs
              ``'domain_ancillary'``      Domain ancillary constructs
              ``'dimension_coordinate'``  Dimension coordinate constructs
              ``'auxiliary_coordinate'``  Auxiliary coordinate constructs
              ``'cell_measure'``          Cell measure constructs
              ==========================  ================================
    
            *Parameter example:*
              To create fields from auxiliary coordinate constructs:
              ``extra='auxiliary_coordinate'`` or
              ``extra=['auxiliary_coordinate']``.
    
            *Parameter example:*
              To create fields from domain ancillary and cell measure
              constructs: ``extra=['domain_ancillary',
              'cell_measure']``.
    
            An extra field construct created via the *extra* parameter
            will have a domain limited to that which can be inferred
            from the corresponding netCDF variable, but without the
            connections that are defined by the parent netCDF data
            variable. It is possible to create independent fields from
            metadata constructs that do incorporate as much of the
            parent field construct's domain as possible by using the
            `~cfdm.Field.convert` method of a returned field
            construct, instead of setting the *extra* parameter.
    
        verbose: `bool`, optional
            If True then print a description of how the contents of
            the netCDF file were parsed and mapped to CF data model
            constructs.
    
        warnings: `bool`, optional
            If True then print warnings when an output field construct
            is incomplete due to structural non-compliance of the
            dataset. By default such warnings are not displayed.
            
        ignore_read_error: `bool`, optional
            If True then ignore any file which raises an IOError
            whilst being read, as would be the case for an empty file,
            unknown file format, etc. By default the IOError is
            raised.
        
        fmt: `str`, optional
            Only read files of the given format, ignoring all other
            files. Valid formats are ``'NETCDF'`` for CF-netCDF files,
            ``'CFA'`` for CFA-netCDF files and ``'PP'`` for PP files
            and 'FF' for UM fields files. By default files of any of
            these formats are read.
    
        aggregate: `bool` or `dict`, optional
            If True (the default) or a (possibly empty) dictionary
            then aggregate the fields read in from all input files
            into as few fields as possible by passing all of the
            fields found the input files to the `cf.aggregate`, and
            returning the output of this function call.
    
            If *aggregate* is a dictionary then it is used to
            configure the aggregation process passing its contents as
            keyword arguments to the `cf.aggregate` function.
    
            If *aggregate* is False then the fields are not
            aggregated.
    
        squeeze: `bool`, optional
            If True then remove size 1 axes from each field's data
            array.
    
        unsqueeze: `bool`, optional
            If True then insert size 1 axes from each field's domain
            into its data array.
    
        select, select_options: optional TODO
            Only return fields which satisfy the given conditions on
            their property values. Only fields which, prior to any
            aggregation, satisfy ``f.match(description=select,
            **select_options) == True`` are returned. See
            `cf.Field.match` for details.
    
        field: (sequence of) `str`, optional TODO
            Create independent fields from field components. The
            *field* parameter may be one, or a sequence, of:
    
              ======================  ====================================
              *field*                 Field components
              ======================  ====================================
              ``'field_ancillary'``   Field ancillary objects
              ``'domain_ancillary'``  Domain ancillary objects
              ``'dimension'``         Dimension coordinate objects
              ``'auxiliary'``         Auxiliary coordinate objects
              ``'measure'``           Cell measure objects
              ``'all'``               All of the above
              ======================  ====================================
    
              *Parameter example:*
                To create fields from auxiliary coordinate objects:
                ``field='auxiliary'`` or ``field=['auxiliary']``.
    
              *Parameter example:*
                To create fields from domain ancillary and cell
                measure objects: ``field=['domain_ancillary',
                'measure']``.
    
            .. versionadded:: 3.0.0
    
        recursive: `bool`, optional
            If True then recursively read sub-directories of any
            directories specified with the *files* parmaeter.
    
        followlinks: `bool`, optional
            If True, and *recursive* is True, then also search for
            files in sub-directories which resolve to symbolic
            links. By default directories which resolve to symbolic
            links are ignored. Ignored of *recursive* is False. Files
            which are symbolic links are always followed.
    
            Note that setting ``recursive=True, followlinks=True`` can
            lead to infinite recursion if a symbolic link points to a
            parent directory of itself.
    
        um: `dict`, optional
            For Met Office (UK) PP files and Met Office (UK) fields
            files only, provide extra decoding instructions. This
            option is ignored for input files which are not PP or
            fields files. In most cases, how to decode a file is
            inferrable from the file's contents, but if not then each
            key/value pair in the dictionary sets a decoding option as
            follows:
    
              ===============  ===========================================
              Key              Value
              ===============  ===========================================
              ``'fmt'``        The file format (``'PP'`` or ``'FF'``)
    
              ``'word_size'``  The word size in bytes (``4`` or ``8``)
    
              ``'endian'``     The byte order (``'big'`` or ``'little'``)
    
              ``'version'``    The Unified Model version to be used when
                               decoding the header. Valid versions are,
                               for example, ``4.2``, ``'6.6.3'`` and
                               ``'8.2'``. The default version is
                               ``4.5``. In general, a given version is
                               ignored if it can be inferred from the
                               header (which is usually the case for files
                               created by the UM at versions 5.3 and
                               later). The exception to this is when the
                               given version has a third element (such as
                               the 3 in 6.6.3), in which case any version
                               in the header is ignored.
              ===============  ===========================================
    
            If format is specified as PP then the word size and byte
            order default to ``4`` and ``'big'`` repsectively.
    
            *Parameter example:*
                To specify that the input files are 32-bit, big-endian
                PP files: ``um={'fmt': 'PP'}``
    
            *Parameter example:*
                To specify that the input files are 32-bit,
                little-endian PP files from version 5.1 of the Unified
                Model: ``um={'fmt': 'PP', 'endian': 'little',
                'version': 5.1}``
    
            .. versionadded:: 1.5
    
        umversion: deprecated at version 3.0.0
            Use the *um* parameter instead.
    
        field: deprecated at version 3.0.0
            Use the *extra* parameter instead.
    
        follow_symlinks: deprecated at version 3.0.0
            Use the *followlinks* parameter instead.
    
        select_options: deprecated at version 3.0.0
    
    :Returns:
        
        `FieldList`
            A list of the fields found in the input file(s). The list
            may be empty.
    
    **Examples:**
    
    >>> x = cfdm.read('file.nc')
    >>> print(type(x))
    <type 'list'>
    
    Read a file and create field constructs from CF-netCDF data
    variables as well as from the netCDF variables that correspond to
    particular types metadata constructs:
    
    >>> f = cfdm.read('file.nc', extra='domain_ancillary')
    >>> g = cfdm.read('file.nc', extra=['dimension_coordinate', 
    ...                                 'auxiliary_coordinate'])
    
    Read a file that contains external variables:
    
    >>> h = cfdm.read('parent.nc')
    >>> i = cfdm.read('parent.nc', external='external.nc')
    >>> j = cfdm.read('parent.nc', external=['external1.nc', 'external2.nc'])
    
    >>> f = cf.read('file*.nc')
    >>> f
    [<CF Field: pmsl(30, 24)>,
     <CF Field: z-squared(17, 30, 24)>,
     <CF Field: temperature(17, 30, 24)>,
     <CF Field: temperature_wind(17, 29, 24)>]
    
    >>> cf.read('file*.nc')[0:2]
    [<CF Field: pmsl(30, 24)>,
     <CF Field: z-squared(17, 30, 24)>]
    
    >>> cf.read('file*.nc')[-1]
    <CF Field: temperature_wind(17, 29, 24)>
    
    >>> cf.read('file*.nc', select='units=K)
    [<CF Field: temperature(17, 30, 24)>,
     <CF Field: temperature_wind(17, 29, 24)>]
    
    >>> cf.read('file*.nc', select='ncvar%ta')
    <CF Field: temperature(17, 30, 24)>

    '''
    if field:
        _DEPRECATION_ERROR_FUNCTION_KWARGS('cf.read', {'field': field},
                                           "Use keyword 'extra' instead") # pragma: no cover

    if select_options:
        _DEPRECATION_ERROR_FUNCTION_KWARGS('cf.read',
                                           {'select_options': select_options}) # pragma: no cover
        
    if follow_symlinks:
        _DEPRECATION_ERROR_FUNCTION_KWARGS('cf.read',
                                           {'follow_symlinks': follow_symlinks},
                                           "Use keyword 'followlink' instead.") # pragma: no cover

    # Parse select
    if isinstance(select, str):
        select = (select,)
        
    if squeeze and unsqueeze:
        raise ValueError("squeeze and unsqueeze can not both be True")

    if follow_symlinks and not recursive:
        raise ValueError(
            "Can't set follow_symlinks={0} when recursive={1}".format(
                follow_symlinks, recursive))

    # Initialize the output list of fields
    field_list = FieldList()

    if isinstance(aggregate, dict):
        aggregate_options = aggregate.copy()
        aggregate         = True
    else:
        aggregate_options = {}

    aggregate_options['copy'] = False
    
    # Parse the extra parameter
    if extra is None:
        extra = ()
    elif isinstance(extra, str):
        extra = (extra,)

    # Count the number of fields (in all files) and the number of
    # files
    field_counter = -1
    file_counter  = 0

    for file_glob in flat(files):

        # Expand variables
        file_glob = os.path.expanduser(os.path.expandvars(file_glob))

        if file_glob.startswith('http://'):
            # Do not glob a URL
            files2 = (file_glob,)
        else:
            # Glob files on disk
            files2 = glob(file_glob)
            
            if not files2 and not ignore_read_error:
                open(file_glob, 'rb')
                
            files3 = []
            for x in files2:
                if isdir(x):
                    # Walk through directories, possibly recursively
                    for path, subdirs, filenames in os.walk(x, followlinks=followlinks):
                        files3.extend(os.path.join(path, f) for f in filenames)
                        if not recursive:                            
                            break
                else:
                    files3.append(x)
            #--- End: for
            
            files2 = files3

        for filename in files2:
            if verbose:
                print('File: {0}'.format(filename)) # pragma: no cover
                
            # --------------------------------------------------------
            # Read the file into fields
            # --------------------------------------------------------
            fields = _read_a_file(filename, external=external,
                                  ignore_read_error=ignore_read_error,
                                  verbose=verbose, warnings=warnings,
                                  aggregate=aggregate,
                                  aggregate_options=aggregate_options,
                                  selected_fmt=fmt, um=um,
                                  extra=extra,
                                  height_at_top_of_model=height_at_top_of_model,
                                  chunk=chunk)
            
            # --------------------------------------------------------
            # Select matching fields
            # --------------------------------------------------------
            if select:
                fields = fields.select(*select)

            # --------------------------------------------------------
            # Add this file's fields to those already read from other
            # files
            # --------------------------------------------------------
            field_list.extend(fields)
   
            field_counter = len(field_list)
            file_counter += 1
        #--- End: for            
    #--- End: for     

    # Print some informative messages
    if verbose:
        print("Read {0} field{1} from {2} file{3}".format( 
            field_counter, _plural(field_counter),
            file_counter , _plural(file_counter))) # pragma: no cover
   
    # ----------------------------------------------------------------
    # Aggregate the output fields
    # ----------------------------------------------------------------
    if aggregate and len(field_list) > 1:
        if verbose:
            org_len = len(field_list) # pragma: no cover
            
        field_list = cf_aggregate(field_list, **aggregate_options)
        
        if verbose:
            n = len(field_list) # pragma: no cover
            print('{0} input field{1} aggregated into {2} field{3}'.format(
                org_len, _plural(org_len), 
                n, _plural(n))) # pragma: no cover
    #--- End: if

#    # ----------------------------------------------------------------
#    # Add standard names to UM fields
#    # ----------------------------------------------------------------
#    for f in field_list:
#        standard_name = getattr(f, '_standard_name', None)
#        if standard_name is not None:
#            f.standard_name = standard_name
#            del f._standard_name
#    #--- End: for

    # ----------------------------------------------------------------
    # Squeeze size one dimensions from the data arrays. Do one of:
    # 
    # 1) Squeeze the fields, i.e. remove all size one dimensions from
    #    all field data arrays
    #
    # 2) Unsqueeze the fields, i.e. Include all size 1 domain
    #    dimensions in the data array.
    #
    # 3) Nothing
    # ----------------------------------------------------------------
    if squeeze:
        for f in field_list:
            f.squeeze(inplace=True) 
    elif unsqueeze:
        for f in field_list:
            f.unsqueeze(inplace=True)
    #--- End: if
    
    if nfields is not None and len(field_list) != nfields:
        raise ValueError(
            "{} field{} requested but {} fields found in file{}".format(
                nfields, _plural(nfields), len(field_list), _plural(file_counter)))

    return field_list


def _plural(n): # pragma: no cover
    '''Return a suffix which reflects a word's plural.

    '''
    return 's' if n !=1 else '' # pragma: no cover


def _read_a_file(filename, aggregate=True, aggregate_options={},
                 ignore_read_error=False, verbose=False,
                 warnings=False, external=None, selected_fmt=None,
                 um=None, extra=None, height_at_top_of_model=None,
                 chunk=True):
    '''Read the contents of a single file into a field list.

    :Parameters:
    
        filename: `str`
            The file name.
    
        aggregate_options: `dict`, optional
            The keys and values of this dictionary may be passed as
            keyword parameters to an external call of the aggregate
            function.
    
        ignore_read_error: `bool`, optional
            If True then return an empty field list if reading the
            file produces an IOError, as would be the case for an
            empty file, unknown file format, etc. By default the
            IOError is raised.
        
        verbose: `bool`, optional
            If True then print information to stdout.
        
    :Returns:
    
        `FieldList`
            The fields in the file.

    '''
    # Find this file's type
    fmt       = None
    word_size = None
    endian    = None
    umversion = 405

    if um:
        ftype = 'UM' 
        fmt         = um.get('fmt')
        word_size   = um.get('word_size')
        endian      = um.get('endian')
        umversion   = um.get('version')
        if fmt in ('PP', 'pp'):
            fmt = fmt.upper()
            # For PP format, there is a default word size and
            # endian-ness
            if word_size is None:
                word_size = 4
            if endian is None:
                endian = 'big'
        #--- End: if

        if umversion is not None:
            umversion = float(str(umversion).replace('.', '0', 1))
    else:
        try:
            ftype = file_type(filename)        
        except Exception as error:
            if not ignore_read_error:
                raise Exception(error)

            if verbose:
                print('WARNING: {}'.format(error)) # pragma: no cover
                
            return FieldList()            
    #--- End: if

    extra_read_vars = {'chunk'            : chunk,
                       'fmt'              : selected_fmt,
                       'ignore_read_error': ignore_read_error,
                       'cfa'              : False,
    }
    
    # ----------------------------------------------------------------
    # Still here? Read the file into fields.
    # ----------------------------------------------------------------
    if ftype == 'netCDF' and (selected_fmt in (None, 'NETCDF', 'CFA')):
        fields = netcdf.read(filename, external=external, extra=extra,
                             verbose=verbose, warnings=warnings,
                             extra_read_vars=extra_read_vars)
        
    elif ftype == 'UM' and (selected_fmt in (None, 'PP', 'FF')):
        fields = UM.read(filename, um_version=umversion,
                         verbose=verbose, set_standard_name=True,
                         height_at_top_of_model=height_at_top_of_model,
                         fmt=fmt, word_size=word_size, endian=endian,
                         chunk=chunk)

        # PP fields are aggregated intrafile prior to interfile
        # aggregation
        if aggregate:
            # For PP fields, the default is strict_units=False
            if 'strict_units' not in aggregate_options:
                aggregate_options['relaxed_units'] = True
    else:
        fields = ()

    # ----------------------------------------------------------------
    # Check for cyclic dimensions
    # ----------------------------------------------------------------
    for f in fields:
        f.autocyclic()
    
    # ----------------------------------------------------------------
    # Return the fields
    # ----------------------------------------------------------------
    return FieldList(fields)


def file_type(filename):
    '''TODO

    :Parameters:
    
        filename: `str`
            The file name.
    
    :Returns:
    
        `str`
            The format type of the file.
    
    **Examples:**
    
    >>> ftype = file_type(filename)

    '''
    # ----------------------------------------------------------------
    # NetCDF
    # ----------------------------------------------------------------
    if netcdf.is_netcdf_file(filename):
        return 'netCDF'
  
    # ----------------------------------------------------------------
    # PP or FF
    # ----------------------------------------------------------------
    if UM.is_um_file(filename):
        return 'UM'

    # Still here?
    raise IOError("Can't determine format of file {}".format(filename))
