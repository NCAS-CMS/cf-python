import logging
import os

from ctypes.util import find_library
from glob        import glob
from os.path     import isdir

from .netcdf import NetCDFRead
from .um     import UMRead

from ..cfimplementation import implementation

from ..fieldlist import FieldList

from ..aggregate import aggregate as cf_aggregate

from ..decorators import _manage_log_level_via_verbosity

from ..functions import flat, _DEPRECATION_ERROR_FUNCTION_KWARGS


# --------------------------------------------------------------------
# Create an implementation container and initialize a read object for
# each format
# --------------------------------------------------------------------
_implementation = implementation()
netcdf = NetCDFRead(_implementation)
UM = UMRead(_implementation)


logger = logging.getLogger(__name__)


@_manage_log_level_via_verbosity
def read(files, external=None, verbose=None, warnings=False,
         ignore_read_error=False, aggregate=True, nfields=None,
         squeeze=False, unsqueeze=False, fmt=None, select=None,
         extra=None, recursive=False, followlinks=False, um=None,
         chunk=True, field=None, height_at_top_of_model=None,
         select_options=None, follow_symlinks=False, mask=True,
         warn_valid=False):
    '''Read field constructs from netCDF, CDL, PP or UM fields files.

    NetCDF files may be on disk or on an OPeNDAP server.

    Any amount of files of any combination of file types may be read.

    **NetCDF unlimited dimensions**

    Domain axis constructs that correspond to NetCDF unlimited
    dimensions may be accessed with the
    `~cf.DomainAxis.nc_is_unlimited` and
    `~cf.DomainAxis.nc_set_unlimited` methods of a domain axis
    construct.

    **CF-compliance**

    If the dataset is partially CF-compliant to the extent that it is
    not possible to unambiguously map an element of the netCDF dataset
    to an element of the CF data model, then a field construct is
    still returned, but may be incomplete. This is so that datasets
    which are partially conformant may nonetheless be modified in
    memory and written to new datasets.

    Such "structural" non-compliance would occur, for example, if the
    "coordinates" attribute of a CF-netCDF data variable refers to
    another variable that does not exist, or refers to a variable that
    spans a netCDF dimension that does not apply to the data
    variable. Other types of non-compliance are not checked, such
    whether or not controlled vocabularies have been adhered to. The
    structural compliance of the dataset may be checked with the
    `~cf.Field.dataset_compliance` method of the field construct, as
    well as optionally displayed when the dataset is read by setting
    the warnings parameter.


    **CDL files**

    A file is considered to be a CDL representation of a netCDF
    dataset if it is a text file whose first non-comment line starts
    with the seven characters "netcdf " (six letters followed by a
    space). A comment line is identified as one which starts with any
    amount white space (including none) followed by "//" (two
    slashes). It is converted to a temporary netCDF4 file using the
    external ``ncgen`` command, and the temporary file persists until
    the end of the Python session, at which time it is automatically
    deleted. The CDL file may omit data array values (as would be the
    case, for example, if the file was created with the ``-h`` or
    ``-c`` option to ``ncdump``), in which case the the relevant
    constructs in memory will be created with data with all missing
    values.

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

    **Performance**

    Descriptive properties are always read into memory, but lazy
    loading is employed for all data arrays, which means that no data
    is read into memory until the data is required for inspection or
    to modify the array contents. This maximises the number of field
    constructs that may be read within a session, and makes the read
    operation fast.

    .. seealso:: `cf.aggregate`, `cf.load_stash2standard_name`,
                 `cf.write`, `cf.Field.convert`,
                 `cf.Field.dataset_compliance`

    :Parameters:

        files: (arbitrarily nested sequence of) `str`
            A string or arbitrarily nested sequence of strings giving
            the file names, directory names, or OPenDAP URLs from
            which to read field constructs. Various type of expansion
            are applied to the names:

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
                                  Python `glob` module is replaced by
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
            Create extra, independent field constructs from netCDF
            variables that correspond to particular types metadata
            constructs. The *extra* parameter may be one, or a
            sequence, of:

            ==========================  ===============================
            *extra*                     Metadata constructs
            ==========================  ===============================
            ``'field_ancillary'``       Field ancillary constructs
            ``'domain_ancillary'``      Domain ancillary constructs
            ``'dimension_coordinate'``  Dimension coordinate constructs
            ``'auxiliary_coordinate'``  Auxiliary coordinate constructs
            ``'cell_measure'``          Cell measure constructs
            ==========================  ===============================

            This parameter replaces the deprecated *field* parameter.

            *Parameter example:*
              To create field constructs from auxiliary coordinate
              constructs: ``extra='auxiliary_coordinate'`` or
              ``extra=['auxiliary_coordinate']``.

            *Parameter example:*
              To create field constructs from domain ancillary and
              cell measure constructs: ``extra=['domain_ancillary',
              'cell_measure']``.

            An extra field construct created via the *extra* parameter
            will have a domain limited to that which can be inferred
            from the corresponding netCDF variable, but without the
            connections that are defined by the parent netCDF data
            variable. It is possible to create independent fields from
            metadata constructs that do incorporate as much of the
            parent field construct's domain as possible by using the
            `~cf.Field.convert` method of a returned field construct,
            instead of setting the *extra* parameter.

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
            convey how the contents of the netCDF file were parsed and
            mapped to CF data model constructs.

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
            ``'CFA'`` for CFA-netCDF files, ``'UM'`` for PP or UM
            fields files, and ``'CDL'`` for CDL text files. By default
            files of any of these formats are read.

        aggregate: `bool` or `dict`, optional
            If True (the default) or a dictionary (possibly empty)
            then aggregate the field constructs read in from all input
            files into as few field constructs as possible by passing
            all of the field constructs found the input files to the
            `cf.aggregate`, and returning the output of this function
            call.

            If *aggregate* is a dictionary then it is used to
            configure the aggregation process passing its contents as
            keyword arguments to the `cf.aggregate` function.

            If *aggregate* is False then the field constructs are not
            aggregated.

        squeeze: `bool`, optional
            If True then remove size 1 axes from each field construct's
            data array.

        unsqueeze: `bool`, optional
            If True then insert size 1 axes from each field
            construct's domain into its data array.

        select: (sequence of) str, optional
            Only return field constructs which have given identities,
            i.e. those that satisfy
            ``f.match_by_identity(*select)``. See
            `cf.Field.match_by_identity` for details.

        recursive: `bool`, optional
            If True then recursively read sub-directories of any
            directories specified with the *files* parameter.

        followlinks: `bool`, optional
            If True, and *recursive* is True, then also search for
            files in sub-directories which resolve to symbolic
            links. By default directories which resolve to symbolic
            links are ignored. Ignored of *recursive* is False. Files
            which are symbolic links are always followed.

            Note that setting ``recursive=True, followlinks=True`` can
            lead to infinite recursion if a symbolic link points to a
            parent directory of itself.

            This parameter replaces the deprecated *follow_symlinks*
            parameter.

        mask: `bool`, optional
            If False then do not mask by convention when reading the
            data of field or metadata constructs from disk. By default
            data is masked by convention.

            The masking by convention of a netCDF array depends on the
            values of any of the netCDF variable attributes
            ``_FillValue``, ``missing_value``, ``valid_min``,
            ``valid_max`` and ``valid_range``.

            The masking by convention of a PP or UM array depends on
            the value of BMDI in the lookup header. A value other than
            ``-1.0e30`` indicates the data value to be masked.

            See
            https://ncas-cms.github.io/cf-python/tutorial.html#data-mask
            for details.

            .. versionadded:: 3.4.0

        warn_valid: `bool`, optional
            If True then print a warning for the presence of
            ``valid_min``, ``valid_max`` or ``valid_range`` properties
            on field contructs and metadata constructs that have
            data. By default no such warning is issued.

            "Out-of-range" data values in the file, as defined by any
            of these properties, are automatically masked by default,
            which may not be as intended. See the *mask* parameter for
            turning off all automatic masking.

            See
            https://ncas-cms.github.io/cf-python/tutorial.html#data-mask
            for details.

            .. versionadded:: 3.4.0

        um: `dict`, optional
            For Met Office (UK) PP files and Met Office (UK) fields
            files only, provide extra decoding instructions. This
            option is ignored for input files which are notPP or
            fields files. In most cases, how to decode a file is
            inferrable from the file's contents, but if not then each
            key/value pair in the dictionary sets a decoding option as
            follows:

            ============================  =====================================
            Key                           Value
            ============================  =====================================
            ``'fmt'``                     The file format (``'PP'`` or
                                          ``'FF'``)

            ``'word_size'``               The word size in bytes (``4`` or
                                          ``8``)

            ``'endian'``                  The byte order (``'big'`` or
                                          ``'little'``)

            ``'version'``                 The UM version to be used
                                          when decoding the
                                          header. Valid versions
                                          are, for example, ``4.2``,
                                          ``'6.6.3'`` and
                                          ``'8.2'``. The default
                                          version is ``4.5``. In
                                          general, a given version
                                          is ignored if it can be
                                          inferred from the header
                                          (which is usually the case
                                          for files created by the
                                          UM at versions 5.3 and
                                          later). The exception to
                                          this is when the given
                                          version has a third
                                          element (such as the 3 in
                                          6.6.3), in which case any
                                          version in the header is
                                          ignored.

            ``'height_at_top_of_model'``  The height (in metres) of
                                          the upper bound of the top
                                          model level. By default
                                          the height at top model is
                                          taken from the top level's
                                          upper bound defined by
                                          BRSVD1 in the lookup
                                          header. If the height at
                                          top model can not be
                                          determined from the header and is
                                          not provided then no
                                          "atmosphere_hybrid_height_coordinate"
                                          dimension coordinate
                                          construct will be created.
            ============================  =====================================

            If format is specified as ``'PP'`` then the word size and
            byte order default to ``4`` and ``'big'`` respectively.

            This parameter replaces the deprecated *umversion* and
            *height_at_top_of_model* parameters.

            *Parameter example:*
              To specify that the input files are 32-bit, big-endian
              PP files: ``um={'fmt': 'PP'}``

            *Parameter example:*
              To specify that the input files are 32-bit,
              little-endian PP files from version 5.1 of the UM:
              ``um={'fmt': 'PP', 'endian': 'little', 'version': 5.1}``

            .. versionadded:: 1.5

        umversion: deprecated at version 3.0.0
            Use the *um* parameter instead.

        height_at_top_of_model: deprecated at version 3.0.0
            Use the *um* parameter instead.

        field: deprecated at version 3.0.0
            Use the *extra* parameter instead.

        follow_symlinks: deprecated at version 3.0.0
            Use the *followlinks* parameter instead.

        select_options: deprecated at version 3.0.0
            Use methods on the returned `FieldList` instead.

    :Returns:

        `FieldList`
            The field constructs found in the input file(s). The list
            may be empty.

    **Examples:**

    >>> x = cf.read('file.nc')

    Read a file and create field constructs from CF-netCDF data
    variables as well as from the netCDF variables that correspond to
    particular types metadata constructs:

    >>> f = cf.read('file.nc', extra='domain_ancillary')
    >>> g = cf.read('file.nc', extra=['dimension_coordinate',
    ...                               'auxiliary_coordinate'])

    Read a file that contains external variables:

    >>> h = cf.read('parent.nc')
    >>> i = cf.read('parent.nc', external='external.nc')
    >>> j = cf.read('parent.nc', external=['external1.nc', 'external2.nc'])

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

    >>> cf.read('file*.nc', select='units=K')
    [<CF Field: temperature(17, 30, 24)>,
     <CF Field: temperature_wind(17, 29, 24)>]

    >>> cf.read('file*.nc', select='ncvar%ta')
    <CF Field: temperature(17, 30, 24)>

    '''
    if field:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'cf.read', {'field': field}, "Use keyword 'extra' instead"
        )  # pragma: no cover

    if select_options:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'cf.read', {'select_options': select_options}
        )  # pragma: no cover

    if follow_symlinks:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'cf.read', {'follow_symlinks': follow_symlinks},
            "Use keyword 'followlink' instead."
        )  # pragma: no cover

    if height_at_top_of_model is not None:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'cf.read', {'height_at_top_of_model': height_at_top_of_model},
            "Use keyword 'um' instead."
        )  # pragma: no cover

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
        aggregate = True
    else:
        aggregate_options = {}

    aggregate_options['copy'] = False

    # Parse the extra parameter
    if extra is None:
        extra = ()
    elif isinstance(extra, str):
        extra = (extra,)

    ftypes = set()

    # Count the number of fields (in all files) and the number of
    # files
    field_counter = -1
    file_counter = 0

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
                    for path, subdirs, filenames in os.walk(
                            x, followlinks=followlinks):
                        files3.extend(os.path.join(path, f) for f in filenames)
                        if not recursive:
                            break
                else:
                    files3.append(x)
            # --- End: for

            files2 = files3

        for filename in files2:
            logger.info('File: {0}'.format(filename))  # pragma: no cover

            if um:
                ftype = 'UM'
            else:
                try:
                    ftype = file_type(filename)
                except Exception as error:
                    if not ignore_read_error:
                        message = error

#                        if not find_library("umfile"):
#                            message += ("\n\n"
#                                "Note: Unable to detect the UM read C library needed "
#                                "to recognise and read PP and UM fields files. "
#                                "This indicates a compilation problem during the "
#                                "cf installation (though note it does not affect "
#                                "any other cf functionality, notably netCDF file "
#                                "processing). If processing of PP and FF files is "
#                                "required, ensure 'GNU make' is available and "
#                                "reinstall cf-python to try to build the library. "
#                                "Note a warning will be given if the build fails."
#                            )

                        raise ValueError(message)

                    logger.warning(
                        'WARNING: {}'.format(error))  # pragma: no cover

                    continue
            # --- End: if

            ftypes.add(ftype)

            # --------------------------------------------------------
            # Read the file into fields
            # --------------------------------------------------------
            fields = _read_a_file(
                filename, ftype=ftype,
                external=external,
                ignore_read_error=ignore_read_error,
                verbose=verbose, warnings=warnings,
                aggregate=aggregate,
                aggregate_options=aggregate_options,
                selected_fmt=fmt, um=um,
                extra=extra,
                height_at_top_of_model=height_at_top_of_model,
                chunk=chunk,
                mask=mask,
                warn_valid=warn_valid,
            )

            # --------------------------------------------------------
            # Select matching fields (not from UM files)
            # --------------------------------------------------------
            if select and ftype != 'UM':
                fields = fields.select_by_identity(*select)

            # --------------------------------------------------------
            # Add this file's fields to those already read from other
            # files
            # --------------------------------------------------------
            field_list.extend(fields)

            field_counter = len(field_list)
            file_counter += 1
        # --- End: for
    # --- End: for

    logger.info(
        "Read {0} field{1} from {2} file{3}".format(
            field_counter, _plural(field_counter), file_counter,
            _plural(file_counter)
        )
    )  # pragma: no cover

    # ----------------------------------------------------------------
    # Aggregate the output fields
    # ----------------------------------------------------------------
    if aggregate and len(field_list) > 1:
        org_len = len(field_list)  # pragma: no cover

        field_list = cf_aggregate(field_list, **aggregate_options)

        n = len(field_list)  # pragma: no cover
        logger.info('{0} input field{1} aggregated into {2} field{3}'.format(
            org_len, _plural(org_len), n, _plural(n))
        )  # pragma: no cover
    # --- End: if

    # ----------------------------------------------------------------
    # Sort by netCDF variable name
    # ----------------------------------------------------------------
    if len(field_list) > 1:
        field_list.sort(key=lambda f: f.nc_get_variable(''))

    # ----------------------------------------------------------------
    # Add standard names to UM/PP fields (post aggregation)
    # ----------------------------------------------------------------
    for f in field_list:
        standard_name = f._custom.get('standard_name', None)
        if standard_name is not None:
            f.set_property('standard_name', standard_name)
            del f._custom['standard_name']
    # --- End: for

    # ----------------------------------------------------------------
    # Select matching fields from UM/PP fields (post setting of
    # standard names)
    # ----------------------------------------------------------------
    if select and 'UM' in ftypes:
        field_list = field_list.select_by_identity(*select)

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
    # --- End: if

    if nfields is not None and len(field_list) != nfields:
        raise ValueError(
            "{} field{} requested but {} fields found in file{}".format(
                nfields, _plural(nfields), len(field_list),
                _plural(file_counter)
            )
        )

    return field_list


def _plural(n):  # pragma: no cover
    '''Return a suffix which reflects a word's plural.

    '''
    return 's' if n != 1 else ''  # pragma: no cover


@_manage_log_level_via_verbosity
def _read_a_file(filename, ftype=None, aggregate=True,
                 aggregate_options=None, ignore_read_error=False,
                 verbose=None, warnings=False, external=None,
                 selected_fmt=None, um=None, extra=None,
                 height_at_top_of_model=None, chunk=True, mask=True,
                 warn_valid=False):
    '''Read the contents of a single file into a field list.

    :Parameters:

        filename: `str`
            The file name.

        ftype: `str`
            TODO

        aggregate_options: `dict`, optional
            The keys and values of this dictionary may be passed as
            keyword parameters to an external call of the aggregate
            function.

        ignore_read_error: `bool`, optional
            If True then return an empty field list if reading the
            file produces an IOError, as would be the case for an
            empty file, unknown file format, etc. By default the
            IOError is raised.

        mask: `bool`, optional
            If False then do not mask by convention when reading data
            from disk. By default data is masked by convention.

            .. versionadded:: 3.4.0

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
            a maximum of ``3``) the more description that is printed.

    :Returns:

        `FieldList`
            The fields in the file.

    '''
    if aggregate_options is None:
        aggregate_options = {}

    # Find this file's type
    fmt = None
    word_size = None
    endian = None
    height_at_top_of_model = None
    umversion = 405

    if um:
        # ftype = 'UM'
        fmt = um.get('fmt')
        word_size = um.get('word_size')
        endian = um.get('endian')
        umversion = um.get('version')
        height_at_top_of_model = um.get('height_at_top_of_model')
        if fmt in ('PP', 'pp', 'pP', 'Pp'):
            fmt = fmt.upper()
            # For PP format, there is a default word size and
            # endian-ness
            if word_size is None:
                word_size = 4

            if endian is None:
                endian = 'big'
        # --- End: if

        if umversion is not None:
            umversion = float(str(umversion).replace('.', '0', 1))
#    else:
#        try:
#            ftype = file_type(filename)
#        except Exception as error:
#            if not ignore_read_error:
#                raise Exception(error)
#
#            logger.warning('WARNING: {}'.format(error))  # pragma: no cover
#
#            return FieldList()
    # --- End: if

    extra_read_vars = {
        'chunk': chunk,
        'fmt': selected_fmt,
        'ignore_read_error': ignore_read_error,
        # 'cfa' defaults to False. If the file has
        # "CFA" in its Conventions global attribute
        # then 'cfa' will be changed to True in
        # netcdf.read
        'cfa': False,
    }

    # ----------------------------------------------------------------
    # Still here? Read the file into fields.
    # ----------------------------------------------------------------
    if ftype == 'CDL':
        # Create a temporary netCDF file from input CDL
        ftype = 'netCDF'
        cdl_filename = filename
        filename = netcdf.cdl_to_netcdf(filename)
        extra_read_vars['fmt'] = 'NETCDF'

        if not netcdf.is_netcdf_file(filename):
            if ignore_read_error:
                logger.warning(
                    "WARNING: Can't determine format of file {} generated "
                    "from CDL file {}".format(filename, cdl_filename)
                )  # pragma: no cover

                return FieldList()
            else:
                raise IOError(
                    "Can't determine format of file {} generated from CDL "
                    "file {}".format(filename, cdl_filename)
                )
    # --- End: if

    if ftype == 'netCDF' and extra_read_vars['fmt'] in (None, 'NETCDF', 'CFA'):
        fields = netcdf.read(filename, external=external, extra=extra,
                             verbose=verbose, warnings=warnings,
                             extra_read_vars=extra_read_vars,
                             mask=mask, warn_valid=warn_valid)

    elif ftype == 'UM' and extra_read_vars['fmt'] in (None, 'UM'):
        fields = UM.read(filename, um_version=umversion,
                         verbose=verbose, set_standard_name=False,
                         height_at_top_of_model=height_at_top_of_model,
                         fmt=fmt, word_size=word_size, endian=endian,
                         chunk=chunk)  # , mask=mask, warn_valid=warn_valid)

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
    '''Return the file format.

    :Parameters:

        filename: `str`
            The file name.

    :Returns:

        `str`
            The format type of the file. One of ``'netCDF'``, ``'UM'``
            or ``'CDL'``.

    **Examples:**

    >>> file_type(filename)
    'netCDF'

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

    # ----------------------------------------------------------------
    # CDL
    # ----------------------------------------------------------------
    if netcdf.is_cdl_file(filename):
        return 'CDL'

    # Still here?
    raise IOError("Can't determine format of file {}".format(filename))
