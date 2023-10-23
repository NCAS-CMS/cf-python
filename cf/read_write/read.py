import logging
import os
import tempfile
from glob import glob
from numbers import Integral
from os.path import isdir
from re import Pattern
from urllib.parse import urlparse

from cfdm import is_log_level_info
from numpy.ma.core import MaskError

from ..aggregate import aggregate as cf_aggregate
from ..cfimplementation import implementation
from ..decorators import _manage_log_level_via_verbosity
from ..domainlist import DomainList
from ..fieldlist import FieldList
from ..functions import _DEPRECATION_ERROR_FUNCTION_KWARGS, flat
from ..query import Query
from .netcdf import NetCDFRead
from .um import UMRead

_cached_temporary_files = {}

# --------------------------------------------------------------------
# Create an implementation container and initialise a read object for
# each format
# --------------------------------------------------------------------
_implementation = implementation()
netcdf = NetCDFRead(_implementation)
UM = UMRead(_implementation)


logger = logging.getLogger(__name__)


@_manage_log_level_via_verbosity
def read(
    files,
    external=None,
    verbose=None,
    warnings=False,
    ignore_read_error=False,
    aggregate=True,
    nfields=None,
    squeeze=False,
    unsqueeze=False,
    fmt=None,
    cdl_string=False,
    select=None,
    extra=None,
    recursive=False,
    followlinks=False,
    um=None,
    chunk=True,
    field=None,
    height_at_top_of_model=None,
    select_options=None,
    follow_symlinks=False,
    mask=True,
    warn_valid=False,
    chunks="auto",
    domain=False,
    cfa=None,
):
    """Read field or domain constructs from files.

    The following file formats are supported: CF-netCDF, CFA-netCDF,
    CDL, PP and UM fields datasets.

    Input datasets are mapped to constructs in memory which are
    returned as elements of a `FieldList` or if the *domain* parameter
    is True, a `DomainList`.

    NetCDF files may be on disk or on an OPeNDAP server.

    Any amount of files of any combination of file types may be read.

    **NetCDF unlimited dimensions**

    Domain axis constructs that correspond to NetCDF unlimited
    dimensions may be accessed with the
    `~cf.DomainAxis.nc_is_unlimited` and
    `~cf.DomainAxis.nc_set_unlimited` methods of a domain axis
    construct.

    **NetCDF hierarchical groups**

    Hierarchical groups in CF provide a mechanism to structure
    variables within netCDF4 datasets. Field constructs are
    constructed from grouped datasets by applying the well defined
    rules in the CF conventions for resolving references to
    out-of-group netCDF variables and dimensions. The group structure
    is preserved in the field construct's netCDF interface. Groups
    were incorporated into CF-1.8. For files with groups that state
    compliance to earlier versions of the CF conventions, the groups
    will be interpreted as per the latest release of CF.

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
    done prior to any field construct aggregation (see the *aggregate*
    parameter).

    When reading PP and UM fields files, the *relaxed_units* aggregate
    option is set to `True` by default, because units are not always
    available to field constructs derived from UM fields files or PP
    files.

    **Performance**

    Descriptive properties are always read into memory, but lazy
    loading is employed for all data arrays which means that, in
    general, data is not read into memory until the data is required
    for inspection or to modify the array contents. This maximises the
    number of field constructs that may be read within a session, and
    makes the read operation fast. The exceptions to the lazy reading
    of data arrays are:

    * Data that define purely structural elements of other data arrays
      that are compressed by convention (such as a count variable for
      a ragged contiguous array). These are always read from disk.

    * If field or domain aggregation is in use (as it is by default,
      see the *aggregate* parameter), then the data of metadata
      constructs may have to be read to determine how the contents of
      the input files may be aggregated. This won't happen for a
      particular field or domain's metadata, though, if it can be
      ascertained from descriptive properties alone that it can't be
      aggregated with anything else (as would be the case, for
      instance, when a field has a unique standard name).

    However, when two or more field or domain constructs are
    aggregated to form a single construct then the data arrays of some
    metadata constructs (coordinates, cell measures, etc.)  must be
    compared non-lazily to ascertain if aggregation is possible.

    .. seealso:: `cf.aggregate`, `cf.write`, `cf.Field`, `cf.Domain`,
                 `cf.load_stash2standard_name`, `cf.unique_constructs`

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

            As a special case, if the `cdl_string` parameter is set to
            True, the interpretation of `files` changes so that each
            value is assumed to be a string of CDL input rather
            than the above.

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

        verbose: `int` or `str` or `None`, optional
            If an integer from ``-1`` to ``3``, or an equivalent string
            equal ignoring case to one of:

            * ``'DISABLE'`` (``0``)
            * ``'WARNING'`` (``1``)
            * ``'INFO'`` (``2``)
            * ``'DETAIL'`` (``3``)
            * ``'DEBUG'`` (``-1``)

            set for the duration of the method call only as the minimum
            cut-off for the verboseness level of displayed output (log)
            messages, regardless of the globally-configured `cf.log_level`.
            Note that increasing numerical value corresponds to increasing
            verbosity, with the exception of ``-1`` as a special case of
            maximal and extreme verbosity.

            Otherwise, if `None` (the default value), output messages will
            be shown according to the value of the `cf.log_level` setting.

            Overall, the higher a non-negative integer or equivalent string
            that is set (up to a maximum of ``3``/``'DETAIL'``) for
            increasing verbosity, the more description that is printed to
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

        cdl_string: `bool`, optional
            If True and the format to read is CDL, read a string
            input, or sequence of string inputs, each being interpreted
            as a string of CDL rather than names of locations from
            which field constructs can be read from, as standard.

            By default, each string input or string element in the input
            sequence is taken to be a file or directory name or an
            OPenDAP URL from which to read field constructs, rather
            than a string of CDL input, including when the `fmt`
            parameter is set as CDL.

            Note that when `cdl_string` is True, the `fmt` parameter is
            ignored as the format is assumed to be CDL, so in that case
            it is not necessary to also specify ``fmt='CDL'``.

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

        select: (sequence of) `str` or `Query` or `re.Pattern`, optional
            Only return field constructs whose identities match the
            given values(s), i.e. those fields ``f`` for which
            ``f.match_by_identity(*select)`` is `True`. See
            `cf.Field.match_by_identity` for details.

            This is equivalent to, but faster than, not using the
            *select* parameter but applying its value to the returned
            field list with its `cf.FieldList.select_by_identity`
            method. For example, ``fl = cf.read(file,
            select='air_temperature')`` is equivalent to ``fl =
            cf.read(file).select_by_identity('air_temperature')``.

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
            on field constructs and metadata constructs that have
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
            option is ignored for input files which are not PP or
            fields files. In most cases, how to decode a file is
            inferrable from the file's contents, but if not then each
            key/value pair in the dictionary sets a decoding option as
            follows:

            ============================  =====================================
            Key                           Value
            ============================  =====================================
            ``'fmt'``                     The file format (``'PP'`` or
                                          ``'FF'``)

            ``'word_size'``               The word size in bytes
                                          (``4`` or ``8``).

            ``'endian'``                  The byte order (``'big'`` or
                                          ``'little'``).

            ``'version'``                 The UM version to be used
                                          when decoding the
                                          header. Valid versions are,
                                          for example, ``4.2``,
                                          ``'6.6.3'`` and
                                          ``'8.2'``. In general, a
                                          given version is ignored if
                                          it can be inferred from the
                                          header (which is usually the
                                          case for files created by
                                          the UM at versions 5.3 and
                                          later). The exception to
                                          this is when the given
                                          version has a third element
                                          (such as the 3 in 6.6.3), in
                                          which case any version in
                                          the header is ignored.

                                          The default version is
                                          ``4.5``.

            ``'height_at_top_of_model'``  The height (in metres) of
                                          the upper bound of the top
                                          model level. By default the
                                          height at top model is taken
                                          from the top level's upper
                                          bound defined by BRSVD1 in
                                          the lookup header. If the
                                          height can't be determined
                                          from the header, or the
                                          given height is less than or
                                          equal to 0, then a
                                          coordinate reference system
                                          will still be created that
                                          contains the 'a' and 'b'
                                          formula term values, but
                                          without an atmosphere hybrid
                                          height dimension coordinate
                                          construct.

                                          .. note:: A current
                                             limitation is that if
                                             pseudolevels and
                                             atmosphere hybrid height
                                             coordinates are defined
                                             by same the lookup
                                             headers then the height
                                             **can't be determined
                                             automatically**. In this
                                             case the height may be
                                             found after reading as
                                             the maximum value of the
                                             bounds of the domain
                                             ancillary construct
                                             containing the 'a'
                                             formula term. The file
                                             can then be re-read with
                                             this height as a *um*
                                             parameter.
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

        chunks: `str`, `int`, `None`, or `dict`, optional
                Specify the `dask` chunking of dimensions for data in
                the input files.

                By default, ``'auto'`` is used to specify the array
                chunking, which uses a chunk size in bytes defined by
                the `cf.chunksize` function, preferring square-like
                chunk shapes across all data dimensions.

                If *chunks* is a `str` then each data array uses this
                chunk size in bytes, preferring square-like chunk
                shapes across all data dimensions. Any string value
                accepted by the *chunks* parameter of the
                `dask.array.from_array` function is permitted.

                *Parameter example:*
                  A chunksize of 2 MiB may be specified as
                  ``'2097152'`` or ``'2 MiB'``.

                If *chunks* is `-1` or `None` then for each there is no
                chunking, i.e. every data array has one chunk
                regardless of its size.

                If *chunks* is a positive `int` then each data array
                dimension has chunks with this number of elements.

                If *chunks* is a `dict`, then each of its keys
                identifies dimension in the file, with a value that
                defines the chunking for that dimension whenever it is
                spanned by data.

                Each dictionary key identifies a file dimension in one
                of three ways: 1. the netCDF dimension name, preceded
                by ``ncdim%`` (e.g. ``'ncdim%lat'``); 2. the "standard
                name" attribute of a CF-netCDF coordinate variable
                that spans the dimension (e.g. ``'latitude'``); or
                3. the "axis" attribute of a CF-netCDF coordinate
                variable that spans the dimension (e.g. ``'Y'``).

                The dictionary values may be `str`, `int` or `None`,
                with the same meanings as those types for the *chunks*
                parameter but applying only to the specified
                dimension. A `tuple` or `list` of integers that sum to
                the dimension size may also be given.

                Not specifying a file dimension in the dictionary is
                equivalent to it being defined with a value of
                ``'auto'``.

                *Parameter example:*
                  ``{'T': '0.5 MiB', 'Y': [36, 37], 'X': None}``

                *Parameter example:*
                  If a netCDF file contains dimensions ``time``,
                  ``z``, ``lat`` and ``lon``, then ``{'ncdim%time':
                  12, 'ncdim%lat', None, 'ncdim%lon': None}`` will
                  ensure that all ``time`` axes have a chunksize of
                  12; and all ``lat`` and ``lon`` axes are not
                  chunked; and all ``z`` axes are chunked to comply as
                  closely as possible with the default chunks size.

                  If the netCDF also contains a ``time`` coordinate
                  variable with a ``standard_name`` attribute of
                  ``'time'`` and an ``axis`` attribute of ``'T'``,
                  then the same chunking could be specified with
                  either ``{'time': 12, 'ncdim%lat', None, 'ncdim%lon':
                  None}`` or ``{'T': 12, 'ncdim%lat', None,
                  'ncdim%lon': None}``.

                .. note:: The *chunks* parameter is ignored for PP and
                          UM fields files, for which the chunking is
                          pre-determined by the file format.

                .. versionadded:: 3.14.0

        domain: `bool`, optional
            If True then return only the domain constructs that are
            explicitly defined by CF-netCDF domain variables, ignoring
            all CF-netCDF data variables. By default only the field
            constructs defined by CF-netCDF data variables are
            returned.

            CF-netCDF domain variables are only defined from CF-1.9,
            so older datasets automatically contain no CF-netCDF
            domain variables.

            The unique domain constructs of the dataset are easily
            found with the `cf.unique_constructs` function. For
            example::

               >>> d = cf.read('file.nc', domain=True)
               >>> ud = cf.unique_constructs(d)
               >>> f = cf.read('file.nc')
               >>> ufd = cf.unique_constructs(x.domain for x in f)

            Domain constructs can not be read from UM or PP datasets.

            .. versionadded:: 3.11.0

        cfa: `dict`, optional
            Configure the reading of CFA-netCDF files. The dictionary
            may have any subset of the following key/value pairs to
            override the information read from the file:

            * ``'substitutions'``: `dict`

              A dictionary whose key/value pairs define text
              substitutions to be applied to the fragment file
              names. Each key may be specified with or without the
              ``${...}`` syntax. For instance, the following are
              equivalent: ``{'base': 'sub'}``, ``{'${base}': 'sub'}``.
              The substitutions are used in conjunction with, and take
              precedence over, any that are stored in the CFA-netCDF
              file by the ``substitutions`` attribute of the ``file``
              CFA aggregation instruction variable.

              *Example:*
                ``{'base': 'file:///data/'}``

            .. versionadded:: 3.15.0

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

        chunk: deprecated at version 3.14.0
            Use the *chunks* parameter instead.

    :Returns:

        `FieldList` or `DomainList`
            The field or domain constructs found in the input
            dataset(s). The list may be empty.

    **Examples**

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

    """
    if field:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "cf.read",
            {"field": field},
            "Use keyword 'extra' instead",
            removed_at="4.0.0",
        )  # pragma: no cover

    if select_options:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "cf.read", {"select_options": select_options}, removed_at="4.0.0"
        )  # pragma: no cover

    if follow_symlinks:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "cf.read",
            {"follow_symlinks": follow_symlinks},
            "Use keyword 'followlink' instead.",
            removed_at="4.0.0",
        )  # pragma: no cover

    if height_at_top_of_model is not None:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "cf.read",
            {"height_at_top_of_model": height_at_top_of_model},
            "Use keyword 'um' instead.",
            removed_at="4.0.0",
        )  # pragma: no cover

    if chunk is not True:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "cf.read",
            {"chunk": chunk},
            "Use keyword 'chunks' instead.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    # Parse select
    if isinstance(select, (str, Query, Pattern)):
        select = (select,)

    # Check chunks
    if chunks is not None and not isinstance(chunks, (str, Integral, dict)):
        raise ValueError(
            "'chunks' parameter must be of type str, int, None or dict. "
            f"Got: {chunks!r}"
        )

    # Manage input parameters where contradictions are possible:
    if cdl_string and fmt:
        if fmt == "CDL":
            if is_log_level_info(logger):
                logger.info(
                    "It is not necessary to set the cf.read fmt as 'CDL' when "
                    "cdl_string is True, since that implies CDL is the format."
                )  # pragma: no cover
        else:
            raise ValueError(
                "cdl_string can only be True when the format is CDL, though "
                "fmt is ignored in that case so there is no need to set it."
            )
    if squeeze and unsqueeze:
        raise ValueError("squeeze and unsqueeze can not both be True")
    if follow_symlinks and not recursive:
        raise ValueError(
            f"Can't set follow_symlinks={follow_symlinks!r} "
            f"when recursive={recursive!r}"
        )

    info = is_log_level_info(logger)

    # Parse the 'cfa' parameter
    if cfa is None:
        cfa_options = {}
    else:
        cfa_options = cfa.copy()
        keys = ("substitutions",)
        if not set(cfa_options).issubset(keys):
            raise ValueError(
                "Invalid dictionary key to the 'cfa' parameter."
                f"Valid keys are {keys}. Got: {cfa_options}"
            )

    if "substitutions" in cfa_options:
        substitutions = cfa_options["substitutions"].copy()
        for base, sub in tuple(substitutions.items()):
            if not (base.startswith("${") and base.endswith("}")):
                # Add missing ${...}
                substitutions[f"${{{base}}}"] = substitutions.pop(base)
    else:
        substitutions = {}

    cfa_options["substitutions"] = substitutions

    # Initialise the output list of fields/domains
    if domain:
        out = DomainList()
    else:
        out = FieldList()

    if isinstance(aggregate, dict):
        aggregate_options = aggregate.copy()
        aggregate = True
    else:
        aggregate_options = {}

    aggregate_options["copy"] = False

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

    if cdl_string:
        files2 = []

        # 'files' input may be a single string or a sequence of them and to
        # handle both cases it is easiest to convert former to a one-item seq.
        if isinstance(files, str):
            files = [files]

        for cdl_file in files:
            c = tempfile.NamedTemporaryFile(
                mode="w",
                dir=tempfile.gettempdir(),
                prefix="cf_",
                suffix=".cdl",
            )

            c_name = c.name
            with open(c_name, "w") as f:
                f.write(cdl_file)

            # ----------------------------------------------------------------
            # Need to cache the TemporaryFile object so that it doesn't get
            # deleted too soon
            # ----------------------------------------------------------------
            _cached_temporary_files[c_name] = c

            files2.append(c.name)

        files = files2

    for file_glob in flat(files):
        # Expand variables
        file_glob = os.path.expanduser(os.path.expandvars(file_glob))

        scheme = urlparse(file_glob).scheme
        if scheme in ("https", "http"):
            # Do not glob a URL
            files2 = (file_glob,)
        else:
            # Glob files on disk
            files2 = glob(file_glob)

            if not files2 and not ignore_read_error:
                open(file_glob, "rb")

            files3 = []
            for x in files2:
                if isdir(x):
                    # Walk through directories, possibly recursively
                    for path, subdirs, filenames in os.walk(
                        x, followlinks=followlinks
                    ):
                        files3.extend(os.path.join(path, f) for f in filenames)
                        if not recursive:
                            break
                else:
                    files3.append(x)

            files2 = files3

        for filename in files2:
            if info:
                logger.info(f"File: {filename}")  # pragma: no cover

            if um:
                ftype = "UM"
            else:
                try:
                    ftype = file_type(filename)
                except Exception as error:
                    if not ignore_read_error:
                        raise ValueError(error)

                    logger.warning(f"WARNING: {error}")  # pragma: no cover
                    continue

            if domain and ftype == "UM":
                raise ValueError(
                    f"Can't read PP/UM file {filename} into domain constructs"
                )

            ftypes.add(ftype)

            # --------------------------------------------------------
            # Read the file
            # --------------------------------------------------------
            file_contents = _read_a_file(
                filename,
                ftype=ftype,
                external=external,
                ignore_read_error=ignore_read_error,
                verbose=verbose,
                warnings=warnings,
                aggregate=aggregate,
                aggregate_options=aggregate_options,
                selected_fmt=fmt,
                um=um,
                extra=extra,
                height_at_top_of_model=height_at_top_of_model,
                chunks=chunks,
                mask=mask,
                warn_valid=warn_valid,
                select=select,
                domain=domain,
                cfa_options=cfa_options,
            )

            # --------------------------------------------------------
            # Select matching fields (not from UM files, yet)
            # --------------------------------------------------------
            if select and ftype != "UM":
                file_contents = file_contents.select_by_identity(*select)

            # --------------------------------------------------------
            # Add this file's contents to that already read from other
            # files
            # --------------------------------------------------------
            out.extend(file_contents)

            field_counter = len(out)
            file_counter += 1

    if info:
        logger.info(
            f"Read {field_counter} field{_plural(field_counter)} from "
            f"{file_counter} file{_plural(file_counter)}"
        )  # pragma: no cover

    # ----------------------------------------------------------------
    # Aggregate the output fields/domains
    # ----------------------------------------------------------------
    if aggregate and len(out) > 1:
        org_len = len(out)  # pragma: no cover

        out = cf_aggregate(out, **aggregate_options)

        n = len(out)  # pragma: no cover
        if info:
            logger.info(
                f"{org_len} input field{_plural(org_len)} aggregated into "
                f"{n} field{_plural(n)}"
            )  # pragma: no cover

    # ----------------------------------------------------------------
    # Sort by netCDF variable name
    # ----------------------------------------------------------------
    if len(out) > 1:
        out.sort(key=lambda f: f.nc_get_variable(""))

    # ----------------------------------------------------------------
    # Add standard names to UM/PP fields (post aggregation)
    # ----------------------------------------------------------------
    for f in out:
        standard_name = f._custom.get("standard_name", None)
        if standard_name is not None:
            f.set_property("standard_name", standard_name, copy=False)
            del f._custom["standard_name"]

    # ----------------------------------------------------------------
    # Select matching fields from UM/PP fields (post setting of
    # standard names)
    # ----------------------------------------------------------------
    if select and "UM" in ftypes:
        out = out.select_by_identity(*select)

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
    if not domain:
        if squeeze:
            for f in out:
                f.squeeze(inplace=True)
        elif unsqueeze:
            for f in out:
                f.unsqueeze(inplace=True)

    if nfields is not None and len(out) != nfields:
        raise ValueError(
            f"{nfields} field{_plural(nfields)} requested but "
            f"{len(out)} field/domain constucts found in "
            f"file{_plural(file_counter)}"
        )

    return out


def _plural(n):  # pragma: no cover
    """Return a suffix which reflects a word's plural."""
    return "s" if n != 1 else ""  # pragma: no cover


@_manage_log_level_via_verbosity
def _read_a_file(
    filename,
    ftype=None,
    aggregate=True,
    aggregate_options=None,
    ignore_read_error=False,
    verbose=None,
    warnings=False,
    external=None,
    selected_fmt=None,
    um=None,
    extra=None,
    height_at_top_of_model=None,
    mask=True,
    warn_valid=False,
    chunks="auto",
    select=None,
    domain=False,
    cfa_options=None,
):
    """Read the contents of a single file into a field list.

    :Parameters:

        filename: `str`
            See `cf.read` for details.

        ftype: `str`
            The file format to interpret the file. Recognised formats are
            ``'netCDF'``, ``'CDL'``, ``'UM'`` and ``'PP'``.

        aggregate_options: `dict`, optional
            See `cf.read` for details.

        ignore_read_error: `bool`, optional
            See `cf.read` for details.

        mask: `bool`, optional
            See `cf.read` for details.

        verbose: `int` or `str` or `None`, optional
            See `cf.read` for details.

        select: optional
            For `read. Ignored for a netCDF file.

        domain: `bool`, optional
            See `cf.read` for details.

        cfa_options: `dict`, optional
            See `cf.read` for details.

            .. versionadded:: 3.15.0

    :Returns:

        `FieldList` or `DomainList`
            The field or domain constructs in the dataset.

    """
    if aggregate_options is None:
        aggregate_options = {}

    # Find this file's type
    fmt = None
    word_size = None
    endian = None
    height_at_top_of_model = None
    umversion = 405

    if um:
        fmt = um.get("fmt")
        word_size = um.get("word_size")
        endian = um.get("endian")
        umversion = um.get("version", umversion)
        height_at_top_of_model = um.get("height_at_top_of_model")

        if fmt is not None:
            fmt = fmt.upper()

        if umversion is not None:
            umversion = float(str(umversion).replace(".", "0", 1))

    extra_read_vars = {
        "chunks": chunks,
        "fmt": selected_fmt,
        "ignore_read_error": ignore_read_error,
        "cfa_options": cfa_options,
    }

    # ----------------------------------------------------------------
    # Still here? Read the file into fields or domains.
    # ----------------------------------------------------------------
    originally_cdl = ftype == "CDL"
    if originally_cdl:
        # Create a temporary netCDF file from input CDL
        ftype = "netCDF"
        cdl_filename = filename
        filename = netcdf.cdl_to_netcdf(filename)
        extra_read_vars["fmt"] = "NETCDF"

        if not netcdf.is_netcdf_file(filename):
            error_msg = (
                f"Can't determine format of file {filename} generated "
                f"from CDL file {cdl_filename}"
            )
            if ignore_read_error:
                logger.warning(error_msg)  # pragma: no cover
                return FieldList()
            else:
                raise IOError(error_msg)

    if ftype == "netCDF" and extra_read_vars["fmt"] in (None, "NETCDF", "CFA"):
        # See https://github.com/NCAS-CMS/cfdm/issues/128 for context on the
        # try/except here, which acts as a temporary fix pending decisions on
        # the best way to handle CDL with only header or coordinate info.
        try:
            out = netcdf.read(
                filename,
                external=external,
                extra=extra,
                verbose=verbose,
                warnings=warnings,
                extra_read_vars=extra_read_vars,
                mask=mask,
                warn_valid=warn_valid,
                domain=domain,
            )
        except MaskError:
            # Some data required for field interpretation is missing,
            # manifesting downstream as a NumPy MaskError.
            if originally_cdl:
                raise ValueError(
                    "Unable to convert CDL without data to field construct(s) "
                    "because there is insufficient information provided by "
                    "the header and/or coordinates alone in this case."
                )
            else:
                raise ValueError(
                    "Unable to convert netCDF to field or domain construct "
                    "because there is missing data."
                )

    elif ftype == "UM" and extra_read_vars["fmt"] in (None, "UM"):
        if domain:
            raise ValueError(
                "Can't set domain=True when reading UM or PP datasets"
            )

        out = UM.read(
            filename,
            um_version=umversion,
            verbose=verbose,
            set_standard_name=False,
            height_at_top_of_model=height_at_top_of_model,
            fmt=fmt,
            word_size=word_size,
            endian=endian,
            select=select,
        )

        # PP fields are aggregated intrafile prior to interfile
        # aggregation
        if aggregate:
            # For PP fields, the default is strict_units=False
            if "strict_units" not in aggregate_options:
                aggregate_options["relaxed_units"] = True

    # ----------------------------------------------------------------
    # Return the fields
    # ----------------------------------------------------------------
    if domain:
        return DomainList(out)

    return FieldList(out)


def file_type(filename):
    """Return the file format.

    :Parameters:

        filename: `str`
            The file name.

    :Returns:

        `str`
            The format type of the file. One of ``'netCDF'``, ``'UM'``
            or ``'CDL'``.

    **Examples**

    >>> file_type(filename)
    'netCDF'

    """
    # ----------------------------------------------------------------
    # NetCDF
    # ----------------------------------------------------------------
    if netcdf.is_netcdf_file(filename):
        return "netCDF"

    # ----------------------------------------------------------------
    # PP or FF
    # ----------------------------------------------------------------
    if UM.is_um_file(filename):
        return "UM"

    # ----------------------------------------------------------------
    # CDL
    # ----------------------------------------------------------------
    if netcdf.is_cdl_file(filename):
        return "CDL"

    # Still here?
    raise IOError(f"Can't determine format of file {filename}")
