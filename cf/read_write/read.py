import logging
from re import Pattern

import cfdm

from ..aggregate import aggregate as cf_aggregate
from ..cfimplementation import implementation
from ..decorators import _manage_log_level_via_verbosity
from ..domainlist import DomainList
from ..fieldlist import FieldList
from ..functions import _DEPRECATION_ERROR_FUNCTION_KWARGS
from ..query import Query

logger = logging.getLogger(__name__)


class read(cfdm.read):
    """Read field or domain constructs from files.

    The following file formats are supported: netCDF, CDL, Zarr,
    Kerchunk, PP, and UM fields file.

    NetCDF and Zarr datasets may be on local disk, on an OPeNDAP
    server, or in an S3 object store.

    CDL, PP, and UM fields files must be on local disk.

    Any amount of files of any combination of file types may be read.

    Input datasets are mapped to `Field` constructs which are returned
    as elements of a `FieldList`, or if the *domain* parameter is
    True, `Domain` constructs returned as elements of a
    `DomainList`. The returned constructs are sorted by the netCDF
    variable names of their corresponding data or domain variables.

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
    ``-c`` option to ``ncdump``), in which case the relevant
    constructs in memory will be created with data with all missing
    values.

    **PP and UM fields files**

    32-bit and 64-bit Met Office (UK) PP files and Met Office (UK)
    fields files of any endian-ness can be read.

    2-d "slices" within a single file are combined, where possible,
    into field constructs with 3-d, 4-d or 5-d data. This is done
    prior to any field construct aggregation (see the *aggregate*
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
    metadata constructs (coordinates, cell measures, etc.) must be
    compared non-lazily to ascertain if aggregation is possible.

    .. seealso:: `cf.aggregate`, `cf.write`, `cf.Field`, `cf.Domain`,
                 `cf.load_stash2standard_name`, `cf.unique_constructs`

    :Parameters:

        {{read datasets:}}

        {{read recursive: `bool`, optional}}

        {{read followlinks: `bool`, optional}}

        {{read cdl_string: `bool`, optional}}

        {{read dataset_type: `None` or (sequence of) `str`, optional}}

            Valid file types are:

            ==============  ==========================================
            *dataset_type*  Description
            ==============  ==========================================
            ``'netCDF'``    A netCDF-3 or netCDF-4 dataset
            ``'CDL'``       A text CDL file of a netCDF dataset
            ``'Zarr'``      A Zarr v2 (xarray) or Zarr v3 dataset
            ``'Kerchunk'``  A Kerchunked dataset
            ``'UM'``        A UM fields file or PP dataset
            ==============  ==========================================

            .. versionadded:: 3.18.0

        {{read external: (sequence of) `str`, optional}}

        {{read extra: (sequence of) `str`, optional}}

        {{read verbose: `int` or `str` or `None`, optional}}

        {{read warnings: `bool`, optional}}

        {{read um: `dict` or `None`, optional}}

            .. versionadded:: 1.5

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

        {{read squeeze: `bool`, optional}}

        {{read unsqueeze: `bool`, optional}}

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

        {{read warn_valid: `bool`, optional}}

            .. versionadded:: 3.4.0

        {{read mask: `bool`, optional}}

            .. versionadded:: 3.4.0

        {{read unpack: `bool`}}

            .. versionadded:: 3.17.0

        {{read domain: `bool`, optional}}

            .. versionadded:: 3.11.0

        {{read backend: `None` or (sequence of) `str`, optional}}

            .. versionadded:: 3.21.0

        {{read backend_options: `None` or `dict`, optional}}

            .. versionadded:: 3.21.0

        {{read storage_options: `dict` or `None`, optional}}

            .. versionadded:: 3.17.0

        {{read cache: `bool`, optional}}

            .. versionadded:: 3.17.0

        {{read dask_chunks: `str`, `int`, `None`, or `dict`, optional}}

              .. versionadded:: 3.17.0

        {{read store_dataset_chunks: `bool`, optional}}

            .. versionadded:: 3.17.0

        {{read store_dataset_shards: `bool`, optional}}

            .. versionadded:: 3.19.0

        {{read cfa: `dict`, optional}}

            .. versionadded:: 3.15.0

        {{read cfa_write: (sequence of) `str`, optional}}

            .. versionadded:: 3.17.0

        {{read cfa_filesystem: `None` or filesystem, optional}}

            .. versionadded:: 3.21.0

        {{read cfa_backend: `None` or (sequence of) `str`, optional}}

            .. versionadded:: 3.21.0

        {{cfa_backend_options: `None` or `dict`, optional}}

            .. versionadded:: 3.21.0

        {{read to_memory: (sequence of) `str`, optional}}

            .. versionadded:: 3.17.0

        {{read group_dimension_search: `str`, optional}}

            .. versionadded:: 3.19.0

        {{read filesystem: optional}}

            .. versionadded:: 3.20.0

        {{read _noncompliance_report: `bool`, optional}}

            ..versionadded:: 3.21.0

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
            Use the *dask_chunks* parameter instead.

        chunks: deprecated at version 3.17.0
            Use the *dask_chunks* parameter instead.

        fmt: deprecated at version 3.17.0
            Use the dataset_type* parameter instead.

        ignore_read_error: deprecated at version 3.17.0
            Use the *dataset_type* parameter instead.

        file_type: deprecated at version 3.18.0
            Use the *dataset_type* parameter instead.

        netcdf_backend: Deprecated at version 3.21.0
            Use *backend* instead.

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

    implementation = implementation()

    @_manage_log_level_via_verbosity
    def __new__(
        cls,
        datasets,
        external=None,
        verbose=None,
        warnings=False,
        aggregate=True,
        nfields=None,
        squeeze=False,
        unsqueeze=False,
        dataset_type=None,
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
        unpack=True,
        warn_valid=False,
        dask_chunks="storage-aligned",
        store_dataset_chunks=True,
        store_dataset_shards=True,
        domain=False,
        cfa=None,
        cfa_write=None,
        to_memory=None,
        backend=None,
        backend_options=None,
        storage_options=None,
        cache=True,
        chunks="auto",
        ignore_read_error=False,
        fmt=None,
        file_type=None,
        group_dimension_search="closest_ancestor",
        filesystem=None,
        legacy_um_backend=False,
        netcdf_backend=None,
    ):
        """Read field or domain constructs from a dataset."""
        kwargs = locals()

        if field:
            _DEPRECATION_ERROR_FUNCTION_KWARGS(
                "cf.read",
                {"field": field},
                "Use keyword 'extra' instead",
                removed_at="4.0.0",
            )  # pragma: no cover

        if select_options:
            _DEPRECATION_ERROR_FUNCTION_KWARGS(
                "cf.read",
                {"select_options": select_options},
                removed_at="4.0.0",
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
                "Use keyword 'dask_chunks' instead.",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        if chunks != "auto":
            _DEPRECATION_ERROR_FUNCTION_KWARGS(
                "cf.read",
                {"chunk": chunk},
                "Use keyword 'dask_chunks' instead.",
                version="3.14.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        if fmt is not None:
            _DEPRECATION_ERROR_FUNCTION_KWARGS(
                "cf.read",
                {"fmt": fmt},
                "Use keyword 'dataset_type' instead.",
                version="3.17.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        if ignore_read_error:
            _DEPRECATION_ERROR_FUNCTION_KWARGS(
                "cf.read",
                {"ignore_read_error": ignore_read_error},
                "Use keyword 'dataset_type' instead.",
                version="3.17.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        if file_type is not None:
            _DEPRECATION_ERROR_FUNCTION_KWARGS(
                "cf.read",
                {"file_type": file_type},
                "Use keyword 'dataset_type' instead.",
                version="3.18.0",
                removed_at="5.0.0",
            )  # pragma: no cover

        return super().__new__(**kwargs)

    def _finalise(self):
        """Actions to take after all datasets have been read.

        Called by `__new__`.

        .. versionadded:: 3.18.0

        :Returns:

            `None`

        """
        # ----------------------------------------------------------------
        # Select matching constructs from netCDF datasets (before
        # aggregation)
        # ----------------------------------------------------------------
        select = self.select
        if select:
            self.constructs = self.constructs.select_by_identity(*select)

        # ----------------------------------------------------------------
        # Aggregate the output fields or domains
        # ----------------------------------------------------------------
        self.aggregate = self.aggregate and len(self.constructs) > 1
        if self.aggregate:
            UM = False  # True if there is at least one UM field
            non_UM = False  # True if there is at least one non-UM field
            for f in self.constructs:
                try:
                    if f.get_property("um_identity", "").startswith("UM_"):
                        UM = True
                    else:
                        non_UM = True
                except Exception:
                    non_UM = True

                if UM and non_UM:
                    break

            if UM and non_UM:
                self.aggregate = False
                logger.warning(
                    "Won't aggregate fields from a mixture of UM and "
                    "non-UM sources (a field from a UM source is defined as "
                    "having a string-valued um_identity property that starts "
                    "with 'UM_')."
                    "\n"
                    "Aggregation may still be possible with cf.aggregate."
                )
                # This is because the aggregation of UM fields
                # requires
                # `aggregate_options["field_identity"]="um_identity"`
                # in order to overcome the many-to-one relationship between
                # STASH codes and standard names.

        if self.aggregate:
            aggregate_options = self.aggregate_options

            if UM:
                # Set extra aggregate options for fields created from
                # UM data.
                #
                # We can't trust the the standard_name to provide the
                # identity for UM fields (multiple STASH codes can
                # have the same standard name), so instead we have to
                # the um_identity property (which encapsulates the
                # submodel, stash/field code and UM version).
                aggregate_options["field_identity"] = "um_identity"

                if "strict_units" not in aggregate_options:
                    aggregate_options["relaxed_units"] = True

            # Do the aggregation
            self.constructs = cf_aggregate(
                self.constructs, **aggregate_options
            )

        super()._finalise()

    def _initialise(self):
        """Actions to take before any datasets have been read.

        Called by `__new__`.

        .. versionadded:: 3.18.0

        :Returns:

            `None`

        """
        super()._initialise()

        # Initialise the list of output constructs
        if self.field:
            self.constructs = FieldList()
        elif self.domain:
            self.constructs = DomainList()

        # Recognised UM dataset formats
        self.UM_dataset_types = set(("UM",))

        # Allowed dataset formats
        self.allowed_dataset_types.update(self.UM_dataset_types)

        # ------------------------------------------------------------
        # Parse the 'um' keyword parameter
        # ------------------------------------------------------------
        kwargs = self.kwargs
        um = kwargs["um"]
        if not um:
            um = {}

        self.um = um

        # ------------------------------------------------------------
        # Parse the 'select' keyword parameter
        # ------------------------------------------------------------
        select = kwargs["select"]
        if isinstance(select, (str, Query, Pattern)):
            select = (select,)

        self.select = select

        # ------------------------------------------------------------
        # Parse the 'aggregate' keyword parameter
        # ------------------------------------------------------------
        aggregate = kwargs["aggregate"]
        if isinstance(aggregate, dict):
            aggregate_options = aggregate.copy()
            aggregate = True
        else:
            aggregate_options = {}

        aggregate_options["copy"] = False

        self.aggregate = aggregate
        self.aggregate_options = aggregate_options
