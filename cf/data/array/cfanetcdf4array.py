from .mixin import CFAMixin
from .netcdf4array import NetCDF4Array


class CFANetCDF4Array(CFAMixin, NetCDF4Array):
    """A CFA-netCDF array accessed with `netCDF4`.

    .. versionadded:: NEXTVERSION

    """

    def __init__(
        self,
        filename=None,
        address=None,
        dtype=None,
        mask=True,
        unpack=True,
        units=False,
        calendar=False,
        instructions=None,
        substitutions=None,
        term=None,
        attributes=None,
        storage_options=None,
        source=None,
        copy=True,
        x=None,
    ):
        """**Initialisation**

        :Parameters:

            filename: (sequence of) `str`, optional
                The name of the CFA-netCDF file containing the
                array. If a sequence then it must contain one element.

            address: (sequence of) `str`, optional
                The name of the CFA-netCDF aggregation variable for the
                array. If a sequence then it must contain one element.

            dtype: `numpy.dtype`
                The data type of the aggregated data array. May be
                `None` if the numpy data-type is not known (which can
                be the case for netCDF string types, for example).

            mask: `bool`
                If True (the default) then mask by convention when
                reading data from disk.

                A netCDF array is masked depending on the values of any of
                the netCDF variable attributes ``valid_min``,
                ``valid_max``, ``valid_range``, ``_FillValue`` and
                ``missing_value``.

            {{init unpack: `bool`, optional}}

                .. versionadded:: NEXTVERSION

            units: `str` or `None`, optional
                The units of the aggregated data. Set to `None` to
                indicate that there are no units.

            calendar: `str` or `None`, optional
                The calendar of the aggregated data. Set to `None` to
                indicate the CF default calendar, if applicable.

            instructions: `str`, optional
                The ``aggregated_data`` attribute value as found on
                the CFA netCDF variable. If set then this will be used
                to improve the performance of `__dask_tokenize__`.

            substitutions: `dict`, optional
                A dictionary whose key/value pairs define text
                substitutions to be applied to the fragment file
                names. Each key must be specified with the ``${...}``
                syntax, for instance ``{'${base}': 'sub'}``.

                .. versionadded:: 3.15.0

            term: `str`, optional
                The name of a non-standard aggregation instruction
                term from which the array is to be created, instead of
                creating the aggregated data in the standard
                terms. If set then *address* must be the name of the
                term's CFA-netCDF aggregation instruction variable,
                which must be defined on the fragment dimensions and
                no others. Each value of the aggregation instruction
                variable will be broadcast across the shape of the
                corresponding fragment.

                *Parameter example:*
                  ``address='cfa_tracking_id', term='tracking_id'``

                .. versionadded:: 3.15.0

            storage_options: `dict` or `None`, optional
                Key/value pairs to be passed on to the creation of
                `s3fs.S3FileSystem` file systems to control the
                opening of fragment files in S3 object stores. Ignored
                for files not in an S3 object store, i.e. those whose
                names do not start with ``s3:``.

                By default, or if `None`, then *storage_options* is
                taken as ``{}``.

                If the ``'endpoint_url'`` key is not in
                *storage_options* or is not in a dictionary defined by
                the ``'client_kwargs`` key (which is always the case
                when *storage_options* is `None`), then one will be
                automatically inserted for accessing a fragment S3
                file. For example, for a file name of
                ``'s3://store/data/file.nc'``, an ``'endpoint_url'``
                key with value ``'https://store'`` would be created.

                *Parameter example:*
                  ``{'key: 'scaleway-api-key...', 'secret':
                  'scaleway-secretkey...', 'endpoint_url':
                  'https://s3.fr-par.scw.cloud', 'client_kwargs':
                  {'region_name': 'fr-par'}}``

                .. versionadded:: NEXTVERSION

            {{init source: optional}}

            {{init copy: `bool`, optional}}

        """
        if source is not None:
            super().__init__(source=source, copy=copy)

            try:
                fragment_shape = source.get_fragment_shape()
            except AttributeError:
                fragment_shape = None

            try:
                instructions = source._get_component("instructions")
            except AttributeError:
                instructions = None

            try:
                aggregated_data = source.get_aggregated_data(copy=False)
            except AttributeError:
                aggregated_data = {}

            try:
                substitutions = source.get_substitutions()
            except AttributeError:
                substitutions = None

            try:
                term = source.get_term()
            except AttributeError:
                term = None

        elif filename is not None:
            shape, fragment_shape, aggregated_data = self._parse_cfa(
                x, term, substitutions
            )
            super().__init__(
                filename=filename,
                address=address,
                shape=shape,
                dtype=dtype,
                mask=mask,
                units=units,
                calendar=calendar,
                copy=copy,
            )
        else:
            super().__init__(
                filename=filename,
                address=address,
                dtype=dtype,
                mask=mask,
                units=units,
                calendar=calendar,
                copy=copy,
            )

            fragment_shape = None
            aggregated_data = None
            instructions = None
            term = None

        self._set_component("fragment_shape", fragment_shape, copy=False)
        self._set_component("aggregated_data", aggregated_data, copy=False)
        self._set_component("instructions", instructions, copy=False)
        self._set_component("term", term, copy=False)

        if substitutions is not None:
            self._set_component(
                "substitutions", substitutions.copy(), copy=False
            )
