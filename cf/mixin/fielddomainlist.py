from ..functions import _DEPRECATION_ERROR_ARG


class FieldDomainList:
    """A sequence of domain constructs corresponding to fields."""

    def select_by_construct(self, *identities, OR=False, **conditions):
        """Select elements by their metadata constructs.

        To find the inverse of the selection, use a list comprehension
        with the `!match_by_construct` method of the construct
        elements. For example, to select all constructs that do *not*
        have a "latitude" metadata construct:

           >>> gl = cf.{{class}}(
           ...     f for f in fl if not f.match_by_construct('latitude')
           ... )

        .. note:: The API changed at version 3.1.0

        .. versionadded:: 3.0.0

        .. seealso: `select`, `__call__`, `select_by_units`,
                    `select_by_naxes`, `select_by_rank`,
                    `select_by_property`

        :Parameters:

            identities: optional
                Identify metadata constructs that have an identity,
                defined by their `!identities` methods, that matches
                any of the given values.

                If no identities nor conditions (see the *conditions*
                parameter) are provided then all constructs are
                selected.

                {{value match}}

                {{displayed identity}}

                If a cell method construct identity is given (such as
                ``'method:mean'``) then it will only be compared with
                the most recently applied cell method operation.

                Alternatively, one or more cell method constructs may
                be identified in a single string with a CF-netCDF cell
                methods-like syntax for describing both the collapse
                dimensions, the collapse method, and any cell method
                construct qualifiers. If N cell methods are described
                in this way then they will collectively identify the N
                most recently applied cell method operations. For
                example, ``'T: maximum within years T: mean over
                years'`` will be compared with the most two most
                recently applied cell method operations.

                *Parameter example:*
                  ``'latitude'``

                *Parameter example:*
                  ``'T'``

                *Parameter example:*
                  ``'long_name=Cell Area'``

                *Parameter example:*
                  ``'cellmeasure1'``

                *Parameter example:*
                  ``'measure:area'``

                *Parameter example:*
                  ``cf.eq('time')'``

                *Parameter example:*
                  ``re.compile('^lat')``

                *Parameter example:*
                  ``'domainancillary2', 'longitude'``

                *Parameter example:*
                  ``'area: mean T: maximum'``

                *Parameter example:*
                  ``'grid_latitude', 'area: mean T: maximum'``

            conditions: optional
                Identify the metadata constructs that have any of the
                given identities or construct keys, and whose data satisfy
                conditions.

                A construct identity or construct key (defined in the
                same way as by the *identities* parameter) is given as
                a keyword name and a condition on its data is given as
                the keyword value.

                The condition is satisfied if any of its data values
                equals the value provided.

                If no conditions nor identities (see the *identities*
                parameter) are provided then all list elements are
                selected.

                *Parameter example:*
                  ``longitude=180.0``

                *Parameter example:*
                  ``time=cf.dt('1959-12-16')``

                *Parameter example:*
                  ``latitude=cf.ge(0)``

                *Parameter example:*
                  ``latitude=cf.ge(0), air_pressure=500``

                *Parameter example:*
                  ``**{'latitude': cf.ge(0), 'long_name=soil_level': 4}``

            OR: `bool`, optional
                If True then return `True` if at least one metadata
                construct matches at least one of the criteria given
                by the *identities* or *conditions* arguments. By
                default `True` is only returned if the field
                constructs matches each of the given criteria.

            mode: deprecated at version 3.1.0
                Use the *OR* parameter instead.

            constructs: deprecated at version 3.1.0

        :Returns:

            `bool`
                The matching field constructs.

        **Examples**

            TODO

        """
        if identities:
            if identities[0] == "or":
                _DEPRECATION_ERROR_ARG(
                    self,
                    "select_by_construct",
                    "or",
                    message="Use 'OR=True' instead.",
                    version="3.1.0",
                )  # pragma: no cover

            if identities[0] == "and":
                _DEPRECATION_ERROR_ARG(
                    self,
                    "select_by_construct",
                    "and",
                    message="Use 'OR=False' instead.",
                    version="3.1.0",
                )  # pragma: no cover

        return type(self)(
            f
            for f in self
            if f.match_by_construct(*identities, OR=OR, **conditions)
        )

    def select_by_ncvar(self, *ncvars):
        """Select list elements by netCDF variable name.

        To find the inverse of the selection, use a list comprehension
        with the `!match_by_ncvar` method of the construct
        elements. For example, to select all constructs which do *not*
        have a netCDF name of 'tas':

           >>> gl = cf.{{class}}(
           ...     f for f in fl if not f.match_by_ncvar('tas')
           ... )

        .. versionadded:: 3.0.0

        .. seealso:: `select`, `select_by_identity`, `select_by_property`,

        :Parameters:

            ncvars: optional
                Select constructs from the list. May be one or more
                netCDF names of constructs.

                A construct is selected if it matches any of the given
                names.

                A netCDF variable name is specified by a string (e.g.
                ``'tas'``, etc.); a `Query` object

                (e.g. ``cf.eq('tas')``); or a compiled regular
                expression (e.g. ``re.compile('^air_')``) that selects
                the constructs whose netCDF variable names match via
                `re.search`.

                If no netCDF variable names are provided then all are
                selected.

        :Returns:

            `{{class}}`
                The matching constructs.

        **Examples**

        >>> fl = cf.{{class}}([cf.example_field(0), cf.example_field(1)])
        >>> fl
        [<CF Field: specific_humidity(latitude(5), longitude(8)) 1>,
         <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>]
        >>> f[0].nc_get_variable()
        'humidity'
        >>> f[1].nc_get_variable()
        'temp'

        >>> fl.select_by_ncvar('humidity')
        [<CF Field: specific_humidity(cf_role=timeseries_id(4), ncdim%timeseries(9))>]
        >>> fl.select_by_ncvar('humidity', 'temp')
        [<CF Field: specific_humidity(cf_role=timeseries_id(4), ncdim%timeseries(9))>,
         <CF Field: air_temperature(cf_role=timeseries_id(4), ncdim%timeseries(9)) Celsius>]
        >>> fl.select_by_ncvar()
        [<CF Field: specific_humidity(cf_role=timeseries_id(4), ncdim%timeseries(9))>,
         <CF Field: air_temperature(cf_role=timeseries_id(4), ncdim%timeseries(9)) Celsius>]

        >>> import re
        >>> fl.select_by_ncvar(re.compile('^hum'))
        [<CF Field: specific_humidity(cf_role=timeseries_id(4), ncdim%timeseries(9))>]

        """
        return type(self)(f for f in self if f.match_by_ncvar(*ncvars))

    def select_by_property(self, *mode, **properties):
        """Select list elements by property.

        To find the inverse of the selection, use a list comprehension
        with the `!match_by_property` method of the construct
        elements. For example, to select all constructs which do *not*
        have a long_name property of "Pressure":

           >>> gl = cf.{{class}}(
           ...     f for f in fl if not f.match_by_property(long_name='Pressure')
           ... )

        .. versionadded:: 3.0.0

        .. seealso:: `select`, `select_by_identity`, `select_by_ncvar`

        :Parameters:

            mode: optional
                Define the behaviour when multiple properties are
                provided.

                By default (or if the *mode* parameter is ``'and'``) a
                construct is selected if it matches all of the given
                properties, but if the *mode* parameter is ``'or'``
                then a construct will be selected when at least one of
                its properties matches.

            properties: optional
                Select the constructs with the given properties.

                A property value is given by a keyword parameter of
                the property name. The value may be a scalar or vector
                (e.g. ``'air_temperature'``, ``4``, ``['foo',
                'bar']``); or a compiled regular expression
                (e.g. ``re.compile('^ocean')``), for which all
                constructs whose methods match (via `re.search`) are
                selected.

                By default a construct is selected if it matches all
                of the given properties, but it may alternatively be
                selected when at least one of its properties matches
                (see the *mode* positional parameter).

        :Returns:

            `{{class}}`
                The matching constructs.

        **Examples**

        See `cf.{{class}}.select_by_identity`

        """
        return type(self)(
            f for f in self if f.match_by_property(*mode, **properties)
        )

    def select_by_rank(self, *ranks):
        """Select list elements by the number of domain axis constructs.

        .. versionadded:: 3.0.0

        .. seealso: `select`, `__call__`, `select_by_units`,
                    `select_by_naxes`, `select_by_construct`,
                    `select_by_property`, `cf.Field.match_by_identity`

        :Parameters:

            ranks: optional
                Define conditions on the number of domain axis constructs.

                A condition is one of:

                  * `int`
                  * a `Query` object

                The condition is satisfied if the number of domain
                axis constructs equals the condition value.

                *Parameter example:*
                  To see if the field construct has 4 domain axis
                  constructs: ``4``

                *Parameter example:*
                  To see if the field construct has at least 3 domain
                  axis constructs: ``cf.ge(3)``

        :Returns:

            `bool`
                The matching field constructs.

        **Examples**

            TODO

        """

        return type(self)(f for f in self if f.match_by_rank(*ranks))
