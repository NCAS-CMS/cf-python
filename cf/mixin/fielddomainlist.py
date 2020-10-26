from copy import copy

import logging

import cfdm

from .mixin_container import Container

from .functions import (_DEPRECATION_ERROR,
                        _DEPRECATION_ERROR_KWARGS,
                        _DEPRECATION_ERROR_METHOD,
                        _DEPRECATION_ERROR_DICT)

from .decorators import (_deprecated_kwarg_check,
                         _manage_log_level_via_verbosity)


logger = logging.getLogger(__name__)


class ConstructList(list,
                    cfdm.Container):
    '''An ordered sequence of constructs

    The elements of the list are construct of the same type.

    The list supports the python list-like operations (such as
    indexing and methods like `!append`).

    >>> fl = cf.{{class}}()
    >>> len(fl)
    0
    >>> fl = cf.FieldList(f)
    >>> len(fl)
    1
    >>> fl = cf.FieldList([f, g])
    >>> len(fl)
    2
    >>> fl = cf.FieldList(cf.FieldList([f] * 3))
    >>> len(fl)
    3
    >>> len(fl + fl)
    6

    Such methods provide functionality similar to that of a
    :ref:`built-in list <python:tut-morelists>`. The main difference
    is that when an element needs to be assesed for equality its
    `!equals` method is used, rather than the ``==`` operator.

    '''
    def __init__(self, constructs=None):
        '''**Initialization**

    :Parameters:

        constructs: (sequence of) constructs
             Create a new list with these constructs.

        '''
        super(cfdm.Container, self).__init__()

        if constructs is not None:
            if getattr(fields, 'construct_type', None) is not None:
                self.append(constructs)
            else:
                self.extend(constructs)


        if key is None:
            key = lambda f: f.identity()

        return super().sort(key=key, reverse=reverse)

    def select_by_ncvar(self, *ncvars):
        '''Select list elements by netCDF variable name.

    To find the inverse of the selection, use a list comprehension
    with the `!match_by_ncvar` method of the constuct elements. For
    example, to select all constructs which do *not* have a netCDF
    name of 'tas':

       >>> gl = cf.{{class}}(
       ...     f for f in fl if not f.match_by_ncvar('tas')
       ... )

    .. versionadded:: 3.0.0

    .. seealso:: `select`, `select_by_identity`, `select_by_property`,

    :Parameters:

        ncvars: optional
            Select constructs from the list. May be one or more:

              * The netCDF name of a construct.

            A construct is selected if it matches any of the given
            names.

            A netCDF variable name is specified by a string (e.g.
            ``'tas'``, etc.); a `Query` object
            (e.g. ``cf.eq('tas')``); or a compiled regular expression
            (e.g. ``re.compile('^air_')``) that selects the constructs
            whose netCDF variable names match via `re.search`.

            If no netCDF variable names are provided then all are
            selected.

    :Returns:

        `{{class}}`
            The matching constructs.

    **Examples:**

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

        '''
        return type(self)(f for f in self if f.match_by_ncvar(*ncvars))

    def select_by_property(self, *mode, **properties):
        '''Select list elements by property.

    To find the inverse of the selection, use a list comprehension
    with the `!match_by_property` method of the constuct elements. For
    example, to select all constructs which do *not* have a long_name
    property of "Pressure":

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
            properties, but if the *mode* parameter is ``'or'`` then a
            construct will be selected when at least one of its
            properties matches.

        properties: optional
            Select the constructs with the given properties. May be
            one or more of:

              * The property of a construct.

            By default a construct is selected if it matches all of
            the given properties, but it may alternatively be selected
            when at least one of its properties matches (see the
            *mode* positional parameter).

            A property value is given by a keyword parameter of the
            property name. The value may be a scalar or vector
            (e.g. ``'air_temperature'``, ``4``, ``['foo', 'bar']``);
            or a compiled regular expression
            (e.g. ``re.compile('^ocean')``), for which all constructs
            whose methods match (via `re.search`) are selected.

    :Returns:

        `{{class}}`
            The matching constructs.

    **Examples:**

    TODO

        '''
        return type(self)(
            f for f in self if f.match_by_property(*mode, **properties))

    def select_by_rank(self, *ranks):
        '''Select list elements by the number of domain axis constructs.

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

            The condition is satisfied if the number of domain axis
            constructs equals the condition value.

            *Parameter example:*
              To see if the field construct has 4 domain axis
              constructs: ``4``

            *Parameter example:*
              To see if the field construct has at least 3 domain axis
              constructs: ``cf.ge(3)``

    :Returns:

        `bool`
            The matching field constructs.

    **Examples:**

        TODO

        '''

        return type(self)(f for f in self if f.match_by_rank(*ranks))

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def select(self, *identities, **kwargs):
        '''Alias of `cf.{{class}}.select_by_identity`.

    To find the inverse of the selection, use a list comprehension
    with the `!match_by_identity` method of the constuct elements. For
    example, to select all constructs whose identity is *not*
    ``'air_temperature'``:

       >>> gl = cf.{{class}}(f for f in fl
       ...                   if not f.match_by_identity('air_temperature'))

    .. seealso:: `select_by_identity`, `__call__`

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'select', kwargs,
                "Use methods 'select_by_units', 'select_by_construct', "
                "'select_by_properties', 'select_by_naxes', 'select_by_rank' "
                "instead."
            )  # pragma: no cover

        if identities and isinstance(identities[0], (list, tuple, set)):
            _DEPRECATION_ERROR(
                "Use of a {!r} for identities has been deprecated. Use the "
                "* operator to unpack the arguments instead.".format(
                    identities[0].__class__.__name__)
            )  # pragma: no cover

        for i in identities:
            if isinstance(i, dict):
                _DEPRECATION_ERROR_DICT(
                    "Use methods 'select_by_units', 'select_by_construct', "
                    "'select_by_properties', 'select_by_naxes', "
                    "'select_by_rank' instead."
                )  # pragma: no cover

            if isinstance(i, str) and ':' in i:
                error = True
                if '=' in i:
                    index0 = i.index('=')
                    index1 = i.index(':')
                    error = index0 > index1

                if error:
                    _DEPRECATION_ERROR(
                        "The identity format {!r} has been deprecated at "
                        "version 3.0.0. Try {!r} instead.".format(
                            i,  i.replace(':', '=', 1))
                    )  # pragma: no cover
        # --- End: for

        return self.select_by_identity(*identities)

# --- End: class
