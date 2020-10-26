from copy import copy


from . import mixin
from . import abstract

from .functions import (_DEPRECATION_ERROR,
                        _DEPRECATION_ERROR_KWARGS,
                        _DEPRECATION_ERROR_METHOD,
                        _DEPRECATION_ERROR_DICT)


class FieldList(mixin.FieldDomainList,
                abstract.ConstructList):
    '''An ordered sequence of fields.

    Each element of a field list is a field construct.

    A field list supports the python list-like operations (such as
    indexing and methods like `!append`). These methods provide
    functionality similar to that of a :ref:`built-in list
    <python:tut-morelists>`. The main difference is that when a field
    construct element needs to be assesed for equality its
    `~cf.Field.equals` method is used, rather than the ``==``
    operator.

    '''
    def __init__(self, fields=None):
        '''**Initialization**

    :Parameters:

        fields: (sequence of) `Field`, optional
             Create a new list with these field constructs.

        '''
        super().__init__(constructs=fields)

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def concatenate(self, axis=0, _preserve=True):
        '''Join the sequence of fields together.

    This is different to `cf.aggregate` because it does not account
    for all metadata. For example, it assumes that the axis order is
    the same in each field.

    .. versionadded:: 1.0

    .. seealso:: `cf.aggregate`, `Data.concatenate`

    :Parameters:

        axis: `int`, optional
            TODO

    :Returns:

        `Field`
            TODO

        '''
        return self[0].concatenate(self, axis=axis, _preserve=_preserve)

    def select_by_construct(self, *identities, OR=False, **conditions):
        '''Select field constructs by metadata constructs.

    To find the inverse of the selection, use a list comprehension
    with the !match_by_construct` method of the constuct elements. For
    example, to select all constructs that do *not* have a "latitude"
    metadata construct:

       >>> gl = cf.{{class}}(
       ...     f for f in fl if not f.match_by_constructs('latitude')
       ... )

    .. note:: The API changed at version 3.1.0

    .. versionadded:: 3.0.0

    .. seealso: `select`, `__call__`, `select_by_units`,
                `select_by_naxes`, `select_by_rank`,
                `select_by_property`, `cf.Field.match_by_construct`

    :Parameters:

        identities: optional
            Identify the metadata constructs by one or more of

            * A metadata construct identity.

              {{construct selection identity}}

            * The key of a metadata construct (although beware that
              construct keys may differ arbitrarily between list
              elements).

            If no identities nor conditions (see the *conditions*
            parameter) are provided then all list elements are
            selected.

            If a cell method construct identity is given (such as
            ``'method:mean'``) then it will only be compared with the
            most recently applied cell method operation.

            Alternatively, one or more cell method constucts may be
            identified in a single string with a CF-netCDF cell
            methods-like syntax for describing both the collapse
            dimensions, the collapse method, and any cell method
            construct qualifiers. If N cell methods are described in
            this way then they will collectively identify the N most
            recently applied cell method operations. For example,
            ``'T: maximum within years T: mean over years'`` will be
            compared with the most two most recently applied cell
            method operations.

            *Parameter example:*
              `'latitude'``

            *Parameter example:*
              ``'T'

            *Parameter example:*
              ``'latitude'``

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

            A construct identity or construct key (as defined by the
            *identities* parameter) is given as a keyword name and a
            condition on its data is given as the keyword value.

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
            construct matches at least one of the criteria given by
            the *identities* or *conditions* arguments. By default
            `True` is only returned if the field constructs matches
            each of the given criteria.

        mode: deprecated at version 3.1.0
            Use the *OR* parameter instead.

        constructs: deprecated at version 3.1.0

    :Returns:

        `bool`
            The matching field constructs.

    **Examples:**

        TODO

        '''
        if identities:
            if identities[0] == 'or':
                _DEPRECATION_ERROR_ARG(
                    self, 'select_by_construct', 'or',
                    message="Use 'OR=True' instead.", version='3.1.0'
                )  # pragma: no cover

            if identities[0] == 'and':
                _DEPRECATION_ERROR_ARG(
                    self, 'select_by_construct', 'and',
                    message="Use 'OR=False' instead.", version='3.1.0'
                )  # pragma: no cover
        # --- End: if

        return type(self)(
            f for f in self
            if f.match_by_construct(*identities, OR=OR, **conditions)
        )

    def select_by_naxes(self, *naxes):
        '''Select field constructs by property.

    To find the inverse of the selection, use a list comprehension
    with `~cf.Field.match_by_naxes` method of the constuct
    elements. For example, to select all constructs which do *not*
    have 3-dimensional data:

       >>> gl = cf.FieldList(
       ...     f for f in fl if not f.match_by_naxes(3)
       ... )

    .. versionadded:: 3.0.0

    .. seealso:: `select`, `select_by_identity`,
                 `select_by_construct`, `select_by_property`,
                 `select_by_rank`, `select_by_units`

    :Parameters:

        naxes: optional
            Select field constructs whose data spans a particular
            number of domain axis constructs.

            A number of domain axis constructs is given by an `int`.

            If no numbers are provided then all field constructs are
            selected.

    :Returns:

        `FieldList`
            The matching field constructs.

    **Examples:**

    TODO

        '''
        return type(self)(f for f in self if f.match_by_naxes(*naxes))

    def select_by_units(self, *units, exact=True):
        '''Select field constructs by units.

    To find the inverse of the selection, use a list comprehension
    with `~cf.Field.match_by_units` method of the constuct
    elements. For example, to select all constructs whose units are
    *not* ``'km'``:

       >>> gl = cf.FieldList(
       ...     f for f in fl if not f.match_by_units('km')
       ... )

    .. versionadded:: 3.0.0

    .. seealso:: `select`, `select_by_identity`,
                 `select_by_construct`, `select_by_naxes`,
                 `select_by_rank`, `select_by_property`

    :Parameters:

        units: optional
            Select field constructs. By default all field constructs
            are selected. May be one or more of:

              * The units of a field construct.

            Units are specified by a string or compiled regular
            expression (e.g. 'km', 'm s-1', ``re.compile('^kilo')``,
            etc.) or a `Units` object (e.g. ``Units('km')``,
            ``Units('m s-1')``, etc.).

        exact: `bool`, optional
            If `False` then select field constructs whose units are
            equivalent to any of those given by *units*. For example,
            metres and are equivelent to kilometres. By default, field
            constructs whose units are exactly one of those given by
            *units* are selected. Note that the format of the units is
            not important, i.e. 'm' is exactly the same as 'metres'
            for this purpose.

    :Returns:

        `FieldList`
            The matching field constructs.

    **Examples:**

    >>> gl = fl.select_by_units('metres')
    >>> gl = fl.select_by_units('m')
    >>> gl = fl.select_by_units('m', 'kilogram')
    >>> gl = fl.select_by_units(Units('m'))
    >>> gl = fl.select_by_units('km', exact=False)
    >>> gl = fl.select_by_units(Units('km'), exact=False)
    >>> gl = fl.select_by_units(re.compile('^met'))
    >>> gl = fl.select_by_units(Units('km'))
    >>> gl = fl.select_by_units(Units('kg m-2'))

        '''
        return type(self)(f for f in self
                          if f.match_by_units(*units, exact=exact))

    def select_field(self, *identities, default=ValueError()):
        '''Select a unique field construct by its identity.

    .. versionadded:: 3.0.4

    .. seealso:: `select`, `select_by_identity`

    :Parameters:

        identities: optional
            Select the field construct by one or more of

            * A construct identity.

              {{construct selection identity}}

        default: optional
            Return the value of the *default* parameter if a unique
            field construct can not be found.

            {{default Exception}}

    :Returns:

        `Field`
            The unique matching field construct.

    **Examples:**

    >>> fl
    [<CF Field: specific_humidity(latitude(73), longitude(96)) 1>,
     <CF Field: specific_humidity(latitude(73), longitude(96)) 1>,
     <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>]
    >>> fl.select_field('air_temperature')
    <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
    >>> f.select_field('specific_humidity')
    ValueError: Multiple fields found
    >>> f.select_field('specific_humidity', 'No unique field')
    'No unique field'
    >>> f.select_field('snowfall_amount')
    ValueError: No fields found

        '''
        out = self.select_by_identity(*identities)

        if not out:
            return self._default(
                default,
                "No fields found from {}".format(identities)
            )
        
        if len(out) > 1:
            return self._default(
                default,
                "Multiple fields found from {!r}".format(identities)
            )

        return out[0]

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def select(self, *identities, **kwargs):
        '''Alias of `cf.{{class}}.select_by_identity`.

    To find the inverse of the selection, use a list comprehension
    with the `~cf.{{class}}_by_identity` method of the constuct
    elements. For example, to select all constructs whose identity is
    *not* ``'air_temperature'``:

       >>> gl = cf.{{class}}(
       ...     f for f in fl if not f.match_by_identity('air_temperature')
       ... )

    .. seealso:: `select_by_identity`, `select_field`

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

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    def _parameters(self, d):
        '''Deprecated at version 3.0.0.

        '''
        _DEPRECATION_ERROR_METHOD(self, '_parameters')  # pragma: no cover

    def _deprecated_method(self, name):
        '''Deprecated at version 3.0.0.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, '_deprecated_method')  # pragma: no cover

    def set_equals(self, other, rtol=None, atol=None,
                   ignore_data_type=False, ignore_fill_value=False,
                   ignore_properties=(), ignore_compression=False,
                   ignore_type=False, traceback=False):
        '''Deprecated at version 3.0.0. Use method 'equals' with
    unordered=True instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'set_equals',
            "Use method 'equals' with unordered=True instead."
        )  # pragma: no cover

    def select1(self, *args, **kwargs):
        '''Deprecated at version 3.0.0. Use method 'fl.select_field' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'select1', "Use method 'fl.select_field' instead."
        )  # pragma: no cover


# --- End: class
