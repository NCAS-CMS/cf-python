from copy import copy

import logging

from .functions import (_DEPRECATION_ERROR,
                        _DEPRECATION_ERROR_KWARGS,
                        _DEPRECATION_ERROR_METHOD,
                        _DEPRECATION_ERROR_DICT)

from .decorators import (_deprecated_kwarg_check,
                         _manage_log_level_via_verbosity)


logger = logging.getLogger(__name__)


class FieldList(list):
    '''An ordered sequence of fields.

    Each element of a field list is a field construct.

    A field list supports the python list-like operations (such as
    indexing and methods like `!append`).

    >>> fl = cf.FieldList()
    >>> len(fl)
    0
    >>> f
    <CF Field: air_temperaturetime(12), latitude(73), longitude(96) K>
    >>> fl = cf.FieldList(f)
    >>> len(fl)
    1
    >>> fl = cf.FieldList([f, f])
    >>> len(fl)
    2
    >>> fl = cf.FieldList(cf.FieldList([f] * 3))
    >>> len(fl)
    3
    >>> len(fl + fl)
    6

    These methods provide functionality similar to that of a
    :ref:`built-in list <python:tut-morelists>`. The main difference
    is that when a field element needs to be assesed for equality its
    `~cf.Field.equals` method is used, rather than the ``==``
    operator.

    '''
    def __init__(self, fields=None):
        '''**Initialization**

    :Parameters:

        fields: (sequence of) `Field`, optional
             Create a new field list with these fields.

        '''
        if fields is not None:
            if getattr(fields, 'construct_type', None) == 'field':
                self.append(fields)
            else:
                self.extend(fields)

    def __call__(self, *identities):
        '''Alias for `cf.FieldList.select_by_identity`.

        '''
        return self.select_by_identity(*identities)

    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        out = [repr(f) for f in self]
        out = ',\n '.join(out)
        return '['+out+']'

    # ----------------------------------------------------------------
    # Overloaded list methods
    # ----------------------------------------------------------------
    def __add__(self, x):
        '''The binary arithmetic operation ``+``

    f.__add__(x) <==> f + x

    :Returns:

        `FieldList`

    **Examples:**

    >>> h = f + g
    >>> f += g

        '''
        return type(self)(list.__add__(self, x))

    def __contains__(self, y):
        '''Called to implement membership test operators.

    x.__contains__(y) <==> y in x

    Each field in the field list is compared with the field's
    `~cf.Field.equals` method, as opposed to the ``==`` operator.

    Note that ``y in fl`` is equivalent to ``any(f.equals(y) for f in fl)``.

        '''
        for f in self:
            if f.equals(y):
                return True
        # --- End: for

        return False

    def __mul__(self, x):
        '''The binary arithmetic operation ``*``

    f.__mul__(x) <==> f * x

    :Returns:

        `FieldList`

    **Examples:**

    >>> h = f * 2
    >>> f *= 2

        '''
        return type(self)(list.__mul__(self, x))

    def __eq__(self, other):
        '''The rich comparison operator ``==``

    f.__eq__(x) <==> f==x

    Each field in the field list is compared with the field's
    `~cf.Field.equals` method, as opposed to the ``==`` operator.

    Note that ``f==x`` is equivalent to ``f.equals(x)``.

    :Returns:

        `bool`

        '''
        return self.equals(other)

    def __getslice__(self, i, j):
        '''Called to implement evaluation of f[i:j]

    f.__getslice__(i, j) <==> f[i:j]

    :Returns:

        `FieldList`

    **Examples:**

    >>> g = f[0:1]
    >>> g = f[1:-4]
    >>> g = f[:1]
    >>> g = f[1:]

        '''
        return type(self)(list.__getslice__(self, i, j))

    def __getitem__(self, index):
        '''Called to implement evaluation of f[index]

    f.__getitem_(index) <==> f[index]

    :Returns:

        `Field` or `FieldList`
            If *index* is an integer then a field construct is
            returned. If *index* is a slice then a field list is returned,
            which may be empty.

    **Examples:**

    >>> g = f[0]
    >>> g = f[-1:-4:-1]
    >>> g = f[2:2:2]

        '''
        out = list.__getitem__(self, index)

        if isinstance(out, list):
            return type(self)(out)

        return out

    def __ne__(self, other):
        '''The rich comparison operator ``!=``

    f.__ne__(x) <==> f!=x

    Each field in the field list is compared with the field's
    `~cf.Field.equals` method, as opposed to the ``==`` operator.

    Note that ``f!=x`` is equivalent to ``not f.equals(x)``.

    :Returns:

        `bool`

        '''
        return not self.equals(other)

    # ???
    __len__ = list.__len__
    __setitem__ = list.__setitem__
    append = list.append
    extend = list.extend
    insert = list.insert
    pop = list.pop
    reverse = list.reverse
    sort = list.sort

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def close(self):
        '''Close all files referenced by each field construct.

    Note that a closed file will be automatically reopened if its
    contents are subsequently required.

    :Returns:

        `None`

    **Examples:**

    >>> fl.close()

        '''
        for f in self:
            f.close()

    def count(self, value):
        '''Return number of occurrences of value

    Each field in the field list is compared with the field's
    `~cf.Field.equals` method, as opposed to the ``==`` operator.

    Note that ``fl.count(value)`` is equivalent to
    ``sum(f.equals(value) for f in fl)``.

    .. seealso:: `cf.Field.equals`, `list.count`

    **Examples:**

    >>> f = cf.FieldList([a, b, c, a])
    >>> f.count(a)
    2
    >>> f.count(b)
    1
    >>> f.count(a+1)
    0

        '''
        return len([None for f in self if f.equals(value)])

    def index(self, value, start=0, stop=None):
        '''Return first index of value.

    Each field in the field list is compared with the field's
    `~cf.Field.equals` method, as opposed to the ``==`` operator.

    An exception is raised if there is no such field.

    .. seealso:: `list.index`

        '''
        if start < 0:
            start = len(self) + start

        if stop is None:
            stop = len(self)
        elif stop < 0:
            stop = len(self) + stop

        for i, f in enumerate(self[start:stop]):
            if f.equals(value):
                return i + start
        # --- End: for

        raise ValueError(
            "{0!r} is not in {1}".format(value, self.__class__.__name__))

    def remove(self, value):
        '''Remove first occurrence of value.

    Each field in the field list is compared with its
    `~cf.Field.equals` method, as opposed to the ``==`` operator.

    .. seealso:: `list.remove`

        '''
        for i, f in enumerate(self):
            if f.equals(value):
                del self[i]
                return
        # --- End: for

        raise ValueError(
            "{0}.remove(x): x not in {0}".format(self.__class__.__name__))

    def sort(self, key=None, reverse=False):
        '''Sort of the field list in place.

    By default the field list is sorted by the identities of its field
    construct elements.

    The sort is stable.

    .. versionadded:: 1.0.4

    .. seealso:: `reverse`

    :Parameters:

        key: function, optional
            Specify a function of one argument that is used to extract
            a comparison key from each field construct. By default the
            field list is sorted by field identity, i.e. the default
            value of *key* is ``lambda f: f.identity()``.

        reverse: `bool`, optional
            If set to `True`, then the field list elements are sorted
            as if each comparison were reversed.

    :Returns:

        `None`

    **Examples:**

    >>> fl
    [<CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
     <CF Field: ocean_meridional_overturning_streamfunction(time(12), region(4), depth(40), latitude(180)) m3 s-1>,
     <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>,
     <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>]
    >>> fl.sort()
    >>> fl
    [<CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>,
     <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
     <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
     <CF Field: ocean_meridional_overturning_streamfunction(time(12), region(4), depth(40), latitude(180)) m3 s-1>]
    >>> fl.sort(reverse=True)
    >>> fl
    [<CF Field: ocean_meridional_overturning_streamfunction(time(12), region(4), depth(40), latitude(180)) m3 s-1>,
     <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
     <CF Field: eastward_wind(time(3), air_pressure(5), grid_latitude(110), grid_longitude(106)) m s-1>,
     <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>]

    >>> [f.datum(0) for f in fl]
    [masked,
     -0.12850454449653625,
     -0.12850454449653625,
     236.51275634765625]
    >>> fl.sort(key=lambda f: f.datum(0), reverse=True)
    >>> [f.datum(0) for f in fl]
    [masked,
     236.51275634765625,
     -0.12850454449653625,
     -0.12850454449653625]

    >>> from operator import attrgetter
    >>> [f.long_name for f in fl]
    ['Meridional Overturning Streamfunction',
     'U COMPNT OF WIND ON PRESSURE LEVELS',
     'U COMPNT OF WIND ON PRESSURE LEVELS',
     'air_temperature']
    >>> fl.sort(key=attrgetter('long_name'))
    >>> [f.long_name for f in fl]
    ['air_temperature',
     'Meridional Overturning Streamfunction',
     'U COMPNT OF WIND ON PRESSURE LEVELS',
     'U COMPNT OF WIND ON PRESSURE LEVELS']

        '''
        if key is None:
            key = lambda f: f.identity()

        return super().sort(key=key, reverse=reverse)

    def __deepcopy__(self, memo):
        '''Called by the `copy.deepcopy` standard library function.

        '''
        return self.copy()

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

    def copy(self, data=True):
        '''Return a deep copy.

    ``f.copy()`` is equivalent to ``copy.deepcopy(f)``.

    :Returns:

            The deep copy.

    **Examples:**

    >>> g = f.copy()
    >>> g is f
    False
    >>> f.equals(g)
    True
    >>> import copy
    >>> h = copy.deepcopy(f)
    >>> h is f
    False
    >>> f.equals(g)
    True

        '''
        return type(self)([f.copy(data=data) for f in self])

    @_deprecated_kwarg_check('traceback')
    @_manage_log_level_via_verbosity
    def equals(self, other, rtol=None, atol=None, verbose=None,
               ignore_data_type=False, ignore_fill_value=False,
               ignore_properties=(), ignore_compression=False,
               ignore_type=False, ignore=(), traceback=False,
               unordered=False):
        '''Whether two field lists are the same.

    Equality requires the two field lists to have the same length and
    for the field construct elements to be equal pair-wise, using
    their `~cf.Field.equals` methods.

    Any type of object may be tested but, in general, equality is only
    possible with another field list, or a subclass of one. See the
    *ignore_type* parameter.

    Equality is between teo field constructs is strict by
    default. This means that for two field constructs to be considered
    equal they must have corresponding metadata constructs and for
    each pair of constructs:

    * the same descriptive properties must be present, with the same
      values and data types, and vector-valued properties must also
      have same the size and be element-wise equal (see the
      *ignore_properties* and *ignore_data_type* parameters), and

    ..

    * if there are data arrays then they must have same shape and data
      type, the same missing data mask, and be element-wise equal (see
      the *ignore_data_type* parameter).

    Two real numbers ``x`` and ``y`` are considered equal if
    ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
    differences) and ``rtol`` (the tolerance on relative differences)
    are positive, typically very small numbers. See the *atol* and
    *rtol* parameters.

    If data arrays are compressed then the compression type and the
    underlying compressed arrays must be the same, as well as the
    arrays in their uncompressed forms. See the *ignore_compression*
    parameter.

    NetCDF elements, such as netCDF variable and dimension names, do
    not constitute part of the CF data model and so are not checked on
    any construct.

    :Parameters:
        other:
            The object to compare for equality.

        atol: float, optional
            The tolerance on absolute differences between real
            numbers. The default value is set by the `cfdm.ATOL`
            function.

        rtol: float, optional
            The tolerance on relative differences between real
            numbers. The default value is set by the `cfdm.RTOL`
            function.

        ignore_fill_value: `bool`, optional
            If `True` then the "_FillValue" and "missing_value"
            properties are omitted from the comparison, for the field
            construct and metadata constructs.

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
            convey information about differences that lead to inequality.

        ignore_properties: sequence of `str`, optional
            The names of properties of the field construct (not the
            metadata constructs) to omit from the comparison. Note
            that the "Conventions" property is always omitted by
            default.

        ignore_data_type: `bool`, optional
            If `True` then ignore the data types in all numerical
            comparisons. By default different numerical data types
            imply inequality, regardless of whether the elements are
            within the tolerance for equality.

        ignore_compression: `bool`, optional
            If `True` then any compression applied to underlying arrays
            is ignored and only uncompressed arrays are tested for
            equality. By default the compression type and, if
            applicable, the underlying compressed arrays must be the
            same, as well as the arrays in their uncompressed forms

        ignore_type: `bool`, optional
            Any type of object may be tested but, in general, equality
            is only possible with another field list, or a subclass of
            one. If *ignore_type* is True then
            ``FieldList(source=other)`` is tested, rather than the
            ``other`` defined by the *other* parameter.

        unordered: `bool`, optional
            TODO

    :Returns:

        `bool`
            Whether the two field lists are equal.

    **Examples:**

    >>> fl.equals(fl)
    True
    >>> fl.equals(fl.copy())
    True
    >>> fl.equals(fl[:])
    True
    >>> fl.equals('not a FieldList instance')
    False

        '''
        if ignore:
            _DEPRECATION_ERROR_KWARGS(
                self, 'equals', {'ignore': ignore},
                "Use keyword 'ignore_properties' instead."
            )  # pragma: no cover

        # Check for object identity
        if self is other:
            return True

        # Check that each object is of compatible type
        if ignore_type:
            if not isinstance(other, self.__class__):
                other = type(self)(source=other, copy=False)
        elif not isinstance(other, self.__class__):
            logger.info(
                "{0}: Incompatible type: {1}".format(
                    self.__class__.__name__, other.__class__.__name__)
            )  # pragma: no cover
            return False

        # Check that there are equal numbers of fields
        len_self = len(self)
        if len_self != len(other):
            logger.info(
                "{0}: Different numbers of field construct: "
                "{1}, {2}".format(
                    self.__class__.__name__,
                    len_self, len(other))
            )  # pragma: no cover
            return False

        if not unordered or len_self == 1:
            # ----------------------------------------------------
            # Check the lists pair-wise
            # ----------------------------------------------------
            for i, (f, g) in enumerate(zip(self, other)):
                if not f.equals(g, rtol=rtol, atol=atol,
                                ignore_fill_value=ignore_fill_value,
                                ignore_properties=ignore_properties,
                                ignore_compression=ignore_compression,
                                ignore_data_type=ignore_data_type,
                                ignore_type=ignore_type,
                                verbose=verbose):
                    logger.info(
                        "{0}: Different field constructs at element {1}: "
                        "{2!r}, {3!r}".format(
                            self.__class__.__name__, i, f, g)
                    )  # pragma: no cover
                    return False
        else:
            # ----------------------------------------------------
            # Check the lists set-wise
            # ----------------------------------------------------
            # Group the variables by identity
            self_identity = {}
            for f in self:
                self_identity.setdefault(f.identity(), []).append(f)

            other_identity = {}
            for f in other:
                other_identity.setdefault(f.identity(), []).append(f)

            # Check that there are the same identities
            if set(self_identity) != set(other_identity):
                logger.info("{}: Different sets of identities: {}, {}".format(
                    self.__class__.__name__,
                    set(self_identity),
                    set(other_identity)))  # pragma: no cover
                return False

            # Check that there are the same number of variables
            # for each identity
            for identity, fl in self_identity.items():
                gl = other_identity[identity]
                if len(fl) != len(gl):
                    logger.info(
                        "{0}: Different numbers of {1!r} {2}s: "
                        "{3}, {4}".format(
                            self.__class__.__name__,
                            identity,
                            fl[0].__class__.__name__,
                            len(fl), len(gl)
                        )
                    )  # pragma: no cover
                    return False
            # --- End: for

            # For each identity, check that there are matching pairs
            # of equal fields.
            for identity, fl in self_identity.items():
                gl = other_identity[identity]

                for f in fl:
                    found_match = False
                    for i, g in enumerate(gl):
                        if f.equals(g, rtol=rtol, atol=atol,
                                    ignore_fill_value=ignore_fill_value,
                                    ignore_properties=ignore_properties,
                                    ignore_compression=ignore_compression,
                                    ignore_data_type=ignore_data_type,
                                    ignore_type=ignore_type,
                                    verbose=verbose):
                            found_match = True
                            del gl[i]
                            break
                # --- End: for

                if not found_match:
                    logger.info(
                        "{0}: No {1} equal to: {2!r}".format(
                            self.__class__.__name__,
                            g.__class__.__name__, f)
                    )  # pragma: no cover
                    return False
        # --- End: if

        # ------------------------------------------------------------
        # Still here? Then the field lists are equal
        # ------------------------------------------------------------
        return True

    def select_by_construct(self, *identities, OR=False, **conditions):
        '''Select field constructs by metadata constructs.

    To find the inverse of the selection, use a list comprehension
    with the `~cf.Field.match_by_construct` method of the field
    constucts. For example, to select all field constructs that do
    *not* have a "latitude" metadata construct:

       >>> gl = cf.FieldList(f for f in fl
       ...                   if not f.match_by_constructs('latitude'))

    .. note:: The API changed at version 3.1.0

    .. versionadded:: 3.0.0

    .. seealso: `select`, `__call__`, `select_by_units`,
                `select_by_naxes`, `select_by_rank`,
                `select_by_property`, `cf.Field.match_by_identity`,
                `cf.Field.subspace`

    :Parameters:

        identities: optional
            Identify the metadata constructs that have any of the
            given identities or construct keys.

            A construct identity is specified by a string
            (e.g. ``'latitude'``, ``'long_name=time'``,
            ``'ncvar%lat'``, etc.); or a compiled regular expression
            (e.g. ``re.compile('^atmosphere')``) that selects the
            relevant constructs whose identities match via
            `re.search`.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            six identities:

               >>> x.identities()
               ['time',
                'long_name=Time',
                'foo=bar',
                'standard_name=time',
                'ncvar%t',
                'T']

            A construct key may optionally have the ``'key%'``
            prefix. For example ``'dimensioncoordinate2'`` and
            ``'key%dimensioncoordinate2'`` are both acceptable keys.

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identity* argument.

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
              ``'measure:area'``

            *Parameter example:*
              ``'latitude'``

            *Parameter example:*
              ``'long_name=Longitude'``

            *Parameter example:*
              ``'domainancillary2', 'ncvar%areacello'``

        conditions: optional
            Identify the metadata constructs that have any of the
            given identities or construct keys, and whose data satisfy
            conditions.

            A construct identity or construct key (as defined by the
            *identities* parameter) is given as a keyword name and a
            condition on its data is given as the keyword value.

            The condition is satisfied if any of its data values
            equals the value provided.

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
#        if constructs:
#            for key, value in constructs.items():
#                if value is None:
#                    message = ("Since its value is None, use {!r} as a "
#                               "positional argument instead".format(value))
#                else:
#                    message = ("Evaluating criteria on data values is no "
#                               "longer possible with this method.")
#
#                _DEPRECATION_ERROR_KWARGS(self, 'select_by_construct',
#                                          kwargs={key: value},
#                                          message=message,
#                                          version='3.1.0') # pragma: no cover
#        # --- End: if

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

    def select_by_identity(self, *identities):
        '''Select field constructs by identity.

    To find the inverse of the selection, use a list comprehension
    with the `~cf.Field.match_by_identity` method of the field
    constucts. For example, to select all field constructs whose
    identity is *not* ``'air_temperature'``:

       >>> gl = cf.FieldList(f for f in fl
       ...                   if not f.match_by_identity('air_temperature'))

    .. versionadded:: 3.0.0

    .. seealso:: `select`, `__call__`, `select_by_units`,
                 `select_by_construct`, `select_by_naxes`,
                 `select_by_rank`, `select_by_property`,
                 `cf.Field.match_by_identity`

    :Parameters:

        identities: optional
            Select field constructs. By default all field constructs
            are selected. May be one or more of:

              * The identity of a field construct.

            A construct identity is specified by a string (e.g.
            ``'air_temperature'``, ``'long_name=Air Temperature',
            ``'ncvar%tas'``, etc.); or a compiled regular expression
            (e.g. ``re.compile('^air_')``) that selects the relevant
            constructs whose identities match via `re.search`.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            five identities:

               >>> x.identities()
               ['air_temperature',
                'long_name=Air Temperature',
                'foo=bar',
                'standard_name=air_temperature',
                'ncvar%tas']

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identities* argument.

    :Returns:

        `FieldList`
            The matching field constructs.

    **Examples:**

    >>> fl
    [<CF Field: specific_humidity(latitude(73), longitude(96)) 1>,
     <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>]
    >>> fl.select('air_temperature')
    [<CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>]

        '''
        return type(self)(f for f in self if f.match_by_identity(*identities))

    def select_by_naxes(self, *naxes):
        '''Select field constructs by property.

    To find the inverse of the selection, use a list comprehension
    with `~cf.Field.match_by_naxes` method of the field constucts. For
    example, to select all field constructs which do *not* have
    3-dimensional data:

       >>> gl = cf.FieldList(f for f in fl if not f.match_by_naxes(3))

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

    def select_by_rank(self, *ranks):
        '''Select field constructs by the number of domain axis constructs.

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

    def select_by_ncvar(self, *ncvars):
        '''Select field constructs by netCDF variable name.

    To find the inverse of the selection, use a list comprehension
    with `~cf.Field.match_by_ncvar` method of the field constucts. For
    example, to select all field constructs which do *not* have a
    netCDF name of 'tas':

       >>> gl = cf.FieldList(f for f in fl if not f.match_by_ncvar('tas'))

    .. versionadded:: 3.0.0

    .. seealso:: `select`, `select_by_identity`,
                 `select_by_construct`, `select_by_naxes`,
                 `select_by_rank`, `select_by_units`

    :Parameters:

        ncvars: optional
            Select field constructs. May be one or more:

              * The netCDF name of a field construct.

            A field construct is selected if it matches any of the
            given names.

            A netCDF variable name is specified by a string (e.g.
            ``'tas'``, etc.); a `Query` object
            (e.g. ``cf.eq('tas')``); or a compiled regular expression
            (e.g. ``re.compile('^air_')``) that selects the field
            constructs whose netCDF variable names match via
            `re.search`.

            If no netCDF variable names are provided then all field
            are selected.

    :Returns:

        `FieldList`
            The matching field constructs.

    **Examples:**

    >>> fl = cf.FieldList([cf.example_field(0), cf.example_field(1)])
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
        '''Select field constructs by property.

    To find the inverse of the selection, use a list comprehension
    with `~cf.Field.match_by_property` method of the field
    constucts. For example, to select all field constructs which do
    *not* have a long_name property of 'Air Pressure':

       >>> gl = cf.FieldList(f for f in fl if not
       ...                   f.match_by_property(long_name='Air Pressure))

    .. versionadded:: 3.0.0

    .. seealso:: `select`, `select_by_identity`,
                 `select_by_construct`, `select_by_naxes`,
                 `select_by_rank`, `select_by_units`

    :Parameters:

        mode: optional
            Define the behaviour when multiple properties are
            provided.

            By default (or if the *mode* parameter is ``'and'``) a
            field construct is selected if it matches all of the given
            properties, but if the *mode* parameter is ``'or'`` then a
            field construct will be selected when at least one of its
            properties matches.

        properties: optional
            Select field constructs. May be one or more of:

              * The property of a field construct.

            By default a field construct is selected if it matches all
            of the given properties, but it may alternatively be
            selected when at least one of its properties matches (see
            the *mode* positional parameter).

            A property value is given by a keyword parameter of the
            property name. The value may be a scalar or vector
            (e.g. ``'air_temperature'``, ``4``, ``['foo', 'bar']``);
            or a compiled regular expression
            (e.g. ``re.compile('^ocean')``), for which all constructs
            whose methods match (via `re.search`) are selected.

    :Returns:

        `FieldList`
            The matching field constructs.

    **Examples:**

    TODO

        '''
        return type(self)(
            f for f in self if f.match_by_property(*mode, **properties))

    def select_by_units(self, *units, exact=True):
        '''Select field constructs by units.

    To find the inverse of the selection, use a list comprehension
    with `~cf.Field.match_by_units` method of the field constucts. For
    example, to select all field constructs whose units are *not*
    ``'km'``:

       >>> gl = cf.FieldList(f for f in fl if not f.match_by_units('km'))

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

    def select_field(self, identity, default=ValueError()):
        '''Select a unique field construct by its identity.

    .. versionadded:: 3.0.4

    .. seealso:: `select`, `select_by_identity`

    :Parameters:

        identity:
            Select the field construct. May be:

              * The identity of a field construct.

            A construct identity is specified by a string (e.g.
            ``'air_temperature'``, ``'long_name=Air Temperature',
            ``'ncvar%tas'``, etc.); or a compiled regular expression
            (e.g. ``re.compile('^air_')``) that selects the relevant
            constructs whose identities match via `re.search`.

            Each construct has a number of identities, and is selected
            if any of them match any of those provided. A construct's
            identities are those returned by its `!identities`
            method. In the following example, the construct ``x`` has
            five identities:

               >>> x.identities()
               ['air_temperature',
                'long_name=Air Temperature',
                'foo=bar',
                'standard_name=air_temperature',
                'ncvar%tas']

            Note that in the output of a `print` call or `!dump`
            method, a construct is always described by one of its
            identities, and so this description may always be used as
            an *identity* argument.

        default: optional
            Return the value of the *default* parameter if a unique
            field construct can not be found. If set to an `Exception`
            instance then it will be raised instead.

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
        out = self.select_by_identity(identity)

        if not out or len(out) > 1:
            if isinstance(default, Exception):
                if not default.args:
                    if not out:
                        message = "No fields found"
                    else:
                        message = "Multiple fields found"

                    default = copy(default)
                    default.args = (message,)

                raise default

            return default

        return out[0]

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def select(self, *identities, **kwargs):
        '''Alias of `cf.FieldList.select_by_identity`.

    To find the inverse of the selection, use a list comprehension
    with the `~cf.Field.match_by_identity` method of the field
    constucts. For example, to select all field constructs whose
    identity is *not* ``'air_temperature'``:

       >>> gl = cf.FieldList(f for f in fl
       ...                   if not f.match_by_identity('air_temperature'))

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
