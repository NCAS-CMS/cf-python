from copy import deepcopy

from ..functions import ATOL, RTOL

from ..query import Query
from ..units import Units

from ..functions import (_DEPRECATION_ERROR_METHOD,
                         _DEPRECATION_ERROR)


class Properties:
    '''Mixin class for a container of descriptive properties.

    .. versionadded:: 3.0.0

    '''
    _special_properties = ()

    # ----------------------------------------------------------------
    # Private attributes
    # ----------------------------------------------------------------
    @property
    def _ATOL(self):
        '''Return the tolerance on absolute differences between real numbers,
    as returned by the `cf.ATOL` function.

    This is used by, for example, the `_equals` method.

        '''
        return ATOL()

    @property
    def _RTOL(self):
        '''Return the tolerance on relative differences between real numbers,
    as returned by the `cf.RTOL` function.

    This is used by, for example, the `_equals` method.

        '''
        return RTOL()

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _matching_values(self, value0, value1, units=False):
        '''Whether two values match.

    The definition of "match" depends on the types of *value0* and
    *value1*. See the code for the details.

    :Parameters:

        value0:
            The first value to be matched.

        value1:
            The second value to be matched.

        units: `bool`, optional

    :Returns:

        `bool`
            Whether or not the two values match.

        '''
        if value1 is None:
            return False

        if isinstance(value0, Query):
            return bool(value0.evaluate(value1))  # TODO vectors

        try:
            # value0 is a re.compile object
            return bool(value0.search(value1))
        except (AttributeError, TypeError):
            if units and isinstance(value0, str):
                return Units(value0).equals(Units(value1))

            return self._equals(value1, value0)

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def id(self):
        '''A canonical identity.

    The `id` attribute can be used to unambiguously identify
    constructs in the absence of a `standard_name` property.

    Note that `id` is not a CF property and so is not read from, or
    written to, datasets.

    **Examples:**

    >>> f.id = 'um01002'
    >>> f.id
    'um01002'
    >>> del f.id

        '''
        try:
            return self._custom['id']
        except KeyError:
            raise AttributeError("{} doesn't have attribute 'id'".format(
                self.__class__.__name__))

    @id.setter
    def id(self, value):   self._custom['id'] = value

    @id.deleter
    def id(self):
        try:
            del self._custom['id']
        except KeyError:
            raise AttributeError("{} doesn't have attribute 'id'".format(
                self.__class__.__name__))

    # ----------------------------------------------------------------
    # CF properties
    # ----------------------------------------------------------------
    @property
    def calendar(self):
        '''The calendar CF property.

    The calendar used for encoding time data. See
    http://cfconventions.org/latest.html for details.

    **Examples:**

    >>> f.calendar = 'noleap'
    >>> f.calendar
    'noleap'
    >>> del f.calendar

    >>> f.set_property('calendar', 'proleptic_gregorian')
    >>> f.has_property('calendar')
    True
    >>> f.get_property('calendar')
    'proleptic_gregorian'
    >>> f.del_property('calendar')

        '''
        return self.get_property('calendar', default=AttributeError())

    @calendar.setter
    def calendar(self, value): self.set_property('calendar', value)

    @calendar.deleter
    def calendar(self):
        self.del_property('calendar', default=AttributeError())

    @property
    def comment(self):
        '''The comment CF property.

    Miscellaneous information about the data or methods used to
    produce it. See http://cfconventions.org/latest.html for details.

    **Examples:**

    >>> f.comment = 'This simulation was done on an HP-35 calculator'
    >>> f.comment
    'This simulation was done on an HP-35 calculator'
    >>> del f.comment

    >>> f.set_property('comment', 'a comment')
    >>> f.has_property('comment')
    True
    >>> f.get_property('comment')
    'a comment'
    >>> f.del_property('comment')

        '''
        return self.get_property('comment', default=AttributeError())

    @comment.setter
    def comment(self, value): self.set_property('comment', value)

    @comment.deleter
    def comment(self):
        self.del_property('comment', default=AttributeError())

    @property
    def history(self):
        '''The history CF property.

    A list of the applications that have modified the original
    data. See http://cfconventions.org/latest.html for details.

    **Examples:**

    >>> f.history = 'created on 2012/10/01'
    >>> f.history
    'created on 2012/10/01'
    >>> del f.history

    >>> f.set_property('history', 'created on 2012/10/01')
    >>> f.has_property('history')
    True
    >>> f.get_property('history')
    'created on 2012/10/01'
    >>> f.del_property('history')

        '''
        return self.get_property('history', default=AttributeError())

    @history.setter
    def history(self, value):
        self.set_property('history', value)

    @history.deleter
    def history(self):
        self.del_property('history', default=AttributeError())

    @property
    def leap_month(self):
        '''The leap_month CF property.

    Specifies which month is lengthened by a day in leap years for a
    user defined calendar. See http://cfconventions.org/latest.html
    for details.

    **Examples:**

    >>> f.leap_month = 2
    >>> f.leap_month
    2
    >>> del f.leap_month

    >>> f.set_property('leap_month', 11)
    >>> f.has_property('leap_month')
    True
    >>> f.get_property('leap_month')
    11
    >>> f.del_property('leap_month')

        '''
        return self.get_property('leap_month', default=AttributeError())

    @leap_month.setter
    def leap_month(self, value):
        self.set_property('leap_month', value)

    @leap_month.deleter
    def leap_month(self):
        self.del_property('leap_month', default=AttributeError())

    @property
    def leap_year(self):
        '''The leap_year CF property.

    Provides an example of a leap year for a user defined calendar. It
    is assumed that all years that differ from this year by a multiple
    of four are also leap years. See
    http://cfconventions.org/latest.html for details.

    **Examples:**

    >>> f.leap_year = 1984
    >>> f.leap_year
    1984
    >>> del f.leap_year

    >>> f.set_property('leap_year', 1984)
    >>> f.has_property('leap_year')
    True
    >>> f.get_property('leap_year')
    1984
    >>> f.del_property('leap_year')

        '''
        return self.get_property('leap_year', default=AttributeError())

    @leap_year.setter
    def leap_year(self, value):
        self.set_property('leap_year', value)

    @leap_year.deleter
    def leap_year(self):
        self.del_property('leap_year', default=AttributeError())

    @property
    def long_name(self):
        '''The long_name CF property.

    A descriptive name that indicates a nature of the data. This name
    is not standardized. See http://cfconventions.org/latest.html for
    details.

    **Examples:**

    >>> f.long_name = 'zonal_wind'
    >>> f.long_name
    'zonal_wind'
    >>> del f.long_name

    >>> f.set_property('long_name', 'surface air temperature')
    >>> f.has_property('long_name')
    True
    >>> f.get_property('long_name')
    'surface air temperature'
    >>> f.del_property('long_name')

        '''
        return self.get_property('long_name', default=AttributeError())

    @long_name.setter
    def long_name(self, value):
        self.set_property('long_name', value)

    @long_name.deleter
    def long_name(self):
        self.del_property('long_name', default=AttributeError())

    @property
    def month_lengths(self):
        '''The month_lengths CF property.

    Specifies the length of each month in a non-leap year for a user
    defined calendar. See http://cfconventions.org/latest.html for
    details.

    Stored as a tuple but may be set as any array-like object.

    **Examples:**

    >>> f.month_lengths = numpy.array(
    ...     [34, 31, 32, 30, 29, 27, 28, 28, 28, 32, 32, 34])
    >>> f.month_lengths
    (34, 31, 32, 30, 29, 27, 28, 28, 28, 32, 32, 34)
    >>> del f.month_lengths

    >>> f.set_property('month_lengths',
    ...                [34, 31, 32, 30, 29, 27, 28, 28, 28, 32, 32, 34])
    >>> f.has_property('month_lengths')
    True
    >>> f.get_property('month_lengths')
    (34, 31, 32, 30, 29, 27, 28, 28, 28, 32, 32, 34)
    >>> f.del_property('month_lengths')

        '''
        return self.get_property('month_lengths', default=AttributeError())

    @month_lengths.setter
    def month_lengths(self, value):
        self.set_property('month_lengths', tuple(value))

    @month_lengths.deleter
    def month_lengths(self):
        self.del_property('month_lengths', default=AttributeError())

    @property
    def standard_name(self):
        '''The standard_name CF property.

    A standard name that references a description of a data in the
    standard name table
    (http://cfconventions.org/standard-names.html). See
    http://cfconventions.org/latest.html for details.

    **Examples:**

    >>> f.standard_name = 'time'
    >>> f.standard_name
    'time'
    >>> del f.standard_name

    >>> f.set_property('standard_name', 'time')
    >>> f.has_property('standard_name')
    True
    >>> f.get_property('standard_name')
    'time'
    >>> f.del_property('standard_name')

        '''
        return self.get_property('standard_name', default=AttributeError())

    @standard_name.setter
    def standard_name(self, value): self.set_property('standard_name', value)

    @standard_name.deleter
    def standard_name(self):
        self.del_property('standard_name', default=AttributeError())

    @property
    def units(self):
        '''The units CF property.

    The units of the data. The value of the `units` property is a
    string that can be recognized by UNIDATA's Udunits package
    (http://www.unidata.ucar.edu/software/udunits). See
    http://cfconventions.org/latest.html for details.

    **Examples:**

    >>> f.units = 'K'
    >>> f.units
    'K'
    >>> del f.units

    >>> f.set_property('units', 'm.s-1')
    >>> f.has_property('units')
    True
    >>> f.get_property('units')
    'm.s-1'
    >>> f.del_property('units')

        '''
        return self.get_property('units', default=AttributeError())

    @units.setter
    def units(self, value): self.set_property('units', value)

    @units.deleter
    def units(self):
        self.del_property('units', default=AttributeError())

    @property
    def valid_max(self):
        '''The valid_max CF property.

    The largest valid value of the data. See
    http://cfconventions.org/latest.html for details.

    **Examples:**

    >>> f.valid_max = 100.0
    >>> f.valid_max
    100.0
    >>> del f.valid_max

    >>> f.set_property('valid_max', 100.0)
    >>> f.has_property('valid_max')
    True
    >>> f.get_property('valid_max')
    100.0
    >>> f.del_property('valid_max')

        '''
        return self.get_property('valid_max', default=AttributeError())

    @valid_max.setter
    def valid_max(self, value): self.set_property('valid_max', value)

    @valid_max.deleter
    def valid_max(self):
        self.del_property('valid_max', default=AttributeError())

    @property
    def valid_min(self):
        '''The valid_min CF property.

    The smallest valid value of the data. See
    http://cfconventions.org/latest.html for details.

    **Examples:**

    >>> f.valid_min = 8.0
    >>> f.valid_min
    8.0
    >>> del f.valid_min

    >>> f.set_property('valid_min', 8.0)
    >>> f.has_property('valid_min')
    True
    >>> f.get_property('valid_min')
    8.0
    >>> f.del_property('valid_min')

        '''
        return self.get_property('valid_min', default=AttributeError())

    @valid_min.setter
    def valid_min(self, value): self.set_property('valid_min', value)

    @valid_min.deleter
    def valid_min(self):
        self.del_property('valid_min', default=AttributeError())

    @property
    def valid_range(self):
        '''The valid_range CF property.

    The smallest and largest valid values the data. See
    http://cfconventions.org/latest.html for details.

    Stored as a tuple but may be set as any array-like object.

    **Examples:**

    >>> f.valid_range = numpy.array([100., 400.])
    >>> f.valid_range
    (100.0, 400.0)
    >>> del f.valid_range

    >>> f.set_property('valid_range', [100.0, 400.0])
    >>> f.has_property('valid_range')
    True
    >>> f.get_property('valid_range')
    (100.0, 400.0)
    >>> f.del_property('valid_range')

        '''
        return tuple(
            self.get_property('valid_range'), default=AttributeError())

    @valid_range.setter
    def valid_range(self, value):
        self.set_property('valid_range', tuple(value))

    @valid_range.deleter
    def valid_range(self):
        self.del_property('valid_range', default=AttributeError())

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def get_property(self, prop, default=ValueError()):
        '''Get a CF property.

    .. versionadded:: 3.0.0

    .. seealso:: `clear_properties`, `del_property`, `has_property`,
                 `properties`, `set_property`

    :Parameters:

        prop: `str`
            The name of the CF property.

            *Parameter example:*
              ``prop='long_name'``

        default: optional
            Return the value of the *default* parameter if the
            property does not exist. If set to an `Exception` instance
            then it will be raised instead.

    :Returns:

            The value of the named property or the default value, if
            set.

    **Examples:**


    >>> f.set_property('project', 'CMIP7')
    >>> f.has_property('project')
    True
    >>> f.get_property('project')
    'CMIP7'
    >>> f.del_property('project')
    'CMIP7'
    >>> f.has_property('project')
    False
    >>> print(f.del_property('project', None))
    None
    >>> print(f.get_property('project', None))
    None

        '''
        # Get a special property
        if prop in self._special_properties:
            try:
                return getattr(self, prop)
            except AttributeError as err:
                return self._default(default, err)
        # --- End: if

        # Still here? Then get a non-special property
        return super().get_property(prop, default=default)

    def has_property(self, prop):
        '''Whether a property has been set.

    .. versionadded:: 3.0.0

    .. seealso:: `clear_properties`, `del_property`, `get_property`,
                 `properties`, `set_property`

    :Parameters:

        prop: `str`
            The name of the property.

            *Parameter example:*
               ``prop='long_name'``

    :Returns:

         `bool`
            `True` if the property has been set, otherwise `False`.

    **Examples:**

    >>> f.set_property('project', 'CMIP7')
    >>> f.has_property('project')
    True
    >>> f.get_property('project')
    'CMIP7'
    >>> f.del_property('project')
    'CMIP7'
    >>> f.has_property('project')
    False
    >>> print(f.del_property('project', None))
    None
    >>> print(f.get_property('project', None))
    None

        '''
        # Get a special property
        if prop in self._special_properties:
            return hasattr(self, prop)

        # Still here? Then get a non-special property
        return super().has_property(prop)

    def del_property(self, prop, default=ValueError()):
        '''Remove a property.

    .. versionadded:: 3.0.0

    .. seealso:: `clear_properties`, `get_property`, `has_property`,
                 `properties`, `set_property`

    :Parameters:

        prop: `str`
            The name of the property.

            *Parameter example:*
               ``prop='long_name'``

        default: optional
            Return the value of the *default* parameter if the
            property does not exist. If set to an `Exception` instance
            then it will be raised instead.

    :Returns:

            The removed property.

    **Examples:**

    >>> f.set_property('project', 'CMIP7')
    >>> f.has_property('project')
    True
    >>> f.get_property('project')
    'CMIP7'
    >>> f.del_property('project')
    'CMIP7'
    >>> f.has_property('project')
    False
    >>> print(f.del_property('project', None))
    None
    >>> print(f.get_property('project', None))
    None

        '''
        # Get a special attribute
        if prop in self._special_properties:
            try:
                out = getattr(self, prop)
            except AttributeError as err:
                return self._default(default, err)
            else:
                delattr(self, prop)
                return out
        # --- End: if

        # Still here? Then del a non-special attribute
        return super().del_property(prop, default=default)

    def match_by_identity(self, *identities):
        '''Whether or not the construct identity satisfies conditions.

    .. versionadded:: 3.0.0

    .. seealso:: `match`, `match_by_property`, `match_by_ncvar`

    :Parameters:

        identities: optional
            Define one or more conditions on the identities.

            A construct identity is specified by a string
            (e.g. ``'latitude'``, ``'long_name=time', ``'ncvar%lat'``,
            etc.); a `Query` object (e.g. ``cf.eq('longitude')``); or
            a compiled regular expression
            (e.g. ``re.compile('^atmosphere')``) that is compared with
            the construct's identities via `re.search`.

            A construct has a number of identities, and the condition
            is satisfied if any of the construct's identities, as
            returned by the `identities` method, equals the condition
            value. A construct's identities are those returned by its
            `!identities` method. In the following example, the
            construct ``x`` has six identities:

               >>> x.identities()
               ['time',
                'long_name=Time',
                'foo=bar',
                'standard_name=time',
                'ncvar%t',
                'T']

    :Returns:

        `bool`
            Whether or not at least one of the conditions are met.

    **Examples:**

    >>> f.match_by_identity('time')

    >>> f.match_by_identity(re.compile('^air'))

    >>> f.match_by_identity('air_pressure', 'air_temperature')

    >>> f.match_by_identity('ncvar%t')

        '''
        if not identities:
            return True

        self_identities = self.identities()

        ok = False
        for value0 in identities:
            for value1 in self_identities:
                ok = self._matching_values(value0, value1)
                if ok:
                    break
            # --- End: for

            if ok:
                break
        # --- End: for

        return ok

    def match_by_ncvar(self, *ncvars):
        '''Whether or not the netCDF variable name satisfies conditions.

    .. versionadded:: 3.0.0

    .. seealso:: `match`, `match_by_identity`, `match_by_property`

    :Parameters:

        ncvars: optional
            Define one or more conditions on the netCDF variable name.

            A netCDF variable name is specified by a string
            (e.g. ``'lat'``), etc.); a `Query` object
            (e.g. ``cf.eq('lon')``); or a compiled regular expression
            (e.g. ``re.compile('^t')``) that is compared with the
            construct's netCDF variable name via `re.search`.

            The condition is satisfied if the netCDF variable name, as
            returned by the `nc_get_variable` method, exists and
            equals the condition value.

    :Returns:

        `bool`
            Whether or not at least one of the conditions are met.

    **Examples:**

    >>> f.nc_get_variable()
    'time'
    >>> f.match_by_ncvar('time')
    True
    >>> f.match_by_ncvar('t', 'time')
    True
    >>> f.match_by_ncvar('t')
    False
    >>> f.match_by_ncvar()
    True
    >>> import re
    >>> f.match_by_ncvar(re.compile('^t'))
    True

        '''
        if not ncvars:
            return True

        try:
            ncvar = self.nc_get_variable(None)
        except AttributeError:
            ncvar = None

        if ncvar is None:
            return False

        ok = False
        for value0 in ncvars:
            ok = self._matching_values(value0, ncvar)
            if ok:
                break
        # --- End: for

        return ok

    def match_by_property(self, *mode, **properties):
        '''Whether or not properties satisfy conditions.

    .. versionadded:: 3.0.0

    .. seealso:: `match`, `match_by_identity`, `match_by_ncvar`

    :Parameters:

        mode: optional
            Define the behaviour when multiple conditions are
            provided.

            By default (or if the *mode* parameter is ``'and'``) the
            match is `True` if the field construct satisfies all of
            the given conditions, but if the *mode* parameter is
            ``'or'`` then the match is `True` when at least one of the
            conditions is satisfied.

        properties: optional
            Define conditions on properties.

            Properties are identified by the name of a keyword
            parameter.

            The keyword parameter value defines a condition to be
            applied to the property, and is either a string
            (e.g. ``'latitude'``); a `Query` object
            (e.g. ``cf.eq('longitude')``); or a compiled regular
            expression (e.g. ``re.compile('^atmosphere')``) that is
            compared with the property value via `re.search`.

            If the value is `None` then the condition is one of
            existence and is satisfied if the property exists,
            regardless of its value. Otherwise the condition is
            satisfied if the property equals the value.

            *Parameter example:*
              To see if there is a 'standard_name' property of 'time':
              ``standard_name='time'``.

            *Parameter example:*
              To see if there is a 'valid_min' property of that is
              less than -999: ``valid_min=cf.lt('-999)``.

            *Parameter example:*
              To see if there is a 'long_name' property with any
              value: ``long_name=None``.

    :Returns:

        `bool`
            Whether or not the conditions are met.

    **Examples:**

    >>> f.match_by_property(standard_name='longitude')

    >>> f.match_by_property(
    ...     standard_name='longitude', foo='cf.set(['bar', 'not_bar'])

    >>> f.match_by_property(long_name=re.compile('^lon'))

        '''
        _or = False

        # Parse mode
        if mode:
            if len(mode) > 1:
                raise ValueError(
                    "Can provide at most one positional argument")

            x = mode[0]
            if x == 'or':
                _or = True
            elif x != 'and':
                raise ValueError(
                    "Positional argument, if provided, must one of 'or', "
                    "'and'"
                )
        # --- End: if

        if not properties:
            return True

        self_properties = self.properties()

        ok = True
        for name, value0 in properties.items():
            if value0 is None:
                ok = (name in self_properties)
            else:
                value1 = self_properties.get(name)
                ok = self._matching_values(value0, value1,
                                           units=(name == 'units'))

            if _or:
                if ok:
                    break
            elif not ok:
                break
        # --- End: for

        return ok

    def properties(self):
        '''Return all properties.

    .. seealso:: `clear_properties`, `get_property`, `has_property`
                 `set_properties`

    :Returns:

        `dict`
            The properties.

    **Examples:**

    >>> f.properties()
    {}
    >>> f.set_properties(
    ...     {'standard_name': 'air_pressure', 'long_name': 'Air Pressure'})
    >>> f.properties()
    {'standard_name': 'air_pressure',
     'foo': 'bar',
     'long_name': 'Air Pressure'}
    >>> f.set_properties({'standard_name': 'air_pressure', 'foo': 'bar'})
    >>> f.properties()
    {'standard_name': 'air_pressure',
     'foo': 'bar',
     'long_name': 'Air Pressure'}
    >>> f.clear_properties()
    {'standard_name': 'air_pressure',
     'foo': 'bar',
     'long_name': 'Air Pressure'}
    >>> f.properties()
    {}

        '''
        out = super().properties()

        for prop in self._special_properties:
            value = getattr(self, prop, None)
            if value is None:
                out.pop(prop, None)
            else:
                out[prop] = value
        # --- End: for

        return out

    def set_properties(self, properties, copy=True):
        '''Set properties.

    .. versionadded:: 3.0.0

    .. seealso:: `clear_properties`, `properties`, `set_property`

    :Parameters:

        properties: `dict`
            Store the properties from the dictionary supplied.

            *Parameter example:*
              ``properties={'standard_name': 'altitude', 'foo': 'bar'}``

        copy: `bool`, optional
            If `False` then any property values provided by the
            *properties* parameter are not copied before insertion. By
            default they are deep copied.

    :Returns:

        `None`

    **Examples:**

    >>> f.properties()
    {}
    >>> f.set_properties(
    ...     {'standard_name': 'air_pressure', 'long_name': 'Air Pressure'})
    >>> f.properties()
    {'standard_name': 'air_pressure',
     'foo': 'bar',
     'long_name': 'Air Pressure'}
    >>> f.set_properties({'standard_name': 'air_pressure', 'foo': 'bar'})
    >>> f.properties()
    {'standard_name': 'air_pressure',
     'foo': 'bar',
     'long_name': 'Air Pressure'}
    >>> f.clear_properties()
    {'standard_name': 'air_pressure',
     'foo': 'bar',
     'long_name': 'Air Pressure'}
    >>> f.properties()
    {}

        '''
        super().set_properties(properties, copy=copy)

        for prop in self._special_properties:
            value = properties.get(prop)
            if value is not None:
                setattr(self, prop, value)

    def set_property(self, prop, value, copy=True):
        '''Set a property.

    .. versionadded:: 3.0.0

    .. seealso:: `clear_properties`, `del_property`, `get_property`,
                 `has_property`, `properties`

    :Parameters:

        prop: `str`
            The name of the property to be set.

        value:
            The value for the property.

        copy: `bool`, optional
            If `True` then set a deep copy of *value*.

    :Returns:

         `None`

    **Examples:**

    >>> f.set_property('project', 'CMIP7')
    >>> f.has_property('project')
    True
    >>> f.get_property('project')
    'CMIP7'
    >>> f.del_property('project')
    'CMIP7'
    >>> f.has_property('project')
    False
    >>> print(f.del_property('project', None))
    None
    >>> print(f.get_property('project', None))
    None

        '''
        # Get a special property
        if prop in self._special_properties:
            if copy:
                value = deepcopy(value)

            setattr(self, prop, value)
            return

        # Still here? Then set a non-special property
        return super().set_property(prop, value, copy=copy)

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def match(self, *identities, **kwargs):
        '''Alias for `match_by_identity`.

        '''
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, 'match', kwargs, "Use 'match_by_*' methods instead."
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
                    "Use 'match_by_*' methods instead.")  # pragma: no cover
        # --- End: for

        return self.match_by_identity(*identities)

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    def setprop(self, *args, **kwargs):
        '''Delete a CF property.

    Deprecated at version 3.0.0. Use method 'set_property' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'setprop', "Use method 'set_property' instead"
        )  # pragma: no cover

    def delprop(self, prop):
        '''Delete a CF property.

    Deprecated at version 3.0.0. Use method 'del_property' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'delprop', "Use method 'del_property' instead"
        )  # pragma: no cover

    def hasprop(self, prop):
        '''Whether a property has been set.

    Deprecated at version 3.0.0. Use method 'has_property' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'hasprop', "Use method 'has_property' instead"
        )  # pragma: no cover

    def getprop(self, prop):
        '''Get a property.

    Deprecated at version 3.0.0. Use method 'get_property' instead.

        '''
        _DEPRECATION_ERROR_METHOD(
            self, 'getprop', "Use method 'get_property' instead"
        )  # pragma: no cover


# --- End: class
