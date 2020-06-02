import logging

from operator import __and__ as operator_and
from operator import __or__  as operator_or

from .functions  import equals  as _equals
from .functions  import inspect as _inspect
from .units      import Units

from .data.data import Data

from .functions  import (_DEPRECATION_ERROR_FUNCTION_KWARGS,
                         _DEPRECATION_ERROR_ATTRIBUTE,
                         _DEPRECATION_ERROR_FUNCTION)

from .decorators import (_deprecated_kwarg_check,
                         _manage_log_level_via_verbosity)


logger = logging.getLogger(__name__)


class Query:
    '''Encapsulate a condition.

    A condition that may be applied to any object may be stored in a
    `Query` object. A `Query` object encapulates a condition, such as
    "strictly less than 3". When applied to an object, via its
    `evaluate` method or the Python `==` operator, the condition is
    evaulated in the context of that object.

       >>> c = cf.Query('lt', 3)
       >>> c
       <CF Query: (lt 3)>
       >>> c.evaluate(2)
       True
       >>> c == 2
       True
       >>> c != 2
       False
       >>> c.evaluate(3)
       False
       >>> c == cf.Data([1, 2, 3])
       <CF Data(3): [True, True, False]>
       >>> c == numpy.array([1, 2, 3])
       array([True, True, False])

    The following operators are supported when constructing `Query`
    instances:

    =========  ===================================
    Operator   Description
    =========  ===================================
    ``'lt'``   A "strictly less than" condition
    ``'le'``   A "less than or equal" condition
    ``'gt'``   A "strictly greater than" condition
    ``'ge'``   A "greater than or equal" condition
    ``'eq'``   An "equal" condition
    ``'ne'``   A "not equal" condition
    ``'wi'``   A "within a range" condition
    ``'wo'``   A "without a range" condition
    ``'set'``  A "member of set" condition
    =========  ===================================

    **Complex conditions**

    Multiple conditions may be combined with the Python bitwise "and"
    (`&`) and "or" (`|`) operators to form a new `Query` object.

       >>> ge3 = cf.Query('ge', 3)
       >>> lt5 = cf.Query('lt', 5)
       >>> c = ge3 & lt5
       >>> c
       >>> <CF Query: [(ge 3) & (lt 5)]>
       >>> c == 2
       False
       >>> c != 2
       True
       >>> c = ge3 | lt5
       >>> c
       <CF Query: [(ge 3) | (lt 5)]>
       >>> c == 2
       True
       >>> c &= cf.Query('set', [1, 3, 5])
       >>> c
       <CF Query: [[(ge 3) | (lt 5)] & (set [1, 3, 5])]>
       >>> c == 2
       False
       >>> c == 3
       True

    A condition can be applied to an attribute of an object.

       >>> upper_bounds_ge_minus4 = cf.Query('ge', -4, attr='upper_bounds')
       >>> x
       <CF DimensionCoordinate: grid_longitude(9) degrees>
       >>> print(x.bounds.array)
       [[-4.92 -4.48]
        [-4.48 -4.04]
        [-4.04 -3.6 ]
        [-3.6  -3.16]
        [-3.16 -2.72]
        [-2.72 -2.28]
        [-2.28 -1.84]
        [-1.84 -1.4 ]
        [-1.4  -0.96]]
       >>> print((upper_bounds_ge_minus4 == x).array)
       [False False  True  True  True  True  True  True  True]
       >>> upper_bounds_ge_minus4 = cf.Query('ge', -4, attr='upper_bounds')

    A condition can also be applied to attributes of attributes of an
    object.

       >>> t
       <CF DimensionCoordinate: time(4) >
       >>> t.lower_bounds.month.array
       array([12,  3,  6,  9])
       >>> c = cf.Query('ge', 8, attr='lower_bounds.month')
       >>> c == t
       <CF Data(4): [True, ..., True]>
       >>> (c == t).array
       array([ True,  False, False, True])


    **The query interface**

    In general, the query operator must be permitted between the value
    of the condition and the operand for which it is being
    evaulated. For example, when the value is an `int`, the query
    works if the operand is also an `int`, but fails if it is a
    `list`:

       >>> c = cf.Query('lt', 2)
       >>> c == 1
       True
       >>> c == [1, 2, 3]
       TypeError: '<' not supported between instances of 'list' and 'int'

    This behaviour is overridden if the operand has an appropriate
    "query interface" method. When such a method exists, it is used
    instead of the equivalent built-in Python operator.

    ======================  ==============================================
    Query interface method  Description
    ======================  ==============================================
    `__query_lt__`          Called when a ``'lt'`` condition is evaulated
    `__query_le__`          Called when a ``'le'`` condition is evaulated
    `__query_gt__`          Called when a ``'gt'`` condition is evaulated
    `__query_ge__`          Called when a ``'ge'`` condition is evaulated
    `__query_eq__`          Called when an ``'eq'`` condition is evaulated
    `__query_ne__`          Called when a ``'ne'`` condition is evaulated
    `__query_wi__`          Called when a ``'wi'`` condition is evaulated
    `__query_wo__`          Called when a ``'wo'`` condition is evaulated
    `__query_set__`         Called when a ``'set'`` condition is evaulated
    ======================  ==============================================

    In all cases the query value is the only, mandatory argument of
    the method.

       >>> class myList(list):
       ...     pass
       ...
       >>> class myList_with_override(list):
       ...     def __query_lt__(self, value):
       ...         """Apply the < operator element-wise"""
       ...         return type(self)([x < value for x in self])
       ...
       >>> c == myList([1, 2, 3])
       TypeError: '<' not supported between instances of 'myList' and 'int'
       >>> c == myList_with_override([1, 2, 3])
       [True, False, False]

    When the condition is on an attribute, or nested attributes, of
    the operand, the query interface method is looked for on the
    attribute object, rather than the parent object.

    If the value has units then the argument passed to query interface
    method is automatically a `Data` object that attaches the units to
    the value.

    '''
    isquery = True

    @_deprecated_kwarg_check('exact')
    def __init__(self, operator, value, units=None, attr=None, exact=True):
        '''**Initialization**

    :Parameters:

        operator: `str`
            The query operator.

        value:
            The value of the condition.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``. See
            also the `addattr` method.

        exact: deprecated at version 3.0.0.
            Use `re.compile` objects in *value* instead.

        '''
        if units is not None:
            value_units = getattr(value, 'Units', None)
            if value_units is None:
                value = Data(value, units)
            elif not value_units.equivalent(Units(units)):
                raise ValueError(
                    "'{}' and '{}' are not equivalent units therefore the "
                    "query does not make physical sense.".format(
                        value_units, Units(units))
                )
        # --- End: if

        self._operator = operator
        self._value = value
        self._compound = False

        if attr:
            self._attr = tuple(attr.split('.'))
        else:
            self._attr = ()

        self._bitwise_operator = None

        self._NotImplemented_RHS_Data_op = True

    def __deepcopy__(self, memo):
        '''Used if copy.deepcopy is called on the variable.

        '''
        return self.copy()

    def __eq__(self, x):
        '''The rich comparison operator ``==``

    x.__eq__(y) <==> x==y

    x.__eq__(y) <==> x.evaluate(y)

        '''
        return self._evaluate(x, ())

    def __ne__(self, x):
        '''The rich comparison operator ``!=``

    x.__ne__(y) <==> x!=y

    x.__ne__(y) <==> (x==y)==False

        '''
        # Note that it is important to use the == operator
        return self._evaluate(x, ()) == False  # ignore PEP8 E712 due to above

    def __and__(self, other):
        '''The binary bitwise operation ``&``

    x.__and__(y) <==> x&y

        '''
        Q = type(self)
        new = Q.__new__(Q)

        new._operator = None
        new._compound = (self, other)
        new._bitwise_operator = operator_and
        new._attr = ()

        new._NotImplemented_RHS_Data_op = True

        return new

    def __iand__(self, other):
        '''The augmented bitwise assignment ``&=``

    x.__iand__(y) <==> x&=y

        '''
        return self & other

    def __or__(self, other):
        '''The binary bitwise operation ``|``

    x.__or__(y) <==> x|y

        '''
        Q = type(self)
        new = Q.__new__(Q)

        new._operator = None
        new._compound = (self, other)
        new._bitwise_operator = operator_or
        new._attr = ()

        new._NotImplemented_RHS_Data_op = True

        return new

    def __ior__(self, other):
        '''The augmented bitwise assignment ``|=``

    x.__ior__(y) <==> x|=y

        '''
        return self | other

    def __repr__(self):
        '''Called by the `repr` built-in function.

    x.__repr__() <==> repr(x)

        '''
        return '<CF {}: {}>'.format(self.__class__.__name__, self)

    def __str__(self):
        '''Called by the `str` built-in function.

    x.__str__() <==> str(x)

        '''
        attr = '.'.join(self._attr)

        if not self._compound:
            out = '{}({} {})'.format(attr, self._operator, self._value)
        else:
            bitwise_operator = repr(self._bitwise_operator)
            if 'and_' in bitwise_operator:
                bitwise_operator = '&'
            elif 'or_' in bitwise_operator:
                bitwise_operator = '|'

            out = '{}[{} {} {}]'.format(attr, self._compound[0],
                                        bitwise_operator,
                                        self._compound[1])

        return out

    @property
    def attr(self):
        '''TODO

    **Examples:**

    >>> q = cf.Query('ge', 4)
    >>> print(q.attr)
    None
    >>> q = cf.Query('le', 6, attr='year')
    >>> q.attr
    'year'
    >>> q.addattr('foo')
    >>> q.attr
    'year'asdasdas TODO

        '''
        return self._attr

    @property
    def operator(self):
        '''TODO

    Compound conditions return `None`.

    **Examples:**

    >>> q = cf.Query('ge', 4)
    >>> q.operator
    'ge'
    >>> q |= cf.Query('le', 6)
    >>> print(q.operator)
    None

        '''
        return self._operator

    @property
    def value(self):
        '''TODO

    An exception is raised for compound conditions.

    **Examples:**

    >>> q = cf.Query('ge', 4)
    >>> q.value
    4
    >>> q |= cf.Query('le', 6)
    >>> q.value
    AttributeError: Compound query doesn't have attribute 'value'

        '''
        if not self._compound:
            return self._value

        raise AttributeError("Compound query doesn't have attribute 'value'")

    def addattr(self, attr):
        '''Return a `Query` object with a new left hand side operand attribute
    to be used during evaluation. TODO

    If another attribute has previously been specified, then the new
    attribute is considered to be an attribute of the existing
    attribute.

    :Parameters:

        attr: `str`
            The attribute name.

    :Returns:

        `Query`
            The new query object.

    **Examples:**

    >>> q = cf.eq(2001)
    >>> q
    <CF Query: (eq 2001)>
    >>> q = q.addattr('year')
    >>> q
    <CF Query: year(eq 2001)>

    >>> q = cf.lt(2)
    >>> q = q.addattr('A')
    >>> q = q.addattr('B')
    >>> q
    <CF Query: A.B(lt 2)>
    >>> q = q.addattr('C')
    >>> q
    <CF Query: A.B.C(lt 2)>

        '''
        Q = type(self)
        new = Q.__new__(Q)

        new.__dict__ = self.__dict__.copy()
        new._attr += (attr,)

        new._NotImplemented_RHS_Data_op = True

        return new

    def copy(self):
        '''Return a deep copy.

    ``q.copy()`` is equivalent to ``copy.deepcopy(q)``.

    :Returns:

            The deep copy.

    **Examples:**

    >>> r = q.copy()

        '''
        return self  # TODO

    def dump(self, display=True):
        '''Return a string containing a full description of the instance.

    :Parameters:

        display: `bool`, optional

            If `False` then return the description as a string. By
            default the description is printed.

    :Returns:

        `None` or `str`
            The description. If *display* is True then the description
            is printed and `None` is returned. Otherwise the
            description is returned as a string.

        '''
        string = str(self)

        if display:
            print(string)
        else:
            return(string)

    @_deprecated_kwarg_check('traceback')
    @_manage_log_level_via_verbosity
    def equals(self, other, verbose=None, traceback=False):
        '''TODO

        '''
        if self._compound:
            if not other._compound:
                logger.info(
                    "{}: Different compound components".format(
                        self.__class__.__name__)
                )  # pragma: no cover
                return False

            if self._bitwise_operator != other._bitwise_operator:
                logger.info(
                    "{}: Different compound operators: {!r}, {!r}".format(
                        self.__class__.__name__, self._bitwise_operator,
                        other._bitwise_operator
                    )
                )  # pragma: no cover
                return False

            if not self._compound[0].equals(other._compound[0]):
                if not self._compound[0].equals(other._compound[1]):
                    logger.info(
                        "{}: Different compound components".format(
                            self.__class__.__name__)
                    )  # pragma: no cover
                    return False
                if not self._compound[1].equals(other._compound[0]):
                    logger.info(
                        "{}: Different compound components".format(
                            self.__class__.__name__)
                    )  # pragma: no cover
                    return False
            elif not self._compound[1].equals(other._compound[1]):
                logger.info(
                    "{}: Different compound components".format(
                        self.__class__.__name__)
                )  # pragma: no cover
                return False

        elif other._compound:
            logger.info(
                "{}: Different compound components".format(
                    self.__class__.__name__)
            )  # pragma: no cover
            return False

        for attr in ('_NotImplemented_RHS_Data_op',
                     '_attr',
                     '_value',
                     '_operator'):
            if not _equals(getattr(self, attr, None),
                           getattr(other, attr, None),
                           verbose=verbose):
                logger.info(
                    "{}: Different {!r} attributes: {!r}, {!r}".format(
                        self.__class__.__name__, attr,
                        getattr(self, attr, None),
                        getattr(other, attr, None)
                    )
                )  # pragma: no cover
                return False
        # --- End: for

        return True

    def evaluate(self, x):
        '''Evaluate the query operation for a given left hand side operand.

    Note that for the query object ``q`` and any object, ``x``,
    ``x==q`` is equivalent to ``q.evaluate(x)`` and ``x!=q`` is
    equivalent to ``q.evaluate(x)==False``.

    :Parameters:

        x:
            The object for the left hand side operand of the query.

    :Returns:

            The result of the query. The nature of the result is
            dependent on the object type of *x*.

    **Examples:**

    >>> q = cf.Query('lt', 5.5)
    >>> q.evaluate(6)
    False

    >>> q = cf.Query('wi', (1,2))
    >>> array = numpy.arange(4)
    >>> array
    array([0, 1, 2, 3])
    >>> q.evaluate(array)
    array([False,  True,  True, False], dtype=bool)

        '''
        return self._evaluate(x, ())

    def _evaluate(self, x, parent_attr):
        '''Evaluate the query operation for a given object.

    .. seealso:: `evaluate`

    :Parameters:

        x:
            See `evaluate`.

        parent_attr: `tuple`

    :Returns:

        See `evaluate`.

        '''
        compound = self._compound
        attr = parent_attr + self._attr

        # ------------------------------------------------------------
        # Evaluate a compound condition
        # ------------------------------------------------------------
        if compound:
            c = compound[0]._evaluate(x, attr)
            d = compound[1]._evaluate(x, attr)
            return self._bitwise_operator(c, d)

        # ------------------------------------------------------------
        # Still here? Then evaluate a simple (non-compoundnd)
        # condition.
        # ------------------------------------------------------------
        for a in attr:
            x = getattr(x, a)

        operator = self._operator
        value = self._value

        if operator == 'eq':
            try:
                return bool(value.search(x))
            except AttributeError:
                return x == value
            except TypeError:
                raise ValueError(
                    "Can't perform regular expression search on a "
                    "non-string: {!r}".format(x)
                )
        # --- End: if

        if operator == 'ne':
            try:
                return not bool(value.search(x))
            except AttributeError:
                return x != value
            except TypeError:
                raise ValueError(
                    "Can't perform regular expression search on a "
                    "non-string: {!r}".format(x)
                )
        # --- End: if

        if operator == 'lt':
            _lt = getattr(x, '__query_lt__', None)
            if _lt is not None:
                return _lt(value)

            return x < value

        if operator == 'le':
            _le = getattr(x, '__query_le__', None)
            if _le is not None:
                return _le(value)

            return x <= value

        if operator == 'gt':
            _gt = getattr(x, '__query_gt__', None)
            if _gt is not None:
                return _gt(value)

            return x > value

        if operator == 'ge':
            _ge = getattr(x, '__query_ge_', None)
            if _ge is not None:
                return _ge(value)

            return x >= value

        if operator == 'wi':
            _wi = getattr(x, '__query_wi__', None)
            if _wi is not None:
                return _wi(value)

            return (x >= value[0]) & (x <= value[1])

        if operator == 'wo':
            _wo = getattr(x, '__query_wo__', None)
            if _wo is not None:
                return _wo(value)

            return (x < value[0]) | (x > value[1])

#        if operator == 'contains':
#            _contain = getattr(x, '__query_contains__', None)
#            if _contain is not None:
#                return _contain(value)
#            else:
#                return x == value
#        # --- End: if

        if operator == 'set':
            if isinstance(x, str):
                for v in value:
                    try:
                        if v.search(x):
                            return True
                    except AttributeError:
                        if x == v:
                            return True
                # --- End: for

                return False
            else:
                _set = getattr(x, '__query_set__', None)
                if _set is not None:
                    return _set(value)

                i = iter(value)
                v = next(i)
                out = (x == v)
                for v in i:
                    out |= (x == v)

                return out
        # --- End: if

    def inspect(self):
        '''Inspect the object for debugging.

    .. seealso:: `cf.inspect`

    :Returns:

        `None`

        '''
        print(_inspect(self))  # pragma: no cover

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def exact(self):
        '''TODO Deprecated at version 3.0.0. Use re.compile objects instead.

        '''
        _DEPRECATION_ERROR_ATTRIBUTE(
            self, 'exact', "Use 're.compile' objects instead."
        )  # pragma: no cover

    def equivalent(self, other, traceback=False):
        '''Deprecated at version 3.0.0.

        '''
        _DEPRECATION_ERROR_FUNCTION(self, 'equivalent')


# --- End: class


# --------------------------------------------------------------------
# Constructor functions
# --------------------------------------------------------------------
def lt(value, units=None, attr=None):
    '''A `Query` object for a "strictly less than" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.le`, `cf.set`, `cf.wi`, `cf.wo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> c = cf.lt(5)
    >>> c == 2
    True
    >>> c == cf.Data(2, 'metres')
    <CF Data(): True>

    >>> c = cf.lt(5, 'metres')
    >>> c == 2
    True
    >>> c == cf.Data(50, 'centimetres')
    <CF Data(): True>

    >>> c = cf.lt(cf.Data(5, 'metres'))
    >>> c == 2
    True
    >>> c == cf.Data(50, 'centimetres')
    <CF Data(): True>

    >>> import numpy
    >>> c = cf.lt(numpy.array([2, 5]), attr='shape')
    >>> c
    <CF Query: shape(lt [2 5])>
    >>> c == numpy.arange(9).reshape(1, 9)
    array([ True, False])

    '''
    return Query('lt', value, units=units, attr=attr)


def le(value, units=None, attr=None):
    '''A `Query` object for a "less than or equal" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> q = cf.le(5)
    >>> q
    <CF Query: x le 5>
    >>> q.evaluate(5)
    True
    >>> q.evaluate(6)
    False

    '''
    return Query('le', value, units=units, attr=attr)


def gt(value, units=None, attr=None):
    '''A `Query` object for a "strictly greater than" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.ne`, `cf.le`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional

            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> q = cf.gt(5)
    >>> q
    <CF Query: x gt 5>
    >>> q.evaluate(6)
    True
    >>> q.evaluate(5)
    False

    '''
    return Query('gt', value, units=units, attr=attr)


def ge(value, units=None, attr=None):
    '''A `Query` object for a "greater than or equal" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.gt`, `cf.ne`, `cf.le`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> q = cf.ge(5)
    >>> q
    <CF Query: x ge 5>
    >>> q.evaluate(5)
    True
    >>> q.evaluate(4)
    False

    >>> cf.ge(10, 'm')
    <CF Query: (ge <CF Data: 10 m>)>
    >>> cf.ge(100, units=Units('kg'))
    <CF Query: (ge <CF Data: 100 kg>)>

    >>> cf.ge(2, attr='month')
    <CF Query: month(ge 2)>

    '''
    return Query('ge', value, units=units, attr=attr)


def eq(value, units=None, attr=None, exact=True):
    '''A `Query` object for an "equal" condition.

    .. seealso:: `cf.contains`, `cf.ge`, `cf.gt`, `cf.ne`, `cf.le`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

        exact: deprecated at version 3.0.0.
            Use `re.compile` objects in *value* instead.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> q = cf.eq(5)
    >>> q
    <CF Query: x eq 5>
    >>> q.evaluate(5)
    True
    >>> q == 4
    False

    >>> q = cf.eq('air', exact=False)
    >>> q == 'air_temperature'
    True

    >>> q = cf.eq('.*temp', exact=False)
    >>> q == 'air_temperature'
    True

    '''
    if not exact:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'eq', exact=True)  # pragma: no cover

    return Query('eq', value, units=units, attr=attr)


def ne(value, units=None, attr=None, exact=True):
    '''A `Query` object for a "not equal" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.le`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

        exact: deprecated at version 3.0.0.
            Use `re.compile` objects in *value* instead.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> q = cf.ne(5)
    >>> q
    <CF Query: x ne 5>
    >>> q.evaluate(4)
    True
    >>> q.evaluate(5)
    False

    '''
    if not exact:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'ne', exact=True)  # pragma: no cover

    return Query('ne', value, units=units, attr=attr)


def wi(value0, value1, units=None, attr=None):
    '''A `Query` object for a "within a range" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.le`, `cf.lt`, `cf.set`, `cf.wo`

    :Parameters:

        value0:
             The lower bound of the range.

        value1:
             The upper bound of the range.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> q = cf.wi(5, 7)
    >>> q
    <CF Query: wi (5, 7)>
    >>> q.evaluate(6)
    True
    >>> q.evaluate(4)
    False

    '''
    return Query('wi', [value0, value1], units=units, attr=attr)


def wo(value0, value1, units=None, attr=None):
    '''A `Query` object for a "without a range" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.le`, `cf.lt`, `cf.set`, `cf.wi`

    :Parameters:

        value0:
             The lower bound of the range.

        value1:
             The upper bound of the range.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> q = cf.wo(5)
    >>> q
    <CF Query: x wo (5, 7)>
    >>> q.evaluate(4)
    True
    >>> q.evaluate(6)
    False

    '''
    return Query('wo', [value0, value1], units=units, attr=attr)


def set(values, units=None, attr=None, exact=True):
    '''A `Query` object for a "member of set" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.le`, `cf.lt`, `cf.wi`, `cf.wo`

    :Parameters:

        value: sequence
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

        exact: deprecated at version 3.0.0.
            Use `re.compile` objects in *value* instead.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> c = cf.set([3, 5])
    >>> c == 4
    False
    >>> c == 5
    True
    >>> c == numpy.array([2, 3, 4, 5])
    array([False  True False  True])

    '''
    if not exact:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'set', exact=True)  # pragma: no cover

    return Query('set', values, units=units, attr=attr)


def contains(value, units=None):
    '''A `Query` object for a "cell contains" condition.

    .. versionadded:: 3.0.0

    .. seealso:: `cf.cellsize`, `cf.cellge`, `cf.cellgt`, `cf.cellne`,
                 `cf.cellle`, `cf.celllt`, `cf.cellwi`, `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> cf.contains(8)
    <CF Query: [lower_bounds(le 8) & upper_bounds(ge 8)]>
    >>> cf.contains(30, 'degrees_east')
    <CF Query: [lower_bounds(le 30 degrees_east) & upper_bounds(ge 30 degrees_east)]>
    >>>  cf.contains(cf.Data(10, 'km'))
    <CF Query: [lower_bounds(le 10 km) & upper_bounds(ge 10 km)]>

    >>> c
    <CF DimensionCoordinate: longitude(4) degrees_east>
    >>> print(c.bounds.array)
    [[  0   90]
     [ 90  180]
     [180  270]
     [270  360]]
    >>> print((cf.contains(100) == c).array)
    [False True False False]
    >>> print((cf.contains(9999) == c).array)
    [False False False False]

    '''
    return (Query('le', value, units=units, attr='lower_bounds') &
            Query('ge', value, units=units, attr='upper_bounds'))


def year(value):
    '''A `Query` object for a "year" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.djf`,
                 `cf.mam`, `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> d = cf.dt(2002, 6, 16)
    >>> d == cf.year(2002)
    True
    >>> d == cf.year(cf.le(2003))
    True
    >>> d == cf.year(2001)
    False
    >>> d == cf.year(cf.wi(2003, 2006))
    False

    '''
    if isinstance(value, Query):
        return value.addattr('year')
    else:
        return Query('eq', value, attr='year')


def month(value):
    '''A `Query` object for a "month of the year" condition.

    .. seealso:: `cf.year`, `cf.day`, `cf.hour`, `cf.minute`,
                 `cf.second`, `cf.seasons`, `cf.djf`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> d = cf.dt(2002, 6, 16)
    >>> d == cf.month(6)
    True
    >>> d == cf.month(cf.le(7))
    True
    >>> d == cf.month(7)
    False
    >>> d == cf.month(cf.wi(1, 6))
    True

    '''
    if isinstance(value, Query):
        return value.addattr('month')
    else:
        return Query('eq', value, attr='month')


def day(value):
    '''A `Query` object for a "day of the month" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.hour`, `cf.minute`,
                 `cf.second`, `cf.seasons`, `cf.djf`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> d = cf.dt(2002, 6, 16)
    >>> d == cf.day(16)
    True
    >>> d == cf.day(cf.le(19))
    True
    >>> d == cf.day(7)
    False
    >>> d == cf.day(cf.wi(1, 21))
    True

    '''
    if isinstance(value, Query):
        return value.addattr('day')
    else:
        return Query('eq', value, attr='day')


def hour(value):
    '''A `Query` object for a "hour of the day" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.minute`,
                 `cf.second`, `cf.seasons`, `cf.djf`, `cf.mam`, `cf.jja`,
                 `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    In this context, any object which has a `!hour` attribute is
    considered to be a date-time variable.

    If *value* is a `Query` object then ``cf.hour(value)`` is
    equivalent to ``value.addattr('hour')``. Otherwise
    ``cf.hour(value)`` is equivalent to ``cf.eq(value, attr='hour')``.

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.minute`,
                 `cf.second`

    :Parameters:

        value:
           Either the value that the hour is to be compared with, or a
           `Query` object for testing the hour.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> d = cf.dt(2002, 6, 16, 18)
    >>> d == cf.hour(18)
    True
    >>> d == cf.hour(cf.le(19))
    True
    >>> d == cf.hour(7)
    False
    >>> d == cf.hour(cf.wi(6, 23))
    True

    '''
    if isinstance(value, Query):
        return value.addattr('hour')
    else:
        return Query('eq', value, attr='hour')


def minute(value):
    '''A `Query` object for a "minute of the hour" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.second`, `cf.seasons`, `cf.djf`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> d = cf.dt(2002, 6, 16, 18, 30, 0)
    >>> d == cf.minute(30)
    True
    >>> d == cf.minute(cf.le(45))
    True
    >>> d == cf.minute(7)
    False
    >>> d == cf.minute(cf.wi(15, 45))
    True

    '''
    if isinstance(value, Query):
        return value.addattr('minute')
    else:
        return Query('eq', value, attr='minute')


def second(value):
    '''A `Query` object for a "second of the minute" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.seasons`, `cf.djf`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> d = cf.dt(2002, 6, 16, 18, 30, 0)
    >>> d == cf.second(0)
    True
    >>> d == cf.second(cf.le(30))
    True
    >>> d == cf.second(30)
    False
    >>> d == cf.second(cf.wi(0, 30))
    True

    '''
    if isinstance(value, Query):
        return value.addattr('second')
    else:
        return Query('eq', value, attr='second')


def cellsize(value, units=None):
    '''A `Query` object for a "cell size" condition.

    .. seealso:: `cf.contains`, `cf.cellge`, `cf.cellgt`, `cf.cellne`,
                 `cf.cellle`, `cf.celllt`, `cf.cellwi`, `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> cf.cellsize(cf.lt(5, 'km'))
    <CF Query: cellsize(lt <CF Data: 5 km>)>
    >>> cf.cellsize(5)
    <CF Query: cellsize(eq 5)>
    >>> cf.cellsize(cf.Data(5, 'km'))
    <CF Query: cellsize(eq <CF Data: 5 km>)>
    >>> cf.cellsize(cf.Data(5, 'km'))
    <CF Query: cellsize(eq <CF Data: 5 km>)>
    >>> cf.cellsize(5, units='km')
    <CF Query: cellsize(eq <CF Data: 5 km>)>

    '''
    if isinstance(value, Query):
        return value.addattr('cellsize')
    else:
        return Query('eq', value, units=units, attr='cellsize')


def cellwi(value0, value1, units=None):
    '''A `Query` object for a "cell bounds lie within range" condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.cellle`, `cf.celllt`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    TODO

    '''
    return (Query('ge', value0, units=units, attr='lower_bounds') &
            Query('le', value1, units=units, attr='upper_bounds'))


def cellwo(value0, value1, units=None):
    '''A `Query` object for a "cell bounds lie without range" condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.cellle`, `cf.celllt`,
                 `cf.cellwi`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    TODO

    '''
    return (Query('lt', value0, units=units, attr='lower_bounds') &
            Query('gt', value1, units=units, attr='upper_bounds'))


def cellgt(value, units=None):
    '''A `Query` object for a "cell bounds strictly greater than"
    condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`
                 `cf.cellne`, `cf.cellle`, `cf.celllt`, `cf.cellwi`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    TODO

    '''
    return Query('gt', value, units=units, attr='lower_bounds')


def cellge(value, units=None):
    '''A `Query` object for a "cell bounds greater than or equal"
    condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellgt`,
                 `cf.cellne`, `cf.cellle`, `cf.celllt`, `cf.cellwi`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.


    :Returns:

        `Query`
            The query object.

    **Examples:**

    TODO

    '''
    return Query('ge', value, units=units, attr='lower_bounds')


def celllt(value, units=None):
    '''A `Query` object for a “cell bounds strictly less than” condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.cellle`, `cf.cellwi`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    TODO

    '''
    return Query('lt', value, units=units, attr='upper_bounds')


def cellle(value, units=None):
    '''A `Query` object for a "cell bounds less than or equal" condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.celllt`, `cf.cellwi`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the operand
            being tested are assumed, if applicable. If *units* is
            specified and *value* already has units (such as those attached
            to a `Data` object), then the pair of units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples:**

    TODO

    '''
    return Query('le', value, units=units, attr='upper_bounds')


def jja():
    '''A `Query` object for a "month of year in June, July or August"
    condition.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.djf`,
                 `cf.mam`, `cf.son`

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> f
    <CF Field: air_temperature(time(365), latitude(64), longitude(128)) K>
    >>> f.subspace(time=cf.jja())
    <CF Field: air_temperature(time(92), latitude(64), longitude(128)) K>

    '''
    return Query('wi', (6, 8), attr='month')


def son():
    '''A `Query` object for a "month of year in September, October,
    November" condition.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.djf`,
                 `cf.mam`, `cf.jja`

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> f
    <CF Field: air_temperature(time(365), latitude(64), longitude(128)) K>
    >>> f.subspace(time=cf.son())
    <CF Field: air_temperature(time(91), latitude(64), longitude(128)) K>

    '''
    return Query('wi', (9, 11), attr='month')


def djf():
    '''A `Query` object for a "month of year in December, January,
    February" condition.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> f
    <CF Field: air_temperature(time(365), latitude(64), longitude(128)) K>
    >>> f.subspace(time=cf.djf())
    <CF Field: air_temperature(time(90), latitude(64), longitude(128)) K>

    '''
    q = Query('ge', 12) | Query('le', 2)
    return q.addattr('month')


def mam():
    '''A `Query` object for a "month of year in March, April, May"
    condition.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.djf`,
                 `cf.jja`, `cf.son`

    :Returns:

        `Query`
            The query object.

    **Examples:**

    >>> f
    <CF Field: air_temperature(time(365), latitude(64), longitude(128)) K>
    >>> f.subspace(time=cf.mam())
    <CF Field: air_temperature(time(92), latitude(64), longitude(128)) K>

    '''
    return Query('wi', (3, 5), attr='month')


def seasons(n=4, start=12):
    '''A customizable list of `Query` objects for "seasons in a year"
    conditions.

    Note that any date-time that lies within a particular season will
    satisfy that query.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`, `cf.minute`,
                 `cf.second`, `cf.djf`, `cf.mam`, `cf.jja`, `cf.son`

    TODO

    .. seealso:: `cf.mam`, `cf.jja`, `cf.son`, `cf.djf`

    :Parameters:

        n: `int`, optional
            The number of seasons in the year. By default there are
            four seasons.

        start: `int`, optional
            The start month of the first season of the year. By
            default this is 12 (December).

    :Returns:

        `list` of `Query`
            The query objects.

    **Examples:**

    >>> cf.seasons()
    [<CF Query: month[(ge 12) | (le 2)]>,
     <CF Query: month(wi (3, 5))>,
     <CF Query: month(wi (6, 8))>,
     <CF Query: month(wi (9, 11))>]

    >>> cf.seasons(4, 1)
    [<CF Query: month(wi (1, 3))>,
     <CF Query: month(wi (4, 6))>,
     <CF Query: month(wi (7, 9))>,
     <CF Query: month(wi (10, 12))>]

    >>> cf.seasons(3, 6)
    [<CF Query: month(wi (6, 9))>,
     <CF Query: month[(ge 10) | (le 1)]>,
     <CF Query: month(wi (2, 5))>]

    >>> cf.seasons(3)
    [<CF Query: month[(ge 12) | (le 3)]>,
     <CF Query: month(wi (4, 7))>,
     <CF Query: month(wi (8, 11))>]

    >>> cf.seasons(3, 6)
    [<CF Query: month(wi (6, 9))>,
     <CF Query: month[(ge 10) | (le 1)]>,
     <CF Query: month(wi (2, 5))>]

    >>> cf.seasons(12)
    [<CF Query: month(eq 12)>,
     <CF Query: month(eq 1)>,
     <CF Query: month(eq 2)>,
     <CF Query: month(eq 3)>,
     <CF Query: month(eq 4)>,
     <CF Query: month(eq 5)>,
     <CF Query: month(eq 6)>,
     <CF Query: month(eq 7)>,
     <CF Query: month(eq 8)>,
     <CF Query: month(eq 9)>,
     <CF Query: month(eq 10)>,
     <CF Query: month(eq 11)>]

    >>> cf.seasons(1, 4)
    [<CF Query: month[(ge 4) | (le 3)]>]

    '''
    if 12 % n:
        raise ValueError(
            "Number of seasons must divide into 12. Got %s" % n)

    if not 1 <= start <= 12 or int(start) != start:
        raise ValueError(
            "Start month must be integer between 1 and 12. Got %s" % start)

    out = []

    inc = int(12 / n)

    start = int(start)

    m0 = start
    for i in range(int(n)):
        m1 = ((m0 + inc) % 12) - 1
        if not m1:
            m1 = 12
        elif m1 == -1:
            m1 = 11

        if m0 < m1:
            q = Query('wi', (m0, m1))
        elif m0 > m1:
            q = Query('ge', m0) | Query('le', m1)
        else:
            q = Query('eq', m0)

        out.append(q.addattr('month'))

        m0 = m1 + 1
        if m0 > 12:
            m0 = 1
    # --- End: for

    return out


# --------------------------------------------------------------------
# Deprecated functions
# --------------------------------------------------------------------
def dtge(*args, **kwargs):
    '''Return a `Query` object for a variable being not earlier than a
    date-time.

    Deprecated at version 3.0.0. Use 'cf.ge' with a datetime object
    value instead.

    '''
    _DEPRECATION_ERROR_FUNCTION(
        'dtge', "Use 'cf.ge' with a datetime object value instead."
    )  # pragma: no cover


def dtgt(*args, **kwargs):
    '''Deprecated at version 3.0.0. Use 'cf.gt' with a datetime object
    value instead.

    '''
    _DEPRECATION_ERROR_FUNCTION(
        'dtgt', "Use 'cf.gt' with a datetime object value instead."
    )  # pragma: no cover


def dtle(*args, **kwargs):
    '''Deprecated at version 3.0.0. Use 'cf.le' with a datetime object
    value instead.

    '''
    _DEPRECATION_ERROR_FUNCTION(
        'dtle', "Use 'cf.le' with a datetime object value instead."
    )  # pragma: no cover


def dtlt(*args, **kwargs):
    '''Deprecated at version 3.0.0. Use 'cf.lt' with a datetime object
    value instead.

    '''
    _DEPRECATION_ERROR_FUNCTION(
        'dtlt', "Use 'cf.lt' with a datetime object value instead."
    )  # pragma: no cover


def dteq(*args, **kwargs):
    '''Deprecated at version 3.0.0. Use 'cf.eq' with a datetime object
    value instead.

    '''
    _DEPRECATION_ERROR_FUNCTION(
        'dteq', "Use 'cf.eq' with a datetime object value instead."
    )  # pragma: no cover


def dtne(*args, **kwargs):
    '''Deprecated at version 3.0.0. Use 'cf.ne' with a datetime object
    value instead.

    '''
    _DEPRECATION_ERROR_FUNCTION(
        'dtne', "Use 'cf.ne' with a datetime object value instead."
    )  # pragma: no cover


def contain(value, units=None, attr=None):
    '''Return a `Query` object for coordinate cells containing a value.

    Deprecated at version 3.0.0. Use function 'cf.contains' instead.

    '''
    _DEPRECATION_ERROR_FUNCTION(
        'cf.contain',
        "Use function 'cf.contains' instead")  # pragma: no cover
