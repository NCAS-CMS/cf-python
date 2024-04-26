import logging
from copy import deepcopy
from operator import __and__ as operator_and
from operator import __or__ as operator_or

import numpy as np
from cfdm import is_log_level_info

from .data import Data
from .decorators import (
    _deprecated_kwarg_check,
    _display_or_return,
    _manage_log_level_via_verbosity,
)
from .functions import (
    _DEPRECATION_ERROR_ATTRIBUTE,
    _DEPRECATION_ERROR_FUNCTION,
    _DEPRECATION_ERROR_FUNCTION_KWARGS,
)
from .functions import atol as cf_atol
from .functions import equals as cf_equals
from .functions import inspect as cf_inspect
from .functions import rtol as cf_rtol
from .units import Units

logger = logging.getLogger(__name__)

# Alias for builtin set, since there is a 'set' function
builtin_set = set


class Query:
    """Encapsulate a condition for subsequent evaluation.

    A condition that may be applied to any object may be stored in a
    `Query` object. A `Query` object encapsulates a condition, such as
    "strictly less than 3". When applied to an object, via its
    `evaluate` method or the Python `==` operator, the condition is
    evaluated in the context of that object.

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

    =============  ===================================
    Operator       Description
    =============  ===================================
    ``'lt'``       A "strictly less than" condition
    ``'le'``       A "less than or equal" condition
    ``'gt'``       A "strictly greater than" condition
    ``'ge'``       A "greater than or equal" condition
    ``'eq'``       An "equal" condition
    ``'ne'``       A "not equal" condition
    ``'wi'``       A "within a range" condition
    ``'wo'``       A "without a range" condition
    ``'set'``      A "member of set" condition
    ``'isclose'``  An "is close" condition
    =============  ===================================

    **Compound queries**

    Multiple conditions may be combined with the Python bitwise "and"
    (`&`) and "or" (`|`) operators to form a new `Query` object.

       >>> ge3 = cf.Query('ge', 3)
       >>> lt5 = cf.Query('lt', 5)
       >>> c = ge3 & lt5
       >>> c
       <CF Query: [(ge 3) & (lt 5)]>
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
    evaluated. For example, when the value is an `int`, the query
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
    `__query_lt__`          Called when a ``'lt'`` condition is evaluated
    `__query_le__`          Called when a ``'le'`` condition is evaluated
    `__query_gt__`          Called when a ``'gt'`` condition is evaluated
    `__query_ge__`          Called when a ``'ge'`` condition is evaluated
    `__query_eq__`          Called when an ``'eq'`` condition is evaluated
    `__query_ne__`          Called when a ``'ne'`` condition is evaluated
    `__query_wi__`          Called when a ``'wi'`` condition is evaluated
    `__query_wo__`          Called when a ``'wo'`` condition is evaluated
    `__query_set__`         Called when a ``'set'`` condition is evaluated
    `__query_isclose__`     Called when an ``'isclose'`` condition is
                            evaluated.
    ======================  ==============================================

    In general, each method must have the query value as it's only
    parameter. The only exception is for `__query_isclose__`, which
    also requires the absolute and relative numerical tolerances to be
    provided.

    When the condition is on an attribute, or nested attributes, of
    the operand, the query interface method is looked for on the
    attribute object, rather than the parent object.

    If the value has units then the argument passed to query interface
    method is automatically a `Data` object that attaches the units to
    the value.

    For example:

       >>> class myList(list):
       ...     pass
       ...
       >>> class myList_with_interface(list):
       ...     def __query_lt__(self, value):
       ...         return type(self)([x < value for x in self])
       ...
       >>> c == myList([1, 2, 3])
       TypeError: '<' not supported between instances of 'myList' and 'int'
       >>> c == myList_with_interface([1, 2, 3])
       [True, False, False]

    """

    isquery = True

    @_deprecated_kwarg_check("exact", version="3.0.0", removed_at="4.0.0")
    def __init__(
        self,
        operator,
        value,
        units=None,
        attr=None,
        exact=True,
        rtol=None,
        atol=None,
        open_lower=False,
        open_upper=False,
    ):
        """**Initialisation**

        :Parameters:

            operator: `str`
                The query operator.

            value:
                The value of the condition.

            units: `str` or `Units`, optional
                The units of *value*. By default, the same units as
                the operand being tested are assumed, if
                applicable. If *units* is specified and *value*
                already has units (such as those attached to a `Data`
                object), then the pair of units must be equivalent.

            attr: `str`, optional
                Apply the condition to the attribute, or nested
                attributes, of the operand, rather than the operand
                itself. Nested attributes are specified by separating
                them with a ``.``. For example, the "month" attribute
                of the "bounds" attribute is specified as
                ``'bounds.month'``. See also the `addattr` method.

            rtol: number, optional
                Only applicable to the ``'isclose'`` operator. The
                tolerance on relative numerical differences. If
                `None`, the default, then the value returned by
                `cf.rtol` is used at evaluation time.

                .. versionadded:: 3.15.2

            atol: number, optional
                Only applicable to the ``'isclose'`` operator. The
                tolerance on absolute numerical differences. If
                `None`, the default, then the value returned by
                `cf.atol` is used at evaluation time.

                .. versionadded:: 3.15.2

            open_lower: `bool`, optional
                Only applicable to the ``'wi'`` operator.
                If True, open the interval at the lower
                bound so that value0 is excluded from the
                range. By default the interval is closed
                so that value0 is included.

                .. versionadded:: 3.16.2

            open_upper: `bool`, optional
                Only applicable to the ``'wi'`` operator.
                If True, open the interval at the upper
                bound so that value1 is excluded from the
                range. By default the interval is closed
                so that value1 is included.

                .. versionadded:: 3.16.2

            exact: deprecated at version 3.0.0.
                Use `re.compile` objects in *value* instead.

        """
        if units is not None:
            value_units = getattr(value, "Units", None)
            if value_units is None:
                value = Data(value, units)
            elif not value_units.equivalent(Units(units)):
                raise ValueError(
                    f"'{value_units}' and '{Units(units)}' are not "
                    f"equivalent units therefore the query does not make "
                    "physical sense."
                )

        self._operator = operator
        self._value = value
        self._compound = False

        if attr:
            self._attr = tuple(attr.split("."))
        else:
            self._attr = ()

        self._bitwise_operator = None

        self.query_type = operator

        self._NotImplemented_RHS_Data_op = True

        if rtol is not None or atol is not None:
            if operator != "isclose":
                raise ValueError(
                    "Can only set the 'rtol' and 'atol' parameters "
                    "for the 'isclose' operator"
                )

            self._rtol = rtol
            self._atol = atol

        if open_lower or open_upper:
            if operator != "wi":
                raise ValueError(
                    "Can only set the 'open_lower' and 'open_upper' "
                    "parameters for the 'wi' operator"
                )

            self._open_lower = open_lower
            self._open_upper = open_upper

    def __dask_tokenize__(self):
        """Return a hashable value fully representative of the object.

        .. versionadded:: 3.15.2

        """
        compound = self._compound
        if compound:
            return (
                compound[0].__dask_tokenize__(),
                self._bitwise_operator,
                compound[1].__dask_tokenize__(),
            )

        value = self._value
        if isinstance(value, Data):
            value = (
                value.tolist(),
                value.Units.formatted(definition=True),
            )
        else:
            value = (value,)

        operator = self._operator
        if operator == "isclose":
            value += (self.rtol, self.atol)

        if operator == "wi":
            value += (self.open_lower, self.open_upper)

        return (self.__class__, operator, self._attr) + value

    def __deepcopy__(self, memo):
        """Used if copy.deepcopy is called on the variable."""
        return self.copy()

    def __eq__(self, x):
        """The rich comparison operator ``==``

        x.__eq__(y) <==> x==y

        x.__eq__(y) <==> x.evaluate(y)

        """
        return self._evaluate(x, ())

    def __ne__(self, x):
        """The rich comparison operator ``!=``

        x.__ne__(y) <==> x!=y

        x.__ne__(y) <==> (x==y)==False

        """
        # Note that it is important to use the == operator
        return self._evaluate(x, ()) == False  # ignore PEP8 E712 due to above

    def __and__(self, other):
        """The binary bitwise operation ``&``

        Combine two queries with a logical And operation. If the
        `!value` of both queries is the same then it will be retained
        on the compound query.

        x.__and__(y) <==> x&y

        """
        Q = type(self)
        new = Q.__new__(Q)

        new._operator = None
        new._compound = (self.copy(), other.copy())
        new._bitwise_operator = operator_and
        new._attr = ()

        # If the value of the two queries is the same then retain it
        # on the compound query
        value0 = self._value
        value1 = other._value
        new_value = None
        if value0 is not None and value1 is not None:
            try:
                if (value0 == value1).all():
                    new_value = deepcopy(value0)
            except AttributeError:
                if value0 == value1:
                    new_value = deepcopy(value0)
            except ValueError:
                pass

        new._value = new_value
        new._NotImplemented_RHS_Data_op = True

        return new

    def __iand__(self, other):
        """The augmented bitwise assignment ``&=``

        x.__iand__(y) <==> x&=y

        """
        return self & other

    def __or__(self, other):
        """The binary bitwise operation ``|``

        Combine two queries with a logical Or operation. If the
        `!value` of both queries is the same then it will be retained
        on the compound query.

        x.__or__(y) <==> x|y

        """
        Q = type(self)
        new = Q.__new__(Q)

        new._operator = None
        new._compound = (self, other)
        new._bitwise_operator = operator_or
        new._attr = ()

        # If the value of the two queries is the same then retain it
        # on the compound query
        value0 = self._value
        value1 = other._value
        new_value = None
        if value0 is not None and value1 is not None:
            try:
                if (value0 == value1).all():
                    new_value = deepcopy(value0)
            except AttributeError:
                if value0 == value1:
                    new_value = deepcopy(value0)
            except ValueError:
                pass

        new._value = new_value
        new._NotImplemented_RHS_Data_op = True

        return new

    def __ior__(self, other):
        """The augmented bitwise assignment ``|=``

        x.__ior__(y) <==> x|=y

        """
        return self | other

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return f"<CF {self.__class__.__name__}: {self}>"

    def __str__(self):
        """Called by the `str` built-in function.

        x.__str__() <==> str(x)

        """
        attr = ".".join(self._attr)
        operator = self._operator
        compound = self._compound

        # For "wi" queries only, open intervals are supported. For "wi" _value
        # is a list of two values, with representation from string list form
        # of '[a, b]' which corresponds to the standard mathematical notation
        # for a closed interval, the default. But an open endpoint is indicated
        # by a parenthesis, so adjust repr. to convert square bracket(s).
        repr_value = str(self._value)
        if self.open_lower:
            repr_value = "(" + repr_value[1:]

        if self.open_upper:
            repr_value = repr_value[:-1] + ")"

        if not compound:
            out = f"{attr}({operator} {repr_value}"
            rtol = self.rtol
            if rtol is not None:
                out += f" rtol={rtol}"

            atol = self.atol
            if atol is not None:
                out += f" atol={atol}"

            out += ")"
            return out

        bitwise_operator = repr(self._bitwise_operator)
        if "and_" in bitwise_operator:
            bitwise_operator = "&"
        elif "or_" in bitwise_operator:
            bitwise_operator = "|"

        return f"{attr}[{compound[0]} {bitwise_operator} {compound[1]}]"

    @property
    def atol(self):
        """The tolerance on absolute numerical differences.

        Returns the tolerance on absolute numerical differences that
        is used when evaluating numerically tolerant conditions
        (i.e. those defined by the ``'isclose'`` operator). If `None`
        then the value returned by `cf.atol` is used instead.

        For compound queries `atol` is always `None`, even if some of
        the constituent conditions have a different value.

        .. versionadded:: 3.15.2

        .. seealso:: `rtol`, `setdefault`

        """
        return getattr(self, "_atol", None)

    @property
    def attr(self):
        """The object attribute on which to apply the query condition.

        For compound queries `attr` is always ``()``, even if some of
        the constituent conditions have a different value.

        .. seealso:: `addattr`

        **Examples**

        >>> q = cf.Query('ge', 4)
        >>> print(q.attr)
        ()
        >>> r = cf.Query('le', 6, attr='year')
        >>> r.attr
        ('year',)
        >>> (q | r).attr
        ()

        """
        return self._attr

    @property
    def operator(self):
        """The query operator.

        For compound queries `operator` is always ``None``, regardless
        of the operators of the constituent conditions.

        **Examples**

        >>> q = cf.Query('ge', 4)
        >>> q.operator
        'ge'
        >>> q |= cf.Query('le', 6)
        >>> print(q.operator)
        None

        """
        return self._operator

    @property
    def Units(self):
        """Return the units of the query.

        .. versionadded:: 3.15.2

        :Returns:

            `Units`
                The units of the query value.

        **Examples**

        >>> cf.eq(9).Units
        <Units: >
        >>> cf.eq(9, 'm s-1').Units
        <Units: m s-1>
        >>> cf.eq(cf.Data(9, 'km')).Units
        <Units: km>

        >>> (cf.eq(9) | cf.gt(10)).Units
        <Units: >
        >>> (cf.eq(9, 'm') | cf.gt(10, 'm')).Units
        <Units: >
        >>> (cf.eq(9, 'm') | cf.gt(9, 'm')).Units
        <Units: m>
        >>> (cf.eq(9, 'm') | cf.gt(45)).Units
        <Units: m>

        >>> (cf.eq(9, 'm') | cf.gt(9, 'day')).Units
        AttributeError: <CF Query: [(eq 9 m) | (gt 9 day)]> has indeterminate units

        """
        value = self._value
        if value is not None:
            try:
                return value.Units
            except AttributeError:
                return Units()

        compound = self._compound
        if compound:
            # Still here? Then we have compund units with no common
            # value, so see if the units of each embedded query are
            # equivalent.
            q0, q1 = compound
            units0 = getattr(q0, "Units", Units())
            units1 = getattr(q1, "Units", Units())
            if not units0:
                if not units1:
                    return Units()

                return units1

            if not units1:
                return units0

            if units0.equivalent(units1):
                return units0

        raise AttributeError(f"{self!r} has indeterminate units")

    @property
    def open_lower(self):
        """True if the interval is open at the (excludes the) lower bound.

        .. versionadded:: 3.16.2

        .. seealso:: `open_upper`

        """
        return getattr(self, "_open_lower", False)

    @property
    def open_upper(self):
        """True if the interval is open at the (excludes the) upper bound.

        .. versionadded:: 3.16.2

        .. seealso:: `open_lower`

        """
        return getattr(self, "_open_upper", False)

    @property
    def rtol(self):
        """The tolerance on relative numerical differences.

        Returns the tolerance on relative numerical differences that
        is used when evaluating numerically tolerant conditions
        (i.e. those defined by the ``'isclose'`` operator). If `None`
        then the value returned by `cf.rtol` is used instead.

        For compound queries `rtol` is always `None`, even if some of
        the constituent conditions have a different value.

        .. versionadded:: 3.15.2

        .. seealso:: `atol`, `setdefault`

        """
        return getattr(self, "_rtol", None)

    @Units.setter
    def Units(self, value):
        self.set_condition_units(value)

    @property
    def value(self):
        """The value of the condition encapsulated by the query.

        An exception is raised for compound conditions.

        **Examples**

        >>> q = cf.Query('ge', 4)
        >>> q.value
        4
        >>> q |= cf.Query('le', 6)
        >>> q.value
        AttributeError: Compound query doesn't have attribute 'value'

        """
        value = self._value
        if value is None:
            raise AttributeError(
                "Compound query doesn't have attribute 'value'"
            )

        return value

    def addattr(self, attr):
        """Redefine the query to be on an object's attribute.

        If another attribute has previously been specified, then the new
        attribute is considered to be an attribute of the existing
        attribute.

        .. seealso:: `attr`

        :Parameters:

            attr: `str`
                The attribute name.

        :Returns:

            `Query`
                The new query object.

        **Examples**

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

        """
        Q = type(self)
        new = Q.__new__(Q)

        new.__dict__ = self.__dict__.copy()
        new._attr += (attr,)

        new._NotImplemented_RHS_Data_op = True

        return new

    def copy(self):
        """Return a deep copy.

        ``q.copy()`` is equivalent to ``copy.deepcopy(q)``.

        :Returns:

                The deep copy.

        **Examples**

        >>> r = q.copy()

        """
        Q = type(self)
        new = Q.__new__(Q)

        d = self.__dict__.copy()
        new.__dict__ = d

        compound = d["_compound"]
        if compound:
            d["_compound"] = deepcopy(compound)
        else:
            d["_value"] = deepcopy(d["_value"])

        return new

    @_display_or_return
    def dump(self, display=True):
        """Return a string containing a full description of the
        instance.

        :Parameters:

            display: `bool`, optional

                If `False` then return the description as a string. By
                default the description is printed.

        :Returns:

            `None` or `str`
                The description. If *display* is True then the description
                is printed and `None` is returned. Otherwise the
                description is returned as a string.

        """
        return str(self)

    @_deprecated_kwarg_check("traceback", version="3.0.0", removed_at="4.0.0")
    @_manage_log_level_via_verbosity
    def equals(self, other, verbose=None, traceback=False):
        """True if two `Query` objects are the same."""
        if self._compound:
            if not other._compound:
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Different compound components"
                    )  # pragma: no cover

                return False

            if self._bitwise_operator != other._bitwise_operator:
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Different compound "
                        f"operators: {self._bitwise_operator!r}, "
                        f"{other._bitwise_operator!r}"
                    )  # pragma: no cover

                return False

            if not self._compound[0].equals(other._compound[0]):
                if not self._compound[0].equals(other._compound[1]):
                    if is_log_level_info(logger):
                        logger.info(
                            f"{self.__class__.__name__}: Different compound components"
                        )  # pragma: no cover

                    return False
                if not self._compound[1].equals(other._compound[0]):
                    if is_log_level_info(logger):
                        logger.info(
                            f"{self.__class__.__name__}: Different compound components"
                        )  # pragma: no cover

                    return False
            elif not self._compound[1].equals(other._compound[1]):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Different compound components"
                    )  # pragma: no cover

                return False

        elif other._compound:
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Different compound components"
                )  # pragma: no cover

            return False

        for attr in (
            "_NotImplemented_RHS_Data_op",
            "_attr",
            "_value",
            "_operator",
            "_rtol",
            "_atol",
            "_open_lower",
            "_open_upper",
        ):
            x = getattr(self, attr, None)
            y = getattr(other, attr, None)
            if not cf_equals(x, y, verbose=verbose):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Different {attr!r} "
                        f"attributes: {x!r}, {y!r}"
                    )  # pragma: no cover

                return False

        return True

    def evaluate(self, x):
        """Evaluate the query operation for a given left hand side
        operand.

        Note that for the query object ``q`` and any object, ``x``,
        ``x==q`` is equivalent to ``q.evaluate(x)`` and ``x!=q`` is
        equivalent to ``q.evaluate(x)==False``.

        :Parameters:

            x:
                The object for the left hand side operand of the query.

        :Returns:

                The result of the query. The nature of the result is
                dependent on the object type of *x*.

        **Examples**

        >>> q = cf.Query('lt', 5.5)
        >>> q.evaluate(6)
        False

        >>> q = cf.Query('wi', (1,2))
        >>> array = numpy.arange(4)
        >>> array
        array([0, 1, 2, 3])
        >>> q.evaluate(array)
        array([False,  True,  True, False], dtype=bool)

        """
        return self._evaluate(x, ())

    def _evaluate(self, x, parent_attr):
        """Evaluate the query operation for a given object.

        .. seealso:: `evaluate`

        :Parameters:

            x:
                See `evaluate`.

            parent_attr: `tuple`

        :Returns:

            See `evaluate`.

        """
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

        # TODO: Once Python 3.9 is no longer supported, this is a good
        #       candidate for PEP 622 – Structural Pattern Matching
        #       (https://peps.python.org/pep-0622)

        if operator == "gt":
            _gt = getattr(x, "__query_gt__", None)
            if _gt is not None:
                return _gt(value)

            return x > value

        if operator == "wi":
            _wi = getattr(x, "__query_wi__", None)
            if _wi is not None:
                return _wi(value)

            if self.open_lower:
                lower_bound = x > value[0]
            else:
                lower_bound = x >= value[0]

            if self.open_upper:
                upper_bound = x < value[1]
            else:
                upper_bound = x <= value[1]

            return lower_bound & upper_bound

        if operator == "eq":
            try:
                return bool(value.search(x))
            except AttributeError:
                return x == value
            except TypeError:
                raise ValueError(
                    "Can't perform regular expression search on a "
                    f"non-string: {x!r}"
                )

        if operator == "isclose":
            rtol = self.rtol
            atol = self.atol
            if atol is None:
                atol = cf_atol().value

            if rtol is None:
                rtol = cf_rtol().value

            _isclose = getattr(x, "__query_isclose__", None)
            if _isclose is not None:
                return _isclose(value, rtol, atol)

            return np.isclose(x, value, rtol=rtol, atol=atol)

        if operator == "ne":
            try:
                return not bool(value.search(x))
            except AttributeError:
                return x != value
            except TypeError:
                raise ValueError(
                    "Can't perform regular expression search on a "
                    f"non-string: {x!r}"
                )

        if operator == "lt":
            _lt = getattr(x, "__query_lt__", None)
            if _lt is not None:
                return _lt(value)

            return x < value

        if operator == "le":
            _le = getattr(x, "__query_le__", None)
            if _le is not None:
                return _le(value)

            return x <= value

        if operator == "ge":
            _ge = getattr(x, "__query_ge_", None)
            if _ge is not None:
                return _ge(value)

            return x >= value

        if operator == "wo":
            _wo = getattr(x, "__query_wo__", None)
            if _wo is not None:
                return _wo(value)

            return (x < value[0]) | (x > value[1])

        if operator == "set":
            if isinstance(x, str):
                for v in value:
                    try:
                        if v.search(x):
                            return True
                    except AttributeError:
                        if x == v:
                            return True

                return False
            else:
                _set = getattr(x, "__query_set__", None)
                if _set is not None:
                    return _set(value)

                i = iter(value)
                v = next(i)
                out = x == v
                for v in i:
                    out |= x == v

                return out

    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))  # pragma: no cover

    def iscontains(self):
        """Return True if the query is a "cell contains" condition.

        .. versionadded:: 3.14.0

        .. seealso:: `cf.contains`

        :Returns:

            `bool`
                Whether or not the query is a "cell contains"
                condition.

        **Examples**

        >>> q = cf.contains(15)
        >>> q
        <CF Query: [lower_bounds(le 15) & upper_bounds(ge 15)]>
        >>> q.iscontains()
        True
        >>> q |= cf.lt(6)
        >>> q.iscontains()
        False

        >>> r = cf.wi(6, 10)
        >>> r.iscontains()
        False

        """
        if not (
            self._compound
            and self._value is not None
            and self._bitwise_operator is operator_and
        ):
            return False

        sig = builtin_set()
        for q in self._compound:
            if q._compound:
                # A compound query of compound queries is not a "cell
                # contains" condition
                break

            sig.add((q.attr, q.operator))

        return sig == builtin_set(
            ((("lower_bounds",), "le"), (("upper_bounds",), "ge"))
        )

    def set_condition_units(self, units):
        """Set units of condition values in-place.

        .. versionadded:: 3.13.1

        :Parameters:

            units: `str` or `Units`

                The units to be set on all condition values.

        :Returns:

            `None`

        **Examples**

        >>> q = cf.lt(9)
        >>> q
        <CF Query: (lt 9)>
        >>> q.set_condition_units('km')
        >>> q
        <CF Query: (lt 9 km)>
        >>> q.set_condition_units('seconds')
            ...
        ValueError: Units <Units: seconds> are not equivalent to query condition units <Units: m>

        >>> q = cf.lt(9, units='m')
        >>> q
        <CF Query: (lt 9 m)>
        >>> q.set_condition_units('km')
        >>> q
        <CF Query: (lt 0.009 km)>

        >>> q = cf.lt(9)
        >>> r = cf.ge(3000, units='m')
        >>> s = q & r
        >>> s
        <CF Query: [(lt 9) & (ge 3000 m)]>
        >>> s.set_condition_units('km')
        >>> s
        <CF Query: [(lt 9 km) & (ge 3 km)]>
        >>> q
        <CF Query: (lt 9)>
        >>> r
        <CF Query: (ge 3000 m)>

        """

        def get_and_set_value_units(v, u):
            """Helper method to simplify setting of specified units."""
            v_units = getattr(v, "Units", None)
            if v_units is None:  # Value 'v' has no units
                v = Data(v, units=u)
            else:  # Value 'v' already has units
                try:
                    v = v.copy()
                    v.Units = u
                except ValueError:
                    raise ValueError(
                        f"Units {u!r} are not equivalent to "
                        f"query condition units {v_units!r}"
                    )

            return v

        units = Units(units)

        compound = self._compound
        if compound:
            for r in compound:
                r.set_condition_units(units)
            return

        value = self._value
        if value is None:
            return

        if self.operator in ("wi", "wo", "set"):
            # Value is a sequence of things that may or may not
            # already have units
            new_values = []
            for v in value:
                v = get_and_set_value_units(v, units)
                new_values.append(v)

            value = new_values
        else:
            value = get_and_set_value_units(value, units)

        self._value = value

    def setdefault(self, rtol=None, atol=None):
        """Set condition parameters in-place that are not already set.

        For compound queries the parameters are set recursively on the
        constituent conditions.

        .. versionadded:: 3.15.2

        .. seealso:: `atol`, `rtol`

        :Parameters:

            rtol: number, optional
                Only applicable to the ``'isclose'`` operator and
                ignore for all other operators. Set the given
                tolerance on relative numerical differences, unless
                already set. Ignored if `None`.

            atol: number, optional
                Only applicable to the ``'isclose'`` operator and
                ignore for all other operators. Set the given
                tolerance on absolute numerical differences, unless
                already set. Ignored if `None`.

        :Returns:

            `None`

        **Examples**

        >>> q = cf.isclose(9)
        >>> q
        <CF Query: (isclose 9)>
        >>> q.setdefault(rtol=1e-5, atol=1e-08)
        <CF Query: (isclose 9 rtol=1e-05 atol=1e-08)>

        >>> q = cf.isclose(9, rtol=9e-9)
        >>> q.setdefault(rtol=1e-5, atol=1e-08)
        >>> q
        <CF Query: (isclose 9 rtol=9e-09 atol=1e-08)>

        >>> q = cf.lt(9) & (cf.eq(1) | cf.isclose(4, rtol=9e-9))
        >>> q
        <CF Query: [(lt 9) & [(eq 1) | (isclose 4 rtol=9e-09)]]>
        >>> q.setdefault(rtol=1e-5, atol=1e-08)
        >>> q
        <CF Query: [(lt 9) & [(eq 1) | (isclose 4 rtol=9e-09 atol=1e-08)]]>

        """
        compound = self._compound
        if compound:
            for q in compound:
                q.setdefault(rtol=rtol, atol=atol)

            return

        if self.operator == "isclose":
            if rtol is not None and self.rtol is None:
                self._rtol = rtol

            if atol is not None and self.atol is None:
                self._atol = atol

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    @property
    def exact(self):
        """Deprecated at version 3.0.0.

        Use `re.compile` objects instead.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "exact",
            "Use 're.compile' objects instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def equivalent(self, other, traceback=False):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_FUNCTION(
            self, "equivalent", version="3.0.0", removed_at="4.0.0"
        )


# --------------------------------------------------------------------
# Constructor functions
# --------------------------------------------------------------------
def lt(value, units=None, attr=None):
    """A `Query` object for a "strictly less than" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.le`, `cf.set`, `cf.wi`, `cf.wo`, `cf.isclose`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples**

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

    """
    return Query("lt", value, units=units, attr=attr)


def le(value, units=None, attr=None):
    """A `Query` object for a "less than or equal" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`, `cf.isclose`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> q = cf.le(5)
    >>> q
    <CF Query: x le 5>
    >>> q.evaluate(5)
    True
    >>> q.evaluate(6)
    False

    """
    return Query("le", value, units=units, attr=attr)


def gt(value, units=None, attr=None):
    """A `Query` object for a "strictly greater than" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.ne`, `cf.le`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`, `cf.isclose`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> q = cf.gt(5)
    >>> q
    <CF Query: x gt 5>
    >>> q.evaluate(6)
    True
    >>> q.evaluate(5)
    False

    """
    return Query("gt", value, units=units, attr=attr)


def ge(value, units=None, attr=None):
    """A `Query` object for a "greater than or equal" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.gt`, `cf.ne`, `cf.le`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`, `cf.isclose`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples**

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

    """
    return Query("ge", value, units=units, attr=attr)


def eq(value, units=None, attr=None, exact=True):
    """A `Query` object for an "equal" condition.

    .. seealso:: `cf.contains`, `cf.ge`, `cf.gt`, `cf.ne`, `cf.le`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`, `cf.isclose`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

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

    **Examples**

    >>> q = cf.eq(5)
    >>> q
    <CF Query: x eq 5>
    >>> q.evaluate(5)
    True
    >>> q == 4
    False

    >>> q = cf.eq('air')
    >>> q == 'air_temperature'
    True

    >>> q = cf.eq('.*temp')
    >>> q == 'air_temperature'
    True

    """
    if not exact:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "eq", exact=True
        )  # pragma: no cover

    return Query("eq", value, units=units, attr=attr)


def ne(value, units=None, attr=None, exact=True):
    """A `Query` object for a "not equal" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.le`,
                 `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`, `cf.isclose`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

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

    **Examples**

    >>> q = cf.ne(5)
    >>> q
    <CF Query: x ne 5>
    >>> q.evaluate(4)
    True
    >>> q.evaluate(5)
    False

    """
    if not exact:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "ne", exact=True
        )  # pragma: no cover

    return Query("ne", value, units=units, attr=attr)


def isclose(value, units=None, attr=None, rtol=None, atol=None):
    """A `Query` object for an "is close" condition.

    .. versionadded:: 3.15.2

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ne`, `cf.ge`, `cf.gt`,
                 `cf.le`, `cf.lt`, `cf.set`, `cf.wi`, `cf.wo`

    :Parameters:

        value:
            The value to be used in the isclose test. May be any
            scalar `cf.Data` object, or else any numerical object that
            can be operated on with ``np.isclose``.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

        rtol: number, optional
            The tolerance on relative numerical differences. If
            `None`, the default, then the value returned by `cf.rtol`
            is used at evaluation time.

        atol: number, optional
            The tolerance on absolute numerical differences. If
            `None`, the default, then the value returned by `cf.atol`
            is used at evaluation time.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> q = cf.isclose(9)
    >>> q
    <CF Query: (isclose 9)>
    >>> q.evaluate(9.000001)
    False
    >>> with cf.configuration(rtol=0.001, atol=0.01):
    ...     print(q.evaluate(9.000001))
    ...
    True
    >>> q.evaluate(9.000001)
    False

    >>> q = cf.isclose(9, rtol=0.001, atol=0.01)
    >>> q
    <CF Query: (isclose 9 rtol=0.001 atol=0.01)>
    >>> q.evaluate(9.000001)
    True

    >>> q = cf.isclose(9, atol=0.01)
    >>> q
    <CF Query: (isclose 9 atol=0.01)>
    >>> q.evaluate(9.000001)
    True

    """
    return Query(
        "isclose", value, units=units, attr=attr, rtol=rtol, atol=atol
    )


def wi(
    value0,
    value1,
    units=None,
    attr=None,
    open_lower=False,
    open_upper=False,
):
    """A `Query` object for a "within a range" condition.

    The condition is a closed interval by default, inclusive of
    both the endpoints, but can be made open or half-open to exclude
    the endpoints on either end with use of the `open_lower` and
    `open_upper` parameters.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.le`, `cf.lt`, `cf.set`, `cf.wo`, `cf.isclose`

    :Parameters:

        value0:
             The lower bound of the range.

        value1:
             The upper bound of the range.

        open_lower: `bool`, optional
             If True, open the interval at the lower
             bound so that value0 is excluded from the
             range. By default the interval is closed
             so that value0 is included.

             .. versionadded:: 3.16.2

        open_upper: `bool`, optional
             If True, open the interval at the upper
             bound so that value1 is excluded from the
             range. By default the interval is closed
             so that value1 is included.

             .. versionadded:: 3.16.2

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> q = cf.wi(5, 7)
    >>> q
    <CF Query: wi (5, 7)>
    >>> q.evaluate(6)
    True
    >>> q.evaluate(4)
    False
    >>> q.evaluate(5)
    True
    >>> q.evaluate(7)
    True

    The interval can be made open on either side or both. Note that,
    as per mathematical interval notation, square brackets indicate
    closed endpoints and parentheses open endpoints in the representation:

    >>> q = cf.wi(5, 7, open_upper=True)
    >>> q
    <CF Query: (wi [5, 7))>
    >>> q.evaluate(7)
    False
    >>> q = cf.wi(5, 7, open_lower=True)
    >>> q
    <CF Query: (wi (5, 7])>
    >>> q.evaluate(5)
    False
    >>> q = cf.wi(5, 7, open_lower=True, open_upper=True)
    >>> q
    <CF Query: (wi (5, 7))>
    >>> q.evaluate(5)
    False
    >>> q.evaluate(7)
    False

    """
    return Query(
        "wi",
        [value0, value1],
        units=units,
        attr=attr,
        open_lower=open_lower,
        open_upper=open_upper,
    )


def wo(value0, value1, units=None, attr=None):
    """A `Query` object for a "without a range" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.le`, `cf.lt`, `cf.set`, `cf.wi`, `cf.isclose`

    :Parameters:

        value0:
             The lower bound of the range.

        value1:
             The upper bound of the range.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

        attr: `str`, optional
            Apply the condition to the attribute, or nested
            attributes, of the operand, rather than the operand
            itself. Nested attributes are specified by separating them
            with a ``.``. For example, the "month" attribute of the
            "bounds" attribute is specified as ``'bounds.month'``.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> q = cf.wo(5)
    >>> q
    <CF Query: x wo (5, 7)>
    >>> q.evaluate(4)
    True
    >>> q.evaluate(6)
    False

    """
    return Query("wo", [value0, value1], units=units, attr=attr)


def set(values, units=None, attr=None, exact=True):
    """A `Query` object for a "member of set" condition.

    .. seealso:: `cf.contains`, `cf.eq`, `cf.ge`, `cf.gt`, `cf.ne`,
                 `cf.le`, `cf.lt`, `cf.wi`, `cf.wo`, `cf.isclose`

    :Parameters:

        value: sequence
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

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

    **Examples**

    >>> c = cf.set([3, 5])
    >>> c == 4
    False
    >>> c == 5
    True
    >>> c == numpy.array([2, 3, 4, 5])
    array([False  True False  True])

    """
    if not exact:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            "set", exact=True
        )  # pragma: no cover

    return Query("set", values, units=units, attr=attr)


def contains(value, units=None):
    """A `Query` object for a "cell contains" condition.

    .. versionadded:: 3.0.0

    .. seealso:: `cf.Query.iscontains`, `cf.cellsize`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.cellle`, `cf.celllt`,
                 `cf.cellwi`, `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> q = cf.contains(8)
    >>> q
    <CF Query: [lower_bounds(le 8) & upper_bounds(ge 8)]>
    >>> q.value
    8

    >>> q = cf.contains(30, 'degrees_east')
    >>> q
    <CF Query: [lower_bounds(le 30 degrees_east) & upper_bounds(ge 30 degrees_east)]>
    >>> q.value
    <CF Data(): 30 degrees_east>

    >>> cf.contains(cf.Data(10, 'km'))
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

    """
    return Query("le", value, units=units, attr="lower_bounds") & Query(
        "ge", value, units=units, attr="upper_bounds"
    )


def year(value):
    """A `Query` object for a "year" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.djf`,
                 `cf.mam`, `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> d = cf.dt(2002, 6, 16)
    >>> d == cf.year(2002)
    True
    >>> d == cf.year(cf.le(2003))
    True
    >>> d == cf.year(2001)
    False
    >>> d == cf.year(cf.wi(2003, 2006))
    False

    """
    if isinstance(value, Query):
        return value.addattr("year")
    else:
        return Query("eq", value, attr="year")


def month(value):
    """A `Query` object for a "month of the year" condition.

    .. seealso:: `cf.year`, `cf.day`, `cf.hour`, `cf.minute`,
                 `cf.second`, `cf.seasons`, `cf.djf`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> d = cf.dt(2002, 6, 16)
    >>> d == cf.month(6)
    True
    >>> d == cf.month(cf.le(7))
    True
    >>> d == cf.month(7)
    False
    >>> d == cf.month(cf.wi(1, 6))
    True

    """
    if isinstance(value, Query):
        return value.addattr("month")
    else:
        return Query("eq", value, attr="month")


def day(value):
    """A `Query` object for a "day of the month" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.hour`, `cf.minute`,
                 `cf.second`, `cf.seasons`, `cf.djf`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> d = cf.dt(2002, 6, 16)
    >>> d == cf.day(16)
    True
    >>> d == cf.day(cf.le(19))
    True
    >>> d == cf.day(7)
    False
    >>> d == cf.day(cf.wi(1, 21))
    True

    """
    if isinstance(value, Query):
        return value.addattr("day")
    else:
        return Query("eq", value, attr="day")


def hour(value):
    """A `Query` object for a "hour of the day" condition.

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

    **Examples**

    >>> d = cf.dt(2002, 6, 16, 18)
    >>> d == cf.hour(18)
    True
    >>> d == cf.hour(cf.le(19))
    True
    >>> d == cf.hour(7)
    False
    >>> d == cf.hour(cf.wi(6, 23))
    True

    """
    if isinstance(value, Query):
        return value.addattr("hour")
    else:
        return Query("eq", value, attr="hour")


def minute(value):
    """A `Query` object for a "minute of the hour" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.second`, `cf.seasons`, `cf.djf`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> d = cf.dt(2002, 6, 16, 18, 30, 0)
    >>> d == cf.minute(30)
    True
    >>> d == cf.minute(cf.le(45))
    True
    >>> d == cf.minute(7)
    False
    >>> d == cf.minute(cf.wi(15, 45))
    True

    """
    if isinstance(value, Query):
        return value.addattr("minute")
    else:
        return Query("eq", value, attr="minute")


def second(value):
    """A `Query` object for a "second of the minute" condition.

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.seasons`, `cf.djf`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Parameter:

        value:
            The query condition's value.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> d = cf.dt(2002, 6, 16, 18, 30, 0)
    >>> d == cf.second(0)
    True
    >>> d == cf.second(cf.le(30))
    True
    >>> d == cf.second(30)
    False
    >>> d == cf.second(cf.wi(0, 30))
    True

    """
    if isinstance(value, Query):
        return value.addattr("second")
    else:
        return Query("eq", value, attr="second")


def cellsize(value, units=None):
    """A `Query` object for a "cell size" condition.

    .. seealso:: `cf.contains`, `cf.cellge`, `cf.cellgt`, `cf.cellne`,
                 `cf.cellle`, `cf.celllt`, `cf.cellwi`, `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> cf.cellsize(cf.lt(5, 'km'))
    <CF Query: cellsize(lt <CF Data: 5 km>)>
    >>> cf.cellsize(5)
    <CF Query: cellsize(eq 5)>
    >>> cf.cellsize(cf.Data(5, 'km'))
    <CF Query: cellsize(eq <CF Data: 5 km>)>
    >>> cf.cellsize(5, units='km')
    <CF Query: cellsize(eq <CF Data: 5 km>)>

    """
    if isinstance(value, Query):
        return value.addattr("cellsize")
    else:
        return Query("eq", value, units=units, attr="cellsize")


def cellwi(value0, value1, units=None):
    """A `Query` object for a "cell bounds lie within range" condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.cellle`, `cf.celllt`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> cf.cellwi(cf.Data(5, 'km'), cf.Data(10, 'km'))
    <CF Query: [lower_bounds(ge 5 km) & upper_bounds(le 10 km)]>
    >>> cf.cellwi(5, 10, units="km")
    <CF Query: [lower_bounds(ge 5 km) & upper_bounds(le 10 km)]>
    >>> cf.cellwi(cf.Data(5, 'km'), cf.Data(10000, 'm'))
    <CF Query: [lower_bounds(ge 5 km) & upper_bounds(le 10000 m)]>
    >>> cf.cellwi(0.2, 0.3)
    <CF Query: [lower_bounds(ge 0.2) & upper_bounds(le 0.3)]>

    """
    return Query("ge", value0, units=units, attr="lower_bounds") & Query(
        "le", value1, units=units, attr="upper_bounds"
    )


def cellwo(value0, value1, units=None):
    """A `Query` object for a "cell bounds lie without range" condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.cellle`, `cf.celllt`,
                 `cf.cellwi`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> cf.cellwo(cf.Data(5, 'km'), cf.Data(10, 'km'))
    <CF Query: [lower_bounds(lt 5 km) & upper_bounds(gt 10 km)]>
    >>> cf.cellwo(5, 10, units="km")
    <CF Query: [lower_bounds(lt 5 km) & upper_bounds(gt 10 km)]>
    >>> cf.cellwo(cf.Data(5, 'km'), cf.Data(10000, 'm'))
    <CF Query: [lower_bounds(lt 5 km) & upper_bounds(gt 10000 m)]>
    >>> cf.cellwo(0.2, 0.3)
    <CF Query: [lower_bounds(lt 0.2) & upper_bounds(gt 0.3)]>

    """
    return Query("lt", value0, units=units, attr="lower_bounds") & Query(
        "gt", value1, units=units, attr="upper_bounds"
    )


def cellgt(value, units=None):
    """A `Query` object for a "cell bounds strictly greater than"
    condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`
                 `cf.cellne`, `cf.cellle`, `cf.celllt`, `cf.cellwi`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> cf.cellgt(cf.Data(300, 'K'))
    <CF Query: lower_bounds(gt 300 K)>
    >>> cf.cellgt(300, units='K')
    <CF Query: lower_bounds(gt 300 K)>
    >>> cf.cellgt(300)
    <CF Query: lower_bounds(gt 300)>

    """
    return Query("gt", value, units=units, attr="lower_bounds")


def cellge(value, units=None):
    """A `Query` object for a "cell bounds greater than or equal"
    condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellgt`,
                 `cf.cellne`, `cf.cellle`, `cf.celllt`, `cf.cellwi`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.


    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> cf.cellge(cf.Data(300, 'K'))
    <CF Query: lower_bounds(ge 300 K)>
    >>> cf.cellge(cf.Data(300, 'K'))
    <CF Query: lower_bounds(ge 300 K)>
    >>> cf.cellge(300)
    <CF Query: lower_bounds(ge 300)>

    """
    return Query("ge", value, units=units, attr="lower_bounds")


def celllt(value, units=None):
    """A `Query` object for a “cell bounds strictly less than”
    condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.cellle`, `cf.cellwi`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> cf.celllt(cf.Data(300, 'K'))
    <CF Query: upper_bounds(lt 300 K)>
    >>> cf.celllt(300, units='K')
    <CF Query: upper_bounds(lt 300 K)>
    >>> cf.celllt(300)
    <CF Query: upper_bounds(lt 300)>

    """
    return Query("lt", value, units=units, attr="upper_bounds")


def cellle(value, units=None):
    """A `Query` object for a "cell bounds less than or equal"
    condition.

    .. seealso:: `cf.cellsize`, `cf.contains`, `cf.cellge`,
                 `cf.cellgt`, `cf.cellne`, `cf.celllt`, `cf.cellwi`,
                 `cf.cellwo`

    :Parameters:

        value:
            The query condition's value.

        units: `str` or `Units`, optional
            The units of *value*. By default, the same units as the
            operand being tested are assumed, if applicable. If
            *units* is specified and *value* already has units (such
            as those attached to a `Data` object), then the pair of
            units must be equivalent.

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> cf.cellle(cf.Data(300, 'K'))
    <CF Query: upper_bounds(le 300 K)>
    >>> cf.cellle(300, units='K')
    <CF Query: upper_bounds(le 300 K)>
    >>> cf.cellle(300)
    <CF Query: upper_bounds(le 300)>

    """
    return Query("le", value, units=units, attr="upper_bounds")


def jja():
    """A `Query` object for a "month of year in June, July or August"
    condition.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.djf`,
                 `cf.mam`, `cf.son`

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> f
    <CF Field: air_temperature(time(365), latitude(64), longitude(128)) K>
    >>> f.subspace(time=cf.jja())
    <CF Field: air_temperature(time(92), latitude(64), longitude(128)) K>

    """
    return Query("wi", (6, 8), attr="month")


def son():
    """A `Query` object for a "month of year in September, October,
    November" condition.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.djf`,
                 `cf.mam`, `cf.jja`

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> f
    <CF Field: air_temperature(time(365), latitude(64), longitude(128)) K>
    >>> f.subspace(time=cf.son())
    <CF Field: air_temperature(time(91), latitude(64), longitude(128)) K>

    """
    return Query("wi", (9, 11), attr="month")


def djf():
    """A `Query` object for a "month of year in December, January,
    February" condition.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.mam`,
                 `cf.jja`, `cf.son`

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> f
    <CF Field: air_temperature(time(365), latitude(64), longitude(128)) K>
    >>> f.subspace(time=cf.djf())
    <CF Field: air_temperature(time(90), latitude(64), longitude(128)) K>

    """
    q = Query("ge", 12) | Query("le", 2)
    return q.addattr("month")


def mam():
    """A `Query` object for a "month of year in March, April, May"
    condition.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`,
                 `cf.minute`, `cf.second`, `cf.seasons`, `cf.djf`,
                 `cf.jja`, `cf.son`

    :Returns:

        `Query`
            The query object.

    **Examples**

    >>> f
    <CF Field: air_temperature(time(365), latitude(64), longitude(128)) K>
    >>> f.subspace(time=cf.mam())
    <CF Field: air_temperature(time(92), latitude(64), longitude(128)) K>

    """
    return Query("wi", (3, 5), attr="month")


def seasons(n=4, start=12):
    """A customisable list of `Query` objects for "seasons in a year"
    conditions.

    Note that any date-time that lies within a particular season will
    satisfy that query.

    .. versionadded:: 1.0

    .. seealso:: `cf.year`, `cf.month`, `cf.day`, `cf.hour`, `cf.minute`,
                 `cf.second`, `cf.djf`, `cf.mam`, `cf.jja`, `cf.son`

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

    **Examples**

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

    """
    if 12 % n:
        raise ValueError("Number of seasons must divide into 12. Got %s" % n)

    if not 1 <= start <= 12 or int(start) != start:
        raise ValueError(
            "Start month must be integer between 1 and 12. Got %s" % start
        )

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
            q = Query("wi", (m0, m1))
        elif m0 > m1:
            q = Query("ge", m0) | Query("le", m1)
        else:
            q = Query("eq", m0)

        out.append(q.addattr("month"))

        m0 = m1 + 1
        if m0 > 12:
            m0 = 1

    return out


# --------------------------------------------------------------------
# Deprecated functions
# --------------------------------------------------------------------
def dtge(*args, **kwargs):
    """Return a `Query` object for a variable being not earlier than a
    date-time.

    Deprecated at version 3.0.0. Use 'cf.ge' with a datetime object
    value instead.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "dtge",
        "Use 'cf.ge' with a datetime object value instead.",
        version="3.0.0",
        removed_at="4.0.0",
    )  # pragma: no cover


def dtgt(*args, **kwargs):
    """Deprecated at version 3.0.0.

    Use 'cf.gt' with a datetime object value instead.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "dtgt",
        "Use 'cf.gt' with a datetime object value instead.",
        version="3.0.0",
        removed_at="4.0.0",
    )  # pragma: no cover


def dtle(*args, **kwargs):
    """Deprecated at version 3.0.0.

    Use 'cf.le' with a datetime object value instead.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "dtle",
        "Use 'cf.le' with a datetime object value instead.",
        version="3.0.0",
        removed_at="4.0.0",
    )  # pragma: no cover


def dtlt(*args, **kwargs):
    """Deprecated at version 3.0.0.

    Use 'cf.lt' with a datetime object value instead.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "dtlt",
        "Use 'cf.lt' with a datetime object value instead.",
        version="3.0.0",
        removed_at="4.0.0",
    )  # pragma: no cover


def dteq(*args, **kwargs):
    """Deprecated at version 3.0.0.

    Use 'cf.eq' with a datetime object value instead.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "dteq",
        "Use 'cf.eq' with a datetime object value instead.",
        version="3.0.0",
        removed_at="4.0.0",
    )  # pragma: no cover


def dtne(*args, **kwargs):
    """Deprecated at version 3.0.0.

    Use 'cf.ne' with a datetime object value instead.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "dtne",
        "Use 'cf.ne' with a datetime object value instead.",
        version="3.0.0",
        removed_at="4.0.0",
    )  # pragma: no cover


def contain(value, units=None, attr=None):
    """Return a `Query` object for coordinate cells containing a value.

    Deprecated at version 3.0.0. Use function 'cf.contains' instead.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "cf.contain",
        "Use function 'cf.contains' instead",
        version="3.0.0",
        removed_at="4.0.0",
    )  # pragma: no cover
