import logging

import cfdm
from cfdm import is_log_level_info

from .decorators import (
    _deprecated_kwarg_check,
    _manage_log_level_via_verbosity,
)
from .functions import (
    _DEPRECATION_ERROR,
    _DEPRECATION_ERROR_DICT,
    _DEPRECATION_ERROR_KWARGS,
)
from .mixin_container import Container

logger = logging.getLogger(__name__)


class ConstructList(list, Container, cfdm.Container):
    """An ordered sequence of constructs.

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

    """

    def __init__(self, constructs=None):
        """**Initialisation**

        :Parameters:

            constructs: (sequence of) constructs
                 Create a new list with these constructs.

        """
        super(cfdm.Container, self).__init__()

        if constructs is not None:
            if getattr(constructs, "construct_type", None) is not None:
                self.append(constructs)
            else:
                self.extend(constructs)

    def __call__(self, *identities):
        """Alias for `cf.{{class}}.select_by_identity`."""
        return self.select_by_identity(*identities)

    def __deepcopy__(self, memo):
        """Called by the `copy.deepcopy` standard library function."""
        return self.copy()

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        out = [repr(f) for f in self]
        out = ",\n ".join(out)
        return "[" + out + "]"

    def __str__(self):
        """Called by the `str` built-in function.

        x.__str__() <==> str(x)

        """
        return repr(self)

    def __docstring_method_exclusions__(self):
        """Return the names of methods to exclude from docstring
        substitutions.

        See `_docstring_method_exclusions` for details.

        """
        return ("append", "extend", "insert", "pop", "reverse", "clear")

    # ----------------------------------------------------------------
    # Overloaded list methods
    # ----------------------------------------------------------------
    def __add__(self, x):
        """The binary arithmetic operation ``+``

        f.__add__(x) <==> f + x

        :Returns:

            `{{class}}`
                The concatenation of the list and another sequence.

        **Examples**

        >>> h = f + g
        >>> f += g

        """
        return type(self)(list.__add__(self, x))

    def __contains__(self, y):
        """Called to implement membership test operators.

        x.__contains__(y) <==> y in x

        {{List comparison}}

        Note that ``x in fl`` is equivalent to
        ``any(f.equals(x) for f in fl)``.

        """
        for f in self:
            if f.equals(y):
                return True

        return False

    def __mul__(self, x):
        """The binary arithmetic operation ``*``

        f.__mul__(x) <==> f * x

        :Returns:

            `{{class}}`
                The list added to itself *n* times.

        **Examples**

        >>> h = f * 2
        >>> f *= 2

        """
        return type(self)(list.__mul__(self, x))

    def __eq__(self, other):
        """The rich comparison operator ``==``

        f.__eq__(x) <==> f == x

        {{List comparison}}

        Note that ``f == x`` is equivalent to ``f.equals(x)``.

        :Returns:

            `bool`

        """
        return self.equals(other)

    def __getslice__(self, i, j):
        """Called to implement evaluation of f[i:j]

        f.__getslice__(i, j) <==> f[i:j]

        :Returns:

            `{{class}}`
                Slice of the list from *i* to *j*.

        **Examples**

        >>> g = f[0:1]
        >>> g = f[1:-4]
        >>> g = f[:1]
        >>> g = f[1:]

        """
        return type(self)(list.__getslice__(self, i, j))

    def __getitem__(self, index):
        """Called to implement evaluation of f[index]

        f.__getitem_(index) <==> f[index]

        :Returns:

                If *index* is an integer then the corresponding
                construct element is returned. If *index* is a slice
                then a new {{class}} is returned, which may be empty.

        **Examples**

        >>> g = f[0]
        >>> g = f[-1:-4:-1]
        >>> g = f[2:2:2]

        """
        out = list.__getitem__(self, index)

        if isinstance(out, list):
            return type(self)(out)

        return out

    def __ne__(self, other):
        """The rich comparison operator ``!=``

        f.__ne__(x) <==> f != x

        {{List comparison}}

        Note that ``f != x`` is equivalent to ``not f.equals(x)``.

        :Returns:

            `bool`

        """
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
        """Close all files referenced by each construct in the list.

        Note that a closed file will be automatically reopened if its
        contents are subsequently required.

        :Returns:

            `None`

        **Examples**

        >>> f.close()

        """
        for f in self:
            f.close()

    def count(self, value):
        """Return number of occurrences of value.

        {{List comparison}}

        Note that ``fl.count(value)`` is equivalent to
        ``sum(f.equals(value) for f in fl)``.

        .. seealso:: `list.count`

        **Examples**

        >>> f = cf.{{class}}([a, b, c, a])
        >>> f.count(a)
        2
        >>> f.count(b)
        1
        >>> f.count('a string')
        0

        """
        return len([None for f in self if f.equals(value)])

    def index(self, value, start=0, stop=None):
        """Return first index of value.

        {{List comparison}}

        An exception is raised if there is no such construct.

        .. seealso:: `list.index`

        """
        if start < 0:
            start = len(self) + start

        if stop is None:
            stop = len(self)
        elif stop < 0:
            stop = len(self) + stop

        for i, f in enumerate(self[start:stop]):
            if f.equals(value):
                return i + start

        raise ValueError(f"{value!r} is not in {self.__class__.__name__}")

    def remove(self, value):
        """Remove first occurrence of value.

        {{List comparison}}

        .. seealso:: `list.remove`

        """
        for i, f in enumerate(self):
            if f.equals(value):
                del self[i]
                return

        raise ValueError(
            f"{self.__class__.__name__}.remove(x): x not in "
            f"{self.__class__.__name__}"
        )

    def sort(self, key=None, reverse=False):
        """Sort of the list in place.

        By default the list is sorted by the identities of its
        constructs, but any sort criteria can be specified with the
        *key* parameter.

        The sort is stable.

        .. versionadded:: 1.0.4

        .. seealso:: `reverse`

        :Parameters:

            key: function, optional
                Specify a function of one argument that is used to
                extract a comparison key from each construct. By
                default the list is sorted by construct identity,
                i.e. the default value of *key* is ``lambda x:
                x.identity()``.

            reverse: `bool`, optional
                If set to `True`, then the list elements are sorted as
                if each comparison were reversed.

        :Returns:

            `None`

        """
        if key is None:
            key = lambda f: f.identity()

        return super().sort(key=key, reverse=reverse)

    def copy(self, data=True):
        """Return a deep copy.

        ``f.copy()`` is equivalent to ``copy.deepcopy(f)``.

        :Returns:

            `{{class}}`
                The deep copy.

        **Examples**

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

        """
        return type(self)([f.copy(data=data) for f in self])

    @_deprecated_kwarg_check("traceback", version="3.0.0", removed_at="4.0.0")
    @_manage_log_level_via_verbosity
    def equals(
        self,
        other,
        rtol=None,
        atol=None,
        verbose=None,
        ignore_data_type=False,
        ignore_fill_value=False,
        ignore_properties=None,
        ignore_compression=False,
        ignore_type=False,
        ignore=(),
        traceback=False,
        unordered=False,
    ):
        """Whether two lists are the same.

        Equality requires the two lists to have the same length and
        for the construct elements to be equal pair-wise, using their
        `!equals` methods.

        Any type of object may be tested but, in general, equality is
        only possible with another {{class}}, or a subclass of
        one. See the *ignore_type* parameter.

        Equality is between the constructs is strict by default. This
        means that for two constructs to be considered equal they must
        have corresponding metadata constructs and for each pair of
        constructs:

        * the same descriptive properties must be present, with the
          same values and data types, and vector-valued properties
          must also have same the size and be element-wise equal (see
          the *ignore_properties* and *ignore_data_type* parameters),
          and

        ..

        * if there are data arrays then they must have same shape and
          data type, the same missing data mask, and be element-wise
          equal (see the *ignore_data_type* parameter).

        {{equals tolerance}}

        If data arrays are compressed then the compression type and
        the underlying compressed arrays must be the same, as well as
        the arrays in their uncompressed forms. See the
        *ignore_compression* parameter.

        NetCDF elements, such as netCDF variable and dimension names,
        do not constitute part of the CF data model and so are not
        checked on any construct.

        :Parameters:
            other:
                The object to compare for equality.

            {{atol: number, optional}}

            {{rtol: number, optional}}

            {{ignore_fill_value: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            {{ignore_properties: (sequence of) `str`, optional}}

            {{ignore_data_type: `bool`, optional}}

            {{ignore_compression: `bool`, optional}}

            unordered: `bool`, optional
                If True then test that the lists contain equal
                constructs in any relative order. By default, construct
                order matters for the list comparison, such that each
                construct is tested for equality with the construct
                at the corresponding position in the list, pair-wise.

        :Returns:

            `bool`
                Whether the two lists are equal.

        **Examples**

        >>> fl.equals(fl)
        True
        >>> fl.equals(fl.copy())
        True
        >>> fl.equals(fl[:])
        True
        >>> fl.equals('a string')
        False

        """
        if ignore:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "equals",
                {"ignore": ignore},
                "Use keyword 'ignore_properties' instead.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        # Check for object identity
        if self is other:
            return True

        # Check that each object is of compatible type
        if ignore_type:
            if not isinstance(other, self.__class__):
                other = type(self)(source=other, copy=False)
        elif not isinstance(other, self.__class__):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Incompatible type: "
                    f"{other.__class__.__name__}"
                )  # pragma: no cover

            return False

        # Check that there are equal numbers of constructs
        len_self = len(self)
        if len_self != len(other):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Different numbers of "
                    f"constructs: {len_self}, {len(other)}"
                )  # pragma: no cover

            return False

        if not unordered or len_self == 1:
            # ----------------------------------------------------
            # Check the lists pair-wise
            # ----------------------------------------------------
            for i, (f, g) in enumerate(zip(self, other)):
                if not f.equals(
                    g,
                    rtol=rtol,
                    atol=atol,
                    ignore_fill_value=ignore_fill_value,
                    ignore_properties=ignore_properties,
                    ignore_compression=ignore_compression,
                    ignore_data_type=ignore_data_type,
                    ignore_type=ignore_type,
                    verbose=verbose,
                ):
                    if is_log_level_info(logger):
                        logger.info(
                            f"{self.__class__.__name__}: Different constructs "
                            f"at element {i}: {f!r}, {g!r}"
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
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Different sets of "
                        "identities: "
                        f"{set(self_identity)}, {set(other_identity)}"
                    )  # pragma: no cover

                return False

            # Check that there are the same number of variables
            # for each identity
            for identity, fl in self_identity.items():
                gl = other_identity[identity]
                if len(fl) != len(gl):
                    if is_log_level_info(logger):
                        logger.info(
                            f"{self.__class__.__name__}: Different numbers of "
                            f"{identity!r} {fl[0].__class__.__name__}s: "
                            f"{len(fl)}, {len(gl)}"
                        )  # pragma: no cover

                    return False

            # For each identity, check that there are matching pairs
            # of equal constructs.
            for identity, fl in self_identity.items():
                gl = other_identity[identity]

                for f in fl:
                    found_match = False
                    for i, g in enumerate(gl):
                        if f.equals(
                            g,
                            rtol=rtol,
                            atol=atol,
                            ignore_fill_value=ignore_fill_value,
                            ignore_properties=ignore_properties,
                            ignore_compression=ignore_compression,
                            ignore_data_type=ignore_data_type,
                            ignore_type=ignore_type,
                            verbose=verbose,
                        ):
                            found_match = True
                            del gl[i]
                            break

                if not found_match:
                    if is_log_level_info(logger):
                        logger.info(
                            f"{self.__class__.__name__}: No "
                            f"{g.__class__.__name__} equal to: {f!r}"
                        )  # pragma: no cover

                    return False

        # ------------------------------------------------------------
        # Still here? Then the lists are equal
        # ------------------------------------------------------------
        return True

    def select_by_identity(self, *identities):
        """Select list elements constructs by identity.

        To find the inverse of the selection, use a list comprehension
        with the `!match_by_identity` method of the constructs. For
        example, to select all constructs whose identity is *not*
        ``'air_temperature'``:

           >>> gl = cf.{{class}}(
           ...     x for x in fl if not f.match_by_identity('air_temperature')
           ... )

        .. versionadded:: 3.0.0

        .. seealso:: `select`, `__call__`, `select_by_ncvar`,
                     `select_by_property`,
                     `{{package}}.{{class}}.match_by_identity`

        :Parameters:

            identities: optional
                Select constructs from the list. By default all
                constructs are selected. May be one or more of:

                * A construct identity.

                  {{construct selection identity}}

                If no identities are provided then all list elements are
                selected.

                *Parameter example:*
                  ``'latitude'``

                *Parameter example:*
                  ``'long_name=Air Temperature'``

                *Parameter example:*
                  ``'air_pressure', 'longitude'``

        :Returns:

            `{{class}}`
                The matching constructs.

        **Examples**

        See `{{package}}.{{class}}.match_by_identity`

        """
        return type(self)(f for f in self if f.match_by_identity(*identities))

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def select(self, *identities, **kwargs):
        """Alias of `cf.{{class}}.select_by_identity`.

        To find the inverse of the selection, use a list comprehension
        with the `!match_by_identity` method of the constructs. For
        example, to select all constructs whose identity is *not*
        ``'air_temperature'``:

           >>> gl = cf.{{class}}(
           ...     f for f in fl if not f.match_by_identity('air_temperature')
           ... )

        .. seealso:: `select_by_identity`, `__call__`

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "select",
                kwargs,
                "Use methods 'select_by_units', 'select_by_construct', "
                "'select_by_properties', 'select_by_naxes', 'select_by_rank' "
                "instead.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        if identities and isinstance(identities[0], (list, tuple, set)):
            _DEPRECATION_ERROR(
                f"Use of a {identities[0].__class__.__name__!r} for "
                "identities has been deprecated. Use the * operator to "
                "unpack the arguments instead.",
                version="3.0.0",
                removed_at="4.0.0",
            )  # pragma: no cover

        for i in identities:
            if isinstance(i, dict):
                _DEPRECATION_ERROR_DICT(
                    "Use methods 'select_by_units', 'select_by_construct', "
                    "'select_by_properties', 'select_by_naxes', "
                    "'select_by_rank' instead.",
                    version="3.0.0",
                    removed_at="4.0.0",
                )  # pragma: no cover

            if isinstance(i, str) and ":" in i:
                error = True
                if "=" in i:
                    index0 = i.index("=")
                    index1 = i.index(":")
                    error = index0 > index1

                if error:
                    _DEPRECATION_ERROR(
                        f"The identity format {i!r} has been deprecated at "
                        f"version 3.0.0. Try {i.replace(':', '=', 1)!r} "
                        "instead.",
                        version="3.0.0",
                        removed_at="4.0.0",
                    )  # pragma: no cover

        return self.select_by_identity(*identities)
