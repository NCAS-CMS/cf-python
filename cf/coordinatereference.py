import logging

import cfdm
from cfdm import is_log_level_info

from . import CoordinateConversion, Datum
from .constants import cr_canonical_units, cr_coordinates, cr_default_values
from .data.data import Data
from .decorators import (
    _deprecated_kwarg_check,
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _manage_log_level_via_verbosity,
)
from .functions import (
    _DEPRECATION_ERROR_ATTRIBUTE,
    _DEPRECATION_ERROR_METHOD,
    allclose,
)
from .functions import atol as cf_atol
from .functions import inspect as cf_inspect
from .functions import rtol as cf_rtol
from .query import Query

_units = {}

logger = logging.getLogger(__name__)


def _totuple(a):
    """Return an N-d (N>0) array as a nested tuple of Python scalars.

    :Parameters:

        a: numpy.ndarray
            The numpy array

    :Returns:

        `tuple`
            The array as an nested tuple of Python scalars.

    """
    try:
        return tuple(_totuple(i) for i in a)
    except TypeError:
        return a


class CoordinateReference(cfdm.CoordinateReference):
    """A coordinate reference construct of the CF data model.

    A coordinate reference construct relates the coordinate values of
    the coordinate system to locations in a planetary reference frame.

    The domain of a field construct may contain various coordinate
    systems, each of which is constructed from a subset of the
    dimension and auxiliary coordinate constructs. For example, the
    domain of a four-dimensional field construct may contain
    horizontal (y-x), vertical (z), and temporal (t) coordinate
    systems. There may be more than one of each of these, if there is
    more than one coordinate construct applying to a particular
    spatiotemporal dimension (for example, there could be both
    latitude-longitude and y-x projection coordinate systems). In
    general, a coordinate system may be constructed implicitly from
    any subset of the coordinate constructs, yet a coordinate
    construct does not need to be explicitly or exclusively associated
    with any coordinate system.

    A coordinate system of the field construct can be explicitly
    defined by a coordinate reference construct which relates the
    coordinate values of the coordinate system to locations in a
    planetary reference frame and consists of the following:

    * References to the dimension coordinate and auxiliary coordinate
      constructs that define the coordinate system to which the
      coordinate reference construct applies. Note that the coordinate
      values are not relevant to the coordinate reference construct,
      only their properties.

    ..

    * A definition of a datum specifying the zeroes of the dimension
      and auxiliary coordinate constructs which define the coordinate
      system. The datum may be implied by the metadata of the
      referenced dimension and auxiliary coordinate constructs, or
      explicitly provided.

    ..

    * A coordinate conversion, which defines a formula for converting
      coordinate values taken from the dimension or auxiliary
      coordinate constructs to a different coordinate system. A
      coordinate reference construct relates the coordinate values of
      the field to locations in a planetary reference frame.


    **NetCDF interface**

    The netCDF grid mapping variable name of a coordinate reference
    construct may be accessed with the `nc_set_variable`,
    `nc_get_variable`, `nc_del_variable` and `nc_has_variable`
    methods.

    """

    def __new__(cls, *args, **kwargs):
        """Store component classes."""
        instance = super().__new__(cls)
        instance._CoordinateConversion = CoordinateConversion
        instance._Datum = Datum
        return instance

    def __getitem__(self, key):
        """Return a parameter value of the datum or the coordinate
        conversion.

        x.__getitem__(key) <==> x[key]

        If the same parameter exists in both the datum and the coordinate
        conversion then an exception is raised.

        .. seealso:: `get`, `coordinate_conversion.get_parameter`,
                     `datum.get_parameter`

        """
        out = []
        try:
            out.append(self.coordinate_conversion.get_parameter(key))
        except ValueError:
            try:
                out.append(self.datum.get_parameter(key))
            except ValueError:
                pass

        if len(out) == 1:
            return out[0]

        if not out:
            raise KeyError(
                f"No {key!r} parameter exists in the coordinate conversion  "
                "nor the datum"
            )

        raise KeyError(
            f"{key!r} parameter exists in both the coordinate conversion and "
            "the datum"
        )

    def __hash__(self):
        """x.__hash__() <==> hash(x)"""
        #        if self.type == 'formula_terms':
        #            raise ValueError("Can't hash a formula_terms %s" %
        #                             self.__class__.__name__)

        h = sorted(self.items())  # TODO
        h.append(self.identity())

        return hash(tuple(h))

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    def _matching_values(self, value0, value1, basic=False):
        """Whether two coordinate reference construct identity values
        match.

        :Parameters:

            value0:
                The first value to be matched.

            value1:
                The second value to be matched.

        :Returns:

            `bool`
                Whether or not the two values match.

        """
        if isinstance(value0, Query):
            return bool(value0.evaluate(value1))  # TODO vectors

        try:
            # re.compile object
            return value0.search(value1)
        except (AttributeError, TypeError):
            return self._equals(value1, value0, basic=basic)

    # ----------------------------------------------------------------
    # Private attributes
    # ----------------------------------------------------------------
    @property
    def _coordinate_identities(self):
        """Return the identity for the coordinate reference construct.

        .. versionadded:: 3.0.0

        """
        return cr_coordinates.get(self.identity(), ())

    def has_bounds(self):
        """Returns False since coordinate reference constructs do not
        have cell bounds.

        **Examples**

        >>> c.has_bounds()
        False

        """
        return False

    #    def canonical(self, field=None):
    #        '''
    #        '''
    #        ref = self.copy()
    #
    #        for term, value in ref.parameters.iteritems():
    #            if value is None or isinstance(value, str):
    #                continue
    #
    #            canonical_units = self.canonical_units(term)
    #            if canonical_units is None:
    #                continue
    #
    #            if isinstance(canonical_units, str):
    #                # units is a standard_name of a coordinate
    #                if field is None:
    #                    raise ValueError("Set the field parameter")
    #                coord = field.coordinate(canonical_units, exact=True)
    #                if coord is not None:
    #                    canonical_units = coord.Units
    #
    #            if canonical_units is not None:
    #                units = getattr(value, 'Units', None)
    #                if units is not None:
    #                    if not canonical_units.equivalent(units):
    #                        raise ValueError("xasdddddddddddddd 87236768 TODO")
    #                    value.Units = canonical_units
    #
    #        return ref

    @classmethod
    def canonical_units(cls, term):
        """Return the canonical units for a standard CF coordinate
        conversion term.

        :Parameters:

            term: `str`
                The name of the term.

        :Returns:

            `Units` or `None`
                The canonical units, or `None` if there are not any.

        **Examples**

        >>> cf.CoordinateReference.canonical_units('perspective_point_height')
        <Units: m>
        >>> cf.CoordinateReference.canonical_units('ptop')
        None

        """
        return cr_canonical_units.get(term, None)

    def close(self):
        """Close all files referenced by coordinate conversion term
        values.

        Deprecated at version 3.14.0. All files are now
        automatically closed when not being accessed.

        :Returns:

            `None`

        **Examples**

        >>> c.close()

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "close",
            "All files are now automatically closed when not being accessed.",
            version="3.14.0",
            removed_at="5.0.0",
        )  # pragma: no cover

    @classmethod
    def default_value(cls, term):
        """Return the default value for an unset standard CF coordinate
        conversion term.

        :Parameters:

            term: `str`
                The name of the term.

        :Returns:

                The default value, or 0.0 if one is not available.

        **Examples**

        >>> cf.CoordinateReference.default_value('ptop')
        0.0
        >>> print(cf.CoordinateReference.default_value('north_pole_grid_latitude'))
        0.0

        """
        return cr_default_values.get(term, 0.0)

    @_deprecated_kwarg_check("traceback", version="3.0.0", removed_at="4.0.0")
    @_manage_log_level_via_verbosity
    def equivalent(
        self, other, atol=None, rtol=None, verbose=None, traceback=False
    ):
        """True if two coordinate references are logically equal, False
        otherwise.

        :Parameters:

            other: cf.CoordinateReference
                The object to compare for equality.

            {{atol: number, optional}}

            {{rtol: number, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

            traceback: deprecated at version 3.0.0
                Use the *verbose* parameter instead.

        :Returns:

            out: `bool`
                Whether or not the two objects are equivalent.

        **Examples**

        >>> a = cf.example_field(6)
        >>> b = cf.example_field(7)
        >>> r = a.coordinate_reference('coordinatereference0')
        >>> s = b.coordinate_reference('coordinatereference0')
        >>> r.equivalent(r)
        True
        >>> r.equivalent(s)
        False
        >>> s.equivalent(r)
        False
        >>> s.equivalent(s)
        True

        """
        if self is other:
            return True

        # Check that each instance is the same type
        if self.__class__ != other.__class__:
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Different types "
                    f"({self.__class__.__name__!r} != "
                    f"{other.__class__.__name__!r})"
                )  # pragma: no cover

            return False

        # ------------------------------------------------------------
        # Check the name
        # ------------------------------------------------------------
        if self.identity() != other.identity():
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Different identities "
                    f"({self.identity()!r} != {other.identity()!r})"
                )  # pragma: no cover

            return False

        # ------------------------------------------------------------
        # Check the domain ancillary terms
        # ------------------------------------------------------------
        ancillaries0 = self.coordinate_conversion.domain_ancillaries()
        ancillaries1 = other.coordinate_conversion.domain_ancillaries()
        if set(ancillaries0) != set(ancillaries1):
            if is_log_level_info(logger):
                logger.info(
                    f"{self.__class__.__name__}: Non-equivalent domain "
                    "ancillary terms"
                )  # pragma: no cover

            return False

        # Check that if one term is None then so is the other
        for term, value0 in ancillaries0.items():
            if (value0 is None) != (ancillaries1[term] is None):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Non-equivalent domain "
                        f"ancillary-valued term {term!r}"
                    )  # pragma: no cover

                return False

        # ------------------------------------------------------------
        # Check the parameter terms and their values
        # ------------------------------------------------------------
        if rtol is None:
            rtol = float(cf_rtol())

        if atol is None:
            atol = float(cf_atol())

        parameters0 = self.coordinate_conversion.parameters()
        parameters1 = other.coordinate_conversion.parameters()

        for term in set(parameters0).union(parameters1):
            value0 = parameters0.get(term, None)
            value1 = parameters1.get(term, None)

            if value1 is None and value0 is None:
                # Term is unset in self and other
                continue

            if value0 is None:
                # Term is unset in self
                value0 = self.default_value(term)

            if value1 is None:
                # Term is unset in other
                value1 = other.default_value(term)

            if not allclose(value0, value1, rtol=rtol, atol=atol):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Non-equivalent "
                        f"coordinate conversion parameter-valued term {term!r}"
                    )  # pragma: no cover

                return False

        parameters0 = self.datum.parameters()
        parameters1 = other.datum.parameters()

        for term in set(parameters0).union(parameters1):
            value0 = parameters0.get(term, None)
            value1 = parameters1.get(term, None)

            if value1 is None and value0 is None:
                # Term is unset in self and other
                continue

            if value0 is None:
                # Term is unset in self
                value0 = self.default_value(term)

            if value1 is None:
                # Term is unset in other
                value1 = other.default_value(term)

            if not allclose(value0, value1, rtol=rtol, atol=atol):
                if is_log_level_info(logger):
                    logger.info(
                        f"{self.__class__.__name__}: Non-equivalent datum "
                        f"parameter-valued term {term!r}"
                    )  # pragma: no cover

                return False

        # Still here?
        return True

    def get(self, key, default=None):
        """Return a parameter value of the datum or the coordinate
        conversion.

        .. versionadded:: 3.0.0

        .. seealso:: `coordinate_conversion.get_parameter`,
                     `datum.get_parameter`, `__getitem__`,

        :Parameters:

            key: `str`
                Coordinate reference construct key.

            default: optional
                Return the value of the *default* parameter if the
                specified key has not been set. If set to
                an `Exception` instance then it will be raised instead.

        :Returns:

                The parameter value.

        """
        try:
            return self[key]
        except KeyError:
            return default

    def inspect(self):
        """Inspect the attributes.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))  # pragma: no cover

    def match_by_identity(self, *identities):
        """Determine whether or not one of the identities matches.

        .. versionadded:: 3.0.0

        .. seealso:: `identities`, `match`

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
                Whether or not the coordinate reference matches one of the
                given identities.

        **Examples**

        >>> c.match_by_identity('time')

        >>> c.match_by_identity(re.compile('^air'))

        >>> c.match_by_identity('air_pressure', 'air_temperature')

        >>> c.match_by_identity('ncvar%t')

        """
        if not identities:
            return True

        self_identities = self.identities()

        x = self.coordinate_conversion.get_parameter("grid_mapping_name", None)
        if x is not None:
            self_identities.insert(0, x)

        x = self.coordinate_conversion.get_parameter("standard_name", None)
        if x is not None:
            self_identities.insert(0, x)

        ok = False
        for value0 in identities:
            for value1 in self_identities:
                ok = self._matching_values(value0, value1, basic=True)
                if ok:
                    break

            if ok:
                break

        return ok

    def match(self, *identities):
        """Alias for `cf.CoordinateReference.match_by_identity`"""
        return self.match_by_identity(*identities)

    @_deprecated_kwarg_check("i", version="3.0.0", removed_at="4.0.0")
    @_inplace_enabled(default=False)
    def change_identifiers(
        self,
        identity_map,
        coordinate=True,
        ancillary=True,
        strict=False,
        inplace=False,
        i=False,
    ):
        """Change the identifiers of a coordinate reference from a
        mapping.

        If an identifier is not in the provided mapping then it is
        set to `None` and thus effectively removed from the coordinate
        reference.

        :Parameters:

            identity_map: dict
                For example: ``{'dim2': 'dim3', 'aux2': 'latitude',
                'aux4': None}``

            strict: `bool`, optional
                If True then coordinate or domain ancillary identifiers
                not set in the *identity_map* dictionary are set to
                `None`. By default they are left unchanged.

            i: `bool`, optional

        :Returns:

            `None`

        **Examples**

        >>> r = cf.CoordinateReference('atmosphere_hybrid_height_coordinate',
        ...                             a='ncvar:ak',
        ...                             b='ncvar:bk')
        >>> r.coordinates
        {'atmosphere_hybrid_height_coordinate'}
        >>> r.change_coord_identitiers({
        ...     'atmosphere_hybrid_height_coordinate', 'dim1', 'ncvar:ak', 'aux0'
        ... })
        >>> r.coordinates
        {'dim1', 'aux0'}

        """
        r = _inplace_enabled_define_and_cleanup(self)

        if not identity_map and not strict:
            if inplace:
                r = None
            return r

        if strict:
            default = None

        if ancillary:
            for (
                term,
                identifier,
            ) in r.coordinate_conversion.domain_ancillaries().items():
                if not strict:
                    default = identifier
                r.coordinate_conversion.set_domain_ancillary(
                    term, identity_map.get(identifier, default), copy=False
                )

        if coordinate:
            for identifier in r.coordinates():
                if not strict:
                    default = identifier

                r.del_coordinate(identifier)
                r.set_coordinate(identity_map.get(identifier, default))

        r.del_coordinate(None, None)

        return r

    def structural_signature(self, rtol=None, atol=None):
        """Return the structural signature of a coordinate reference.

        :Return:

            `tuple`

        """
        if rtol is None:
            rtol = float(cf_rtol())

        if atol is None:
            atol = float(cf_atol())

        s = [self.identity()]
        append = s.append

        for component in ("datum", "coordinate_conversion"):
            x = getattr(self, component)
            for term, value in sorted(x.parameters().items()):
                if isinstance(value, str):
                    append((component + ":" + term, value))
                    continue

                if value is None:
                    # Do not add an unset scalar or vector parameter value
                    # to the structural signature
                    continue

                value = Data.asdata(value, dtype=float)

                cu = self.canonical_units(term)
                if cu is not None:
                    if value.Units.equivalent(cu):
                        value.Units = cu
                    elif value.Units:
                        cu = value.Units
                else:
                    cu = value.Units

                if str(cu) in _units:
                    cu = _units[str(cu)]
                else:
                    ok = 0
                    for units in _units.values():
                        if cu.equals(units):
                            _units[str(cu)] = units
                            cu = units
                            ok = 1
                            break
                    if not ok:
                        _units[str(cu)] = cu

                if allclose(
                    value, self.default_value(term), rtol=rtol, atol=atol
                ):
                    # Do not add a default value to the structural signature
                    continue

                # Convert value to a Python scalar if it's 0-d, or a
                # tuple if it's N-d.
                value = value.array
                if not value.ndim:
                    value = value.item()
                else:
                    value = _totuple(value)

                append(
                    (
                        component + ":" + term,
                        value,
                        cu.formatted(definition=True),
                    )
                )

        # Add the domain ancillary-valued terms which have been set
        terms = self.coordinate_conversion.domain_ancillaries()
        append(
            tuple(
                sorted(
                    [
                        term
                        for term, value in terms.items()
                        if value is not None
                    ]
                )
            )
        )

        return tuple(s)

    # ----------------------------------------------------------------
    # Deprecated attributes and methods
    # ----------------------------------------------------------------
    def __delitem__(self, key):
        """x.__delitem__(key) <==> del x[key]

        Deprecated at version 3.0.0. Use method 'datum.del_parameter',
        'coordinate_conversion.del_parameter' or
        'coordinate_conversion.del_domain_ancillary' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "__getitem__",
            "Use method 'datum.del_parameter', "
            "'coordinate_conversion.del_parameter' or "
            "'coordinate_conversion.del_domain_ancillary' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def conversion(self):
        """Deprecated at version 3.0.0.

        Use attribute 'coordinate_conversion' instead.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "conversion",
            "Use attribute 'coordinate_conversion' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def hasbounds(self):
        """False. Coordinate reference objects do not have cell bounds.

        Deprecated at version 3.0.0. Use method 'has_bounds' instead.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "hasbounds",
            "Use method 'has_bounds' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def ancillaries(self):
        """Deprecated at version 3.0.0.

        Use the 'coordinate_conversion.domain_ancillaries' method
        instead.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "ancillaries",
            "Use the 'coordinate_conversion.domain_ancillaries' method "
            "instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    @property
    def parameters(self):
        """Deprecated at version 3.0.0.

        Use methods 'coordinate_conversion.parameters' and
        'datum.parameters' instead.

        """
        _DEPRECATION_ERROR_ATTRIBUTE(
            self,
            "parameters",
            "Use methods 'coordinate_conversion.parameters' and "
            "'datum.parameters' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def clear(self, coordinates=True, parameters=True, ancillaries=True):
        """Deprecated at version 3.0.0.

        Use methods 'coordinate_conversion.parameters' and
        'datum.parameters' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "parameters",
            "Use methods 'coordinate_conversion.parameters' and "
            "'datum.parameters' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def name(self, default=None, identity=False, ncvar=False):
        """Return a name.

        Deprecated at version 3.0.0. Use the 'identity' method instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "name",
            "Use the 'identity' method instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover

    def all_identifiers(self):
        """Deprecated at version 3.0.0."""
        _DEPRECATION_ERROR_METHOD(
            self, "all_identifiers", version="3.0.0", removed_at="4.0.0"
        )  # pragma: no cover

    def set_term(self, term_type, term, value):
        """Deprecated at version 3.0.0.

        Use method 'datum.set_parameter',
        'coordinate_conversion.set_parameter' or
        'coordinate_conversion.set_domain_ancillary' instead.

        """
        _DEPRECATION_ERROR_METHOD(
            self,
            "set_term",
            "Use method 'datum.set_parameter', "
            "'coordinate_conversion.set_parameter' or "
            "'coordinate_conversion.set_domain_ancillary' instead.",
            version="3.0.0",
            removed_at="4.0.0",
        )  # pragma: no cover
