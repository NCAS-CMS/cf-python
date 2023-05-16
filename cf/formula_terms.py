import logging

from cfdm import is_log_level_debug, is_log_level_detail
from cfdm.core import DocstringRewriteMeta

from .constants import (
    formula_terms_computed_standard_names,
    formula_terms_D1,
    formula_terms_max_dimensions,
    formula_terms_standard_names,
    formula_terms_units,
)
from .docstring import _docstring_substitution_definitions
from .functions import bounds_combination_mode
from .units import Units

logger = logging.getLogger(__name__)


class FormulaTerms(metaclass=DocstringRewriteMeta):
    """Functions for computing non-parametric vertical coordinates from
    the formula defined by a coordinate reference construct.

    {{formula terms links}}

    .. versionaddedd:: 3.8.0

    """

    # The standard names for which there are formulas. There must be a
    # method with the same name.
    standard_names = (
        "atmosphere_ln_pressure_coordinate",
        "atmosphere_sigma_coordinate",
        "atmosphere_hybrid_sigma_pressure_coordinate",
        "atmosphere_hybrid_height_coordinate",
        "atmosphere_sleve_coordinate",
        "ocean_sigma_coordinate",
        "ocean_s_coordinate",
        "ocean_s_coordinate_g1",
        "ocean_s_coordinate_g2",
        "ocean_sigma_z_coordinate",
        "ocean_double_sigma_coordinate",
    )

    def __docstring_substitutions__(self):
        """Define docstring substitutions that apply to this class and
        all of its subclasses.

        These are in addtion to, and take precendence over, docstring
        substitutions defined by the base classes of this class.

        See `_docstring_substitutions` for details.

        .. versionadded:: 3.8.0

        .. seealso:: `_docstring_substitutions`

        :Returns:

            `dict`
                The docstring substitutions that have been applied.

        """
        return _docstring_substitution_definitions

    def __docstring_package_depth__(self):
        """Return the package depth for "package" docstring
        substitutions.

        See `_docstring_package_depth` for details.

        """
        return 0

    # ----------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------
    @staticmethod
    def _domain_ancillary_term(
        f,
        standard_name,
        coordinate_conversion,
        term,
        default_to_zero,
        strict,
        bounds,
    ):
        """Find a domain ancillary construct corresponding to a formula
        term.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            standard_name: `str`
                The standard name of the parametric vertical coordinate.

            coordinate_conversion: `CoordinateConversion`
                The definition of the formula

            term: `str`
                A term of the formula.

                *Parameter example:*
                  ``term='orog'``

            {{default_to_zero: `bool`, optional}}

            strict: `bool`
                If False then allow the computation to occur when

                * A term with a defined standard name (or names) has no
                  standard name.

                * When the computed standard name can not be found by
                  inference from the standard names of the domain
                  ancillary constructs; or from the
                  ``computed_standard_name`` property.

            bounds: `bool`, optional
                If False then do not create bounds of zero for a default
                domain ancillary construct.

        :Returns:

            `DomainAncillary`, `str`
                The domain ancillary construct for the formula term and
                its construct key.

        """
        units = formula_terms_units[standard_name].get(term, None)
        if units is None:
            raise ValueError(
                "Can't calculate non-parametric vertical coordinates: "
                f"{term!r} is not a valid term"
            )

        units = Units(units)

        key = coordinate_conversion.get_domain_ancillary(term, None)
        if key is not None:
            var = f.domain_ancillary(key, default=None)
        else:
            var = None

        if var is not None:
            if is_log_level_detail(logger):
                logger.detail(
                    f"Formula term {term!r}:\n"
                    f"{var.dump(display=False, _level=1)}"
                )  # pragma: no cover

            valid_standard_names = formula_terms_standard_names[standard_name][
                term
            ]

            vsn = var.get_property("standard_name", None)

            if vsn is None:
                if (
                    strict
                    and vsn not in valid_standard_names
                    and len(valid_standard_names) > 1
                ):
                    expected_names = ", ".join(
                        repr(x) for x in valid_standard_names
                    )
                    raise ValueError(
                        "Can't calculate non-parametric vertical coordinates: "
                        f"{term!r} term {var!r} has no standard name. "
                        f"Expected one of {expected_names}"
                    )
            elif vsn not in valid_standard_names:
                expected_names = ", ".join(
                    repr(x) for x in valid_standard_names
                )
                raise ValueError(
                    "Can't calculate non-parametric vertical coordinates: "
                    f"{term!r} term {var!r} has invalid standard name: "
                    f"{vsn!r}. Expected one of {expected_names}"
                )

            if var.ndim > formula_terms_max_dimensions[standard_name][term]:
                raise ValueError(
                    "Can't calculate non-parametric vertical coordinates: "
                    f"{term!r} term {var!r} has incorrect "
                    f"number of dimensions. Expected at most {var.ndim}"
                )

            if not var.Units.equivalent(units):
                raise ValueError(
                    "Can't calculate non-parametric vertical coordinates: "
                    f"{term!r} term {var!r} has incorrect units: "
                    f"{var.Units!r}. Expected units equivalent to {units!r}"
                )
        else:
            if not default_to_zero:
                raise ValueError(
                    "Can't calculate non-parametric vertical coordinates: "
                    f"No {term!r} term domain ancillary construct and "
                    "default_to_zero=False"
                )
            # ----------------------------------------------------
            # Create a default zero-valued domain ancillary
            # ----------------------------------------------------
            var = f._DomainAncillary()

            data = f._Data(0.0, units)
            var.set_data(data)

            if bounds:
                bounds = f._Bounds(data=f._Data((0.0, 0.0), units))
                var.set_bounds(bounds)

            if is_log_level_detail(logger):
                logger.detail(
                    f"Formula term {term!r} (default):\n"
                    f"{var.dump(display=False, _level=1)}"
                )  # pragma: no cover

        return var, key

    @staticmethod
    def _computed_standard_name(f, standard_name, coordinate_reference):
        """Find the standard name of the computed non-parametric
        vertical coordinates.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            standard_name: `str`
                The standard name of the parametric vertical coordinate.

            coordinate_reference: `CoordinateReference`
                A coordinate reference construct of the parent field
                construct.

        :Returns:

            `str` or `None`
                The standard name of the computed vertical coordinates, or
                `None` if one could not be found.

        """
        computed_standard_name = formula_terms_computed_standard_names[
            standard_name
        ]

        if isinstance(computed_standard_name, str):
            # ------------------------------------------------------------
            # There is a unique computed standard name for this formula
            # ------------------------------------------------------------
            if is_log_level_detail(logger):
                logger.detail(
                    f"computed_standard_name: {computed_standard_name!r}"
                )  # pragma: no cover

            return computed_standard_name

        # ----------------------------------------------------------------
        # Still here? Then see if the computed standard name depends on
        # the standard name of one of the formula terms.
        # ----------------------------------------------------------------
        term, mapping = tuple(computed_standard_name.items())[0]

        computed_standard_name = None

        coordinate_conversion = coordinate_reference.coordinate_conversion
        key = coordinate_conversion.get_domain_ancillary(term, None)

        if key is not None:
            var = f.domain_ancillary(key, default=None)
            if var is not None:
                term_standard_name = var.get_property("standard_name", None)
                if term_standard_name is not None:
                    for x, y in mapping.items():
                        if term_standard_name == x:
                            computed_standard_name = y
                            break

        if computed_standard_name is None:
            # ------------------------------------------------------------
            # See if the computed standard name is a set as a coordinate
            # conversion parameter
            # ------------------------------------------------------------
            computed_standard_name = coordinate_conversion.get_parameter(
                "computed_standard_name", None
            )

        if computed_standard_name is None:
            # ------------------------------------------------------------
            # As a last resort use the computed_standard_name property of
            # the parametric vertical coordinate construct
            # ------------------------------------------------------------
            var = None
            key = None

            for key in coordinate_reference.coordinates():
                var = f.coordinate(key, default=None)
                if var is None:
                    continue

                if var.get_property("standard_name", None) != standard_name:
                    continue

                computed_standard_name = var.get_property(
                    "computed_standard_name", None
                )
                break

        if is_log_level_detail(logger):
            logger.detail(
                "computed_standard_name: {}".format(
                    repr(computed_standard_name)
                    if computed_standard_name
                    else ""
                )
            )  # pragma: no cover

        return computed_standard_name

    @staticmethod
    def _vertical_axis(f, keys):
        """Find the vertical axis corresponding to the parametric
        vertical coordinates.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            keys: sequence of `str` or `None`
                The construct keys of 1-d domain ancillary or coordinate
                constructs that span the vertical axis. If a key is `None`
                then that key is ignored.

        :Returns:

            `tuple`
                Either a 1-tuple containing the domain axis construct key
                of the vertical axis, or an empty tuple if no such axis
                could be found.

        """
        axis = ()
        for key in keys:
            if key is None:
                continue

            axis = f.get_data_axes(key)
            break

        if is_log_level_detail(logger):
            logger.detail(f"Vertical axis: {axis!r}")  # pragma: no cover

        return axis

    @staticmethod
    def _conform_eta(f, eta, eta_key, depth, depth_key):
        """Transform the 'eta' term so that broadcasting will work with
        the 'depth' term.

        This entails making sure that the trailing dimensions of 'eta' are
        the same as all of the dimensions of 'depth'.

        For example, if we have ``eta(i,n,j)`` and ``depth(j,i)`` then eta
        will be transformed to ``eta(n,j,i)``.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            eta: `DomainAncillary`
                The 'eta' domain ancillary construct.

            eta_key: `str`
                The construct key of the 'eta' domain ancillary construct.

            depth: `DomainAncillary`
                The 'depth' domain ancillary construct.

            depth_key: `str`
                The construct key of the 'depth' domain ancillary
                construct.

        :Returns:

            `DomainAncillary`, `tuple`
                The conformed 'eta' domain ancillary construct for the
                formula term, and the domain axis construct keys of its
                dimensions.

        """
        eta_axes = f.get_data_axes(eta_key, default=None)
        depth_axes = f.get_data_axes(depth_key, default=None)

        if eta_axes is not None and depth_axes is not None:
            if not set(eta_axes).issuperset(depth_axes):
                raise ValueError(
                    "Can't calculate non-parametric coordinates: "
                    f"'depth' term {depth!r} axes must be a subset of "
                    f"'eta' term {eta!r} axes."
                )

            eta_axes2 = depth_axes
            if len(eta_axes) > len(depth_axes):
                diff = [axis for axis in eta_axes if axis not in depth_axes]
                eta_axes2 = tuple(diff) + depth_axes

            iaxes = [eta_axes.index(axis) for axis in eta_axes2]
            eta = eta.transpose(iaxes)

            eta_axes = eta_axes2

        if is_log_level_debug(logger):
            logger.debug(
                f"Transposed domain ancillary 'eta': {eta!r}\n"
                f"Transposed domain ancillary 'eta' axes: {eta_axes!r}"
            )  # pragma: no cover

        return eta, eta_axes

    @staticmethod
    def _conform_computed(f, computed, computed_axes, k_axis):
        """Move the vertical axis of the computed non-parametric
        vertical coordinates from its current position as the last
        (rightmost) dimension, if applicable.

        **Note:**

        * If the input computed coordinates do not span the vertical axis
          then they are returned unchanged.

        * If the input computed coordinates do span the vertical axis then
          it assumed to be the last (rightmost) dimension.

        * If the input computed coordinates span a unique time axis then
          the vertical axis is moved to position immediately to the right
          of it. Otherwise the vertical axis is moved to the first
          (leftmost) position.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            computed: `DomainAncillary`
                The computed non-parametric vertical coordinates.

            computed_axes: `tuple`
                The construct keys of the domain axes spanned by the
                computed coordinates.

            k_axis: `tuple`
                The construct key of the vertical domain axis. If the
                vertical axis does not appear in the computed
                non-parametric coodinates then this must be an empty
                tuple.

        :Returns:

            `DomainAncillary`, `tuple`
                The conformed computed non-parametric vertical coordinates
                construct and the domain axis construct keys of its
                dimensions.

        """
        ndim = computed.ndim
        if k_axis and ndim >= 2:
            iaxes = list(range(ndim - 1))

            time = f.domain_axis("T", key=True, default=None)
            if time in computed_axes:
                # Move it to the immediate right of the time
                # axis
                iaxes.insert(computed_axes.index(time) + 1, -1)
            else:
                # Move to to position 0, as there is no time
                # axis.
                iaxes.insert(0, -1)

            computed.transpose(iaxes, inplace=True)

            computed_axes = tuple([computed_axes[i] for i in iaxes])

        if is_log_level_detail(logger):
            logger.detail(
                f"Non-parametric coordinate axes: {computed_axes!r}"
            )  # pragma: no cover

        return computed, computed_axes

    @staticmethod
    def _conform_units(term, var, ref_term2, ref_units):
        """Make sure that the units of a variable of the formula are the
        same as the units of another formula term.

        .. versionadded:: 3.8.0

        :Parameters:

            term: `str`
                A term of the formula for which the units are to be
                conformed.

                *Parameter example:*
                  ``term='z2'``

            var: `DomainAncillary` or `Coordinate`
                The variable for the *term* term whose units are to
                conformed to *ref_units*.

            ref_term: `str`
                A term of the formula which defines the reference units.

                *Parameter example:*
                  ``ref_term='href'``

            ref_units: `Units`
                The units of the *ref_term* term.

                *Parameter example:*
                  ``ref_units=cf.Units('1')`

        :Returns:

            `DomainAncillary` or `Coordinate`
                The input *var* construct with conformed units.

        """
        units = var.Units

        if units != ref_units:
            if not units.equivalent(ref_units):
                raise ValueError(
                    f"Terms {ref_term2!r} and {term!r} have incompatible "
                    f"units: {ref_units!r}, {units!r}"
                )

            var = var.copy()
            var.Units = ref_units

        return var

    @staticmethod
    def _check_standard_name_consistency(
        strict, zlev=(None, None), eta=(None, None), depth=(None, None)
    ):
        """Check that there are consistent sets of values for the
        standard_names of formula terms and the computed_standard_name
        needed in defining the ocean sigma coordinate, the ocean
        s-coordinate, the ocean_sigma over z coordinate, and the ocean
        double sigma coordinate.

        .. versionadded:: 3.8.0

        :Parameters:

            zlev: `tuple`, optional
                The 'zlev' term domain ancillary construct and its
                construct key. If the domain ancillary construct was
                created from default values then the construct key will be
                `None`.

            eta: `tuple`, optional
                The 'eta' term domain ancillary construct and its
                construct key. If the domain ancillary construct was
                created from default values then the construct key will be
                `None`.

            depth: `tuple`, optional
                The 'depth' term domain ancillary construct and its
                construct key. If the domain ancillary construct was
                created from default values then the construct key will be
                `None`.

            strict: `bool`
                If False then allow the computation to occur even if none
                of the given the 'zlev', 'eta', and 'depth' domain
                ancillary constructs have a standard name. By default if
                any of these constructs is missing a standard name then an
                exception will be raised.

        :Returns:

            `None`
                A `ValueError` is raised if the standard names are
                inconsistent.

        """
        kwargs = locals()
        kwargs.pop("strict")

        indices = set()

        for term, (var, key) in kwargs.items():
            if var is None:
                continue

            standard_name = var.get_property("standard_name", None)

            if standard_name is None and key is None:
                continue

            try:
                indices.add(formula_terms_D1[term].index(standard_name))
            except ValueError:
                # Note that the existence of a standard name has already
                # been ensured by `_domain_ancillary_term`
                if strict:
                    raise ValueError(
                        f"{term!r} term {var!r} has invalid "
                        f"standard name: {standard_name!r}"
                    )

        if strict and not indices:
            raise ValueError(
                "Terms {} have no standard names. "
                "See Appendix D: Parametric Vertical Coordinates "
                "of the CF conventions.".format(
                    ", ".join(repr(term) for term in kwargs)
                )
            )

        if len(indices) > 1:
            raise ValueError(
                "Terms {} have incompatible standard names. "
                "See Appendix D: Parametric Vertical Coordinates "
                "of the CF conventions.".format(
                    ", ".join(repr(term) for term in kwargs)
                )
            )

    @staticmethod
    def _check_index_term(term, var):
        """Check that an index term contains a single integer.

        .. versionadded:: 3.8.0

        :Parameters:

            term: `str`
                A term of the formula.

                *Parameter example:*
                  ``term='nsigma'``

            eta: `DomainAncillary`
                The corresponding domain ancillary construct.

        :Returns:

            `None`
                A `ValueEerror` is raised if the index term does not
                contain a single integer.

        """
        if var.size != 1 or var.dtype.kind != "i":
            raise ValueError(
                "Can't calculate non-parametric vertical coordinates: "
                f"{term!r} term {var!r} doen't contain exactly one "
                "integer value"
            )

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    @classmethod
    def atmosphere_ln_pressure_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        atmosphere_ln_pressure_coordinate parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "atmosphere_ln_pressure_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        lev, lev_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "lev",
            default_to_zero,
            strict,
            True,
        )

        p0, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "p0",
            default_to_zero,
            strict,
            True,
        )

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [lev_key])
        computed_axes = k_axis

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        # ----------------------------------------------------------------
        with bounds_combination_mode("OR" if lev.has_bounds() else "AND"):
            computed = p0 * (-lev).exp()

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def atmosphere_sigma_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        atmosphere_sigma_coordinate parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "atmosphere_sigma_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        sigma, sigma_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "sigma",
            default_to_zero,
            strict,
            True,
        )

        ptop, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "ptop",
            default_to_zero,
            strict,
            True,
        )

        ps, ps_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "ps",
            default_to_zero,
            strict,
            True,
        )

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [sigma_key])
        computed_axes = g.get_data_axes(ps_key) + k_axis

        # Insert a size one dimension to allow broadcasting
        # over the vertical axis
        if k_axis:
            ps = ps.insert_dimension(-1)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        # ----------------------------------------------------------------
        with bounds_combination_mode("OR" if sigma.has_bounds() else "AND"):
            computed = ptop + (ps - ptop) * sigma

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def atmosphere_hybrid_sigma_pressure_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        atmosphere_hybrid_sigma_pressure_coordinate parametric
        coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "atmosphere_hybrid_sigma_pressure_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        ap_term = "ap" in coordinate_conversion.domain_ancillaries()

        if ap_term:
            ap, ap_key = cls._domain_ancillary_term(
                g,
                standard_name,
                coordinate_conversion,
                "ap",
                default_to_zero,
                strict,
                True,
            )

            a_key = None
        else:
            a, a_key = cls._domain_ancillary_term(
                g,
                standard_name,
                coordinate_conversion,
                "a",
                default_to_zero,
                strict,
                True,
            )

            p0, _ = cls._domain_ancillary_term(
                g,
                standard_name,
                coordinate_conversion,
                "p0",
                default_to_zero,
                strict,
                False,
            )

            ap_key = None

        b, b_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "b",
            default_to_zero,
            strict,
            True,
        )

        ps, ps_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "ps",
            default_to_zero,
            strict,
            False,
        )

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [ap_key, a_key, b_key])
        computed_axes = g.get_data_axes(ps_key) + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            ps = ps.insert_dimension(-1)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        # ----------------------------------------------------------------
        if ap_term:
            with bounds_combination_mode(
                "OR" if ap.has_bounds() and b.has_bounds() else "AND"
            ):
                computed = ap + b * ps
        else:
            with bounds_combination_mode(
                "OR" if a.has_bounds() and b.has_bounds() else "AND"
            ):
                computed = a * p0 + b * ps

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def atmosphere_hybrid_height_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        atmosphere_hybrid_height_coordinate parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "atmosphere_hybrid_height_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        a, a_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "a",
            default_to_zero,
            strict,
            True,
        )

        b, b_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "b",
            default_to_zero,
            strict,
            True,
        )

        orog, orog_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "orog",
            default_to_zero,
            strict,
            False,
        )

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [a_key, b_key])
        computed_axes = g.get_data_axes(orog_key) + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            orog = orog.insert_dimension(-1)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        # ----------------------------------------------------------------
        with bounds_combination_mode(
            "OR" if a.has_bounds() and b.has_bounds() else "AND"
        ):
            computed = a + b * orog

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def atmosphere_sleve_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        atmosphere_sleve_coordinate parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "atmosphere_sleve_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        a, a_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "a",
            default_to_zero,
            strict,
            True,
        )

        b1, b1_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "b1",
            default_to_zero,
            strict,
            True,
        )

        b2, b2_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "b2",
            default_to_zero,
            strict,
            True,
        )

        ztop, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "ztop",
            default_to_zero,
            strict,
            False,
        )

        zsurf1, zsurf1_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "zsurf1",
            default_to_zero,
            strict,
            False,
        )

        zsurf2, zsurf2_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "zsurf2",
            default_to_zero,
            strict,
            False,
        )

        # Make sure that zsurf1 and zsurf2 have the same number of axes
        # and the same axis order
        zsurf1_axes = g.get_data_axes(zsurf1_key, default=None)
        zsurf2_axes = g.get_data_axes(zsurf2_key, default=None)

        if zsurf1_axes is not None and zsurf2_axes is not None:
            if set(zsurf1_axes) != set(zsurf2_axes):
                raise ValueError(
                    "Can't calculate non-parametric coordinates: "
                    "'zsurf1' and 'zsurf2' terms "
                    "domain ancillaries span different domain axes"
                )

            iaxes = [zsurf2_axes.index(axis) for axis in zsurf1_axes]
            zsurf2 = zsurf2.transpose(iaxes)

            zsurf_axes = zsurf1_axes
        elif zsurf1_axes is not None:
            zsurf_axes = zsurf1_axes
        elif zsurf1_axes is not None:
            zsurf_axes = zsurf2_axes
        else:
            zsurf_axes = ()

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [a_key, b1_key, b2_key])
        computed_axes = zsurf_axes + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            zsurf1 = zsurf1.insert_dimension(-1)
            zsurf2 = zsurf2.insert_dimension(-1)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        # ----------------------------------------------------------------
        with bounds_combination_mode(
            "OR"
            if a.has_bounds() and b1.has_bounds() and b2.has_bounds()
            else "AND"
        ):
            computed = a * ztop + b1 * zsurf1 + b2 * zsurf2

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def ocean_sigma_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        ocean_sigma_coordinate parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "ocean_sigma_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        sigma, sigma_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "sigma",
            default_to_zero,
            strict,
            True,
        )

        eta, eta_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "eta",
            default_to_zero,
            strict,
            False,
        )

        depth, depth_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth",
            default_to_zero,
            strict,
            False,
        )

        cls._check_standard_name_consistency(
            strict, eta=(eta, eta_key), depth=(depth, depth_key)
        )

        # Make sure that eta and depth are consistent, which may require
        # eta to be transposed.
        eta, eta_axes = cls._conform_eta(g, eta, eta_key, depth, depth_key)

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [sigma_key])
        computed_axes = eta_axes + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            eta = eta.insert_dimension(-1)
            depth = depth.insert_dimension(-1)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        # ----------------------------------------------------------------
        with bounds_combination_mode("OR" if sigma.has_bounds() else "AND"):
            computed = eta + (eta + depth) * sigma

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def ocean_s_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        ocean_s_coordinate parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "ocean_s_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their construct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        s, s_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "s",
            default_to_zero,
            strict,
            True,
        )

        eta, eta_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "eta",
            default_to_zero,
            strict,
            False,
        )

        depth, depth_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth",
            default_to_zero,
            strict,
            False,
        )

        a, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "a",
            default_to_zero,
            strict,
            True,
        )

        b, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "b",
            default_to_zero,
            strict,
            True,
        )

        depth_c, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth_c",
            default_to_zero,
            strict,
            False,
        )

        cls._check_standard_name_consistency(
            strict, eta=(eta, eta_key), depth=(depth, depth_key)
        )

        # Make sure that eta and depth are consistent, which may require
        # eta to be transposed.
        eta, eta_axes = cls._conform_eta(g, eta, eta_key, depth, depth_key)

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [s_key])
        computed_axes = eta_axes + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            eta = eta.insert_dimension(-1)
            depth = depth.insert_dimension(-1)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates as
        # ----------------------------------------------------------------
        # Ensure that a has the same units as s
        a = cls._conform_units("a", a, "s", s.Units)

        with bounds_combination_mode("OR" if s.has_bounds() else "AND"):
            C = (1 - b) * (a * s).sinh() / a.sinh() + b * (
                (a * (s + 0.5)).tanh() / (2 * (a * 0.5).tanh()) - 0.5
            )

            computed = eta * (s + 1) + depth_c * s + (depth - depth_c) * C

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def ocean_s_coordinate_g1(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        ocean_s_coordinate_g1 parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "ocean_s_coordinate_g1"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        s, s_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "s",
            default_to_zero,
            strict,
            True,
        )

        C, C_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "C",
            default_to_zero,
            strict,
            True,
        )

        eta, eta_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "eta",
            default_to_zero,
            strict,
            False,
        )

        depth, depth_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth",
            default_to_zero,
            strict,
            False,
        )

        depth_c, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth_c",
            default_to_zero,
            strict,
            False,
        )

        cls._check_standard_name_consistency(
            strict, eta=(eta, eta_key), depth=(depth, depth_key)
        )

        # Make sure that eta and depth are consistent, which may require
        # eta to be transposed.
        eta, eta_axes = cls._conform_eta(g, eta, eta_key, depth, depth_key)

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [s_key, C_key])
        computed_axes = eta_axes + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            eta = eta.insert_dimension(-1)
            depth = depth.insert_dimension(-1)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        # ----------------------------------------------------------------
        with bounds_combination_mode(
            "OR" if s.has_bounds() and C.has_bounds() else "AND"
        ):
            S = depth_c * s + (depth - depth_c) * C

            computed = S + eta * (1 + S / depth)

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def ocean_s_coordinate_g2(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        ocean_s_coordinate_g2 parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "ocean_s_coordinate_g2"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        s, s_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "s",
            default_to_zero,
            strict,
            True,
        )

        eta, eta_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "eta",
            default_to_zero,
            strict,
            False,
        )

        depth, depth_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth",
            default_to_zero,
            strict,
            False,
        )

        depth_c, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth_c",
            default_to_zero,
            strict,
            False,
        )

        C, C_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "C",
            default_to_zero,
            strict,
            True,
        )

        cls._check_standard_name_consistency(
            strict, eta=(eta, eta_key), depth=(depth, depth_key)
        )

        # Make sure that eta and depth are consistent, which may require
        # eta to be transposed.
        eta, eta_axes = cls._conform_eta(g, eta, eta_key, depth, depth_key)

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [s_key, C_key])
        computed_axes = eta_axes + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            eta = eta.insert_dimension(-1)
            depth = depth.insert_dimension(-1)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        # ----------------------------------------------------------------
        with bounds_combination_mode(
            "OR" if s.has_bounds() and C.has_bounds() else "AND"
        ):
            S = (depth_c * s + depth * C) / (depth + depth_c)

            computed = eta + (eta + depth) * S

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def ocean_sigma_z_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        ocean_sigma_z_coordinate parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "ocean_sigma_z_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        sigma, sigma_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "sigma",
            default_to_zero,
            strict,
            True,
        )

        eta, eta_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "eta",
            default_to_zero,
            strict,
            False,
        )

        depth, depth_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth",
            default_to_zero,
            strict,
            False,
        )

        depth_c, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth_c",
            default_to_zero,
            strict,
            False,
        )

        nsigma, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "nsigma",
            default_to_zero,
            strict,
            False,
        )

        zlev, zlev_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "zlev",
            default_to_zero,
            strict,
            True,
        )

        cls._check_standard_name_consistency(
            strict,
            zlev=(zlev, zlev_key),
            eta=(eta, eta_key),
            depth=(depth, depth_key),
        )

        # Make sure that eta and depth are consistent, which may require
        # eta to be transposed.
        eta, eta_axes = cls._conform_eta(g, eta, eta_key, depth, depth_key)

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [sigma_key, zlev_key])
        computed_axes = eta_axes + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            eta = eta.insert_dimension(-1)
            depth = depth.insert_dimension(-1)

        cls._check_index_term("nsigma", nsigma)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        #
        # Note: This isn't overly efficient, because we do calculations
        #       for k>nsigma and then overwrite them.
        # ----------------------------------------------------------------
        with bounds_combination_mode(
            "OR" if zlev.has_bounds() and sigma.has_bounds() else "AND"
        ):
            computed = (
                eta + (depth.where(depth > depth_c, depth_c) + eta) * sigma
            )

            nsigma = int(nsigma.data)
            computed[..., nsigma:] = zlev[nsigma:]

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def ocean_double_sigma_coordinate(
        cls, g, coordinate_reference, default_to_zero, strict
    ):
        """Compute non-parametric vertical coordinates from
        ocean_double_sigma_coordinate parametric coordinates.

        .. note:: The vertical axis is the last (rightmost) dimension of
                  the returned computed non-parametric vertical
                  coordinates, if applicable.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                The coordinate reference construct of the parent field
                construct that defines the conversion formula.

            {{default_to_zero: `bool`, optional}}

        :Returns:

            {{Returns formula}}

        """
        standard_name = "ocean_double_sigma_coordinate"

        computed_standard_name = cls._computed_standard_name(
            g, standard_name, coordinate_reference
        )

        # ----------------------------------------------------------------
        # Get the formula terms and their contruct keys
        # ----------------------------------------------------------------
        coordinate_conversion = coordinate_reference.coordinate_conversion

        sigma, sigma_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "sigma",
            default_to_zero,
            strict,
            True,
        )

        depth, depth_key = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "depth",
            default_to_zero,
            strict,
            False,
        )

        z1, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "z1",
            default_to_zero,
            strict,
            True,
        )

        z2, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "z2",
            default_to_zero,
            strict,
            True,
        )

        a, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "a",
            default_to_zero,
            strict,
            True,
        )

        href, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "href",
            default_to_zero,
            strict,
            True,
        )

        k_c, _ = cls._domain_ancillary_term(
            g,
            standard_name,
            coordinate_conversion,
            "k_c",
            default_to_zero,
            strict,
            True,
        )

        # Get the axes of the non-parametric coordinates, putting the
        # vertical axis in postition -1 (the rightmost position).
        k_axis = cls._vertical_axis(g, [sigma_key])
        computed_axes = g.get_data_axes(depth_key) + k_axis

        # Insert a size one dimension to allow broadcasting over the
        # vertical axis
        if k_axis:
            depth = depth.insert_dimension(-1)

        cls._check_index_term("k_c", k_c)

        # ----------------------------------------------------------------
        # Compute the non-parametric coordinates
        #
        # Note: This isn't overly efficient, because we do calculations
        #       for k<=k_c and then overwrite them.
        # ----------------------------------------------------------------
        # Ensure that a, z1, z2, and href all have the same units as depth
        a = cls._conform_units("a", a, "depth", depth.Units)
        z1 = cls._conform_units("z1", z1, "depth", depth.Units)
        z2 = cls._conform_units("z2", z2, "depth", depth.Units)
        href = cls._conform_units("href", href, "depth", depth.Units)

        with bounds_combination_mode("OR" if sigma.has_bounds() else "AND"):
            f = (z1 + z2) * 0.5 + (
                0.5 * (z1 - z2) * (2 * a / (z1 - z2) * (depth - href)).tanh()
            )

            computed = f * sigma

            k_c1 = int(k_c.data) + 1

            computed[..., k_c1:] = f + (depth - f) * (sigma[k_c1:] - 1)

        return (
            standard_name,
            computed_standard_name,
            computed,
            computed_axes,
            k_axis,
        )

    @classmethod
    def formula(
        cls, f, coordinate_reference, default_to_zero=True, strict=True
    ):
        """Compute non-parametric vertical coordinates.

        Dimensional vertical auxiliary coordinate values are computed from
        parametric vertical coordinate values (usually dimensionless) and
        associated domain ancillary constructs, as defined by the formula
        stored in a coordinate reference construct.

        .. versionadded:: 3.8.0

        :Parameters:

            f: `Field`
                The parent field construct.

            coordinate_reference: `CoordinateReference`
                A coordinate reference construct of the parent field
                construct.

            {{default_to_zero: `bool`, optional}}

            strict: `bool`
                If False then allow the computation to occur when

                * A domain ancillary construct has no standard name, but
                  the corresponding term has a standard name that is
                  prescribed

                * When the computed standard name can not be found by
                  inference from the standard names of the domain
                  ancillary constructs, nor from the
                  ``computed_standard_name`` parameter of the relevant
                  coordinate reference construct.

                By default an exception is raised in these cases.

                If a domain ancillary construct does have a standard name,
                but one that is inconsistent with any prescribed standard
                names, then an exception is raised regardless of the value
                of *strict*.

        :Returns:

            5-`tuple`
                * The standard name of the parametric coordinates.

                * The standard name of the computed non-parametric
                  coordinates. This may be `None` if a computed standard
                  name could not be found.

                * The computed coordinates in a `DomainAncillary`
                  construct.

                * A tuple of the domain axis construct keys for the
                  dimensions of the computed non-parametric coordinates.

                * A tuple containing the construct key of the vertical
                  domain axis. If the vertical axis does not appear in the
                  computed non-parametric coodinates then this an empty
                  tuple, instead.

                If the coordinate reference does not define a conversion
                from parametric coordinates to nonparametric coordinates
                then a `None` is returned for all of the tuple elements.

        """
        standard_name = (
            coordinate_reference.coordinate_conversion.get_parameter(
                "standard_name", None
            )
        )

        if standard_name is not None:
            if is_log_level_detail(logger):
                logger.detail(
                    f"standard_name: {standard_name!r}"
                )  # pragma: no cover

            if standard_name in cls.standard_names:
                # --------------------------------------------------------
                # Compute the non-parametric vertical coordinates
                # --------------------------------------------------------
                (
                    standard_name,
                    computed_standard_name,
                    computed,
                    computed_axes,
                    k_axis,
                ) = getattr(cls, standard_name)(
                    f, coordinate_reference, default_to_zero, strict
                )

                # --------------------------------------------------------
                # Move the vertical axis of the computed non-parametric
                # coordinates from its current position as the last
                # (rightmost) dimension, if applicable.
                # --------------------------------------------------------
                computed, computed_axes = cls._conform_computed(
                    f, computed, computed_axes, k_axis
                )

                return (
                    standard_name,
                    computed_standard_name,
                    computed,
                    computed_axes,
                    k_axis,
                )

        # Still here?
        return (None,) * 5
