'''Functions for computing non-parametric vertical coordinates from
the formula defined by a coordinate reference construct.

See Appendix D: Parametric Vertical Coordinates of the CF conventions.

.. versionaddedd:: 3.8.0

'''
import logging

from .units import Units

from .functions import combine_bounds_with_coordinates

from .constants import (
    formula_term_standard_names,
    formula_term_max_dimensions,
    formula_term_units,
    computed_standard_names,
)

logger = logging.getLogger(__name__)


def _domain_ancillary_term(f, standard_name, coordinate_conversion,
                           term, default_to_zero, bounds):
    '''Find a domain ancillary construct cooresponding to a formula term.

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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. If True the a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

        bounds: `bool`, optional
            If False then do not create bounds of zero for a default
            domain ancillary construct.

    :Returns:

        `DomainAncillary`, `str`
            The domain ancillary construct for the formula term and
            its construct key.

    '''
    units = formula_term_units[standard_name].get(term, None)
    if units is None:
        raise ValueError(
            "Can't calculate non-parametric vertical coordinates: "
            "{!r} is not a valid term".format(term)
        )

    units = Units(units)

    key = coordinate_conversion.get_domain_ancillary(term, None)
    if key is not None:
        var = f.domain_ancillary(key, default=None)
    else:
        var = None

    if var is not None:
        logger.detail(
            "  Formula term {!r}:\n{}".format(
                term, var.dump(display=False, _level=1)
            )
        )  # pragma: no cover

        vsn = var.get_property('standard_name', None)
        if (
                vsn is not None
                and vsn not in formula_term_standard_names[standard_name][term]
        ):
            raise ValueError(
                "Can't calculate non-parametric vertical coordinates: "
                "{!r} term {!r} has incorrect "
                "standard name: {!r}".format(term, var, vsn)
            )

        if var.ndim > formula_term_max_dimensions[standard_name][term]:
            raise ValueError(
                "Can't calculate non-parametric vertical coordinates: "
                "{!r} term {!r} has incorrect "
                "number of dimensions. Expected at most {}".format(
                    term, var, var.ndim
                )
            )

        if not var.Units.equivalent(units):
            raise ValueError(
                "Can't calculate non-parametric vertical coordinates: "
                "{!r} term {!r} has incorrect units: {!r} "
                "Expected units equivalent to {!r}".format(
                    term, var, var.Units, units
                )
            )
    else:
        if not default_to_zero:
            raise ValueError(
                "Can't calculate non-parametric vertical coordinates: "
                "No {!r} term domain ancillary construct and "
                "default_to_zero=False".format(term)
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

        logger.detail(
            "  Formula term {!r} (default):\n{}".format(
                term, var.dump(display=False, _level=1)
            )
        )  # pragma: no cover

    return var, key


def _coordinate_term(f, standard_name, coordinate_reference,
                     term=None):
    '''Find a coordinate construct cooresponding to a formula term.

    .. versionadded:: 3.8.0

    :Parameters:

        f: `Field`
            The parent field construct.

        standard_name: `str`
            The standard name of the parametric vertical coordinate.

        coordinate_reference: `CoordinateReference`
            A coordinate reference construct of the parent field
            construct.

        term: `str`
            A term of the formula.

            *Parameter example:*
              ``term='sigma'``

    :Returns:

        `Coordinate`, `str`
            The coordinate construct for the formula term, and its
            construct key. If the *term* is `None` and the coordinate
            construct does not exisit then `None`, `None` is returned
            instead.

    '''
    var = None
    key = None

    for key in coordinate_reference.coordinates():
        var = f.coordinate(key, default=None)
        if var is None:
            continue

        if var.get_property('standard_name', None) == standard_name:
            logger.detail(
                "  Parametric coordinates: {!r}".format(var)
            )  # pragma: no cover

            break

        var = None

    if term is not None:
        if var is None:
            raise ValueError(
                "Can't calculate non-parametric vertical coordinates: "
                "No {!r} term coordinate construct".format(term)
            )

        logger.detail(
            "  Formula term {!r} (default):\n{}".format(
                term, var.dump(display=False, _level=1)
            )
        )  # pragma: no cover

    return var, key


def _computed_standard_name(f, standard_name, coordinate_reference):
    '''Find the standard name of the computed non-parametric vertical
    coordinates.

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

        `str`
            The standard name of the computed vertical coordinate
            values. See Appendix D of the CF conventions.

    '''
    computed_standard_name = computed_standard_names[standard_name]
    if isinstance(computed_standard_name, str):
        # ------------------------------------------------------------
        # There is a unique computed standard name for this formula
        # ------------------------------------------------------------
        logger.detail(
            "  computed_standard_name: {!r}".format(computed_standard_name)
        )  # pragma: no cover

        return computed_standard_name

    # ----------------------------------------------------------------
    # Still here? Then the computed standard name depends on the
    # standdard name of one of the formula terms.
    # ----------------------------------------------------------------
    term, mapping = tuple(computed_standard_name.items())[0]

    computed_standard_name = None

    coordinate_conversion = coordinate_reference.coordinate_conversion
    key = coordinate_conversion.get_domain_ancillary(term, None)

    if key is not None:
        var = f.domain_ancillary(key, default=None)
        if var is not None:
            term_standard_name = var.get_property('standard_name', None)
            if term_standard_name is not None:
                for x, y in mapping.items():
                    if term_standard_name == x:
                        computed_standard_name = y
                        break
    # --- End: if

    if computed_standard_name is None:
        # ------------------------------------------------------------
        # As a last resort use the computed_standard_name property of
        # the parametric vertical coordinate construct
        # ------------------------------------------------------------
        var, _ = _coordinate_term(f, standard_name, coordinate_reference)
        if var is not None:
            computed_standard_name = var.get_property(
                'computed_standard_name', None
            )
    # --- End: if

    logger.detail(
        "  computed_standard_name: {}".format(
            repr(computed_standard_name) if computed_standard_name else ''
        )
    )  # pragma: no cover

    return computed_standard_name


def _vertical_axis(f, *keys):
    '''Find the vertical axis corresponding to the parametric vertical
    coordinates.

    .. versionadded:: 3.8.0

    :Parameters:

        f: `Field`
            The parent field construct.

        keys: one or more of `str` or `None`
            Construct keys of 1-d domain ancillary or coordinate
            construicts that span the vertical axis. If a key is
            `None` then that key is ignored.

    :Returns:

        `tuple`
            Either a 1-tuple containing the domain axis consrtuct key
            of the vertical axis, or an empty tuple if no such axis
            could be found.

    '''
    axis = ()
    for key in keys:
        if key is None:
            continue

        axis = f.get_data_axes(key)
        break

    logger.detail(
        "  Vertical axis: {!r}".format(axis)
    )  # pragma: no cover

    return axis


def _conform_eta(f, eta, eta_key, depth, depth_key):
    '''Trasnform the 'eta' term so that brodcasting will work with the
    'depth' term.

    This entails making dure that the trailing dimensions of 'eta' are
    the same as the all of dimensions of 'depth'.

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

    '''
    eta_axes = f.get_data_axes(eta_key, default=None)
    depth_axes = f.get_data_axes(depth_key, default=None)

    if eta_axes is not None and depth_axes is not None:
        if not set(eta_axes).issuperset(depth_axes):
            raise ValueError(
                "Can't calculate non-parametric coordinates: "
                "'depth' term {!r} axes must be a subset of "
                "'eta' term {!r} axes.".format(depth, eta)
            )

        eta_axes2 = depth_axes
        if len(eta_axes) > len(depth_axes):
            diff = [axis for axis in eta_axes
                    if axis not in depth_axes]
            eta_axes2 = tuple(diff) + depth_axes

        iaxes = [eta_axes.index(axis) for axis in eta_axes2]
        eta = eta.transpose(iaxes)

        eta_axes = eta_axes2

    logger.debug(
        "  Transposed domain ancillary 'eta': {!r}\n"
        "  Transposed domain ancillary 'eta' axes: {!r}".format(
            eta,
            eta_axes
        )
    )  # pragma: no cover

    return eta, eta_axes


def _conform_computed(f, computed, computed_axes, k_axis):
    '''Move the vertical axis of the computed non-parametric vertical
    coordinates from its current position as the last (rightmost)
    dimension, if applicable.

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

    '''
    ndim = computed.ndim
    if k_axis and ndim >= 2:
        iaxes = list(range(ndim - 1))

        time = f.domain_axis('T', key=True, default=None)
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

    logger.detail(
        "  Non-parametric coordinate axes: {!r}".format(computed_axes)
    )  # pragma: no cover

    return computed, computed_axes


def _conform_units(term, var, ref_term2, ref_units):
    '''Make sure that the units of a variable of the formula are the same
    as the units of another formula term.

    .. versionadded:: 3.8.0

    :Parameters:

        term: `str`
            A term of the formula for which the units are to
            conformed.

            *Parameter example:*
              ``term='z2'``

        var: `DomainAncillary` or `Coordinate`
            The variable for the *term* term whose units are to
            conformed to *ref_units*.

        ref_term: `str`
            A term of the formula which defines the reference units.

            *Parameter example:*
              ``term='href'``

        ref_units: `Units`
            The units of the *ref_term* term.

            *Parameter example:*
              ``ref_units=cf.Units('1')`

    :Returns:

        `DomainAncillary` or `Coordinate`
            The input *var* construct with conformed units.

    '''
    units = var.Units

    if units != ref_units:
        if not units.equivalent(ref_units):
            raise ValueError(
                "Terms {!r} and {!r} have incompatible units: "
                "{!r}, {!r} ".format(
                    ref_term, term, ref_units, units
                )
            )

        var = var.copy()
        var.Units = ref_units

    return var


def atmosphere_ln_pressure_coordinate(g, coordinate_reference,
                                      default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'atmosphere_ln_pressure_coordinate'

    coordinate_conversion = coordinate_reference.coordinate_conversion

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    p0, _ = _domain_ancillary_term(g, standard_name,
                                   coordinate_conversion, 'p0',
                                   default_to_zero, True)

    lev, lev_key = _coordinate_term(g, standard_name,
                                    coordinate_conversion,
                                    'lev')

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, lev_key)
    computed_axes = g.get_data_axes(lev_key) + k_axis

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(lev.has_bounds())

    computed = p0 * (-lev).exp()

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def atmosphere_sigma_coordinate(g, coordinate_reference,
                                default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'atmosphere_sigma_coordinate'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    ptop, _ = _domain_ancillary_term(g, standard_name,
                                     coordinate_conversion, 'ptop',
                                     default_to_zero, True)

    ps, ps_key = _domain_ancillary_term(g, standard_name,
                                        coordinate_conversion, 'ps',
                                        default_to_zero, True)

    sigma, sigma_key = _coordinate_term(g, standard_name,
                                        coordinate_conversion,
                                        'sigma')

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, sigma_key)
    computed_axes = g.get_data_axes(ps_key) + k_axis

    # Insert a size one dimension to allow broadcasting
    # over the vertical axis
    if k_axis:
        ps = ps.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(sigma.has_bounds())

    computed = (ptop
                + (ps - ptop) * sigma)

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def atmosphere_hybrid_sigma_pressure_coordinate(g,
                                                coordinate_reference,
                                                default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'atmosphere_hybrid_sigma_pressure_coordinate'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    ap_term = 'ap' in coordinate_conversion.domain_ancillaries()

    if ap_term:
        ap, ap_key = _domain_ancillary_term(g, standard_name,
                                            coordinate_conversion,
                                            'ap', default_to_zero,
                                            True)

        a_key = None
    else:
        a, a_key = _domain_ancillary_term(g, standard_name,
                                          coordinate_conversion, 'a',
                                          default_to_zero, True)

        p0, _ = _domain_ancillary_term(g, standard_name,
                                       coordinate_conversion, 'p0',
                                       default_to_zero, False)

        ap_key = None

    b, b_key = _domain_ancillary_term(g, standard_name,
                                      coordinate_conversion, 'b',
                                      default_to_zero, True)

    ps, ps_key = _domain_ancillary_term(g, standard_name,
                                        coordinate_conversion, 'ps',
                                        default_to_zero, False)

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, ap_key, a_key, b_key)
    computed_axes = g.get_data_axes(ps_key) + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        ps = ps.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    if ap_term:
        old = combine_bounds_with_coordinates(ap.has_bounds()
                                              and b.has_bounds())

        computed = (ap
                    + b * ps)
    else:
        old = combine_bounds_with_coordinates(a.has_bounds()
                                              and b.has_bounds())

        computed = (a * p0
                    + b * ps)

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def atmosphere_hybrid_height_coordinate(g, coordinate_reference,
                                        default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'atmosphere_hybrid_height_coordinate'

    coordinate_conversion = coordinate_reference.coordinate_conversion

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    a, a_key = _domain_ancillary_term(g, standard_name,
                                      coordinate_conversion, 'a',
                                      default_to_zero, True)

    b, b_key = _domain_ancillary_term(g, standard_name,
                                      coordinate_conversion, 'b',
                                      default_to_zero, True)

    orog, orog_key = _domain_ancillary_term(g, standard_name,
                                            coordinate_conversion,
                                            'orog', default_to_zero,
                                            False)

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, a_key, b_key)
    computed_axes = g.get_data_axes(orog_key) + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        orog = orog.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(a.has_bounds()
                                          and b.has_bounds())

    computed = (a
                + b * orog)

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def atmosphere_sleve_coordinate(g, coordinate_reference,
                                default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'atmosphere_sleve_coordinate'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    a, a_key = _domain_ancillary_term(g, standard_name,
                                      coordinate_conversion, 'a',
                                      default_to_zero, True)

    b1, b1_key = _domain_ancillary_term(g, standard_name,
                                        coordinate_conversion, 'b1',
                                        default_to_zero, True)

    b2, b2_key = _domain_ancillary_term(g, standard_name,
                                        coordinate_conversion, 'b2',
                                        default_to_zero, True)

    ztop, _ = _domain_ancillary_term(g, standard_name,
                                     coordinate_conversion, 'ztop',
                                     default_to_zero, False)

    zsurf1, zsurf1_key = _domain_ancillary_term(g, standard_name,
                                                coordinate_conversion,
                                                'zsurf1', False)

    zsurf2, zsurf2_key = _domain_ancillary_term(g, standard_name,
                                                coordinate_conversion,
                                                'zsurf2', False)

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
    k_axis = _vertical_axis(g, a_key, b1_key, b2_key)
    computed_axes = zsurf_axes + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        zsurf1 = zsurf1.insert_dimension(-1)
        zsurf2 = zsurf2.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(a.has_bounds()
                                          and b1.has_bounds()
                                          and b2.has_bounds())

    computed = (a * ztop
                + b1 * zsurf1
                + b2 * zsurf2)

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def ocean_sigma_coordinate(g, coordinate_reference, default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'ocean_sigma_coordinate'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    eta, eta_key = _domain_ancillary_term(g, standard_name,
                                          coordinate_conversion,
                                          'eta', False)

    depth, depth_key = _domain_ancillary_term(g, standard_name,
                                              coordinate_conversion,
                                              'depth',
                                              default_to_zero, False)

    sigma, sigma_key = _coordinate_term(g, standard_name,
                                        coordinate_conversion,
                                        'sigma')

    # Make sure that eta and depth are consistent, which may require
    # eta to be transposed.
    eta, eta_axes = _conform_eta(g, eta, eta_key, depth, depth_key)

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, sigma_key)
    computed_axes = eta_axes + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        eta = eta.insert_dimension(-1)
        depth = depth.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(sigma.has_bounds())

    computed = (eta
                + (eta + depth) * sigma)

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def ocean_s_coordinate(g, coordinate_reference, default_to_zero):
    '''Compute non-parametric vertical coordinates from ocean_s_coordinate
    parametric coordinates.

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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'ocean_s_coordinate'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    eta, eta_key = _domain_ancillary_term(g, standard_name,
                                          coordinate_conversion,
                                          'eta', default_to_zero,
                                          False)

    depth, depth_key = _domain_ancillary_term(g, standard_name,
                                              coordinate_conversion,
                                              'depth',
                                              default_to_zero, False)

    C, C_key = _domain_ancillary_term(g, standard_name,
                                      coordinate_conversion, 'C',
                                      default_to_zero, True)

    a, _ = _domain_ancillary_term(g, standard_name,
                                  coordinate_conversion, 'a',
                                  default_to_zero, True)

    b, _ = _domain_ancillary_term(g, standard_name,
                                  coordinate_conversion, 'b',
                                  default_to_zero, True)

    depth_c, _ = _domain_ancillary_term(g, standard_name,
                                        coordinate_conversion,
                                        'depth_c', default_to_zero,
                                        False)

    s, s_key = _coordinate_term(g, standard_name,
                                coordinate_conversion, 's')

    # Make sure that eta and depth are consistent, which may require
    # eta to be transposed.
    eta, eta_axes = _conform_eta(g, eta, eta_key, depth, depth_key)

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, s_key, C_key)
    computed_axes = eta_axes + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        eta = eta.insert_dimension(-1)
        depth = depth.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(s.has_bounds())

    # Ensure that a has the same units as s
    a = _conform_units('a', a, 's', s.Units)

    C = (
        (1 - b) * (a * s).sinh() / a.sinh()
        + b * (
            (a * (s + 0.5)).tanh() / (2 * (a * 0.5).tanh()) - 0.5
        )
    )

    computed = (eta * (s + 1)
                + depth_c * s
                + (depth - depth_c) * C)

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def ocean_s_coordinate_g1(g, coordinate_reference, default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'ocean_s_coordinate_g1'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    eta, eta_key = _domain_ancillary_term(g, standard_name,
                                          coordinate_conversion,
                                          'eta', False)

    depth, depth_key = _domain_ancillary_term(g, standard_name,
                                              coordinate_conversion,
                                              'depth',
                                              default_to_zero, False)

    depth_c, _ = _domain_ancillary_term(g, standard_name,
                                        coordinate_conversion,
                                        'depth_c', default_to_zero,
                                        False)

    C, C_key = _domain_ancillary_term(g, standard_name,
                                      coordinate_conversion, 'C',
                                      default_to_zero, True)

    s, s_key = _coordinate_term(g, standard_name,
                                coordinate_conversion, 's')

    # Make sure that eta and depth are consistent, which may require
    # eta to be transposed.
    eta, eta_axes = _conform_eta(g, eta, eta_key, depth, depth_key)

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, s_key, C_key)
    computed_axes = eta_axes + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        eta = eta.insert_dimension(-1)
        depth = depth.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(s.has_bounds()
                                          and C.has_bounds())

    S = (depth_c * s
         + (depth - depth_c) * C)

    computed = (S
                + eta * (1 + S / depth))

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def ocean_s_coordinate_g2(g, coordinate_reference, default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'ocean_s_coordinate_g2'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    eta, eta_key = _domain_ancillary_term(g, standard_name,
                                          coordinate_conversion,
                                          'eta', False)

    depth, depth_key = _domain_ancillary_term(g, standard_name,
                                              coordinate_conversion,
                                              'depth',
                                              default_to_zero, False)

    depth_c, _ = _domain_ancillary_term(g, standard_name,
                                        coordinate_conversion,
                                        'depth_c', default_to_zero,
                                        False)

    C, C_key = _domain_ancillary_term(g, standard_name,
                                      coordinate_conversion, 'C',
                                      default_to_zero, True)

    s, s_key = _coordinate_term(g, standard_name,
                                coordinate_conversion, 's')

    # Make sure that eta and depth are consistent, which may require
    # eta to be transposed.
    eta, eta_axes = _conform_eta(g, eta, eta_key, depth, depth_key)

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, s_key, C_key)
    computed_axes = eta_axes + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        eta = eta.insert_dimension(-1)
        depth = depth.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(s.has_bounds()
                                          and C.has_bounds())

    S = (
        (depth_c * s
         + depth * C)
        / (depth + depth_c)
    )

    computed = (eta
                + (eta + depth) * S)

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def ocean_sigma_z_coordinate(g, coordinate_reference, default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'ocean_sigma_z_coordinate'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    eta, eta_key = _domain_ancillary_term(g, standard_name,
                                          coordinate_conversion,
                                          'eta', False)

    depth, depth_key = _domain_ancillary_term(g, standard_name,
                                              coordinate_conversion,
                                              'depth',
                                              default_to_zero, False)

    depth_c, _ = _domain_ancillary_term(g, standard_name,
                                        coordinate_conversion,
                                        'depth_c', default_to_zero,
                                        False)

    nsigma, _ = _domain_ancillary_term(g, standard_name,
                                       coordinate_conversion,
                                       'nsigma', default_to_zero,
                                       False)

    zlev, zlev_key = _domain_ancillary_term(g, standard_name,
                                            coordinate_conversion,
                                            'zlev', default_to_zero,
                                            True)

    sigma, sigma_key = _coordinate_term(g, standard_name,
                                        coordinate_conversion,
                                        'sigma')

    # Make sure that eta and depth are consistent, which may require
    # eta to be transposed.
    eta, eta_axes = _conform_eta(g, eta, eta_key, depth, depth_key)

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, s_key, zlev_key)
    computed_axes = eta_axes + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        eta = eta.insert_dimension(-1)
        depth = depth.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    #
    # Note: This isn't overly efficient, because we do calculations
    #       for k>nsigma and then overwrite them.
    #
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(zlev.has_bounds()
                                          and sigma.has_bounds())

    computed = (
        eta
        + (depth.where(depth > depth_c, depth_c) + eta) * sigma
    )

    nsigma = int(nsigma.item())
    computed[..., nsigma:] = zlev[nsigma:]

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


def ocean_double_sigma_coordinate(g, coordinate_reference,
                                  default_to_zero):
    '''Compute non-parametric vertical coordinates from
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

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `str`, `str`, `DomainAncillary`, `tuple`, `tuple`
            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates.

            * The computed non-parametric coordinates in a
              `DomainAncillary` construct.

            * A tuple of the domain axis construct keys for the
              dimensions of the computed non-parametric coordinates.

            * A tuple containing the construct key of the vertical
              domain axis. If the vertical axis does not appear in the
              computed non-parametric coodinates then this an empty
              tuple, instead.

    '''
    standard_name = 'ocean_double_sigma_coordinate'

    computed_standard_name = _computed_standard_name(g, standard_name,
                                                     coordinate_reference)

    # ----------------------------------------------------------------
    # Get the formula terms and their contruct keys
    # ----------------------------------------------------------------
    depth, depth_key = _domain_ancillary_term(g, standard_name,
                                              coordinate_conversion,
                                              'depth',
                                              default_to_zero, False)

    a, _ = _domain_ancillary_term(g, standard_name,
                                  coordinate_conversion, 'a',
                                  default_to_zero, True)

    z2, _ = _domain_ancillary_term(g, standard_name,
                                   coordinate_conversion, 'z2',
                                   default_to_zero, True)

    z1, _ = _domain_ancillary_term(g, standard_name,
                                   coordinate_conversion, 'z1',
                                   default_to_zero, True)

    z2, _ = _domain_ancillary_term(g, standard_name,
                                   coordinate_conversion, 'z2',
                                   default_to_zero, True)

    k_c, _ = _domain_ancillary_term(g, standard_name,
                                    coordinate_conversion, 'k_c',
                                    default_to_zero, True)

    href, _ = _domain_ancillary_term(g, standard_name,
                                     coordinate_conversion, 'href',
                                     default_to_zero, True)

    sigma, sigma_key = _coordinate_term(g, standard_name,
                                        coordinate_conversion,
                                        'sigma')

    # Get the axes of the non-parametric coordinates, putting the
    # vertical axis in postition -1 (the rightmost position).
    k_axis = _vertical_axis(g, sigma_key)
    computed_axes = g.get_data_axes(depth_key) + k_axis

    # Insert a size one dimension to allow broadcasting over the
    # vertical axis
    if k_axis:
        depth = depth.insert_dimension(-1)

    # ----------------------------------------------------------------
    # Compute the non-parametric coordinates
    #
    # Note: This isn't overly efficient, because we do calculations
    #       for k<=k_c and then overwrite them.
    #
    # ----------------------------------------------------------------
    old = combine_bounds_with_coordinates(sigma.has_bounds())

    # Ensure that a, z1, z2, and href all have the same units as depth
    a = _conform_units('a', a, 'depth', depth.Units)
    z1 = _conform_units('z1', z1, 'depth', depth.Units)
    z2 = _conform_units('z2', z2, 'depth', depth.Units)
    href = _conform_units('href', href, 'depth', depth.Units)

    f = (
        (z1 + z2) * 0.5
        + (
            0.5
            * (z1 - z2)
            * (2 * a / (z1 - z2) * (depth - href)).tanh()
        )
    )

    computed = f * sigma

    k_c = int(k_c.item())
    computed[..., k_c:] = (f
                           + (sigma - 1) * (depth - f))

    combine_bounds_with_coordinates(old)

    return (standard_name, computed_standard_name, computed,
            computed_axes, k_axis)


# --------------------------------------------------------------------
# A mapping of parametric coordinate standard names to the functions
# for computing their correspoonding non-parametric vertical
# coordinates
# --------------------------------------------------------------------
_formula_functions = {
    'atmosphere_ln_pressure_coordinate': atmosphere_ln_pressure_coordinate,
    'atmosphere_sigma_coordinate': atmosphere_sigma_coordinate,
    'atmosphere_hybrid_sigma_pressure_coordinate': atmosphere_hybrid_sigma_pressure_coordinate,
    'atmosphere_hybrid_height_coordinate': atmosphere_hybrid_height_coordinate,
    'atmosphere_sleve_coordinate': atmosphere_sleve_coordinate,
    'ocean_sigma_coordinate': ocean_sigma_coordinate,
    'ocean_s_coordinate': ocean_s_coordinate,
    'ocean_s_coordinate_g1': ocean_s_coordinate_g1,
    'ocean_s_coordinate_g2': ocean_s_coordinate_g2,
    'ocean_sigma_z_coordinate': ocean_sigma_z_coordinate,
    'ocean_double_sigma_coordinate': ocean_double_sigma_coordinate,
}


def formula(f, coordinate_reference, default_to_zero=True):
    '''Compute non-parametric vertical coordinates.

    Dimensional vertical auxiliary coordinate values are computed from
    parametric vertical coordinate values (usually dimensionless) and
    associated domain ancillary constructs, as defined by the formula
    stored in a coordinate reference construct.

    See the "Parametric Vertical Coordinate" sections of the CF
    conventions for more details.

    .. versionadded:: 3.8.0

    :Parameters:

        f: `Field`
            The parent field construct.

        coordinate_reference: `CoordinateReference`
            A coordinate reference construct of the parent field
            construct.

        default_to_zero: `bool`, optional
            If False then do not assume that missing terms have a
            value of zero. By default a missing term is assumed to be
            zero, as described in Appendix D: Parametric Vertical
            Coordinates of the CF conventions.

    :Returns:

        `tuple`
            A 5-tuple containing

            * The standard name of the parametric coordinates.

            * The standard name of the computed non-parametric
              coordinates. The may be `None` if a computed standard
              name could not be found..

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

    '''
    standard_name = coordinate_reference.coordinate_conversion.get_parameter(
        'standard_name', None
    )

    if standard_name is not None:
        logger.detail(
            "  standard_name: {!r}".format(standard_name)
        )  # pragma: no cover

        if standard_name in _formula_functions:
            # --------------------------------------------------------
            # Compute the non-parametric vertical coordinates
            # --------------------------------------------------------
            (standard_name,
             computed_standard_name,
             computed,
             computed_axes,
             k_axis) = (
                 _formula_functions[standard_name](f,
                                                   coordinate_reference,
                                                   default_to_zero)
             )

            # --------------------------------------------------------
            # Move the vertical axis of the computed non-parametric
            # coordinates from its current position as the last
            # (rightmost) dimension, if applicable.
            # --------------------------------------------------------
            computed, computed_axes = _conform_computed(f, computed,
                                                        computed_axes, k_axis)

            return (standard_name, computed_standard_name, computed,
                    computed_axes, k_axis)
    # --- End: if

    # Still here?
    return (None,) * 5
