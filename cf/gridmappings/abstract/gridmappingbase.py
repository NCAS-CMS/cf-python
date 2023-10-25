import itertools
import re

from pyproj import CRS

from ...constants import cr_canonical_units, cr_gm_valid_attr_names_are_numeric
from ...data import Data
from ...data.utils import is_numeric_dtype
from ...units import Units


PROJ_PREFIX = "+proj"

"""
Define this first since it provides the default for several parameters,
e.g. WGS1984_CF_ATTR_DEFAULTS.semi_major_axis is 6378137.0, the radius
of the Earth in metres. Note we use the 'merc' projection to take these
from since that projection includes all the attributes given for
'latlon' instead and with identical values, but also includes further
map parameters with defaults applied across the projections.

At the time of dedicating the code, the value of this is as follows, and
the values documented as defaults in the docstrings are taken from this:

{'crs_wkt': '<not quoted here due to long length>',
 'semi_major_axis': 6378137.0,
 'semi_minor_axis': 6356752.314245179,
 'inverse_flattening': 298.257223563,
 'reference_ellipsoid_name': 'WGS 84',
 'longitude_of_prime_meridian': 0.0,
 'prime_meridian_name': 'Greenwich',
 'geographic_crs_name': 'unknown',
 'horizontal_datum_name': 'World Geodetic System 1984',
 'projected_crs_name': 'unknown',
 'grid_mapping_name': 'mercator',
 'standard_parallel': 0.0,
 'longitude_of_projection_origin': 0.0,
 'false_easting': 0.0,
 'false_northing': 0.0,
 'scale_factor_at_projection_origin': 1.0}
"""
WGS1984_CF_ATTR_DEFAULTS = CRS.from_proj4("+proj=merc").to_cf()


def convert_proj_angular_data_to_cf(proj_data, context=None):
    """Convert a PROJ angular data component into CF Data with CF Units.

    PROJ units for latitude and longitude are in
    units of decimal degrees, where forming a string by adding
    a suffix character indicates alternative units of
    radians if the suffix is 'R' or 'r'. If a string, a suffix
    of 'd', 'D' or '°' confirm units of decimal degrees.

    Note that `cf.convert_cf_angular_data_to_proj` and
    this function are not strict inverse functions,
    since the former will always convert to the *simplest*
    way to specify the PROJ input, namely with no suffix for
    degrees(_X) units and the 'R' suffix for radians, whereas
    the input might have 'D' or 'r' etc. instead.

    .. versionadded:: GMVER

    :Parameters:

        proj_data: `str`
            The PROJ angular data component, for example "90", "90.0",
             "90.0D", "1.0R", or "1r".

            For details on valid PROJ data components in PROJ strings,
            notably indicating units, see:

            https://proj.org/en/9.2/usage/projections.html#projection-units

        context: `str` or `None`, optional
            The physical context of the conversion, where 'lat' indicates
            a latitude value and 'lon' indicates a longitude, such that
            indication of either context will return cf.Data with values
            having units appropriate to that context, namely 'degrees_north'
            or 'degrees_east' respectively. If None, 'degrees' or 'radians'
            (depending on the input PROJ units) will be the units.
            The default is None.

    :Returns:

        `Data`
            A cf.Data object with CF-compliant units that corresponds
            to the PROJ data and the context provided.

    """
    cf_compatible = True  # unless find otherwise (True unless proven False)
    if context == "lat":
        cf_units = "degrees_north"
    elif context == "lon":
        cf_units = "degrees_east"
    else:
        # From the CF Conventions Document (3.1. Units):
        # "The COARDS convention prohibits the unit degrees altogether,
        # but this unit is not forbidden by the CF convention because
        # it may in fact be appropriate for a variable containing, say,
        # solar zenith angle. The unit degrees is also allowed on
        # coordinate variables such as the latitude and longitude
        # coordinates of a transformed grid. In this case the coordinate
        # values are not true latitudes and longitudes which must always
        # be identified using the more specific forms of degrees as
        # described in Section 4.1.
        cf_units = "degrees"

    # Only valid input is a valid float or integer (digit with zero or one
    # decimal point only) optionally followed by a single suffix letter
    # indicating decimal degrees or radians with PROJ. Be strict about an
    # exact regex match, because anything not following the pattern (e.g.
    # something with extra letters) will be ambiguous for PROJ units.
    valid_form = re.compile("(-?\d+(\.\d*)?)([rRdD°]?)")
    form = re.fullmatch(valid_form, proj_data)
    if form:
        comps = form.groups()
        suffix = None
        if len(comps) == 3:
            value, float_comp, suffix = comps
        else:
            value, *float_comp = comps

        # Convert string value to relevant numeric form
        if float_comp:
            numeric_value = float(value)
        else:
            numeric_value = int(value)

        if suffix in ("r", "R"):  # radians units
            if context:
                # Convert so we can apply degree_X form of the lat/lon context
                numeric_value = Units.conform(
                    numeric_value, Units("radians"), Units("degrees")
                )
            else:  # Otherwise leave as radians to avoid rounding etc.
                cf_units = "radians"
        elif suffix and suffix not in ("d", "D", "°"):  # 'decimal degrees'
            cf_compatible = False
    else:
        cf_compatible = False

    if not cf_compatible:
        raise ValueError(
            f"PROJ data input not valid: {proj_data}. Ensure a valid "
            "PROJ value and optionally units are supplied."
        )

    return Data(numeric_value, Units(cf_units))


def convert_cf_angular_data_to_proj(data):
    """Convert singleton angular CF Data into a PROJ data component.

    PROJ units for latitude and longitude are in
    units of decimal degrees, where forming a string by adding
    a suffix character indicates alternative units of
    radians if the suffix is 'R' or 'r'. If a string, a suffix
    of 'd', 'D' or '°' confirm units of decimal degrees.

    Note that this function and `convert_proj_angular_data_to_cf`
    are not strict inverse functions, since this function
    will always convert to the *simplest*
    way to specify the PROJ input, namely with no suffix for
    degrees(_X) units and the 'R' suffix for radians, whereas
    the input might have 'D' or 'r' etc. instead.

    .. versionadded:: GMVER

    :Parameters:

        data: `Data`
            A cf.Data object of size 1 containing an angular value
            with CF-compliant angular units, for example
            cf.Data(45, units="degrees_north").

    :Returns:

        `str`
            A PROJ angular data component that corresponds
            to the Data provided.

    """
    if data.size != 1:
        raise ValueError(
            f"Input cf.Data must have size 1, got size: {data.size}"
        )
    if not is_numeric_dtype(data):
        raise TypeError(
            f"Input cf.Data must have numeric data type, got: {data.dtype}"
        )

    units = data.Units
    if not units:
        raise ValueError(
            "Must provide cf.Data with units for unambiguous conversion."
        )
    units_str = units.units

    degrees_unit_prefix = ["degree", "degrees"]
    # Taken from 4.1. Latitude Coordinate and 4.2. Longitude Coordinate:
    # http://cfconventions.org/cf-conventions/cf-conventions.html...
    # ...#latitude-coordinate and ...#longitude-coordinate
    valid_cf_lat_lon_units = [
        s + e
        for s, e in itertools.product(
            degrees_unit_prefix, ("_north", "_N", "N", "_east", "_E", "E")
        )
    ]
    valid_degrees_units = degrees_unit_prefix + valid_cf_lat_lon_units

    if units_str in valid_degrees_units:
        # No need for suffix 'D' for decimal degrees, as that is the default
        # recognised when no suffix is given
        proj_data = f"{data.data.array.item()}"
    elif units_str == "radians":
        proj_data = f"{data.data.array.item()}R"
    else:
        raise ValueError(
            "Unrecognised angular units set on the cf.Data. Valid options "
            f"are: {', '.join(valid_degrees_units)} and radians but got: "
            f"{units_str}"
        )

    return proj_data


def _make_proj_string_comp(spec):
    """Form a PROJ proj-string end from the given PROJ parameters.

    :Parameters:

        spec: `dict`
            A dictionary providing the proj-string specifiers for
            parameters, as keys, with their values as values. Values
            must be convertible to strings.

    """
    proj_string = ""
    for comp, value in spec.items():
        if not isinstance(value, str):
            try:
                value = str(value)
            except TypeError:
                raise TypeError(
                    "Can't create proj-string due to non-representable "
                    f"value {value} for key {comp}"
                )
        proj_string += f" +{comp}={value}"
    return proj_string


def _validate_map_parameter(mp_name, mp_value):
    """Validate map parameters for correct type and canonical units.

    :Parameters:

        mp_name: `str`
            The name of the map parameter to validate. It should be
            a name valid in this way in CF, namely one listed under
            the 'Table F.1. Grid Mapping Attributes' in Appendix
            F: Grid Mappings', therefore listed as a key in
            `cr_gm_valid_attr_names_are_numeric`, else a ValueError
            will be raised early.

        mp_value: `str`, `Data`, numeric or `None`
            The map parameter value being set for the given map
            parameter name. The type, and if numeric or `Data`,
            units, will be validated against the expected values
            for the given map parameter name.

    :Returns:

        `Data`, `str`, or `None`
            The map parameter value, assuming it passes validation,
            conformed to the canonical units of the map parameter
            name, if units are applicable.

    """
    # 0. Check input parameters are valid CF GM map parameters, not
    # for any case something unrecognised that will silently do nothing.
    if mp_name not in cr_gm_valid_attr_names_are_numeric:
        raise ValueError(
            "Unrecognised map parameter provided for the "
            f"Grid Mapping: {mp_name}"
        )

    # 1. If None, can return early:
    if mp_value is None:  # distinguish from 0 or 0.0 etc.
        return None

    # 2. Now ensure the type of the value is as expected.
    expect_numeric = cr_gm_valid_attr_names_are_numeric[mp_name]
    if expect_numeric:
        if (isinstance(mp_value, Data) and not is_numeric_dtype(mp_value)) or (
            not isinstance(mp_value, (Data, int, float))
        ):
            raise TypeError(
                f"Map parameter {mp_name} has an incompatible "
                "data type, expected numeric or Data but got "
                f"{type(mp_value)}"
            )
    elif not expect_numeric and not isinstance(mp_value, str):
        raise TypeError(
            f"Map parameter {mp_name} has an incompatible "
            "data type, expected a string but got "
            f"{type(mp_value)}"
        )

    # 3. Finally ensure the units are valid and conformed to the
    # canonical units, where numeric.
    if expect_numeric:
        canon_units = cr_canonical_units[mp_name]
        if isinstance(mp_value, Data):
            # In this case, is Data which may have units which aren't equal to
            # the canonical ones, so may need to conform the value.
            units = mp_value.Units
            # The units must be checked and might need to be conformed
            if not units.equivalent(canon_units):
                raise ValueError(
                    f"Map parameter {mp_name} value has units that "
                    "are incompatible with the expected units of "
                    f"{canon_units}: {units}"
                )
            elif not units.equals(canon_units):
                conforming_value = Units.conform(
                    mp_value.array.item(), units, canon_units
                )
            else:
                conforming_value = mp_value
        else:
            conforming_value = mp_value

        # Return numeric value as Data with conformed value and canonical units
        return Data(conforming_value, units=canon_units)
    else:
        # Return string value
        return mp_value


class GridMapping():
    """A container for a Grid Mapping recognised by the CF Conventions."""

    # The value of the 'grid_mapping_name' attribute.
    grid_mapping_name = None
    # The PROJ projection identifier shorthand name.
    proj_id = None

    def __init__(
        self,
        # i.e. WGS1984_CF_ATTR_DEFAULTS["reference_ellipsoid_name"], etc.
        reference_ellipsoid_name="WGS 84",
        # The next three parameters are non-zero floats so don't hard-code
        # WGS84 defaults in case of future precision changes:
        semi_major_axis=WGS1984_CF_ATTR_DEFAULTS["semi_major_axis"],
        semi_minor_axis=WGS1984_CF_ATTR_DEFAULTS["semi_minor_axis"],
        inverse_flattening=WGS1984_CF_ATTR_DEFAULTS["inverse_flattening"],
        prime_meridian_name="Greenwich",
        longitude_of_prime_meridian=0.0,
        earth_radius=None,
        **kwargs,
    ):
        """**Initialisation**

        :Parameters:

            reference_ellipsoid_name: `str` or `None`, optional
                The name of a built-in ellipsoid definition.
                The default is "WGS 84".

                .. note:: If used in conjunction with 'earth_radius',
                          the 'earth_radius' parameter takes precedence.

            inverse_flattening: number or scalar `Data`, optional
                The reverse flattening of the ellipsoid (PROJ 'rf'
                value), :math:`\frac{1}{f}`, where f corresponds to
                the flattening value (PROJ 'f' value) for the
                ellipsoid. Unitless, so `Data` must be unitless.
                The default is 298.257223563.

            prime_meridian_name: `str`, optional
                A predeclared name to define the prime meridian (PROJ
                'pm' value). The default is "Greenwich". Supported
                names and corresponding longitudes are listed at:

                https://proj.org/en/9.2/usage/
                projections.html#prime-meridian

                .. note:: If used in conjunction with
                          'longitude_of_prime_meridian', this
                          parameter takes precedence.

            longitude_of_prime_meridian: number or scalar `Data`, optional
                The longitude relative to Greenwich of the
                prime meridian. If provided as a number or `Data` without
                units, the units are taken as 'degrees_east', else the
                `Data` units are taken and must be angular and
                compatible with longitude. The default is 0.0
                degrees_east.

                .. note:: If used in conjunction with
                          'prime_meridian_name', the
                          'prime_meridian_name' parameter takes
                          precedence.

            semi_major_axis: number, scalar `Data` or `None`, optional
                The semi-major axis of the ellipsoid (PROJ 'a' value),
                in units of meters unless units are otherwise specified
                via `Data` units, in which case they must be conformable
                to meters and will be converted. The default is 6378137.0.

            semi_minor_axis: number, scalar `Data` or `None`, optional
                The semi-minor axis of the ellipsoid (PROJ 'b' value)
                in units of meters unless units are otherwise specified
                via `Data` units, in which case they must be conformable
                to meters and will be converted. The default is
                6356752.314245179.

            earth_radius: number, scalar `Data` or `None`, optional
                The radius of the ellipsoid, if a sphere (PROJ 'R' value),
                in units of meters unless units are otherwise specified
                via `Data` units, in which case they must be conformable
                to meters and will be converted.

                If the ellipsoid is not a sphere, then
                set as `None`, the default, to indicate that ellipsoid
                parameters such as the reference_ellipsoid_name or
                semi_major_axis and semi_minor_axis are being set,
                since these will should be used to define a
                non-spherical ellipsoid.

                .. note:: If used in conjunction with
                          'reference_ellipsoid_name', this parameter
                          takes precedence.

        """
        # Validate map parameters that are valid for any GridMapping.
        # These are attributes which describe the ellipsoid and prime meridian,
        # which may be included, when applicable, with any grid mapping, as
        # specified in Appendix F of the Conventions.
        self.earth_radius = _validate_map_parameter(
            "earth_radius", earth_radius
        )
        self.inverse_flattening = _validate_map_parameter(
            "inverse_flattening", inverse_flattening
        )
        self.longitude_of_prime_meridian = _validate_map_parameter(
            "longitude_of_prime_meridian", longitude_of_prime_meridian
        )
        self.prime_meridian_name = _validate_map_parameter(
            "prime_meridian_name", prime_meridian_name
        )
        self.reference_ellipsoid_name = _validate_map_parameter(
            "reference_ellipsoid_name", reference_ellipsoid_name
        )
        self.semi_major_axis = _validate_map_parameter(
            "semi_major_axis", semi_major_axis
        )
        self.semi_minor_axis = _validate_map_parameter(
            "semi_minor_axis", semi_minor_axis
        )

    @classmethod
    def is_latlon_gm(cls):
        """Whether the Grid Mapping is of LatitudeLongitude form.

        :Returns:

            `bool`
                True only if the Grid Mapping is LatitudeLongitude.

        """
        return False

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        # Report parent GridMapping class to indicate classification,
        # but only if it has one (> 2 avoids own class and 'object')
        # base. E.g. we get <CF AzimuthalGridMapping:Orthographic>,
        # <CF GridMapping:AzimuthalGridMapping>, <CF GridMapping>.
        parent_gm = ""
        if len(self.__class__.__mro__) > 2:
            parent_gm = self.__class__.__mro__[1].__name__ + ": "
        return f"<CF {parent_gm}{self.__class__.__name__}>"

    def __str__(self):
        """x.__str__() <==> str(x)"""
        return f"{self.__repr__()[:-1]} {self.get_proj_string()}>"

    def __eq__(self, other):
        """The rich comparison operator ``==``."""
        return self.get_proj_crs() == other.get_proj_crs()

    def __hash__(self, other):
        """The built-in function `hash`, x.__hash__() <==> hash(x)."""
        return hash(self.get_proj_crs())

    def get_proj_string(self, params=None):
        """The value of the PROJ proj-string defining the projection."""
        # TODO enable parameter input and sync'ing
        if not params:
            params = ""
        return f"{PROJ_PREFIX}={self.proj_id}{params}"

    def get_proj_crs(self):
        """Get the PROJ Coordinate Reference System.

        :Returns:

            `pyproj.crs.CRS`
                 The PROJ Coordinate Reference System defined with
                 a `pyproj` `CRS` class that corresponds to the
                 Grid Mapping instance.

        """
        return CRS.from_proj4(self.get_proj_string())

    def has_crs_wkt(self):
        """True if the Grid Mapping has a valid crs_wkt attribute set.

        :Returns:

            `bool`
                 Whether the Grid Mapping instance has a crs_wkt
                 attribute set.

       """
        return self.crs_wkt is not None
