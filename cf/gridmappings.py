from pyproj import CRS

ALL_GRID_MAPPING_ATTR_NAMES = {
    "grid_mapping_name",
    # *Those which describe the ellipsoid and prime meridian:*
    "earth_radius",                       # PROJ '+R' value
    "inverse_flattening",                 # PROJ '+rf' value
    "longitude_of_prime_meridian",
    "prime_meridian_name",                # PROJ '+pm' value
    "reference_ellipsoid_name",           # PROJ '+ellps' value
    "semi_major_axis",                    # PROJ '+a' value
    "semi_minor_axis",                    # PROJ '+b' value
    # *Specific/applicable to only given grid mapping(s):*
    # ...projection origin related:
    "longitude_of_projection_origin",     # PROJ '+lon_0' value
    "latitude_of_projection_origin",      # PROJ '+lat_0' value
    "scale_factor_at_projection_origin",  # PROJ '+k_0' value
    # ...false-Xings:
    "false_easting",                      # PROJ '+x_0' value
    "false_northing",                     # PROJ '+y_0' value
    # ...angle axis related:
    "sweep_angle_axis",                   # PROJ '+sweep' value
    "fixed_angle_axis",
    # ...central meridian related:
    "longitude_of_central_meridian",
    "scale_factor_at_central_meridian",
    # ...pole coordinates related:
    "grid_north_pole_latitude",
    "grid_north_pole_longitude",
    "north_pole_grid_longitude",
    # ...other:
    "standard_parallel",                  # PROJ ['+lat_1', '+lat_2'] values
    "perspective_point_height",           # PROJ '+h' value
    "azimuth_of_central_line",            # PROJ '+alpha' (ignore '+gamma')
    "straight_vertical_longitude_from_pole",
    # *Other, not needed for a specific grid mapping but also listed
    # in 'Table F.1. Grid Mapping Attributes':*
    "crs_wkt",                            # PROJ 'crs_wkt' value
    "geographic_crs_name",
    "geoid_name",
    "geopotential_datum_name",
    "horizontal_datum_name",
    "projected_crs_name",
    "towgs84",                            # PROJ '+towgs84' value
}

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


"""Abstract classes for general Grid Mappings.

Note that default arguments are based upon the PROJ defaults, which can
be cross-referenced via running:

CRS.from_proj4("+proj=<proj_id> <minimal parameters>").to_cf()

where <minimal parameters> is for when required arguments must be provided
to return a coordinate reference instance, and obviously these values
where reported should not be included as defaults. An example is:

CRS.from_proj4("+proj=lcc +lat_1=1").to_cf()

where `'standard_parallel': (1.0, 0.0)` would not be taken as a default.

"""


class GridMapping:
    """A container for a Grid Mapping recognised by the CF Conventions."""

    def __init__(
        self,
        grid_mapping_name=None,
        proj_id=None,
        # i.e. WGS1984_CF_ATTR_DEFAULTS.reference_ellipsoid_name:
        reference_ellipsoid_name="WGS 84",
        semi_major_axis=WGS1984_CF_ATTR_DEFAULTS.semi_major_axis,
        semi_minor_axis=WGS1984_CF_ATTR_DEFAULTS.semi_minor_axis,
        inverse_flattening=WGS1984_CF_ATTR_DEFAULTS.inverse_flattening,
        prime_meridian_name=WGS1984_CF_ATTR_DEFAULTS.prime_meridian_name,
        longitude_of_prime_meridian=WGS1984_CF_ATTR_DEFAULTS.longitude_of_prime_meridian,
        earth_radius=None,
    ):
        """**Initialisation**

        :Parameters:

        Parameters to define the grid mapping:

            grid_mapping_name: `str`
                The value of the 'grid_mapping_name' attribute
                attached to a data variable, for example
                "mercator" to indicate the Mercator projection.

            proj_id: `str`
                The PROJ projection identifier shorthand name that
                corresponds to the specified 'grid_mapping_name'
                attribute, for example "merc" for the Mercator
                projection. This is the initial component in the
                PROJ 'proj-string' to describe the coordinate
                transformation.

                .. note:: Do not specify the full 'proj-string'
                          including parameters, since these are
                          calculated from the class input parameters,
                          and do not include the projection specifier
                          '+proj' as a prefix. Only give the
                          projection ID.

        Parameters to define the ellipsoid size and shape:

            reference_ellipsoid_name: `str` or `None`, optional
                The name of a built-in ellipsoid definition.
                The default is "WGS 84".

                .. note:: If used in conjunction with 'earth_radius',
                          the 'earth_radius' parameter takes precedence.

            inverse_flattening: number, optional
                The reverse flattening of the ellipsoid (PROJ 'rf'
                value), :math:`\frac{1}{f}`, where f corresponds to
                the flattening value (PROJ 'f' value) for the ellipsoid.
                Unitless. The default is 298.257223563.

            prime_meridian_name: `str`, optional
                A predeclared name to define the prime meridian (PROJ
                'pm' value). The default is "Greenwich". Supported
                names and corresponding longitudes are listed at:

                https://proj.org/en/9.2/usage/
                projections.html#prime-meridian

                .. note:: If used in conjunction with
                          'longitude_of_prime_meridian', this
                          parameter takes precedence.

            longitude_of_prime_meridian: `str or `None`, optional
                The longitude relative to Greenwich of the
                prime meridian. The default is 0.0.

                .. note:: If used in conjunction with
                          'prime_meridian_name', the
                          'prime_meridian_name' parameter takes
                          precedence.

            semi_major_axis: number or `None`, optional
                The semi-major axis of the ellipsoid (PROJ 'a' value)
                in units of meters. The default is 6378137.0.

            semi_minor_axis: number or `None`, optional
                The semi-minor axis of the ellipsoid (PROJ 'b' value)
                in units of meters. The default is 6356752.314245179.

            earth_radius: number or `None`, optional
                The radius of the ellipsoid, if a sphere (PROJ 'R' value),
                in units of meters. If the ellipsoid is not a sphere,
                set as `None`, the default, to indicate that ellipsoid
                parameters such as the reference_ellipsoid_name or
                semi_major_axis and semi_minor_axis are being set,
                since these take precendence.

                .. note:: If used in conjunction with
                          'reference_ellipsoid_name', this parameter
                          takes precedence.

        """
        if not grid_mapping_name and not proj_id:
            raise NotImplementedError(
                "Must define a specific Grid Mapping via setting its CF "
                "Conventions 'grid_mapping_name' attribute value with the "
                "grid_mapping_name parameter, as well as the corresponding "
                "base PROJ '+proj' identifier with the proj_id parameter."
            )

        # Defining the Grid Mapping
        self.grid_mapping_name = grid_mapping_name
        self.proj_id = proj_id

        # The attributes which describe the ellipsoid and prime meridian,
        # which may be included, when applicable, with any grid mapping
        self.earth_radius = earth_radius
        self.inverse_flattening = inverse_flattening
        self.longitude_of_prime_meridian = longitude_of_prime_meridian
        self.prime_meridian_name = prime_meridian_name
        self.reference_ellipsoid_name = reference_ellipsoid_name
        self.semi_major_axis = semi_major_axis
        self.semi_minor_axis = semi_minor_axis


class AzimuthalGridMapping(GridMapping):
    """A Grid Mapping with Azimuthal classification.

    .. versionadded:: GMVER

    :Parameters:

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.longitude_of_projection_origin = longitude_of_projection_origin
        self.latitude_of_projection_origin = latitude_of_projection_origin
        self.false_easting = false_easting
        self.false_northing = false_northing


class ConicGridMapping(GridMapping):
    """A Grid Mapping with Conic classification.

    .. versionadded:: GMVER

    :Parameters:

        standard_parallel: number, `str` or 2-`tuple`
            The standard parallel values, either the first (PROJ
            'lat_1' value), the second (PROJ 'lat_2' value) or
            both, given as a 2-tuple of numbers or strings corresponding to
            the first and then the second in order, where `None`
            indicates that a value is not being specified for either. In
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees.

            The default is (0.0, 0.0), that is 0.0 decimal degrees
            for the first and second standard parallel values.

        longitude_of_central_meridian: number or `str`, optional
            The longitude of (natural) origin i.e. central meridian, in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        standard_parallel,
        longitude_of_central_meridian=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.standard_parallel = standard_parallel
        self.longitude_of_central_meridian = longitude_of_central_meridian
        self.latitude_of_projection_origin = latitude_of_projection_origin
        self.false_easting = false_easting
        self.false_northing = false_northing


class CylindricalGridMapping(GridMapping):
    """A Grid Mapping with Cylindrical classification.

    .. versionadded:: GMVER

    :Parameters:

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(self, false_easting=0.0, false_northing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.false_easting = false_easting
        self.false_northing = false_northing


class LatLonGridMapping(GridMapping):
    """A Grid Mapping with Latitude-Longitude nature.

    Such a Grid Mapping is based upon latitude and longitude coordinates
    on a spherical Earth, defining the canonical 2D geographical coordinate
    system so that the figure of the Earth can be described.

    .. versionadded:: GMVER

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PerspectiveGridMapping(AzimuthalGridMapping):
    """A Grid Mapping with Azimuthal classification and perspective view.

    .. versionadded:: GMVER

    :Parameters:

        perspective_point_height: number
            The height of the view point above the surface (PROJ
            'h') value, for example the height of a satellite above
            the Earth, in units of meters.

    """

    def __init__(
        self,
        perspective_point_height,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.perspective_point_height = perspective_point_height


"""Concrete classes for all Grid Mappings supported by the CF Conventions.

For the full listing, see:

https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
cf-conventions.html#appendix-grid-mappings

from which these classes should be kept consistent and up-to-date.
"""


class AlbersEqualArea(ConicGridMapping):
    """The Albers Equal Area grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_albers_equal_area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/aea.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        standard_parallel: number, `str` or 2-`tuple`, optional
            The standard parallel values, either the first (PROJ
            'lat_1' value), the second (PROJ 'lat_2' value) or
            both, given as a 2-tuple of numbers or strings corresponding to
            the first and then the second in order, where `None`
            indicates that a value is not being specified for either. In
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees.

            The default is (0.0, 0.0), that is 0.0 decimal degrees
            for the first and second standard parallel values.

        longitude_of_central_meridian: number or `str`, optional
            The longitude of (natural) origin i.e. central meridian, in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        standard_parallel,
        longitude_of_central_meridian=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__("albers_conical_equal_area", "aea", *args, **kwargs)


class AzimuthalEquidistant(AzimuthalGridMapping):
    """The Azimuthal Equidistant grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#azimuthal-equidistant

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/aeqd.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__("azimuthal_equidistant", "aeqd", *args, **kwargs)


class Geostationary(PerspectiveGridMapping):
    """The Geostationary Satellite View grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_geostationary_projection

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/geos.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        perspective_point_height: number
            The height of the view point above the surface (PROJ
            'h') value, for example the height of a satellite above
            the Earth, in units of meters.

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

        sweep_angle_axis: `str`, optional
            Sweep angle axis of the viewing instrument, which indicates
            the axis on which the view sweeps. Valid options
            are "x" and "y". The default is "y".

            For more information about the nature of this parameter, see:

            https://proj.org/en/9.2/operations/projections/
            geos.html#note-on-sweep-angle

        fixed_angle_axis: `str`, optional
            The axis on which the view is fixed. It corresponds to the
            inner-gimbal axis of the gimbal view model, whose axis of
            rotation moves about the outer-gimbal axis. Valid options
            are "x" and "y". The default is "x".

            .. note:: If the fixed_angle_axis is "x", sweep_angle_axis
                      is "y", and vice versa.

    """

    def __init__(
        self,
        perspective_point_height,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        sweep_angle_axis="y",
        fixed_angle_axis="x",
        *args,
        **kwargs,
    ):
        super().__init__("geostationary", "geos", *args, **kwargs)

        # sweep_angle_axis must be the opposite (of "x" and "y") to
        # fixed_angle_axis.
        if (sweep_angle_axis.lower(), fixed_angle_axis.lower()) not in [
            ("x", "y")("y", "x")
        ]:
            raise ValueError(
                "The sweep_angle_axis must be the opposite value, from 'x' "
                "and 'y', to the fixed_angle_axis."
            )

        # Values "x" and "y" are not case-sensitive, so convert to lower-case
        self.sweep_angle_axis = sweep_angle_axis.lower()
        self.fixed_angle_axis = fixed_angle_axis.lower()


class LambertAzimuthalEqualArea(AzimuthalGridMapping):
    """The Lambert Azimuthal Equal Area grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#lambert-azimuthal-equal-area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/laea.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            "lambert_azimuthal_equal_area", "laea", *args, **kwargs
        )


class LambertConformalConic(ConicGridMapping):
    """The Lambert Conformal Conic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_lambert_conformal

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/lcc.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        standard_parallel: number, `str` or 2-`tuple`
            The standard parallel values, either the first (PROJ
            'lat_1' value), the second (PROJ 'lat_2' value) or
            both, given as a 2-tuple of numbers or strings corresponding to
            the first and then the second in order, where `None`
            indicates that a value is not being specified for either. In
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees.

            The default is (0.0, 0.0), that is 0.0 decimal degrees
            for the first and second standard parallel values.

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        standard_parallel,
        longitude_of_central_meridian=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__("lambert_conformal_conic", "lcc", *args, **kwargs)


class LambertCylindricalEqualArea(CylindricalGridMapping):
    """The Equal Area Cylindrical grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_lambert_cylindrical_equal_area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/cea.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

        standard_parallel: number, `str` or 2-`tuple`, optional
            The standard parallel values, either the first (PROJ
            'lat_1' value), the second (PROJ 'lat_2' value) or
            both, given as a 2-tuple of numbers or strings corresponding to
            the first and then the second in order, where `None`
            indicates that a value is not being specified for either. In
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees.

            The default is (0.0, None), that is 0.0 decimal degrees
            for the first standard parallel value and nothing set for
            the second.

        longitude_of_central_meridian: number or `str`, optional
            The longitude of (natural) origin i.e. central meridian, in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        scale_factor_at_projection_origin: number, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            It is unitless. The default is 1.0.

    """

    def __init__(
        self,
        standard_parallel=(0.0, None),
        false_easting=0.0,
        false_northing=0.0,
        scale_factor_at_projection_origin=1.0,
        longitude_of_central_meridian=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            "lambert_cylindrical_equal_area", "cea", *args, **kwargs
        )

        self.standard_parallel = standard_parallel
        self.longitude_of_central_meridian = longitude_of_central_meridian
        self.scale_factor_at_projection_origin = (
            scale_factor_at_projection_origin
        )


class Mercator(CylindricalGridMapping):
    """The Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/merc.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

        standard_parallel: number, `str` or 2-`tuple`, optional
            The standard parallel values, either the first (PROJ
            'lat_1' value), the second (PROJ 'lat_2' value) or
            both, given as a 2-tuple of numbers or strings corresponding to
            the first and then the second in order, where `None`
            indicates that a value is not being specified for either. In
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees.

            The default is (0.0, None), that is 0.0 decimal degrees
            for the first standard parallel value and nothing set for
            the second.

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        scale_factor_at_projection_origin: number, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            It is unitless. The default is 1.0.

    """

    def __init__(
        self,
        standard_parallel=(0.0, None),
        longitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        scale_factor_at_projection_origin=1.0,
        *args,
        **kwargs,
    ):
        super().__init__("mercator", "merc", *args, **kwargs)

        self.standard_parallel = standard_parallel
        self.longitude_of_projection_origin = longitude_of_projection_origin
        self.scale_factor_at_projection_origin = (
            scale_factor_at_projection_origin
        )


class ObliqueMercator(CylindricalGridMapping):
    """The Oblique Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_oblique_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/omerc.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

        azimuth_of_central_line: number or `str`, optional
            The azimuth i.e. tilt angle of the centerline clockwise
            from north at the center point of the line (PROJ 'alpha'
            value), in units of decimal degrees, where
            forming a string by adding a suffix character
            indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        scale_factor_at_projection_origin: number, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            It is unitless. The default is 1.0.

    """

    def __init__(
        self,
        azimuth_of_central_line=0.0,
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        scale_factor_at_projection_origin=1.0,
        *args,
        **kwargs,
    ):
        super().__init__("oblique_mercator", "omerc", *args, **kwargs)

        self.azimuth_of_central_line = azimuth_of_central_line
        self.latitude_of_projection_origin = latitude_of_projection_origin
        self.longitude_of_projection_origin = longitude_of_projection_origin
        self.scale_factor_at_projection_origin = (
            scale_factor_at_projection_origin
        )


class Orthographic(AzimuthalGridMapping):
    """The Orthographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_orthographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/ortho.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__("orthographic", "ortho", *args, **kwargs)


class PolarStereographic(AzimuthalGridMapping):
    """The Universal Polar Stereographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#polar-stereographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/ups.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        straight_vertical_longitude_from_pole: number or `str`, optional
            The longitude of (natural) origin i.e. central meridian,
            oriented straight up from the North or South Pole, in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        scale_factor_at_projection_origin: number, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            It is unitless. The default is 1.0.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

        standard_parallel: number, `str` or 2-`tuple`, optional
            The standard parallel values, either the first (PROJ
            'lat_1' value), the second (PROJ 'lat_2' value) or
            both, given as a 2-tuple of numbers or strings corresponding to
            the first and then the second in order, where `None`
            indicates that a value is not being specified for either. In
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees.

            The default is (0.0, 0.0), that is 0.0 decimal degrees
            for the first and second standard parallel values.

    """

    def __init__(
        self,
        standard_parallel=(0.0, 0.0),
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
        straight_vertical_longitude_from_pole=0.0,
        false_easting=0.0,
        false_northing=0.0,
        scale_factor_at_projection_origin=1.0,
        *args,
        **kwargs,
    ):
        # TODO check defaults here, they do not appear for
        # CRS.from_proj4("+proj=ups").to_cf() to cross reference!
        super().__init__("polar_stereographic", "ups", *args, **kwargs)

        # See: https://github.com/cf-convention/cf-conventions/issues/445
        if (
            longitude_of_projection_origin
            and straight_vertical_longitude_from_pole
        ):
            raise ValueError(
                "Only one of 'longitude_of_projection_origin' and "
                "'straight_vertical_longitude_from_pole' can be set."
            )

        self.straight_vertical_longitude_from_pole = (
            straight_vertical_longitude_from_pole
        )
        self.standard_parallel = standard_parallel
        self.scale_factor_at_projection_origin = (
            scale_factor_at_projection_origin
        )


class RotatedLatitudeLongitude(LatLonGridMapping):
    """The Rotated Latitude-Longitude grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_rotated_pole

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/eqc.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        grid_north_pole_latitude: number or `str`
            Latitude of the North pole of the unrotated source CRS,
            expressed in the rotated geographic CRS, in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        grid_north_pole_longitude: number or `str`
            Longitude of the North pole of the unrotated source CRS,
            expressed in the rotated geographic CRS, in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        north_pole_grid_longitude: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

    """

    def __init__(
        self,
        grid_north_pole_latitude,
        grid_north_pole_longitude,
        north_pole_grid_longitude=0.0,
        *args,
        **kwargs,
    ):
        super().__init__("rotated_latitude_longitude", "eqc", *args, **kwargs)

        self.grid_north_pole_latitude = grid_north_pole_latitude
        self.grid_north_pole_longitude = grid_north_pole_longitude
        self.north_pole_grid_longitude = north_pole_grid_longitude


class LatitudeLongitude(LatLonGridMapping):
    """The Latitude-Longitude i.e. Plate Carrée grid mapping.

    For alternative names, see e.g:

    https://en.wikipedia.org/wiki/Equirectangular_projection

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_latitude_longitude

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/eqc.html

    for more information.

    .. versionadded:: GMVER

    """

    def __init__(self, *args, **kwargs):
        super().__init__("latitude_longitude", "eqc", *args, **kwargs)


class Sinusoidal(GridMapping):
    """The Sinusoidal (Sanson-Flamsteed) grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_sinusoidal

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/sinu.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        longitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__("sinusoidal", "sinu", *args, **kwargs)

        self.longitude_of_projection_origin = longitude_of_projection_origin
        self.false_easting = false_easting
        self.false_northing = false_northing


class Stereographic(AzimuthalGridMapping):
    """The Stereographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_stereographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/stere.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

        scale_factor_at_projection_origin: number, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            It is unitless. The default is 1.0.

    """

    def __init__(
        self,
        false_easting=0.0,
        false_northing=0.0,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        scale_factor_at_projection_origin=1.0,
        *args,
        **kwargs,
    ):
        super().__init__("stereographic", "stere", *args, **kwargs)

        self.scale_factor_at_projection_origin = (
            scale_factor_at_projection_origin
        )


class TransverseMercator(CylindricalGridMapping):
    """The Transverse Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_transverse_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/tmerc.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

        scale_factor_at_central_meridian: number, optional
            The scale factor at (natural) origin i.e. central meridian.
            It is unitless. The default is 1.0.

        longitude_of_central_meridian: number or `str`, optional
            The longitude of (natural) origin i.e. central meridian, in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

    """

    def __init__(
        self,
        scale_factor_at_central_meridian=1.0,
        longitude_of_central_meridian=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__("transverse_mercator", "tmerc", *args, **kwargs)

        self.scale_factor_at_central_meridian = (
            scale_factor_at_central_meridian
        )
        self.longitude_of_central_meridian = longitude_of_central_meridian
        self.latitude_of_projection_origin = latitude_of_projection_origin


class VerticalPerspective(PerspectiveGridMapping):
    """The Vertical (or Near-sided) Perspective grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#vertical-perspective

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/nsper.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        perspective_point_height: number
            The height of the view point above the surface (PROJ
            'h') value, for example the height of a satellite above
            the Earth, in units of meters.

        longitude_of_projection_origin: number or `str`, optional
            The longitude of projection center (PROJ 'lon_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        latitude_of_projection_origin: number or `str`, optional
            The latitude of projection center (PROJ 'lat_0' value), in
            units of decimal degrees, where forming a string by adding
            a suffix character indicates alternative units of
            radians if the suffix is 'R' or 'r'. If a string, a suffix
            of 'd', 'D' or '°' confirm units of decimal degrees. The default
            is 0.0 decimal degrees.

        false_easting: number, optional
            The false easting (PROJ 'x_0') value, in units of metres.
            The default is 0.0.

        false_northing: number, optional
            The false northing (PROJ 'y_0') value, in units of metres.
            The default is 0.0.

    """

    def __init__(
        self,
        perspective_point_height,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        *args,
        **kwargs,
    ):
        super().__init__("vertical_perspective", "nsper", *args, **kwargs)
