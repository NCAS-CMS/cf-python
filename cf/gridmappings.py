from pyproj import CRS

ALL_GRID_MAPPING_ATTR_NAMES = {
    "grid_mapping_name",
    # *Those which describe the ellipsoid and prime meridian:*
    "earth_radius",  # PROJ +R
    "inverse_flattening",  # PROJ "+rf"
    "longitude_of_prime_meridian",
    "prime_meridian_name",  # PROJ +pm
    "reference_ellipsoid_name",  # PROJ +ellps
    "semi_major_axis",  # PROJ "+a"
    "semi_minor_axis",  # PROJ "+b"
    # *Specific/applicable to only given grid mapping(s):*
    # ...projection origin related:
    "longitude_of_projection_origin",  # PROJ +lon_0
    "latitude_of_projection_origin",  # PROJ +lat_0
    "scale_factor_at_projection_origin",  # PROJ k_0
    # ...false-Xings:
    "false_easting",  # PROJ +x_0
    "false_northing",  # PROJ +y_0
    # ...angle axis related:
    "sweep_angle_axis",  # PROJ +sweep
    "fixed_angle_axis",
    # ...central meridian related:
    "longitude_of_central_meridian",
    "scale_factor_at_central_meridian",
    # ...pole coordinates related:
    "grid_north_pole_latitude",
    "grid_north_pole_longitude",
    "north_pole_grid_longitude",
    # ...other:
    "standard_parallel",  # PROJ ["+lat_1", "+lat_2"] (up to 2)
    "perspective_point_height",  # PROJ "+h"
    "azimuth_of_central_line",  # PROJ +gamma OR +alpha
    "straight_vertical_longitude_from_pole",  # PROJ +south
    # *Other, not needed for a specific grid mapping but also listed
    # in 'Table F.1. Grid Mapping Attributes':*
    "crs_wkt" "geographic_crs_name",  # PROJ "crs_wkt",  # PROJ geoid_crs
    "geoid_name",  # PROJ geoidgrids
    "geopotential_datum_name",
    "horizontal_datum_name",
    "projected_crs_name",
    "towgs84",  # PROJ +towgs84
}


"""Abstract classes for general Grid Mappings."""


class GridMapping:
    """A container for a Grid Mapping recognised by the CF Conventions."""

    def __init__(
        self,
        grid_mapping_name=None,
        proj_id=None,
        earth_radius=None,
        inverse_flattening=None,
        longitude_of_prime_meridian=None,
        prime_meridian_name=None,
        reference_ellipsoid_name=None,
        semi_major_axis=None,
        semi_minor_axis=None,
    ):
        """**Initialisation**

        :Parameters:

            grid_mapping_name: string
                TODO

            proj_id: string
                TODO "_proj" projection name

            earth_radius: number, optional
                TODO

            inverse_flattening: TODO, optional
                TODO

            longitude_of_prime_meridian: TODO, optional
                TODO

            prime_meridian_name: TODO, optional
                TODO

            reference_ellipsoid_name: TODO, optional
                TODO

            semi_major_axis: TODO, optional
                TODO

            semi_minor_axis: TODO, optional
                TODO

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

        longitude_of_projection_origin: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(
        self,
        longitude_of_projection_origin,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
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

        standard_parallel: TODO
            TODO

        longitude_of_central_meridian: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(
        self,
        standard_parallel,
        longitude_of_central_meridian,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
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

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(self, false_easting, false_northing, *args, **kwargs):
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

        perspective_point_height: TODO
            TODO

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

        standard_parallel: TODO
            TODO

        longitude_of_central_meridian: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(
        self,
        standard_parallel,
        longitude_of_central_meridian,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
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

        longitude_of_projection_origin: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(
        self,
        longitude_of_projection_origin,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
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

        longitude_of_projection_origin: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

        perspective_point_height: TODO
            TODO

        sweep_angle_axis: TODO
            TODO

        fixed_angle_axis: TODO
            TODO

    """

    def __init__(
        self,
        longitude_of_projection_origin,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
        perspective_point_height,
        sweep_angle_axis,
        fixed_angle_axis,
        *args,
        **kwargs,
    ):
        super().__init__("geostationary", "geos", *args, **kwargs)

        self.sweep_angle_axis = sweep_angle_axis
        self.fixed_angle_axis = fixed_angle_axis


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

        longitude_of_projection_origin: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(
        self,
        longitude_of_projection_origin,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
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

        standard_parallel: TODO
            TODO

        longitude_of_central_meridian: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(
        self,
        standard_parallel,
        longitude_of_central_meridian,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
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

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

        standard_parallel: TODO
            TODO

        longitude_of_central_meridian: TODO
            TODO

        scale_factor_at_projection_origin: TODO
            TODO

    """

    def __init__(
        self,
        false_easting,
        false_northing,
        standard_parallel,
        longitude_of_central_meridian,
        scale_factor_at_projection_origin,
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

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

        standard_parallel: TODO
            TODO

        longitude_of_projection_origin: TODO
            TODO

        scale_factor_at_projection_origin: TODO
            TODO

    """

    def __init__(
        self,
        false_easting,
        false_northing,
        standard_parallel,
        longitude_of_projection_origin,
        scale_factor_at_projection_origin,
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

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

        azimuth_of_central_line: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        longitude_of_projection_origin: TODO
            TODO

        scale_factor_at_projection_origin: TODO
            TODO

    """

    def __init__(
        self,
        false_easting,
        false_northing,
        azimuth_of_central_line,
        latitude_of_projection_origin,
        longitude_of_projection_origin,
        scale_factor_at_projection_origin,
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

        longitude_of_projection_origin: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(
        self,
        longitude_of_projection_origin,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
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

        straight_vertical_longitude_from_pole: TODO
            TODO

        longitude_of_projection_origin: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        scale_factor_at_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

        standard_parallel: TODO
            TODO

    """

    def __init__(
        self,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
        standard_parallel,
        scale_factor_at_projection_origin,
        longitude_of_projection_origin=None,
        straight_vertical_longitude_from_pole=None,
        *args,
        **kwargs,
    ):
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

        grid_north_pole_latitude: TODO
            TODO

        grid_north_pole_longitude: TODO
            TODO

        north_pole_grid_longitude: TODO
            TODO

    """

    def __init__(
        self,
        grid_north_pole_latitude,
        grid_north_pole_longitude,
        north_pole_grid_longitude,
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

        longitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

    """

    def __init__(
        self,
        longitude_of_projection_origin,
        false_easting,
        false_northing,
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

        longitude_of_projection_origin: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

        scale_factor_at_projection_origin: TODO
            TODO

    """

    def __init__(
        self,
        longitude_of_projection_origin,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
        scale_factor_at_projection_origin,
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

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

        scale_factor_at_central_meridian: TODO
            TODO

        longitude_of_central_meridian: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

    """

    def __init__(
        self,
        false_easting,
        false_northing,
        scale_factor_at_central_meridian,
        longitude_of_central_meridian,
        latitude_of_projection_origin,
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

        longitude_of_projection_origin: TODO
            TODO

        latitude_of_projection_origin: TODO
            TODO

        false_easting: TODO
            TODO

        false_northing: TODO
            TODO

        perspective_point_height: TODO
            TODO

    """

    def __init__(
        self,
        longitude_of_projection_origin,
        latitude_of_projection_origin,
        false_easting,
        false_northing,
        perspective_point_height,
        *args,
        **kwargs,
    ):
        super().__init__("vertical_perspective", "nsper", *args, **kwargs)
