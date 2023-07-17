from pyproj import CRS

ALL_GRID_MAPPING_ATTR_NAMES = {
    "grid_mapping_name",
    # *Those which describe the ellipsoid and prime meridian:*
    "earth_radius",
    "inverse_flattening",
    "longitude_of_prime_meridian",
    "prime_meridian_name",
    "reference_ellipsoid_name",
    "semi_major_axis",
    "semi_minor_axis",
    # *Specific/applicable to only given grid mapping(s):*
    # ...projection origin related:
    "longitude_of_projection_origin",
    "latitude_of_projection_origin",
    "scale_factor_at_projection_origin",
    # ...false-Xings:
    "false_easting",
    "false_northing",
    # ...angle axis related:
    "sweep_angle_axis",
    "fixed_angle_axis",
    # ...central meridian related:
    "longitude_of_central_meridian",
    "scale_factor_at_central_meridian",
    # ...pole coordinates related:
    "grid_north_pole_latitude",
    "grid_north_pole_longitude",
    "north_pole_grid_longitude",
    # ...other:
    "standard_parallel",
    "perspective_point_height",
    "azimuth_of_central_line",
    "straight_vertical_longitude_from_pole",
    # *Other, not needed for a specific grid mapping but also listed
    # in 'Table F.1. Grid Mapping Attributes':*
    "crs_wkt",
    "geographic_crs_name",
    "geoid_name",
    "geopotential_datum_name",
    "horizontal_datum_name",
    "inverse_flattening",
    "projected_crs_name",
    "towgs84",
}


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
                TODO

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


class AzimuthalGridMapping(GridMapping):
    """TODO."""

    def __init__(
        self, longitude_of_projection_origin, latitude_of_projection_origin
    ):
        super().__init__()


class ConicGridMapping(GridMapping):
    """TODO."""

    def __init__(self, standard_parallel, longitude_of_central_meridian):
        super().__init__()


class CylindricalGridMapping(GridMapping):
    """TODO."""

    def __init__(self, false_easting, false_northing):
        super().__init__()


class LatLonGridMapping(GridMapping):
    """TODO."""

    def __init__(self):
        super().__init__()


class PerspectiveGridMapping(AzimuthalGridMapping):
    """TODO."""

    def __init__(
        self, false_easting, false_northing, perspective_point_height
    ):
        super().__init__()


class StereographicGridMapping(AzimuthalGridMapping):
    """TODO."""

    def __init__(self, false_easting, false_northing):
        super().__init__()


class AlbersEqualArea(ConicGridMapping):
    """The Albers Equal Area grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_albers_equal_area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/aea.html

    for more information.

    """

    def __init__(self):
        super().__init__("albers_conical_equal_area", "aea")


class AzimuthalEquidistant(AzimuthalGridMapping):
    """The Azimuthal Equidistant grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#azimuthal-equidistant

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/aeqd.html

    for more information.

    """

    def __init__(self):
        super().__init__("azimuthal_equidistant", "aeqd")


class Geostationary(PerspectiveGridMapping):
    """The Geostationary Satellite View grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_geostationary_projection

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/geos.html

    for more information.

    """

    def __init__(self):
        super().__init__("geostationary", "geos")


class LambertAzimuthalEqualArea(AzimuthalGridMapping):
    """The Lambert Azimuthal Equal Area grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#lambert-azimuthal-equal-area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/laea.html

    for more information.

    """

    def __init__(self):
        super().__init__("lambert_azimuthal_equal_area", "laea")


class LambertConformalConic(ConicGridMapping):
    """The Lambert Conformal Conic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_lambert_conformal

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/lcc.html

    for more information.

    """

    def __init__(self):
        super().__init__("lambert_conformal_conic", "lcc")


class LambertCylindricalEqualArea(CylindricalGridMapping):
    """The Equal Area Cylindrical grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_lambert_cylindrical_equal_area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/cea.html

    for more information.

    """

    def __init__(self):
        super().__init__("lambert_cylindrical_equal_area", "cea")


class Mercator(CylindricalGridMapping):
    """The Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/merc.html

    for more information.

    """

    def __init__(self):
        super().__init__("mercator", "merc")


class ObliqueMercator(CylindricalGridMapping):
    """The Oblique Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_oblique_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/omerc.html

    for more information.

    """

    def __init__(self):
        super().__init__("oblique_mercator", "omerc")


class Orthographic(AzimuthalGridMapping):
    """The Orthographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_orthographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/ortho.html

    for more information.

    """

    def __init__(self):
        super().__init__("orthographic", "ortho")


class PolarStereographic(StereographicGridMapping):
    """The Universal Polar Stereographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#polar-stereographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/ups.html

    for more information.

    """

    def __init__(self):
        super().__init__("polar_stereographic", "ups")


class RotatedLatitudeLongitude(LatLonGridMapping):
    """The Rotated Latitude-Longitude grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_rotated_pole

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/eqc.html

    for more information.

    """

    def __init__(self):
        super().__init__("rotated_latitude_longitude", "eqc")


class LatitudeLongitude(RotatedLatitudeLongitude):
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

    """

    def __init__(self):
        super().__init__("latitude_longitude", "eqc")


class Sinusoidal(GridMapping):
    """The Sinusoidal (Sanson-Flamsteed) grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_sinusoidal

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/sinu.html

    for more information.

    """

    def __init__(self):
        super().__init__("sinusoidal", "sinu")


class Stereographic(StereographicGridMapping):
    """The Stereographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_stereographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/stere.html

    for more information.

    """

    def __init__(self):
        super().__init__("stereographic", "stere")


class TransverseMercator(CylindricalGridMapping):
    """The Transverse Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_transverse_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/tmerc.html

    for more information.

    """

    def __init__(self):
        super().__init__("transverse_mercator", "tmerc")


class VerticalPerspective(PerspectiveGridMapping):
    """The Vertical (or Near-sided) Perspective grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#vertical-perspective

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/nsper.html

    for more information.

    """

    def __init__(self):
        super().__init__("vertical_perspective", "nsper")
