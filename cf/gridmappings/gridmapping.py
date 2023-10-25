# Concrete classes for all Grid Mappings supported by the CF Conventions.
# For the full listing, see:
#     https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
#     cf-conventions.html#appendix-grid-mappings
# from which these classes should be kept consistent and up-to-date.
from .albersequalarea import AlbersEqualArea
from .azimuthalequidistant import AzimuthalEquidistant
from .geostationary import Geostationary
from .lambertazimuthalequalarea import LambertAzimuthalEqualArea
from .lambertconformalconic import LambertConformalConic
from .lambertcylindricalequalarea import LambertCylindricalEqualArea
from .latitudelongitude import LatitudeLongitude
from .mercator import Mercator
from .obliquemercator import ObliqueMercator
from .orthographic import Orthographic
from .polarstereographic import PolarStereographic
from .rotatedlatitudelongitude import RotatedLatitudeLongitude
from .sinusoidal import Sinusoidal
from .stereographic import Stereographic
from .transversemercator import TransverseMercator
from .verticalperspective import VerticalPerspective


class InvalidGridMapping(Exception):
    """Exception for a coordinate reference with unsupported Grid Mapping.

    .. versionadded:: GMVER

    :Parameters:

        TODOGM.

    """

    def __init__(self, gm_name_attr, custom_message=None):
        self.gm_name_attr = grid_mapping_name_attr
        self.custom_message = custom_message

    def __str__(self):
        if self.custom_message:
            return self.custom_message
        elif self.gm_name_attr:
            return (
                f"Coordinate reference construct with grid_mapping_name "
                f"{self.gm_name_attr} corresponds to a Grid Mapping that "
                "is not supported by the CF Conventions."
            )
        else:
            return (
                f"A coordinate reference construct must have an attribute "
                "'Coordinate conversion:grid_mapping_name' defined in order "
                "to interpret it as a GM class. Missing 'grid_mapping_name'."
            )


class GM():
    """A validated Grid Mapping supported by the CF Conventions.

    .. versionadded:: GMVER

    :Parameters:
        TODOGM.

    """

    def __new__(cls, coordinate_reference, **kwargs):
        """TODOGM."""
        if cls is GM:
            name = coordinate_reference.coordinate_conversion.get_parameter(
                "grid_mapping_name", default=None
            )

            # TODO: once cf Python minimum is v.3.10, use the new match/case
            # syntax to consolidate this long if/elif.
            if not name:
                raise InvalidGridMapping(name)
            elif name == "albers_conical_equal_area":
                return AlbersEqualArea(*args, **kwargs)
            elif name == "azimuthal_equidistant":
                return AzimuthalEquidistant(*args, **kwargs)
            elif name == "geostationary":
                return Geostationary(*args, **kwargs)
            elif name == "lambert_azimuthal_equal_area":
                return LambertAzimuthalEqualArea(*args, **kwargs)
            elif name == "lambert_conformal_conic":
                return LambertConformalConic(*args, **kwargs)
            elif name == "lambert_cylindrical_equal_area":
                return LambertCylindricalEqualArea(*args, **kwargs)
            elif name == "latitude_longitude":
                return LatitudeLongitude(*args, **kwargs)
            elif name == "mercator":
                return Mercator(*args, **kwargs)
            elif name == "oblique_mercator":
                return ObliqueMercator(*args, **kwargs)
            elif name == "orthographic":
                return Orthographic(*args, **kwargs)
            elif name == "polar_stereographic":
                return PolarStereographic(*args, **kwargs)
            elif name == "rotated_latitude_longitude":
                return RotatedLatitudeLongitude(*args, **kwargs)
            elif name == "sinusoidal":
                return Sinusoidal(*args, **kwargs)
            elif name == "stereographic":
                return Stereographic(*args, **kwargs)
            elif name == "transverse_mercator":
                return TransverseMercator(*args, **kwargs)
            elif name == "vertical_perspective":
                return VerticalPerspective(*args, **kwargs)
            else:
                raise InvalidGridMapping(name)

    def __init__(self, coordinate_reference):
        """TODOGM."""
        pass  # TODOGM

    def create_crs():
        """TODOGM."""
        pass  # TODOGM
