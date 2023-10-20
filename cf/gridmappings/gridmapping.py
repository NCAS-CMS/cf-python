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
    """Exception for a Grid Mapping which is not supported by CF.

    .. versionadded:: GMVER

    :Parameters:

        TODOGM.

    """

    def __init__(self, grid_mapping, custom_message=None):
        self.grid_mapping = grid_mapping
        self.custom_message = custom_message

    def __str__(self):
        grid_mapping_name = self.grid_mapping.grid_mapping_name
        if self.custom_message:
            return self.custom_message
        elif grid_mapping_name:
            return (
                f"Grid Mapping {self.grid_mapping} with grid_mapping_name "
                f"{grid_mapping_name} is not supported by the CF "
                "Conventions."
            )
        else:
            return (
                f"Grid Mapping {self.grid_mapping} missing grid_mapping_name "
                "and therefore cannot be interpreted."
            )


class GM():
    """A validated Grid Mapping supported by the CF Conventions.

    .. versionadded:: GMVER

    :Parameters:
        TODOGM.

    """

    def __new__(cls, *args, **kwargs):
        """TODOGM."""
        if cls is GM:
            name = cls.grid_mapping_name

            # TODO: once cf Python minimum is v.3.10, use the new match/case
            # syntax to consolidate this long if/elif.
            if not name:
                pass  # TODOGM raise a custom exception
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
                pass  # TODOGM raise a custom exception

    def __init__(self, coordinate_reference):
        """TODOGM."""
        pass  # TODOGM

    def create_crs():
        """TODOGM."""
        pass  # TODOGM
