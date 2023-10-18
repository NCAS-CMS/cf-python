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


class GM():
    """A validated Grid Mapping supported by the CF Conventions."""

    def __new__(cls, *args, **kwargs):
        """TODOGM."""
        if cls is GM:
            name = cls.grid_mapping_name

            # TODO: once cf Python minimum is v.3.10, use the new match/case
            # syntax to consolidate this long if/elif.
            if not name:
                pass  # TODOGM raise a custom exception
            elif name is "albers_conical_equal_area":
                return AlbersEqualArea(*args, **kwargs)
            elif name is "azimuthal_equidistant":
                return AzimuthalEquidistant(*args, **kwargs)
            elif name is "geostationary":
                return Geostationary(*args, **kwargs)
            elif name is "lambert_azimuthal_equal_area":
                return LambertAzimuthalEqualArea(*args, **kwargs)
            elif name is "lambert_conformal_conic":
                return LambertConformalConic(*args, **kwargs)
            elif name is "lambert_cylindrical_equal_area":
                return LambertCylindricalEqualArea(*args, **kwargs)
            elif name is "latitude_longitude":
                return LatitudeLongitude(*args, **kwargs)
            elif name is "mercator":
                return Mercator(*args, **kwargs)
            elif name is "oblique_mercator":
                return ObliqueMercator(*args, **kwargs)
            elif name is "orthographic":
                return Orthographic(*args, **kwargs)
            elif name is "polar_stereographic":
                return PolarStereographic(*args, **kwargs)
            elif name is "rotated_latitude_longitude":
                return RotatedLatitudeLongitude(*args, **kwargs)
            elif name is "sinusoidal":
                return Sinusoidal(*args, **kwargs)
            elif name is "stereographic":
                return Stereographic(*args, **kwargs)
            elif name is "transverse_mercator":
                return TransverseMercator(*args, **kwargs)
            elif name is "vertical_perspective":
                return VerticalPerspective(*args, **kwargs)
            else:
                pass  # TODOGM raise a custom exception

    def __init__(self, coordinate_reference):
        """TODOGM."""
        pass  # TODOGM

    def create_crs():
        """TODOGM."""
        pass  # TODOGM

    def is_latlon_gm():
        """Whether the Grid Mapping is of LatitudeLongitude form.

        :Returns:

            `bool`
                True only if the Grid Mapping is LatitudeLongitude.

        """
        return False
