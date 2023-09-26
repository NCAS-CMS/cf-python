"""
Module for Grid Mappings supported by the CF Conventions.

For the full list of supported Grid Mappings and details, see Appendix F
of the canonical document:

https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
cf-conventions.html#appendix-grid-mappings

This module should be kept up to date with the Appendix, by adding or
amending appropriate classes.

Note that abstract classes to support the creation of concrete classes
for concrete Grid Mappinds are defined in the 'abstract' module, so
not included in the listing below.

"""

from .gridmapping import (
    AlbersEqualArea,
    AzimuthalEquidistant,
    Geostationary,
    LambertAzimuthalEqualArea,
    LambertConformalConic,
    LambertCylindricalEqualArea,
    Mercator,
    ObliqueMercator,
    Orthographic,
    PolarStereographic,
    RotatedLatitudeLongitude,
    LatitudeLongitude,
    Sinusoidal,
    Stereographic,
    TransverseMercator,
    VerticalPerspective,
    _all_concrete_grid_mappings,
)
