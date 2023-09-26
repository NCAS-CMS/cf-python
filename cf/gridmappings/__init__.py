"""
Module for Grid Mappings supported by the CF Conventions.

For the full list of supported Grid Mappings and details, see Appendix F
of the canonical document:

https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
cf-conventions.html#appendix-grid-mappings

This module should be kept up to date with the Appendix, by adding or
amending appropriate classes.

"""

from .gridmapping import (
    GridMapping,
    AzimuthalGridMapping,
    ConicGridMapping,
    CylindricalGridMapping,
    LatLonGridMapping,
    PerspectiveGridMapping,
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
    convert_proj_angular_data_to_cf,
    convert_cf_angular_data_to_proj,
    _all_abstract_grid_mappings,
    _all_concrete_grid_mappings,
    _get_cf_grid_mapping_from_name,
)
