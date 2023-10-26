"""
Abstract classes representing categories of Grid Mapping.

Categories correspond to the type of projection a Grid Mapping is
based upon, for example these are often based upon a geometric shape
that forms the developable surface that is used to flatten the map
of Earth (such as conic for a cone or cylindrical for a cylinder).

The LatLonGridMapping case is special in that it (from the CF Conventions,
Appendix F) 'defines the canonical 2D geographical coordinate system
based upon latitude and longitude coordinates on a spherical Earth'.

"""

from .gridmappingbase import (
    GridMapping,
    convert_proj_angular_data_to_cf,
    convert_cf_angular_data_to_proj,
    _validate_map_parameter,
)
from .azimuthal import AzimuthalGridMapping
from .conic import ConicGridMapping
from .cylindrical import CylindricalGridMapping
from .latlon import LatLonGridMapping
from .perspective import PerspectiveGridMapping
