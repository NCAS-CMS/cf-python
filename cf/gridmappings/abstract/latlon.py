from .gridmappingbase import (
    GridMapping,
    _validate_map_parameter,
)


class LatLonGridMapping(GridMapping):
    """A Grid Mapping with Latitude-Longitude nature.

    Such a Grid Mapping is based upon latitude and longitude coordinates
    on a spherical Earth, defining the canonical 2D geographical coordinate
    system so that the figure of the Earth can be described.

    .. versionadded:: GMVER

    """

    pass
