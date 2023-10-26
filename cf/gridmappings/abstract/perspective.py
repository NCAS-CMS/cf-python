from .gridmappingbase import _validate_map_parameter
from .azimuthal import AzimuthalGridMapping


class PerspectiveGridMapping(AzimuthalGridMapping):
    """A Grid Mapping with Azimuthal classification and perspective view.

    .. versionadded:: GMVER

    :Parameters:

        perspective_point_height: number or scalar `Data`
            The height of the view point above the surface (PROJ
            'h') value, for example the height of a satellite above
            the Earth, in units of meters 'm'. If provided
            as a number or `Data` without units, the units
            are taken as metres 'm', else the `Data` units are
            taken and must be compatible with distance.

    """

    def __init__(self, perspective_point_height, **kwargs):
        super().__init__(**kwargs)

        self.perspective_point_height = _validate_map_parameter(
            "perspective_point_height", perspective_point_height
        )
