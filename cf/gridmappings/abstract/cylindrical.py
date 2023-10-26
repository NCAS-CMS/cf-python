from .gridmappingbase import (
    GridMapping,
    _validate_map_parameter,
)


class CylindricalGridMapping(GridMapping):
    """A Grid Mapping with Cylindrical classification.

    .. versionadded:: GMVER

    :Parameters:

        false_easting: number or scalar `Data`, optional
            The false easting (PROJ 'x_0') value.
            If provided as a number or `Data` without units, the units
            are taken as metres 'm', else the `Data` units are
            taken and must be compatible with distance.
            The default is 0.0 metres.

        false_northing: number or scalar `Data`, optional
            The false northing (PROJ 'y_0') value.
            If provided as a number or `Data` without units, the units
            are taken as metres 'm', else the `Data` units are
            taken and must be compatible with distance.
            The default is 0.0 metres.

    """

    def __init__(self, false_easting=0.0, false_northing=0.0, **kwargs):
        super().__init__(**kwargs)

        self.false_easting = _validate_map_parameter(
            "false_easting", false_easting
        )
        self.false_northing = _validate_map_parameter(
            "false_northing", false_northing
        )
