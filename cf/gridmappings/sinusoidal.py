from .abstract import GridMapping
from .abstract.gridmappingbase import _validate_map_parameter


class Sinusoidal(GridMapping):
    """The Sinusoidal (Sanson-Flamsteed) grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_sinusoidal

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/sinu.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        longitude_of_central_meridian: number or scalar `Data`, optional
            The longitude of (natural) origin i.e. central meridian.
            If provided as a number or `Data` without units, the units
            are taken as 'degrees_east', else the `Data` units are
            taken and must be angular and compatible with longitude.
            The default is 0.0 degrees_east.

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

    grid_mapping_name = "sinusoidal"
    proj_id = "sinu"

    def __init__(
        self,
        longitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.longitude_of_projection_origin = _validate_map_parameter(
            "longitude_of_projection_origin", longitude_of_projection_origin
        )
        self.false_easting = _validate_map_parameter(
            "false_easting", false_easting
        )
        self.false_northing = _validate_map_parameter(
            "false_northing", false_northing
        )
