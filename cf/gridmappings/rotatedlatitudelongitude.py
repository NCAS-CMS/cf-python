from .abstract import LatLonGridMapping
from .abstract.gridmappingbase import _validate_map_parameter


class RotatedLatitudeLongitude(LatLonGridMapping):
    """The Rotated Latitude-Longitude grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_rotated_pole

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        grid_north_pole_latitude: number or scalar `Data`
            Latitude of the North pole of the unrotated source CRS,
            expressed in the rotated geographic CRS.
            If provided as a number or `Data` without units, the units
            are taken as 'degrees_north', else the `Data` units are
            taken and must be angular and compatible with latitude.

        grid_north_pole_longitude: number or scalar `Data`
            Longitude of the North pole of the unrotated source CRS,
            expressed in the rotated geographic CRS.
            If provided as a number or `Data` without units, the units
            are taken as 'degrees_east', else the `Data` units are
            taken and must be angular and compatible with longitude.

        north_pole_grid_longitude: number or scalar `Data`, optional
            The longitude of projection center (PROJ 'lon_0' value).
            If provided as a number or `Data` without units, the units
            are taken as 'degrees_east', else the `Data` units are
            taken and must be angular and compatible with longitude.

    """

    grid_mapping_name = "rotated_latitude_longitude"
    proj_id = "latlong"

    def __init__(
        self,
        grid_north_pole_latitude,
        grid_north_pole_longitude,
        north_pole_grid_longitude=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.grid_north_pole_latitude = _validate_map_parameter(
            "grid_north_pole_latitude", grid_north_pole_latitude
        )
        self.grid_north_pole_longitude = _validate_map_parameter(
            "grid_north_pole_longitude", grid_north_pole_longitude
        )
        self.north_pole_grid_longitude = _validate_map_parameter(
            "north_pole_grid_longitude", north_pole_grid_longitude
        )
