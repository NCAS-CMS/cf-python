from .abstract import CylindricalGridMapping


class TransverseMercator(CylindricalGridMapping):
    """The Transverse Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_transverse_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/tmerc.html

    for more information.

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

        scale_factor_at_projection_origin: number or scalar `Data`, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            Unitless, so `Data` must be unitless. The default is 1.0.

        longitude_of_central_meridian: number or scalar `Data`, optional
            The longitude of (natural) origin i.e. central meridian.
            If provided as a number or `Data` without units, the units
            are taken as 'degrees_east', else the `Data` units are
            taken and must be angular and compatible with longitude.
            The default is 0.0 degrees_east.

        latitude_of_projection_origin: number or scalar `Data`, optional
            The latitude of projection center (PROJ 'lat_0' value).
            If provided as a number or `Data` without units, the units
            are taken as 'degrees_north', else the `Data` units are
            taken and must be angular and compatible with latitude.
            The default is 0.0 degrees_north.

    """

    grid_mapping_name = "transverse_mercator"
    proj_id = "tmerc"

    def __init__(
        self,
        scale_factor_at_central_meridian=1.0,
        longitude_of_central_meridian=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        **kwargs,
    ):
        super().__init__(
            false_easting=false_easting,
            false_northing=false_northing,
            **kwargs,
        )

        self.scale_factor_at_central_meridian = _validate_map_parameter(
            "scale_factor_at_central_meridian",
            scale_factor_at_central_meridian,
        )
        self.longitude_of_central_meridian = _validate_map_parameter(
            "longitude_of_central_meridian", longitude_of_central_meridian
        )
        self.latitude_of_projection_origin = _validate_map_parameter(
            "latitude_of_projection_origin", latitude_of_projection_origin
        )
