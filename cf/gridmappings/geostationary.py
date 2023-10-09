from .abstract import PerspectiveGridMapping


class Geostationary(PerspectiveGridMapping):
    """The Geostationary Satellite View grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_geostationary_projection

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/geos.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        perspective_point_height: number or scalar `Data`
            The height of the view point above the surface (PROJ
            'h') value, for example the height of a satellite above
            the Earth, in units of meters 'm'. If provided
            as a number or `Data` without units, the units
            are taken as metres 'm', else the `Data` units are
            taken and must be compatible with distance.

        longitude_of_projection_origin: number or scalar `Data`, optional
            The longitude of projection center (PROJ 'lon_0' value).
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

        sweep_angle_axis: `str`, optional
            Sweep angle axis of the viewing instrument, which indicates
            the axis on which the view sweeps. Valid options
            are "x" and "y". The default is "y".

            For more information about the nature of this parameter, see:

            https://proj.org/en/9.2/operations/projections/
            geos.html#note-on-sweep-angle

        fixed_angle_axis: `str`, optional
            The axis on which the view is fixed. It corresponds to the
            inner-gimbal axis of the gimbal view model, whose axis of
            rotation moves about the outer-gimbal axis. Valid options
            are "x" and "y". The default is "x".

            .. note:: If the fixed_angle_axis is "x", sweep_angle_axis
                      is "y", and vice versa.

    """

    grid_mapping_name = "geostationary"
    proj_id = "geos"

    def __init__(
        self,
        perspective_point_height,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        sweep_angle_axis="y",
        fixed_angle_axis="x",
        **kwargs,
    ):
        super().__init__(
            perspective_point_height,
            longitude_of_projection_origin=longitude_of_projection_origin,
            latitude_of_projection_origin=latitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            **kwargs,
        )

        # Values "x" and "y" are not case-sensitive, so convert to lower-case
        self.sweep_angle_axis = _validate_map_parameter(
            "sweep_angle_axis", sweep_angle_axis
        ).lower()
        self.fixed_angle_axis = _validate_map_parameter(
            "fixed_angle_axis", fixed_angle_axis
        ).lower()

        # sweep_angle_axis must be the opposite (of "x" and "y") to
        # fixed_angle_axis.
        if (self.sweep_angle_axis, self.fixed_angle_axis) not in [
            ("x", "y"),
            ("y", "x"),
        ]:
            raise ValueError(
                "The sweep_angle_axis must be the opposite value, from 'x' "
                "and 'y', to the fixed_angle_axis."
            )
