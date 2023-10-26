from .abstract import ConicGridMapping


class LambertConformalConic(ConicGridMapping):
    """The Lambert Conformal Conic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_lambert_conformal

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/lcc.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        standard_parallel: 2-`tuple` of number or scalar `Data` or `None`
            The standard parallel value(s): the first (PROJ 'lat_1'
            value) and/or the second (PROJ 'lat_2' value), given
            as a 2-tuple of numbers or strings corresponding to
            the first and then the second in order, where `None`
            indicates that a value is not being set for either.

            If provided as a number or `Data` without units, the units
            for each of the values are taken as 'degrees_north', else
            the `Data` units are taken and must be angular and
            compatible with latitude.

            The default is (0.0, 0.0), that is 0.0 degrees_north
            for the first and second standard parallel values.

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

    """

    grid_mapping_name = "lambert_conformal_conic"
    proj_id = "lcc"
