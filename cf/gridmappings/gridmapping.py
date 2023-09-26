import itertools
import re
from abc import ABC, abstractmethod

from pyproj import CRS

from ..constants import cr_canonical_units, cr_gm_valid_attr_names_are_numeric
from ..data import Data
from ..data.utils import is_numeric_dtype
from ..units import Units

from .abstract import (
    GridMapping,
    AzimuthalGridMapping,
    ConicGridMapping,
    CylindricalGridMapping,
    LatLonGridMapping,
    PerspectiveGridMapping,
)


"""Concrete classes for all Grid Mappings supported by the CF Conventions.

For the full listing, see:

https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
cf-conventions.html#appendix-grid-mappings

from which these classes should be kept consistent and up-to-date.
"""


class AlbersEqualArea(ConicGridMapping):
    """The Albers Equal Area grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_albers_equal_area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/aea.html

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

    grid_mapping_name = "albers_conical_equal_area"
    proj_id = "aea"


class AzimuthalEquidistant(AzimuthalGridMapping):
    """The Azimuthal Equidistant grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#azimuthal-equidistant

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/aeqd.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

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

    grid_mapping_name = "azimuthal_equidistant"
    proj_id = "aeqd"


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


class LambertAzimuthalEqualArea(AzimuthalGridMapping):
    """The Lambert Azimuthal Equal Area grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#lambert-azimuthal-equal-area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/laea.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

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

    grid_mapping_name = "lambert_azimuthal_equal_area"
    proj_id = "laea"


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


class LambertCylindricalEqualArea(CylindricalGridMapping):
    """The Equal Area Cylindrical grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_lambert_cylindrical_equal_area

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/cea.html

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

        standard_parallel: 2-`tuple` of number or scalar `Data`
                           or `None`, optional
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

        longitude_of_central_meridian: number or scalar `Data`, optional
            The longitude of (natural) origin i.e. central meridian.
            If provided as a number or `Data` without units, the units
            are taken as 'degrees_east', else the `Data` units are
            taken and must be angular and compatible with longitude.
            The default is 0.0 degrees_east.

        scale_factor_at_projection_origin: number or scalar `Data`, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            Unitless, so `Data` must be unitless. The default is 1.0.

    """

    grid_mapping_name = "lambert_cylindrical_equal_area"
    proj_id = "cea"

    def __init__(
        self,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallel=(0.0, 0.0),
        scale_factor_at_projection_origin=1.0,
        longitude_of_central_meridian=0.0,
        **kwargs,
    ):
        super().__init__(
            false_easting=false_easting,
            false_northing=false_northing,
            **kwargs,
        )

        self.standard_parallel = (
            _validate_map_parameter("standard_parallel", standard_parallel[0]),
            _validate_map_parameter("standard_parallel", standard_parallel[1]),
        )
        self.longitude_of_central_meridian = _validate_map_parameter(
            "longitude_of_central_meridian", longitude_of_central_meridian
        )
        self.scale_factor_at_projection_origin = _validate_map_parameter(
            "scale_factor_at_projection_origin",
            scale_factor_at_projection_origin,
        )


class Mercator(CylindricalGridMapping):
    """The Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/merc.html

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

        standard_parallel: 2-`tuple` of number or scalar `Data`
                           or `None`, optional
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

        scale_factor_at_projection_origin: number or scalar `Data`, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            Unitless, so `Data` must be unitless. The default is 1.0.

    """

    grid_mapping_name = "mercator"
    proj_id = "merc"

    def __init__(
        self,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallel=(0.0, 0.0),
        longitude_of_projection_origin=0.0,
        scale_factor_at_projection_origin=1.0,
        **kwargs,
    ):
        super().__init__(
            false_easting=false_easting,
            false_northing=false_northing,
            **kwargs,
        )

        self.standard_parallel = (
            _validate_map_parameter("standard_parallel", standard_parallel[0]),
            _validate_map_parameter("standard_parallel", standard_parallel[1]),
        )
        self.longitude_of_projection_origin = _validate_map_parameter(
            "longitude_of_projection_origin", longitude_of_projection_origin
        )
        self.scale_factor_at_projection_origin = _validate_map_parameter(
            "scale_factor_at_projection_origin",
            scale_factor_at_projection_origin,
        )


class ObliqueMercator(CylindricalGridMapping):
    """The Oblique Mercator grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_oblique_mercator

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/omerc.html

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

        azimuth_of_central_line: number or scalar `Data`, optional
            The azimuth i.e. tilt angle of the centerline clockwise
            from north at the center point of the line (PROJ 'alpha'
            value). If provided as a number or `Data` without units,
            the units are taken as 'degrees', else the `Data`
            units are taken and must be angular and compatible.
            The default is 0.0 degrees.

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

        scale_factor_at_projection_origin: number or scalar `Data`, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            Unitless, so `Data` must be unitless. The default is 1.0.

    """

    grid_mapping_name = "oblique_mercator"
    proj_id = "omerc"

    def __init__(
        self,
        azimuth_of_central_line=0.0,
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
        scale_factor_at_projection_origin=1.0,
        false_easting=0.0,
        false_northing=0.0,
        **kwargs,
    ):
        super().__init__(
            false_easting=false_easting,
            false_northing=false_northing,
            **kwargs,
        )

        self.azimuth_of_central_line = _validate_map_parameter(
            "azimuth_of_central_line", azimuth_of_central_line
        )
        self.latitude_of_projection_origin = _validate_map_parameter(
            "latitude_of_projection_origin", latitude_of_projection_origin
        )
        self.longitude_of_projection_origin = _validate_map_parameter(
            "longitude_of_projection_origin", longitude_of_projection_origin
        )
        self.scale_factor_at_projection_origin = _validate_map_parameter(
            "scale_factor_at_projection_origin",
            scale_factor_at_projection_origin,
        )


class Orthographic(AzimuthalGridMapping):
    """The Orthographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_orthographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/ortho.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

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

    grid_mapping_name = "orthographic"
    proj_id = "ortho"


class PolarStereographic(AzimuthalGridMapping):
    """The Universal Polar Stereographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#polar-stereographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/ups.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

        straight_vertical_longitude_from_pole: number or scalar `Data`, optional
            The longitude of (natural) origin i.e. central meridian,
            oriented straight up from the North or South Pole.
            If provided as a number or `Data` without units, the units
            are taken as 'degrees_east', else the `Data` units are
            taken and must be angular and compatible with longitude.
            The default is 0.0 degrees_east.

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

        scale_factor_at_projection_origin: number or scalar `Data`, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            Unitless, so `Data` must be unitless. The default is 1.0.

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

        standard_parallel: 2-`tuple` of number or scalar `Data`
                           or `None`, optional
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

    """

    grid_mapping_name = "polar_stereographic"
    proj_id = "ups"

    def __init__(
        self,
        latitude_of_projection_origin=0.0,
        longitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        standard_parallel=(0.0, 0.0),
        straight_vertical_longitude_from_pole=0.0,
        scale_factor_at_projection_origin=1.0,
        **kwargs,
    ):
        # TODO check defaults here, they do not appear for
        # CRS.from_proj4("+proj=ups").to_cf() to cross reference!
        super().__init__(
            latitude_of_projection_origin=latitude_of_projection_origin,
            longitude_of_projection_origin=longitude_of_projection_origin,
            false_easting=false_easting,
            false_northing=false_northing,
            **kwargs,
        )

        # See: https://github.com/cf-convention/cf-conventions/issues/445
        if (
            longitude_of_projection_origin
            and straight_vertical_longitude_from_pole
        ):
            raise ValueError(
                "Only one of 'longitude_of_projection_origin' and "
                "'straight_vertical_longitude_from_pole' can be set."
            )

        self.straight_vertical_longitude_from_pole = _validate_map_parameter(
            "straight_vertical_longitude_from_pole",
            straight_vertical_longitude_from_pole,
        )
        self.standard_parallel = (
            _validate_map_parameter("standard_parallel", standard_parallel[0]),
            _validate_map_parameter("standard_parallel", standard_parallel[1]),
        )
        self.scale_factor_at_projection_origin = _validate_map_parameter(
            "scale_factor_at_projection_origin",
            scale_factor_at_projection_origin,
        )


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


class LatitudeLongitude(LatLonGridMapping):
    """The Latitude-Longitude grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_latitude_longitude

    for more information.

    .. versionadded:: GMVER

    """

    grid_mapping_name = "latitude_longitude"
    proj_id = "latlong"


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


class Stereographic(AzimuthalGridMapping):
    """The Stereographic grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#_stereographic

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/stere.html

    for more information.

    .. versionadded:: GMVER

    :Parameters:

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

        scale_factor_at_projection_origin: number or scalar `Data`, optional
            The scale factor used in the projection (PROJ 'k_0' value).
            Unitless, so `Data` must be unitless. The default is 1.0.

    """

    grid_mapping_name = "stereographic"
    proj_id = "stere"

    def __init__(
        self,
        false_easting=0.0,
        false_northing=0.0,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        scale_factor_at_projection_origin=1.0,
        **kwargs,
    ):
        super().__init__(
            false_easting=false_easting,
            false_northing=false_northing,
            longitude_of_projection_origin=longitude_of_projection_origin,
            latitude_of_projection_origin=latitude_of_projection_origin,
            **kwargs,
        )

        self.scale_factor_at_projection_origin = _validate_map_parameter(
            "scale_factor_at_projection_origin",
            scale_factor_at_projection_origin,
        )


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


class VerticalPerspective(PerspectiveGridMapping):
    """The Vertical (or Near-sided) Perspective grid mapping.

    See the CF Conventions document 'Appendix F: Grid Mappings' sub-section on
    this grid mapping:

    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/
    cf-conventions.html#vertical-perspective

    or the corresponding PROJ projection page:

    https://proj.org/en/9.2/operations/projections/nsper.html

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

    """

    grid_mapping_name = "vertical_perspective"
    proj_id = "nsper"


# Representing all Grid Mappings repsented by the CF Conventions (Appendix F)
_all_concrete_grid_mappings = (
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
)
