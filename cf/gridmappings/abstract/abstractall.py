import itertools
import re
from abc import ABC, abstractmethod

from pyproj import CRS

from ...constants import cr_canonical_units, cr_gm_valid_attr_names_are_numeric
from ...data import Data
from ...data.utils import is_numeric_dtype
from ...units import Units

from .gridmappingbase import GridMapping


class AzimuthalGridMapping(GridMapping):
    """A Grid Mapping with Azimuthal classification.

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

    def __init__(
        self,
        longitude_of_projection_origin=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.longitude_of_projection_origin = _validate_map_parameter(
            "longitude_of_projection_origin", longitude_of_projection_origin
        )
        self.latitude_of_projection_origin = _validate_map_parameter(
            "latitude_of_projection_origin", latitude_of_projection_origin
        )
        self.false_easting = _validate_map_parameter(
            "false_easting", false_easting
        )
        self.false_northing = _validate_map_parameter(
            "false_northing", false_northing
        )


class ConicGridMapping(GridMapping):
    """A Grid Mapping with Conic classification.

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

    def __init__(
        self,
        standard_parallel,
        longitude_of_central_meridian=0.0,
        latitude_of_projection_origin=0.0,
        false_easting=0.0,
        false_northing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.standard_parallel = (
            _validate_map_parameter("standard_parallel", standard_parallel[0]),
            _validate_map_parameter("standard_parallel", standard_parallel[1]),
        )
        self.longitude_of_central_meridian = _validate_map_parameter(
            "longitude_of_central_meridian", longitude_of_central_meridian
        )
        self.latitude_of_projection_origin = _validate_map_parameter(
            "latitude_of_projection_origin", latitude_of_projection_origin
        )
        self.false_easting = _validate_map_parameter(
            "false_easting", false_easting
        )
        self.false_northing = _validate_map_parameter(
            "false_northing", false_northing
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


class LatLonGridMapping(GridMapping):
    """A Grid Mapping with Latitude-Longitude nature.

    Such a Grid Mapping is based upon latitude and longitude coordinates
    on a spherical Earth, defining the canonical 2D geographical coordinate
    system so that the figure of the Earth can be described.

    .. versionadded:: GMVER

    """

    pass


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


_all_abstract_grid_mappings = (
    GridMapping,
    AzimuthalGridMapping,
    ConicGridMapping,
    CylindricalGridMapping,
    LatLonGridMapping,
    PerspectiveGridMapping,
)
