from .abstract import LatLonGridMapping


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

    @classmethod
    def is_latlon_gm(cls):
        """Whether the Grid Mapping is of LatitudeLongitude form.

        :Returns:

            `bool`
                True only if the Grid Mapping is LatitudeLongitude.

        """
        return True
