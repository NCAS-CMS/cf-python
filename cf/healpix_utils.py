"""General functions useful for HEALPix functionality."""


def _del_healpix_coordinate_reference(f):
    """Remove a healpix grid mapping coordinate reference construct.

    A new latitude_longitude grid mapping coordinate reference will be
    created in-place, if required, to store any generic coordinate
    conversion or datum parameters found in the healpix grid mapping
    coordinate reference.

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain from which to delete the healpix grid
            mapping coordinate reference.

    :Returns:

        `CoordinateReference` or `None`
            The removed healpix grid mapping coordinate reference
            construct, or `None` if there wasn't one.

    """
    cr_key, cr = f.coordinate_reference(
        "grid_mapping_name:healpix", item=True, default=(None, None)
    )
    latlon = f.coordinate_reference(
        "grid_mapping_name:latitude_longitude", default=None
    )

    if cr is not None:
        f.del_construct(cr_key)

        if latlon is None:
            latlon = cr.copy()
            cc = latlon.coordinate_conversion
            cc.del_parameter("grid_mapping_name", None)
            cc.del_parameter("indexing_scheme", None)
            cc.del_parameter("refinement_level", None)
            if cc.parameters() or latlon.datum.parameters():
                # The healpix coordinate reference contains generic
                # coordinate conversion or datum parameters
                latlon.coordinate_conversion.set_parameter(
                    "grid_mapping_name", "latitude_longitude"
                )

                # Remove healpix_index coordinates from the coordinate
                # reference
                for key in f.coordinates(
                    "healpix_index", filter_by_naxes=(1,), todict=True
                ):
                    latlon.del_coordinate(key, None)

                f.set_construct(latlon)

    return cr
