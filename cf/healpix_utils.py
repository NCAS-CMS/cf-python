"""TODOHEALPIX"""


def del_healpix_coordinate_reference(f, axis=None):
    """TODOHEALPIX

    :Parameters:

        f: `Field` or `Domain`

        axis: optional

    :Returns:

        `CoordinateReference` or `None`

    """
    cr_key, cr = f.coordinate_reference(
        "grid_mapping_name:healpix", item=True, default=(None, None)
    )
    latlon = f.coordinate_reference(
        "grid_mapping_name:latitude_longitude", default=None
    )

    if cr is None:
        out = None
    else:
        out = cr.copy()

    if latlon is not None:
        # There is already a latitude_longitude coordinate reference,
        # so delete a healpix coordinate reference.
        if cr is not None:
            f.del_construct(cr_key)

    elif cr is not None:
        cc = cr.coordinate_conversion
        cc.del_parameter("grid_mapping_name", None)
        cc.del_parameter("indexing_scheme", None)
        cc.del_parameter("refinement_level", None)
        if cc.parameters() or cr.datum.parameters():
            # The healpix coordinate reference contains generic
            # coordinate conversion or datum parameters, so rename it
            # as 'latitude_longitude'.
            cr.coordinate_conversion.set_parameter(
                "grid_mapping_name", "latitude_longitude"
            )

            # Remove a healpix_index coordinate from the coordinate
            # reference
            if axis is None:
                axis = f.healpix_axis(None)

            if axis is not None:
                key = f.coordinate(
                    "healpix_index",
                    filter_by_axis=(axis,),
                    axis_mode="exact",
                    key=True,
                    default=None,
                )
                if key is not None:
                    cr.del_coordinate(key)
        else:
            # The healpix coordinate reference contains no generic
            # parameters, so delete it.
            f.del_construct(cr_key)

    return out
