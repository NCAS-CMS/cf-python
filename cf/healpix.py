"""HEALPix functionality."""

import logging

import dask.array as da
import numpy as np
from cfdm import is_log_level_info

logger = logging.getLogger(__name__)

HEALPix_indexing_schemes = ("nested", "ring", "nested_unique")


def _healpix_create_latlon_coordinates(f, pole_longitude):
    """Create latitude and longitude coordinates for a HEALPix grid.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.Field.create_latlon_coordinates`

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain containing the HEALPix grid, which
            will be updated in-place.

        pole_longitude: `None` or number
            The longitude of coordinate bounds that lie exactly on the
            north (south) pole. If `None` then the longitude of such a
            vertex will be the same as the south (north) vertex of the
            same cell. If set to a number, then the longitudes of such
            vertices will all be given that value.

    :Returns:

        (`str`, `str`) or (`None`, `None`)
            The keys of the new latitude and longitude coordinate
            constructs, or `None` if the coordinates could not be
            created.

    """
    from .data.dask_utils import cf_healpix_bounds, cf_healpix_coordinates

    hp = f.healpix_info()

    indexing_scheme = hp.get("indexing_scheme")
    if indexing_scheme not in HEALPix_indexing_schemes:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 1-d latitude and longitude coordinates for "
                f"{f!r}: indexing_scheme in the healpix grid mapping "
                "coordinate reference must be one of "
                f"{HEALPix_indexing_schemes!r}. Got {indexing_scheme!r}"
            )  # pragma: no cover

        return (None, None)

    refinement_level = hp.get("refinement_level")
    if refinement_level is None and indexing_scheme != "nested_unique":
        if is_log_level_info(logger):
            logger.info(
                "Can't create 1-d latitude and longitude coordinates for "
                f"{f!r}: refinement_level has not been set in the healpix "
                "grid mapping coordinate reference"
            )  # pragma: no cover

        return (None, None)

    healpix_index = hp.get("healpix_index")
    if healpix_index is None:
        if is_log_level_info(logger):
            logger.info(
                "Can't create 1-d latitude and longitude coordinates for "
                f"{f!r}: Missing healpix_index coordinates"
            )  # pragma: no cover

        return (None, None)

    # Get the Dask array of HEALPix indices
    dx = healpix_index.data.to_dask_array(
        _force_mask_hardness=False, _force_to_memory=False
    )
    meta = np.array((), dtype="float64")

    # Create latitude coordinates
    dy = dx.map_blocks(
        cf_healpix_coordinates,
        meta=meta,
        indexing_scheme=indexing_scheme,
        refinement_level=refinement_level,
        lat=True,
    )
    lat = f._AuxiliaryCoordinate(
        data=f._Data(dy, "degrees_north", copy=False),
        properties={"standard_name": "latitude"},
        copy=False,
    )

    # Create longitude coordinates
    dy = dx.map_blocks(
        cf_healpix_coordinates,
        meta=meta,
        indexing_scheme=indexing_scheme,
        refinement_level=refinement_level,
        lon=True,
    )
    lon = f._AuxiliaryCoordinate(
        data=f._Data(dy, "degrees_east", copy=False),
        properties={"standard_name": "longitude"},
        copy=False,
    )

    # Create latitude bounds
    dy = da.blockwise(
        cf_healpix_bounds,
        "ij",
        dx,
        "i",
        new_axes={"j": 4},
        meta=meta,
        indexing_scheme=indexing_scheme,
        refinement_level=refinement_level,
        lat=True,
    )
    bounds = f._Bounds(data=dy)
    lat.set_bounds(bounds)

    # Create longitude bounds
    dy = da.blockwise(
        cf_healpix_bounds,
        "ij",
        dx,
        "i",
        new_axes={"j": 4},
        meta=meta,
        indexing_scheme=indexing_scheme,
        refinement_level=refinement_level,
        lon=True,
        pole_longitude=pole_longitude,
    )
    bounds = f._Bounds(data=dy)
    lon.set_bounds(bounds)

    # Set the new latitude and longitude coordinates
    axis = hp["domain_axis_key"]
    lat_key = f.set_construct(lat, axes=axis, copy=False)
    lon_key = f.set_construct(lon, axes=axis, copy=False)

    return lat_key, lon_key


def _healpix_indexing_scheme(healpix_index, hp, new_indexing_scheme):
    """Change the indexing scheme of HEALPix indices.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    M. Reinecke and E. Hivon: Efficient data structures for masks on
    2D grids. A&A, 580 (2015)
    A132. https://doi.org/10.1051/0004-6361/201526549

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.Field.healpix_indexing_scheme`

    :Parameters:

        healpix_index: `Coordinate`
            The healpix_index coordinates, which will be updated
            in-place.

        hp: `dict`
            The HEALPix info dictionary.

        new_indexing_scheme: `str`
            The new indexing scheme.

    :Returns:

        `None`

    """
    from .data.dask_utils import cf_healpix_indexing_scheme

    indexing_scheme = hp["indexing_scheme"]
    refinement_level = hp.get("refinement_level")

    # Change the HEALPix indices
    dx = healpix_index.data.to_dask_array(
        _force_mask_hardness=False, _force_to_memory=False
    )
    dx = dx.map_blocks(
        cf_healpix_indexing_scheme,
        meta=np.array((), dtype="int64"),
        indexing_scheme=indexing_scheme,
        new_indexing_scheme=new_indexing_scheme,
        refinement_level=refinement_level,
    )
    healpix_index.set_data(dx, copy=False)


def _healpix_locate(lat, lon, f):
    """Locate HEALPix cells containing latitude-longitude locations.

    Returns the discrete axis indices of the cells containing the
    latitude-longitude locations.

    If a single latitude is given then it is paired with each
    longitude, and if a single longitude is given then it is paired
    with each latitude. If multiple latitudes and multiple longitudes
    are provided then they are paired element-wise.

    If a cell contains more than one of the given latitude-longitude
    locations then that cell's index appears only once in the output.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.locate`

    :Parameters:

        lat: (sequence of) number
            The latitude(s), in degrees north, for which to find the
            cell indices. Must be in the range [-90, 90], although
            this is not checked.

        lon: (sequence of) number
            The longitude(s), in degrees east, for which to find the
            cell indices.

        f: `Field` or `Domain`
            The Field or Domain containing the HEALPix grid.

    :Returns:

        `numpy.ndarray`
            Indices for the HEALPix axis that contain the
            latitude-longitude locations. Note that these indices
            identify locations along the HEALPix axis, and are not the
            HEALPix indices defined by the indexing scheme.

    """
    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "to allow the location of HEALPix cells"
        )

    hp = f.healpix_info()

    healpix_index = hp.get("healpix_index")
    if healpix_index is None:
        raise ValueError(
            f"Can't locate HEALPix cells for {f!r}: There are no "
            "healpix_index coordinates"
        )

    indexing_scheme = hp.get("indexing_scheme")
    if indexing_scheme is None:
        raise ValueError(
            f"Can't locate HEALPix cells for {f!r}: indexing_scheme has "
            "not been set in the healpix grid mapping coordinate reference"
        )

    if indexing_scheme == "nested_unique":
        # nested_unique indexing scheme
        index = []
        healpix_index = healpix_index.array
        orders = healpix.uniq2pix(healpix_index, nest=True)[0]
        orders = np.unique(orders)
        for order in orders:
            # For this refinement level, find the HEALPix nested
            # indices of the cells that contain the lat-lon points.
            nside = healpix.order2nside(order)
            pix = healpix.ang2pix(nside, lon, lat, nest=True, lonlat=True)
            # Remove duplicate indices
            pix = np.unique(pix)
            # Convert back to HEALPix nested_unique indices
            pix = healpix._chp.nest2uniq(order, pix, pix)
            # Find where these HEALPix indices are located in the
            # healpix_index coordinates
            index.append(da.where(da.isin(healpix_index, pix))[0])

        index = da.unique(da.concatenate(index, axis=0))
    elif indexing_scheme in ("nested", "ring"):
        # nested or ring indexing scheme
        refinement_level = hp.get("refinement_level")
        if refinement_level is None:
            raise ValueError(
                f"Can't locate HEALPix cells for {f!r}: refinement_level "
                "has not been set in the healpix grid mapping coordinate "
                "reference"
            )

        # Find the HEALPix indices of the cells that contain the
        # lat-lon points
        nest = indexing_scheme == "nested"
        nside = healpix.order2nside(refinement_level)
        pix = healpix.ang2pix(nside, lon, lat, nest=nest, lonlat=True)
        # Remove duplicate indices
        pix = np.unique(pix)
        # Find where these HEALPix indices are located in the
        # healpix_index coordinates
        index = da.where(da.isin(healpix_index, pix))[0]
    else:
        raise ValueError(
            f"Can't locate HEALPix cells for {f!r}: indexing_scheme in the "
            "healpix grid mapping coordinate reference must be one of "
            f"{HEALPix_indexing_schemes!r}. Got {indexing_scheme!r}"
        )

    # Return the cell locations as a numpy array of element indices
    return index.compute()


def del_healpix_coordinate_reference(f):
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
    if cr is not None:
        f.del_construct(cr_key)

        latlon = f.coordinate_reference(
            "grid_mapping_name:latitude_longitude", default=None
        )
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


def healpix_info(f):
    """Get information about the HEALPix grid, if there is one.

    .. versionadded:: NEXTVERSION

    :Parameters:

        f: `Field` or `Domain`
            The field or domain.

    :Returns:

        `dict`
            The information about the HEALPix axis, with some or all
            of the following dictionary keys:

            * ``'coordinate_reference_key'``: The construct key of the
                                              healpix coordinate
                                              reference construct.

            * ``'grid_mapping_name:healpix'``: The healpix coordinate
                                               reference construct.

            * ``'indexing_scheme'``: The HEALPix indexing scheme.

            * ``'refinement_level'``: The refinement level of the
                                      HEALPix grid.

            * ``'domain_axis_key'``: The construct key of the HEALPix
                                     domain axis construct.

            * ``'coordinate_key'``: The construct key of the
                                    healpix_index coordinate
                                    construct.

            * ``'healpix_index'``: The healpix_index coordinate
                                   construct.

            The dictionary will be empty if there is no HEALPix axis.

    **Examples**

    >>> f = cf.example_field(12)
    >>> healpix_info(f)
    {'coordinate_reference_key': 'coordinatereference0',
     'grid_mapping_name:healpix': <CF CoordinateReference: grid_mapping_name:healpix>,
     'indexing_scheme': 'nested',
     'refinement_level': 1,
     'domain_axis_key': 'domainaxis1',
     'coordinate_key': 'auxiliarycoordinate0',
     'healpix_index': <CF AuxiliaryCoordinate: healpix_index(48) 1>}

    >>> f = cf.example_field(0)
    >>> healpix_info(f)
    {}

    """
    info = {}

    cr_key, cr = f.coordinate_reference(
        "grid_mapping_name:healpix", item=True, default=(None, None)
    )
    if cr is not None:
        info["coordinate_reference_key"] = cr_key
        info["grid_mapping_name:healpix"] = cr
        parameters = cr.coordinate_conversion.parameters()
        for param in ("indexing_scheme", "refinement_level"):
            value = parameters.get(param)
            if value is not None:
                info[param] = value

    hp_key, healpix_index = f.coordinate(
        "healpix_index",
        filter_by_naxes=(1,),
        item=True,
        default=(None, None),
    )
    if healpix_index is not None:
        info["domain_axis_key"] = f.get_data_axes(hp_key)[0]
        info["coordinate_key"] = hp_key
        info["healpix_index"] = healpix_index

    return info


def healpix_max_refinement_level():
    """Return the maxium permitted HEALPix refinement level.

    The maximum refinement level is the highest refiniment level for
    which all of its HEALPix indices are representable as double
    precision integers.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    .. versionadded:: NEXTVERSION

    :Returns:

        `int`
            The maxium permitted HEALPix refinement level.

    """
    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "to find the HEALPix maximum refinement level"
        )

    return healpix.nside2order(healpix._chp.NSIDE_MAX)
