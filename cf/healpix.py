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

    .. seealso:: `cf.Field.create_latlon_coordinates`,
                 `cf.Field.healpix_to_ugrid`

    :Parameters:

        f: `Field` or `Domain`
            The Field or Domain containing the HEALPix grid, which
            will be updated in-place.

        pole_longitude: `None` or number
            The longitude of HEALPix coordinate bounds that lie
            exactly on the north (south) pole. If `None` then the
            longitude of such a vertex will be the same as the south
            (north) vertex of the same cell. If set to a number, then
            the longitudes of such vertices will all be given that
            value.

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

    # Get the Dask array of HEALPix indices.
    #
    # `cf_healpix_coordinates` anad `cf_healpix_bounds` have their own
    # calls to `cfdm_to_memory`, so we can set _force_to_memory=False.
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
        latitude=True,
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
        longitude=True,
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
        latitude=True,
    )
    bounds = f._Bounds(data=dy)
    lat.set_bounds(bounds, copy=False)

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
        longitude=True,
        pole_longitude=pole_longitude,
    )
    bounds = f._Bounds(data=dy)
    lon.set_bounds(bounds, copy=False)

    # Set the new latitude and longitude coordinates
    axis = hp["domain_axis_key"]
    lat_key = f.set_construct(lat, axes=axis, copy=False)
    lon_key = f.set_construct(lon, axes=axis, copy=False)

    return lat_key, lon_key


def _healpix_increase_refinement_level(x, ncells, iaxis, quantity):
    """Increase the HEALPix refinement level in-place.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.Field.healpix_increase_refinement_level`

    :Parameters:

        x: construct
            The construct containing data that is to be changed
            in-place.

        ncells: `int`
            The number of cells at the new refinement level which are
            contained in one cell at the original refinement level.

        iaxis: `int`
            The position of the HEALPix axis in the construct's data
            dimensions.

        quantity: `str`
            Whether the data represent intensive or extensive
            quantities, specified with ``'intensive'`` and
            ``'extensive'`` respectively.

    :Returns:

        `None`

    """
    from dask.array.core import normalize_chunks

    # Get the Dask array (e.g. dx.shape is (12, 19, 48))
    dx = x.data.to_dask_array(_force_mask_hardness=False)

    # Divide extensive data by the number of new cells
    if quantity == "extensive":
        dx = dx / ncells

    # Add a new size dimension just after the HEALPix dimension
    # (e.g. .shape becomes (12, 19, 48, 1))
    new_axis = iaxis + 1
    dx = da.expand_dims(dx, new_axis)

    # Modify the size of the new dimension to be the number of cells
    # at the new refinement level which are contained in one cell at
    # the original refinement level (e.g. size becomes 16)
    shape = list(dx.shape)
    shape[new_axis] = ncells

    # Work out what the chunks should be for the new dimension
    chunks = list(dx.chunks)
    chunks[new_axis] = "auto"
    chunks = normalize_chunks(chunks, shape, dtype=dx.dtype)

    # Broadcast the data along the new dimension (e.g. dx.shape
    # becomes (12, 19, 48, 16))
    dx = da.broadcast_to(dx, shape, chunks=chunks)

    # Reshape the array so that it has a single, larger HEALPix
    # dimension (e.g. dx.shape becomes (12, 19, 768))
    dx = dx.reshape(
        shape[:iaxis]
        + [shape[iaxis] * shape[new_axis]]
        + shape[new_axis + 1 :]
    )

    x.set_data(dx, copy=False)


def _healpix_increase_refinement_level_indices(
    healpix_index, ncells, refinement_level
):
    """Increase the refinement level of HEALPix indices in-place.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    .. versionadded:: NEXTVERSION

    .. seealso:: `cf.Field.healpix_increase_refinement_level`

    :Parameters:

        healpix_index: `Coordinate`
            The HEALPix indices to be changed. It is assumed they use
            the "nested" indexing scheme.

        ncells: `int`
            The number of cells at the new higher refinement level
            which are contained in one cell at the original refinement
            level.

        refinement_level: `int`
            The new higher refinement level.

    :Returns:

        `None`

    """
    from cfdm import integer_dtype
    from dask.array.core import normalize_chunks

    # Get any cached data values
    cached = healpix_index.data._get_cached_elements().copy()

    # Get the Dask array (e.g. dx.shape is (48,))
    dx = healpix_index.data.to_dask_array(_force_mask_hardness=False)

    # Set the data type to allow for the largest possible HEALPix
    # index at the new refinement level
    dtype = integer_dtype(12 * (4**refinement_level) - 1)
    if dx.dtype != dtype:
        dx = dx.astype(dtype)

    # Change each original HEALpix index to the smallest new HEALPix
    # index that the larger cell contains
    dx = dx * ncells

    # Add a new size dimension just after the HEALPix dimension
    # (e.g. .shape becomes (48, 1))
    new_axis = 1
    dx = da.expand_dims(dx, new_axis)

    # Modify the size of the new dimension to be the number of cells
    # at the new refinement level which are contained in one cell at
    # the original refinement level (e.g. size becomes 16)
    shape = list(dx.shape)
    shape[new_axis] = ncells

    # Work out what the chunks should be for the new dimension
    chunks = list(dx.chunks)
    chunks[new_axis] = "auto"
    chunks = normalize_chunks(chunks, shape, dtype=dx.dtype)

    # Broadcast the data along the new dimension (e.g. dx.shape
    # becomes (48, 16))
    dx = da.broadcast_to(dx, shape, chunks=chunks)

    # Increment the broadcast values along the new dimension, so that
    # dx[i, :] contains all of the nested indices at the higher
    # refinement level that correspond to HEALPix index value i at the
    # original refinement level.
    new_shape = [1] * dx.ndim
    new_shape[new_axis] = shape[new_axis]

    dx += da.arange(ncells, chunks=chunks[new_axis]).reshape(new_shape)

    # Reshape the new array to combine the original HEALPix and
    # broadcast dimensions into a single new HEALPix dimension
    # (e.g. dx.shape becomes (768,))
    dx = dx.reshape(shape[0] * shape[new_axis])

    healpix_index.set_data(dx, copy=False)

    # Set new cached data elements
    data = healpix_index.data
    if 0 in cached:
        x = np.array(cached[0], dtype=dtype) * ncells
        data._set_cached_elements({0: x, 1: x + 1})

    if -1 in cached:
        x = np.array(cached[-1], dtype=dtype) * ncells + (ncells - 1)
        data._set_cached_elements({-1: x})


def _healpix_indexing_scheme(f, hp, new_indexing_scheme):
    """Change the indexing scheme of HEALPix indices in-place.

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

        healpix_index: `Coordinate` TODOHEALPIX
            The healpix_index coordinates, which will be updated TODOHEALPIX
            in-place.

        hp: `dict`
            The HEALPix info dictionary.

        new_indexing_scheme: `str`
            The new indexing scheme.

    :Returns:

        `None`

    """
    from cfdm import integer_dtype

    from .data.dask_utils import cf_healpix_indexing_scheme

    healpix_index = hp["healpix_index"]
    indexing_scheme = hp["indexing_scheme"]
    refinement_level = hp.get("refinement_level")

    # Change the HEALPix indices
    #
    # `cf_healpix_indexing_scheme` has its own call to
    # `cfdm_to_memory`, so we can set _force_to_memory=False.
    dx = healpix_index.data.to_dask_array(
        _force_mask_hardness=False, _force_to_memory=False
    )

    # Find the datatype for the largest possible index at this
    # refinement level
    if new_indexing_scheme == "nested_unique":
        # 16*(4**n) - 1 = 4*(4**n) + 12*(4**n) - 1
        dtype = integer_dtype(16 * (4**refinement_level) - 1)
    else:
        # nested or ring
        dtype = integer_dtype(12 * (4**refinement_level) - 1)

    dx = dx.map_blocks(
        cf_healpix_indexing_scheme,
        dtype=dtype,
        meta=np.array((), dtype=dtype),
        indexing_scheme=indexing_scheme,
        new_indexing_scheme=new_indexing_scheme,
        refinement_level=refinement_level,
    )
    healpix_index.set_data(dx, copy=False)

    # If a Dimension Coordinate is now not monotonically ordered,
    # convert it to an Auxiliary Coordinate
    if healpix_index.construct_type == "dimension_coordinate" and set(
        (indexing_scheme, new_indexing_scheme)
    ) != set(("nested", "nested_unique")):
        f.dimension_to_auxiliary(hp["coordinate_key"], inplace=True)


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
    match indexing_scheme:
        case "nested_unique":
            # nested_unique indexing scheme
            index = []
            healpix_index = healpix_index.array
            orders = healpix.uniq2pix(healpix_index, nest=True)[0]
            orders = np.unique(orders)
            for order in orders:
                # For this refinement level, find the HEALPix nested
                # indices of the cells that contain the lat-lon
                # points.
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

        case "nested" | "ring":
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

        case None:
            raise ValueError(
                f"Can't locate HEALPix cells for {f!r}: indexing_scheme has "
                "not been set in the healpix grid mapping coordinate "
                "reference"
            )

        case _:
            raise ValueError(
                f"Can't locate HEALPix cells for {f!r}: indexing_scheme in "
                "the healpix grid mapping coordinate reference must be one "
                f"of {HEALPix_indexing_schemes!r}. Got {indexing_scheme!r}"
            )

    # Return the cell locations as a numpy array of element indices
    return index.compute()


def del_healpix_coordinate_reference(f):
    """Remove a healpix grid mapping coordinate reference construct.

    If required, a new latitude_longitude grid mapping coordinate
    reference will be created in-place to store any generic coordinate
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
     'healpix_index': <CF DimensionCoordinate: healpix_index(48)>}

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
            "to find the maximum HEALPix refinement level"
        )

    return healpix.nside2order(healpix._chp.NSIDE_MAX)
