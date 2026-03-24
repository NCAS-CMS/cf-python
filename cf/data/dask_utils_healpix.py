"""HEALPix functions intended to be passed to be Dask.

These will typically be functions that operate on Dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

import numpy as np
from cfdm.data.dask_utils import cfdm_to_memory

from cf.functions import healpix_max_refinement_level


def _zuniq2pix(a, nest=False):
    """Convert from the zuniq to ring or nested pixel scheme.

    This function emulates the as-yet-nonexistent function
    `healpix.zuniq2pix`, with which it should be replaced should that
    function ever come into existence (see
    https://github.com/ntessore/healpix/issues/94).

    See
    https://github.com/cds-astro/cds-healpix-rust/blob/v0.7.3/src/nested/mod.rs#L188-L194
    for the algorithm definition.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: array_like
            An array of zuniq indices.

        nest: `bool`, optional
            True if for nested indices, or False (the default) for
            ring indices.

    :Returns:

        (`numpy.ndarray`, `numpy.ndarray`)
            Returns a 2-tuple of a `numpy` array of HEALPix refinement
            levels, and a `numpy` array of HEALPix pixel indices. The
            indices will follow the nested indexing scheme if *nest*
            is True, otherwise the ring indexing scheme.

    """
    if not nest:
        raise NotImplementedError(
            "Can't yet convert from zuniq to ring indices "
            "(consider doing zuniq -> nuniq -> ring instead)"
        )

    if isinstance(a, np.ndarray):
        a = a.astype("int64", copy=False)
    else:
        a = np.array(a, dtype="int64")

    n_trailing_zeros = np.bitwise_count((a & -a) - 1)
    n_trailing_zeros = n_trailing_zeros.astype("int8", copy=False)

    delta_depth = n_trailing_zeros >> 1
    depth = healpix_max_refinement_level() - delta_depth

    a = a >> (n_trailing_zeros + 1)

    return depth, a


def _zuniq2uniq(a):
    """Convert from the zuniq to nuniq pixel scheme.

    This function emulates the as-yet-nonexistent function
    `healpix.zuniq2uniq`, with which it should be replaced should that
    function ever come into existence (see
    https://github.com/ntessore/healpix/issues/94).

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: array_like
            An array of zuniq indices.

    :Returns:

        `numpy.ndarray`
            Returns a `numpy` array of HEALPix pixel indices following
            the nuniq indexing scheme.

    """
    order, a = _zuniq2pix(a, nest=True)
    order = order.astype("int64", copy=False)

    order += 1
    a += 4**order

    return a


def _uniq2zuniq(a):
    """Convert from the nuniq to zuniq pixel scheme.

    This function emulates (in a non-optimal way!) the
    as-yet-nonexistent function `healpix.uniq2zuniq`, with which it
    should be replaced should that function ever come into existence
    (see https://github.com/ntessore/healpix/issues/94).

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: array_like
            An array of nuniq indices.

    :Returns:

        `numpy.ndarray`
            Returns a `numpy` array of HEALPix pixel indices following
            the zuniq indexing scheme.

    """
    import healpix

    order, a = healpix.uniq2pix(a, nest=True)
    order = order.astype("int64", copy=False)

    order = 4 ** (healpix_max_refinement_level() - order)
    a *= 2
    a += 1
    a *= order

    return a


def _pix2zuniq(refinement_level, a, nest=False):
    """Convert the ring or nested to zuniq pixel scheme.

    This function emulates (in a non-optimal way!) the
    as-yet-nonexistent function `healpix.pix2zuniq`, with which it
    should be replaced should that function ever come into existence
    (see https://github.com/ntessore/healpix/issues/94).

    .. versionadded:: NEXTVERSION

    :Parameters:

        refinement_level: `int`
            The refinement level of the indices.

        a: array_like
            An array of nested or ring indices at the given refinement
            level.

        nest: `bool`, optional
            True for nested indices, or False (the default) for ring
            indices.

    :Returns:

        `numpy.ndarray`
            Returns a `numpy` array of HEALPix pixel indices following
            the zuniq indexing scheme.

    """
    if not nest:
        # Convert ring to nested
        import healpix

        nside = healpix.order2nside(refinement_level)
        a = healpix.ring2nest(nside, a)
    elif isinstance(a, np.ndarray):
        a = a.astype("int64", copy=True)
    else:
        a = np.array(a, dtype="int64")

    # Convert nested to zuniq
    a *= 2
    a += 1
    a *= 4 ** (healpix_max_refinement_level() - refinement_level)

    return a


def cf_healpix_bounds(
    a,
    indexing_scheme,
    refinement_level=None,
    latitude=False,
    longitude=False,
    pole_longitude=None,
):
    """Calculate HEALPix cell bounds.

    Latitude or longitude locations of the cell vertices are derived
    from HEALPix indices. Each cell has four bounds which are returned
    in an anticlockwise direction, as seen from above, starting with
    the northern-most vertex.

    See CF Appendix F: Grid Mappings.
    https://doi.org/10.5281/zenodo.14274886

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        indexing_scheme: `str`
            The HEALPix indexing scheme. One of ``'nested'``,
            ``'ring'``, ``'nuniq'``, or ``'zuniq'``.

        refinement_level: `int` or `None`, optional
            The refinement level of the grid within the HEALPix
            hierarchy, starting at 0 for the base tessellation with 12
            cells. Must be an `int` for *indexing_scheme* ``'nested'``
            or ``'ring'``, but is ignored for *indexing_scheme*
            ``'nuniq'`` or ``'zuniq'``, in which case
            *refinement_level* may be `None`.

        latitude: `bool`, optional
            If True then return the bounds' latitudes.

        longitude: `bool`, optional
            If True then return the bounds' longitudes.

        pole_longitude: `None` or number, optional
            Define the longitudes of vertices that lie exactly on the
            north or south pole. If `None` (the default) then the
            longitude of such a vertex on the north (south) pole will
            be the same as the longitude of the south (north) vertex
            of the same cell. If set to a number, then the longitudes
            of all vertices on the north or south pole will be given
            the value *pole_longitude*. Ignored if *longitude* is
            False.

    :Returns:

        `numpy.ndarray`
            A 2-d array containing the HEALPix cell bounds.

    **Examples**

    >>> cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, latitude=True
    )
    array([[41.8103149 , 19.47122063,  0.        , 19.47122063],
           [66.44353569, 41.8103149 , 19.47122063, 41.8103149 ],
           [66.44353569, 41.8103149 , 19.47122063, 41.8103149 ],
           [90.        , 66.44353569, 41.8103149 , 66.44353569]])
    >>> cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, longitude=True
    )
    array([[45. , 22.5, 45. , 67.5],
           [90. , 45. , 67.5, 90. ],
           [ 0. ,  0. , 22.5, 45. ],
           [45. ,  0. , 45. , 90. ]])
    >>> cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, longitude=True,
    ...     pole_longitude=3.14159
    )
    array([[45.     , 22.5    , 45.     , 67.5    ],
           [90.     , 45.     , 67.5    , 90.     ],
           [ 0.     ,  0.     , 22.5    , 45.     ],
           [ 3.14159,  0.     , 45.     , 90.     ]])

    """
    if not a.size:
        return a

    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "for the calculation of latitude/longitude coordinate bounds "
            "of a HEALPix grid"
        )

    a = cfdm_to_memory(a)

    if not a.ndim:
        # Turn a 0-d array into a 1-d array, for convenience.
        a = np.atleast_1d(a)

    # Keep an eye on https://github.com/ntessore/healpix/issues/66
    if a.ndim > 1:
        raise ValueError(
            "Can only calculate HEALPix cell bounds when the "
            f"healpix_index array has one dimension. Got shape: {a.shape}"
        )

    if latitude and longitude:
        raise ValueError(
            "Only one of 'latitude' and 'longitude' must be True."
        )

    if latitude:
        pos = 1
    elif longitude:
        pos = 0
    else:
        raise ValueError("One of 'latitude' and 'longitude' must be True.")

    # Convert zuniq to nuniq
    if indexing_scheme == "zuniq":
        a = _zuniq2uniq(a)
        indexing_scheme = "nuniq"

    # Define the function that's going to calculate the bounds from
    # the HEALPix indices
    match indexing_scheme:
        case "ring":
            bounds_func = healpix._chp.ring2ang_uv
        case "nested" | "nuniq":
            bounds_func = healpix._chp.nest2ang_uv
        case _:
            raise ValueError(
                "Can't calculate HEALPix cell bounds: Unknown "
                "'indexing_scheme' in cf_healpix_bounds: "
                f"{indexing_scheme!r}"
            )

    # Define the cell vertices in an anticlockwise direction, as seen
    # from above, starting with the northern-most vertex. Each vertex
    # is defined in the form needed by `bounds_func`.
    north = (1, 1)
    west = (0, 1)
    south = (0, 0)
    east = (1, 0)
    vertices = (north, west, south, east)

    # Initialise the output bounds array
    b = np.empty((a.size, 4), dtype="float64")

    match indexing_scheme:
        case "nuniq":
            # Create bounds for 'nuniq' indices
            orders, a = healpix.uniq2pix(a, nest=True)
            for order in np.unique(orders):
                nside = healpix.order2nside(order)
                indices = np.where(orders == order)[0]
                for j, (u, v) in enumerate(vertices):
                    thetaphi = bounds_func(nside, a[indices], u, v)
                    b[indices, j] = healpix.lonlat_from_thetaphi(*thetaphi)[
                        pos
                    ]

            del orders, indices

        case "nested" | "ring":
            # Create bounds for 'nested' or 'ring' indices
            nside = healpix.order2nside(refinement_level)
            for j, (u, v) in enumerate(vertices):
                thetaphi = bounds_func(nside, a, u, v)
                b[:, j] = healpix.lonlat_from_thetaphi(*thetaphi)[pos]

    del thetaphi, a

    if longitude:
        # Ensure that longitude bounds are less than 360
        where_ge_360 = np.where(b >= 360)
        if where_ge_360[0].size:
            b[where_ge_360] -= 360.0
            del where_ge_360

        # Vertices on the north or south pole come out with a
        # longitude of NaN, so replace these with sensible values:
        # Either the constant 'pole_longitude', or else the longitude
        # of the cell vertex that is opposite the vertex on the pole.
        north = 0
        south = 2
        for pole, replacement in ((north, south), (south, north)):
            indices = np.argwhere(np.isnan(b[:, pole])).flatten()
            if not indices.size:
                continue

            if pole_longitude is None:
                longitude = b[indices, replacement]
            else:
                longitude = pole_longitude

            b[indices, pole] = longitude

    return b


def cf_healpix_coordinates(
    a, indexing_scheme, refinement_level=None, latitude=False, longitude=False
):
    """Calculate HEALPix cell centre coordinates.

    See CF Appendix F: Grid Mappings.
    https://doi.org/10.5281/zenodo.14274886

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        indexing_scheme: `str`
            The HEALPix indexing scheme. One of ``'nested'``,
            ``'ring'``, ``'nuniq'``, or ``'zuniq'``.

        refinement_level: `int` or `None`, optional
            The refinement level of the grid within the HEALPix
            hierarchy, starting at 0 for the base tessellation with 12
            cells. Must be an `int` for *indexing_scheme* ``'nested'``
            or ``'ring'``, but is ignored for *indexing_scheme*
            ``'nuniq'`` or ``'zuniq'``, in which case
            *refinement_level* may be `None`.

        latitude: `bool`, optional
            If True then return the coordinate latitudes.

        longitude: `bool`, optional
            If True then return the coordinate longitudes.

    :Returns:

        `numpy.ndarray`
            A 1-d array containing the HEALPix cell coordinates.

    **Examples**

    >>> cf_healpix_coordinates(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, latitude=True
    )
    array([19.47122063, 41.8103149 , 41.8103149 , 66.44353569])
    >>> cf_healpix_coordinates(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, longitude=True
    )
    array([45. , 67.5, 22.5, 45. ])

    """
    if not a.size:
        return a

    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "for the calculation of latitude/longitude coordinates of a "
            "HEALPix grid"
        )

    a = cfdm_to_memory(a)

    scalar = not a.ndim
    if scalar:
        # Turn a 0-d array into a 1-d array, for convenience (we'll
        # turn the result back to 0-d at the end).
        a = np.atleast_1d(a)

    if a.ndim > 1:
        raise ValueError(
            "Can only calculate HEALPix cell coordinates when the "
            f"healpix_index array has one dimension. Got shape: {a.shape}"
        )

    if latitude and longitude:
        raise ValueError(
            "Only one of 'latitude' and 'longitude' must be True."
        )

    if latitude:
        pos = 1
    elif longitude:
        pos = 0
    else:
        raise ValueError("One of 'latitude' and 'longitude' must be True.")

    # Convert zuniq to nuniq
    if indexing_scheme == "zuniq":
        a = _zuniq2uniq(a)
        indexing_scheme = "nuniq"

    match indexing_scheme:
        case "nuniq":
            # Create coordinates for 'nuniq' indices
            c = np.empty(a.shape, dtype="float64")
            nest = True
            orders, a = healpix.uniq2pix(a, nest=nest)
            for order in np.unique(orders):
                nside = healpix.order2nside(order)
                indices = np.where(orders == order)[0]
                c[indices] = healpix.pix2ang(
                    nside=nside, ipix=a[indices], nest=nest, lonlat=True
                )[pos]

        case "nested" | "ring":
            # Create coordinates for 'nested' or 'ring' indices
            nest = indexing_scheme == "nested"
            nside = healpix.order2nside(refinement_level)
            c = healpix.pix2ang(
                nside=nside,
                ipix=a,
                nest=nest,
                lonlat=True,
            )[pos]

        case _:
            raise ValueError(
                "Can't calculate HEALPix cell coordinates: Unknown "
                "'indexing_scheme' in cf_healpix_coordinates: "
                f"{indexing_scheme!r}"
            )  # pragma: no cover

    if scalar:
        c = np.squeeze(c)

    return c


def cf_healpix_change_indexing_scheme(
    a,
    indexing_scheme,
    new_indexing_scheme,
    healpix_index_dtype,
    refinement_level=None,
    moc_refinement_level=None,
):
    """Change the indexing scheme of HEALPix indices.

    See CF Appendix F: Grid Mappings.
    https://doi.org/10.5281/zenodo.14274886

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        indexing_scheme: `str`
            The original HEALPix indexing scheme. One of ``'nested'``,
            ``'ring'``, ``'nuniq'``, or ``'zuniq'``.

        new_indexing_scheme: `str`
            The new HEALPix indexing scheme to change to. One of
            ``'nested'``, ``'ring'``, ``'nuniq'``, or ``'zuniq'``.

        healpix_index_dtype: `str` or `numpy.dtype`
            Typecode or data-type to which the new indices will be
            cast. This should be the smallest data type needed for
            storing the largest possible index value at the refinement
            level.

        refinement_level: `int` or `None`, optional
            The refinement level of the grid within the HEALPix
            hierarchy, starting at 0 for the base tessellation with 12
            cells. Must be an `int` for *indexing_scheme* ``'nested'``
            or ``'ring'``, but is ignored for *indexing_scheme*
            ``'nuniq'`` or ``'zuniq'`` (in which case
            *refinement_level* may be `None`).

        moc_refinement_level: `int` or `None`, optional
            When changing from the nuniq or zuniq MOC indexing scheme
            to the ring or nested indexing scheme,
            *moc_refinement_level* must be set to the unique integer
            refinement level known to be represented by the MOC
            indices. If the MOC indices include a refinement level
            that is not equal to *moc_refinement_level* then an
            exception is raised. Ignored when changing from the ring
            or nested indexing scheme.

    :Returns:

        `numpy.ndarray`
            An array containing the new HEALPix indices.

    **Examples**

    >>> cf_healpix_change_indexing_scheme(
    ...     [0, 1, 2, 3], 'nested', 'ring', 1
    ... )
    array([13,  5,  4,  0])
    >>> cf_healpix_change_indexing_scheme(
    ...     [0, 1, 2, 3], 'nested', 'nuniq', 1
    )
    array([16, 17, 18, 19])
    >>> cf_healpix_change_indexing_scheme(
    ...     [16, 17, 18, 19], 'nuniq', 'nested'
    )
    array([0, 1, 2, 3])

    """
    if new_indexing_scheme == indexing_scheme or not a.size:
        # Null operation
        return a

    from cf.functions import healpix_indexing_schemes

    if new_indexing_scheme not in healpix_indexing_schemes():
        raise ValueError(
            "Can't change HEALPix indexing scheme: Unknown "
            "'new_indexing_scheme' in cf_healpix_change_indexing_scheme: "
            f"{new_indexing_scheme!r}"
        )

    a = cfdm_to_memory(a)

    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "for changing the HEALPix indexing scheme"
        )

    match indexing_scheme:
        case "nested":
            match new_indexing_scheme:
                case "ring":
                    nside = healpix.order2nside(refinement_level)
                    a = healpix.nest2ring(nside, a)

                case "nuniq":
                    a = healpix.pix2uniq(refinement_level, a, nest=True)

                case "zuniq":
                    a = _pix2zuniq(refinement_level, a, nest=True)

        case "ring":
            match new_indexing_scheme:
                case "nested":
                    nside = healpix.order2nside(refinement_level)
                    a = healpix.ring2nest(nside, a)

                case "nuniq":
                    a = healpix.pix2uniq(refinement_level, a, nest=False)

                case "zuniq":
                    a = _pix2zuniq(refinement_level, a, nest=False)

        case "nuniq":
            match new_indexing_scheme:
                case "nested" | "ring":
                    if moc_refinement_level is None:
                        raise ValueError(
                            "Must set moc_refinement_level when changing from "
                            f"{indexing_scheme!r} to {new_indexing_scheme!r}"
                        )

                    nest = new_indexing_scheme == "nested"
                    order, a = healpix.uniq2pix(a, nest=nest)

                    order = np.unique(order)
                    if order.size > 1:
                        order = order.tolist()
                    elif order != moc_refinement_level:
                        order = sorted((order.item(0), moc_refinement_level))

                    if len(order) > 1:
                        raise ValueError(
                            "Can't change HEALPix indexing scheme from "
                            f"{indexing_scheme!r} to {new_indexing_scheme!r}: "
                            "HEALPix indices span multiple refinement levels "
                            f"(at least levels {order})"
                        )

                case "zuniq":
                    a = _uniq2zuniq(a)

        case "zuniq":
            match new_indexing_scheme:
                case "nested" | "ring":
                    if moc_refinement_level is None:
                        raise ValueError(
                            "Must set moc_refinement_level when changing from "
                            f"{indexing_scheme!r} to {new_indexing_scheme!r}"
                        )

                    # Convert to zuniq to nested
                    order, a = _zuniq2pix(a, nest=True)

                    order = np.unique(order)
                    if order.size > 1:
                        order = order.tolist()
                    elif order != moc_refinement_level:
                        order = sorted((order.item(0), moc_refinement_level))

                    if len(order) > 1:
                        raise ValueError(
                            "Can't change HEALPix indexing scheme from "
                            f"{indexing_scheme!r} to {new_indexing_scheme!r}: "
                            "HEALPix indices span multiple refinement levels "
                            f"(at least levels {order})"
                        )

                    if new_indexing_scheme == "ring":
                        # Convert to nested to ring
                        nside = healpix.order2nside(order.item())
                        a = healpix.nest2ring(nside, a)

                case "nuniq":
                    a = _zuniq2uniq(a)

        case _:
            raise ValueError(
                "Can't change HEALPix indexing scheme: Unknown "
                "'indexing_scheme' in cf_healpix_change_indexing_scheme: "
                f"{indexing_scheme!r}"
            )

    # Cast the new indices to the given data type
    a = a.astype(healpix_index_dtype, copy=False)
    return a


def cf_healpix_weights(a, indexing_scheme, measure=False, radius=None):
    """Calculate HEALPix cell area weights.

    See CF Appendix F: Grid Mappings.
    https://doi.org/10.5281/zenodo.14274886

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        indexing_scheme: `str`
            The HEALPix indexing scheme. Must be ``'nuniq'`` or
            ``'zuniq'``.

        measure: `bool`, optional
            If True then create weights that are actual cell areas, in
            units of the square of the radius units.

        radius: number, optional
            The radius of the sphere. Must be set if *measure * is
            True, otherwise ignored.

    :Returns:

        `numpy.ndarray`
            An array containing the HEALPix cell weights.

    **Examples**

    >>> cf_healpix_weights(
    ...     [76, 77, 78, 79, 20, 21], 'nuniq'
    )
    array([0.0625, 0.0625, 0.0625, 0.0625, 0.25  , 0.25  ])
    >>> cf_healpix_weights(
    ...     [76, 77, 78, 79, 20, 21], 'nuniq',
    ...     measure=True, radius=6371000
    )
    array([2.65658579e+12, 2.65658579e+12, 2.65658579e+12, 2.65658579e+12,
           1.06263432e+13, 1.06263432e+13])

    """
    if not a.size:
        return a

    if indexing_scheme not in ("nuniq", "zuniq"):
        raise ValueError(
            "cf_healpix_weights: Can only calculate HEALPix weights for the "
            "'nuniq' or 'zuniq' indexing scheme"
        )

    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "for the calculation of cell area weights of a HEALPix grid"
        )

    a = cfdm_to_memory(a)

    # Convert zuniq to nuniq
    if indexing_scheme == "zuniq":
        a = _zuniq2uniq(a)
        indexing_scheme = "nuniq"

    if a.ndim != 1:
        raise ValueError(
            "Can only calculate HEALPix cell area weights when the "
            f"healpix_index array has one dimension. Got shape: {a.shape}"
        )

    # Each cell at refinement level N has weight x/(4**N), where ...
    if measure:
        # Cell weights equal cell areas. Surface area of a sphere is
        # 4*pi*(r**2), number of HEALPix cells at refinement level N
        # is 12*(4**N) => Area of each cell is (pi*(r**2)/3)/(4**N)
        x = np.pi * (radius**2) / 3.0
    else:
        # Normalised weights
        x = 1.0

    orders = healpix.uniq2pix(a, nest=True)[0]
    orders, index, inverse = np.unique(
        orders, return_index=True, return_inverse=True
    )

    # Initialise the output weights array
    w = np.empty(a.shape, dtype="float64")

    # For each refinement level N, put the weights (= x/4**N) into 'w'
    # at the correct locations.
    for N, i in zip(orders, index):
        w = np.where(inverse == inverse[i], x / (4**N), w)

    return w
