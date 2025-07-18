"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

import numpy as np
from cfdm.data.dask_utils import cfdm_to_memory


def cf_healpix_bounds(
    a,
    indexing_scheme,
    refinement_level=None,
    lat=False,
    lon=False,
    pole_longitude=None,
):
    """Calculate HEALPix cell bounds.

    Latitude or longitude locations of the cell vertices are derived
    from HEALPix indices.x Each cell has four bounds, which are
    returned in an anticlockwise direction, as seen from above,
    starting with the eastern-most vertex.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        indexing_scheme: `str`
            The HEALPix indexing scheme. One of ``'nested'``,
            ``'ring'``, or ``'nested_unique'``.

        refinement_level: `int` or `None`, optional
            The refinement level of the grid within the HEALPix
            hierarchy, starting at 0 for the base tessellation with 12
            cells. Must be an `int` for *indexing_scheme* ``'nested'``
            or ``'ring'``, but is ignored for ``'nested_unique'`` (in
            which case *refinement_level* may be `None`).

        lat: `bool`, optional
            If True then return latitude bounds.

        lon: `bool`, optional
            If True then return longitude bounds.

        pole_longitude: `None` or number
            The longitude of coordinate bounds that lie exactly on the
            north or south pole. If `None` (the default) then the
            longitudes of such a point will be identical to its
            opposite vertex. If set to a number, then the longitudes
            of such points will all be that value.

    :Returns:

        `numpy.ndarray`
            A 2-d  array containing the HEALPix cell bounds.

    **Examples**

    >>> cf_healpix_bounds([0, 1, 2, 3], 'nested', 1, lat=True)
    array([[19.47122063, 41.8103149 , 19.47122063,  0.        ],
           [41.8103149 , 66.44353569, 41.8103149 , 19.47122063],
           [41.8103149 , 66.44353569, 41.8103149 , 19.47122063],
           [66.44353569, 90.        , 66.44353569, 41.8103149 ]])
    >>> cf_healpix_bounds([0, 1, 2, 3], 'nested', 1, lon=True)
    array([[45. , 22.5, 45. , 67.5],
           [90. , 45. , 67.5, 90. ],
           [ 0. ,  0. , 22.5, 45. ],
           [45. ,  0. , 45. , 90. ]])
    >>> cf_healpix_bounds([0, 1, 2, 3], 'nested', 1, lon=True,
    ...                   pole_longitude=3.14159)
    array([[45.     , 22.5    , 45.     , 67.5    ],
           [90.     , 45.     , 67.5    , 90.     ],
           [ 0.     ,  0.     , 22.5    , 45.     ],
           [ 3.14159,  0.     , 45.     , 90.     ]])

    """
    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "to allow the calculation of latitude/longitude coordinate "
            "bounds for a HEALPix grid"
        )

    a = cfdm_to_memory(a)

    # Keep an eye on https://github.com/ntessore/healpix/issues/66
    if a.ndim != 1:
        raise ValueError(
            "Can only calculate HEALPix cell bounds when the "
            f"healpix_index array has one dimension. Got shape {a.shape}"
        )

    if lat:
        pos = 1
    elif lon:
        pos = 0

    if indexing_scheme == "ring":
        bounds_func = healpix._chp.ring2ang_uv
    else:
        bounds_func = healpix._chp.nest2ang_uv

    # Define the cell vertices, in an anticlockwise direction, as seen
    # from above, starting with the northern-most vertex.
    east = (1, 0)
    north = (1, 1)
    west = (0, 1)
    south = (0, 0)
    vertices = (north, west, south, east)

    # Initialise the output bounds array
    b = np.empty((a.size, 4), dtype="float64")

    if indexing_scheme == "nested_unique":
        # Create bounds for 'nested_unique' cells
        orders, a = healpix.uniq2pix(a, nest=True)
        for order in np.unique(orders):
            nside = healpix.order2nside(order)
            indices = np.where(orders == order)[0]
            for j, (u, v) in enumerate(vertices):
                thetaphi = bounds_func(nside, a[indices], u, v)
                b[indices, j] = healpix.lonlat_from_thetaphi(*thetaphi)[pos]
    else:
        # Create bounds for 'nested' or 'ring' cells
        nside = healpix.order2nside(refinement_level)
        for j, (u, v) in enumerate(vertices):
            thetaphi = bounds_func(nside, a, u, v)
            b[..., j] = healpix.lonlat_from_thetaphi(*thetaphi)[pos]

    if not pos:
        # Ensure that longitude bounds are less than 360
        where_ge_360 = np.where(b >= 360)
        if where_ge_360[0].size:
            b[where_ge_360] -= 360.0

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
    a, indexing_scheme, refinement_level=None, lat=False, lon=False
):
    """Calculate HEALPix cell coordinates.

    THe coordinates are for the cell centres.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        indexing_scheme: `str`
            The HEALPix indexing scheme. One of ``'nested'``,
            ``'ring'``, or ``'nested_unique'``.

        refinement_level: `int` or `None`, optional
            The refinement level of the grid within the HEALPix
            hierarchy, starting at 0 for the base tessellation with 12
            cells. Must be an `int` for *indexing_scheme* ``'nested'``
            or ``'ring'``, but is ignored for ``'nested_unique'`` (in
            which case *refinement_level* may be `None`).

        lat: `bool`, optional
            If True then return latitude coordinates.

        lon: `bool`, optional
            If True then return longitude coordinates.

    :Returns:

        `numpy.ndarray`
            A 1-d array containing the HEALPix cell coordinates.

    **Examples**

    >>> cf_healpix_coordinates([0, 1, 2, 3], 'nested', 1, lat=True)
    array([19.47122063, 41.8103149 , 41.8103149 , 66.44353569])
    >>> cf_healpix_coordinates([0, 1, 2, 3], 'nested', 1, lon=True)
    array([45. , 67.5, 22.5, 45. ])

    """
    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "to allow the calculation of latitude/longitude coordinates "
            "for a HEALPix grid"
        )

    a = cfdm_to_memory(a)

    if a.ndim != 1:
        raise ValueError(
            "Can only calculate HEALPix cell coordinates when the "
            f"healpix_index array has one dimension. Got shape {a.shape}"
        )

    if lat:
        pos = 1
    elif lon:
        pos = 0

    if indexing_scheme == "nested_unique":
        # Create coordinates for 'nested_unique' cells
        c = np.empty(a.shape, dtype="float64")

        orders, a = healpix.uniq2pix(a, nest=True)
        for order in np.unique(orders):
            nside = healpix.order2nside(order)
            indices = np.where(orders == order)[0]
            c[indices] = healpix.pix2ang(
                nside=nside, ipix=a[indices], nest=True, lonlat=True
            )[pos]
    else:
        # Create coordinates for 'nested' or 'ring' cells
        nest = indexing_scheme == "nested"
        nside = healpix.order2nside(refinement_level)
        c = healpix.pix2ang(
            nside=nside,
            ipix=a,
            nest=nest,
            lonlat=True,
        )[pos]

    return c


def cf_healpix_indexing_scheme(
    a, indexing_scheme, new_indexing_scheme, refinement_level=None
):
    """Change the ordering of HEALPix indices.

    Does not change the position of each cell in the array, but
    redefines their indices according to the new ordering scheme.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        indexing_scheme: `str`
            The original HEALPix indexing scheme. One of ``'nested'``,
            ``'ring'``, or ``'nested_unique'``.

        new_indexing_scheme: `str`
            The new HEALPix indexing scheme to change to. One of
            ``'nested'``, ``'ring'``, or ``'nested_unique'``.

        refinement_level: `int` or `None`, optional
            The refinement level of the grid within the HEALPix
            hierarchy, starting at 0 for the base tessellation with 12
            cells. Must be an `int` for *indexing_scheme* ``'nested'``
            or ``'ring'``, but is ignored for ``'nested_unique'`` (in
            which case *refinement_level* may be `None`).

    :Returns:

        `numpy.ndarray`
            An array containing the new HEALPix indices.

    **Examples**

    >>> cf_healpix_indexing_scheme([0, 1, 2, 3], 'nested', 'ring', 1)
    array([13,  5,  4,  0])
    >>> cf_healpix_indexing_scheme([0, 1, 2, 3], 'nested', 'nested_unique', 1)
    array([16, 17, 18, 19])
    >>> cf_healpix_indexing_scheme([16, 17, 18, 19], 'nested_unique', 'nest', None)
    array([0, 1, 2, 3])

    """
    if new_indexing_scheme == indexing_scheme:
        # Null operation
        return a

    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "to allow the changing of the HEALPix index scheme"
        )

    a = cfdm_to_memory(a)

    if indexing_scheme == "nested":
        if new_indexing_scheme == "ring":
            nside = healpix.order2nside(refinement_level)
            return healpix.nest2ring(nside, a)

        if new_indexing_scheme == "nested_unique":
            return healpix.pix2uniq(refinement_level, a, nest=True)

    elif indexing_scheme == "ring":
        if new_indexing_scheme == "nested":
            nside = healpix.order2nside(refinement_level)
            return healpix.ring2nest(nside, a)

        if new_indexing_scheme == "nested_unique":
            return healpix.pix2uniq(refinement_level, a, nest=False)

    elif indexing_scheme == "nested_unique":
        if new_indexing_scheme in ("nested", "ring"):
            nest = new_indexing_scheme == "nested"
            order, a = healpix.uniq2pix(a, nest=nest)

            refinement_levels = np.unique(order)
            if refinement_levels.size > 1:
                raise ValueError(
                    "Can't change HEALPix indexing scheme from "
                    f"'nested_unique' to {new_indexing_scheme!r} when the "
                    "HEALPix indices span multiple refinement levels (at "
                    f"least levels {refinement_levels.tolist()})"
                )

            return a

    raise ValueError("Failed to change the HEALPix indexing scheme")


def cf_healpix_weights(a, indexing_scheme, measure=False, radius=None):
    """Calculate HEALPix cell area weights.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix 'nested_unique' indices.

        indexing_scheme: `str`
            The HEALPix indexing scheme. Must be ``'nested_unique'``.

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

    >>> cf_healpix_weights([76, 77, 78, 79, 20, 21], 'nested_unique')
    array([0.0625, 0.0625, 0.0625, 0.0625, 0.25  , 0.25  ])
    >>> cf_healpix_weights([76, 77, 78, 79, 20, 21], 'nested_unique',
    ...                    measure=True, radius=6371000)
    array([2.65658579e+12, 2.65658579e+12, 2.65658579e+12, 2.65658579e+12,
           1.06263432e+13, 1.06263432e+13])

    """
    try:
        import healpix
    except ImportError as e:
        raise ImportError(
            f"{e}. Must install healpix (https://pypi.org/project/healpix) "
            "to allow the calculation of cell area weights for a HEALPix grid"
        )

    if indexing_scheme != "nested_unique":
        raise ValueError(
            "cf_healpix_weights: Can only calulate weights for the "
            "'nested_unique' indexing scheme"
        )

    a = cfdm_to_memory(a)

    if measure:
        # Surface area of sphere is 4*pi*(r**2)
        # Number of HEALPix cells at refinement level N is 12*(4**N)
        # => Area of one cell is pi*(r**2)/(3*(4**N))
        x = np.pi * (radius**2) / 3.0
    else:
        x = 1.0

    orders = healpix.uniq2pix(a, nest=True)[0]
    orders, index, inverse = np.unique(
        orders, return_index=True, return_inverse=True
    )

    # Initialise the output weights array
    w = np.empty(a.shape, dtype="float64")

    for order, i in zip(orders, index):
        w = np.where(inverse == inverse[i], x / (4**order), w)

    return w
