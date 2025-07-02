"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

import numpy as np


def cf_HEALPix_bounds(
    a, index_scheme, refinement_level=None, lat=False, lon=False
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

        index_scheme: `str`
            The HEALPix indexing scheme. One of ``'nested'``,
            ``'ring'``, or ``'nuniq'``.

        refinement_level: `int` or `None`, optional
            For a ``'nested'`` or ``'ring'`` ordered grid, the
            refinement level of the grid within the HEALPix hierarchy,
            starting at 0 for the base tesselation with 12 cells. Set
            to `None` for a ``'nuniq'`` ordered grid, for which the
            refinement level is ignored.

        lat: `bool`, optional
            If True then return latitude bounds.

        lon: `bool`, optional
            If True then return longitude bounds.

    :Returns:

        `numpy.ndarray`
            A 2-d  array containing the HEALPix cell bounds.

    **Examples**

    >>> cf_HEALPix_bounds([0, 1, 2, 3], 'nested', 1, lat=True)
    array([[19.47122063, 41.8103149 , 19.47122063,  0.        ],
           [41.8103149 , 66.44353569, 41.8103149 , 19.47122063],
           [41.8103149 , 66.44353569, 41.8103149 , 19.47122063],
           [66.44353569, 90.        , 66.44353569, 41.8103149 ]])
    >>> cf_HEALPix_bounds([0, 1, 2, 3], 'nested', 1, lon=True)
    array([[67.5, 45. , 22.5, 45. ],
           [90. , 90. , 45. , 67.5],
           [45. ,  0. ,  0. , 22.5],
           [90. , 45. ,  0. , 45. ]])

    """
    import healpix

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

    if index_scheme == "nested":
        bounds_func = healpix._chp.nest2ang_uv
    else:
        bounds_func = healpix._chp.ring2ang_uv

    # Define the cell vertices, in an anticlockwise direction, as seen
    # from above, starting with the eastern-most vertex.
    right = (1, 0)
    top = (1, 1)
    left = (0, 1)
    bottom = (0, 0)
    vertices = (right, top, left, bottom)

    # Initialise the output bounds array
    b = np.empty((a.size, 4), dtype="float64")

    if index_scheme == "nuniq":
        # Create bounds for 'nuniq' cells
        orders, a = healpix.uniq2pix(a, nest=False)
        orders, index, inverse = np.unique(
            orders, return_index=True, return_inverse=True
        )
        for order, i in zip(orders, index):
            level = np.where(inverse == inverse[i])[0]
            nside = healpix.order2nside(order)
            for j, (u, v) in enumerate(vertices):
                thetaphi = bounds_func(nside, a[level], u, v)
                b[level, j] = healpix.lonlat_from_thetaphi(*thetaphi)[pos]
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

        # Bounds on the north (south) pole come out with a longitude
        # of NaN, so replace these with a sensible value, i.e. the
        # longitude of the southern (northern) vertex.
        #
        # North pole
        i = np.argwhere(np.isnan(b[:, 1])).flatten()
        if i.size:
            b[i, 1] = b[i, 3]

        # South pole
        i = np.argwhere(np.isnan(b[:, 3])).flatten()
        if i.size:
            b[i, 3] = b[i, 1]

    return b


def cf_HEALPix_change_order(
    a, index_scheme, new_index_scheme, refinement_level
):
    """Change the ordering of HEALPix indices.

    Does not change the position of each cell in the array, but
    redefines their indices according to the new ordering scheme.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        index_scheme: `str`
            The original HEALPix indexing scheme. One of ``'nested'``
            or ``'ring'``.

        new_index_scheme: `str`
            The new HEALPix indexing scheme to change to. One of
            ``'nested'``, ``'ring'``, or ``'nuniq'``.

        refinement_level: `int`
            The refinement level of the original grid within the
            HEALPix hierarchy, starting at 0 for the base tesselation
            with 12 cells.

    :Returns:

        `numpy.ndarray`
            An array containing the new HEALPix indices.

    **Examples**

    >>> cf_HEALPix_change_order([0, 1, 2, 3], 'nested', 'ring', 1)
    array([13,  5,  4,  0])
    >>> cf_HEALPix_change_order([0, 1, 2, 3], 'nuniq', 'ring', 1)
    array([16, 17, 18, 19])

    """
    import healpix

    if index_scheme == "nested":
        if new_index_scheme == "ring":
            return healpix.nest2ring(healpix.order2nside(refinement_level), a)

        if new_index_scheme == "nuniq":
            return healpix._chp.nest2uniq(refinement_level, a)

    elif index_scheme == "ring":
        if new_index_scheme == "nested":
            return healpix.ring2nest(healpix.order2nside(refinement_level), a)

        if new_index_scheme == "nuniq":
            return healpix._chp.ring2uniq(refinement_level, a)

    else:
        raise ValueError(
            "Can't change HEALPix order: Can only change from HEALPix "
            f"order 'nested' or 'ring'. Got {index_scheme!r}"
        )


def cf_HEALPix_coordinates(
    a, index_scheme, refinement_level=None, lat=False, lon=False
):
    """Calculate HEALPix cell coordinates.

    THe coordinates are for the cell centres.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        index_scheme: `str`
            The HEALPix indexing scheme. One of ``'nested'``,
            ``'ring'``, or ``'nuniq'``.

        refinement_level: `int` or `None`, optional
            For a ``'nested'`` or ``'ring'`` ordered grid, the
            refinement level of the grid within the HEALPix hierarchy,
            starting at 0 for the base tesselation with 12 cells.
            Ignored for a ``'nuniq'`` ordered grid.

        lat: `bool`, optional
            If True then return latitude coordinates.

        lon: `bool`, optional
            If True then return longitude coordinates.

    :Returns:

        `numpy.ndarray`
            A 1-d array containing the HEALPix cell coordinates.

    **Examples**

    >>> cf_HEALPix_coordinates([0, 1, 2, 3], 'nested', 1, lat=True)
    array([19.47122063, 41.8103149 , 41.8103149 , 66.44353569])
    >>> cf_HEALPix_coordinates([0, 1, 2, 3], 'nested', 1, lon=True)
    array([45. , 67.5, 22.5, 45. ])

    """
    import healpix

    if a.ndim != 1:
        raise ValueError(
            "Can't calculate HEALPix cell coordinates when the "
            f"healpix_index array has one dimension. Got shape {a.shape}"
        )

    if lat:
        pos = 1
    elif lon:
        pos = 0

    if index_scheme == "nuniq":
        # Create coordinates for 'nuniq' cells
        c = np.empty(a.shape, dtype="float64")

        nest = False
        orders, a = healpix.uniq2pix(a, nest=nest)
        orders, index, inverse = np.unique(
            orders, return_index=True, return_inverse=True
        )
        for order, i in zip(orders, index):
            level = np.where(inverse == inverse[i])[0]
            nside = healpix.order2nside(order)
            c[level] = healpix.pix2ang(
                nside=nside, ipix=a[level], nest=nest, lonlat=True
            )[pos]
    else:
        # Create coordinates for 'nested' or 'ring' cells
        nest = (index_scheme == "nested",)
        nside = healpix.order2nside(refinement_level)
        c = healpix.pix2ang(
            nside=nside,
            ipix=a,
            nest=nest,
            lonlat=True,
        )[pos]

    return c


def cf_HEALPix_nuniq_area_weights(a, measure=False, radius=None):
    """Calculate HEALPix cell area weights for 'nuniq' indices.

    For mathematical details, see section 4 of:

        K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann,
        et al.. HEALPix: A Framework for High‐Resolution
        Discretization and Fast Analysis of Data Distributed on the
        Sphere. The Astrophysical Journal, 2005, 622 (2), pp.759-771.
        https://dx.doi.org/10.1086/427976

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix 'nuniq' indices.

        measure: `bool`, optional
            If True then create weights that are actual cell areas, in
            units of the square of the radius units.

        radius: number, optional
            The radius of the sphere, in units of length. Must be set
            if *measure * is True, otherwise ignored.

    :Returns:

        `numpy.ndarray`
            An array containing the HEALPix cell weights.

    **Examples**

    >>> cf_HEALPix_nuniq_weights([76, 77, 78, 79, 20, 21])
    array([0.0625, 0.0625, 0.0625, 0.0625, 0.25  , 0.25  ])
    >>> cf_HEALPix_nuniq_weights([76, 77, 78, 79, 20, 21],
    ...                          measure=True, radius=6371000)
    array([2.65658579e+12, 2.65658579e+12, 2.65658579e+12, 2.65658579e+12,
           1.06263432e+13, 1.06263432e+13])

    """
    import healpix

    if measure:
        x = np.pi * (radius**2) / 3.0
    else:
        x = 1.0

    orders = healpix.uniq2pix(a)[0]
    orders, index, inverse = np.unique(
        orders, return_index=True, return_inverse=True
    )

    # Initialise the output weights array
    w = np.empty(a.shape, dtype="float64")

    for order, i in zip(orders, index):
        nside = healpix.order2nside(order)
        w = np.where(inverse == inverse[i], x / (nside**2), w)

    return w
