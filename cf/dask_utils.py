"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

import healpix
import numpy as np


def cf_HEALPix_bounds(
    a, healpix_order, refinement_level=None, lat=False, lon=False
):
    """Calculate HEALPix cell bounds.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        healpix_order: `str`
            One of ``'nested'``, ``'ring'``, or ``'nuniq'``.

        refinement_level: `int` or `None`, optional
            For a ``'nested'`` or ``'ring'`` ordered grid, the
            refinement level of the grid within the HEALPix hierarchy,
            starting at 0 for the base tesselation with 12 cells.
            Ignored for a ``'nuniq'`` ordered grid.

        lat: `bool`, optional
            If True then return latitude bounds.

        lon: `bool`, optional
            If True then return longitude bounds.

    :Returns:

        `numpy.ndarray`
            An array containing the HEALPix cell bounds.

    """
    # Keep an eye on https://github.com/ntessore/healpix/issues/66
    if a.ndim != 1:
        raise ValueError(
            "Can't calculate HEALPix cell bounds when "
            f"healpix_index array has shape {a.shape}"
        )

    if lat:
        pos = 1
    elif lon:
        pos = 0

    if healpix_order == "nested":
        bounds_func = healpix._chp.nest2ang_uv
    else:
        bounds_func = healpix._chp.ring2ang_uv

    # Define the cell vertices in an anticlockwise direction, as seen
    # from above.
    right = (1, 0)
    top = (1, 1)
    left = (0, 1)
    bottom = (0, 0)
    vertices = (right, top, left, bottom)

    b = np.empty((a.size, 4), dtype="float64")

    if healpix_order == "nuniq":
        # nuniq
        nsides, a = healpix.uniq2pix(a, nest=False)
        nsides, index, inverse = np.unique(
            nsides, return_index=True, return_inverse=True
        )
        for nside, i in zip(nsides, index):
            level = np.where(inverse == inverse[i])[0]
            for j, (u, v) in enumerate(vertices):
                thetaphi = bounds_func(nside, a[level], u, v)
                b[level, j] = healpix.lonlat_from_thetaphi(*thetaphi)[pos]
    else:
        # nested or ring
        nside = healpix.order2nside(refinement_level)
        for j, (u, v) in enumerate(vertices):
            thetaphi = bounds_func(nside, a, u, v)
            b[..., j] = healpix.lonlat_from_thetaphi(*thetaphi)[pos]

    if not pos:
        # Longitude bounds
        b[np.where(b >= 360)] -= 360.0

        # Bounds on the north or south pole come out with a longitude
        # of NaN, so replace these with a sensible value.
        i = np.argwhere(np.isnan(b[:, 1])).flatten()
        if i.size:
            b[i, 1] = b[i, 3]

        i = np.argwhere(np.isnan(b[:, 3])).flatten()
        if i.size:
            b[i, 3] = b[i, 1]

    return b


def cf_HEALPix_change_order(
    a, healpix_order, new_healpix_order, refinement_level
):
    """Change the ordering of HEALPix indices.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        healpix_order: `str`
            The original HEALPix order. One of ``'nested'`` or
            ``'ring'``.

        new_healpix_order: `str`
            The new HEALPix order to change to. One of ``'nested'``,
            ``'ring'``, or ``'nuniq'``.

        refinement_level: `int`
            The refinement level of the original grid within the
            HEALPix hierarchy, starting at 0 for the base tesselation
            with 12 cells.

    :Returns:

        `numpy.ndarray`
            An array containing the new HEALPix indices.

    """
    nside = healpix.order2nside(refinement_level)

    if healpix_order == "nested":
        if new_healpix_order == "ring":
            return healpix.nest2ring(nside, a)

        if new_healpix_order == "nuniq":
            return healpix._chp.nest2uniq(nside, a)

    elif healpix_order == "ring":
        if new_healpix_order == "nested":
            return healpix.ring2nest(nside, a)

        if new_healpix_order == "nuniq":
            return healpix._chp.ring2uniq(nside, a)

    else:
        raise ValueError(
            "Can't change HEALPix order: Can only change from HEALPix "
            f"order 'nested' or 'ring'. Got {healpix_order!r}"
        )


def cf_HEALPix_coordinates(
    a, healpix_order, refinement_level=None, lat=False, lon=False
):
    """Calculate HEALPix cell coordinates.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        healpix_order: `str`
            One of ``'nested'``, ``'ring'``, or ``'nuniq'``.

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
            An array containing the HEALPix cell coordinates.

    """
    if a.ndim != 1:
        raise ValueError(
            "Can't calculate HEALPix cell coordinates when "
            f"healpix_index array has shape {a.shape}"
        )

    if lat:
        pos = 1
    elif lon:
        pos = 0

    if healpix_order == "nuniq":
        c = np.empty(a.shape, dtype="float64")

        nest = False
        nsides, a = healpix.uniq2pix(a, nest=nest)
        nsides, index, inverse = np.unique(
            nsides, return_index=True, return_inverse=True
        )
        for nside, i in zip(nsides, index):
            level = np.where(inverse == inverse[i])[0]
            c[level] = healpix.pix2ang(
                nside=nside, ipix=a[level], nest=nest, lonlat=True
            )[pos]
    else:
        # nested or ring
        c = healpix.pix2ang(
            nside=healpix.order2nside(refinement_level),
            ipix=a,
            nest=healpix_order == "nested",
            lonlat=True,
        )[pos]

    return c


def cf_HEALPix_nuniq_weights(a, measure=False, radius=None):
    """Calculate HEALPix cell weights for 'nuniq' indices.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`
            The array of HEALPix indices.

        measure: `bool`, optional
            If True then create weights that are actual cell areas, in
            units of the square of the radius units.

        radius: number, optional
            The radius of the sphere, in units of length. Must be set
            if *measure * is True, otherwise ignored.

    :Returns:

        `numpy.ndarray`
            An array containing the HEALPix cell weights.

    """
    if measure:
        x = np.pi * (radius**2) / 3.0
    else:
        x = 1.0

    nsides = healpix.uniq2pix(a)[0]
    nsides, index, inverse = np.unique(
        nsides, return_index=True, return_inverse=True
    )

    w = np.empty(a.shape, dtype="float64")

    for nside, i in zip(nsides, index):
        w = np.where(inverse == inverse[i], x / (nside**2), w)

    return w
