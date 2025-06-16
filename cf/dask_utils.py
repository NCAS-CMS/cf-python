# Define functions to create latitudes and longitudes from
# HEALPix indices
import healpix

def cf_HEALPix_coordinates(
        a,
        healpix_order, refinement_level=None, lat=False, lon=False
):
    """Calculate HEALPix cell coordinates.
    
    .. versionadded:: NEXTVERSION

    :Parameters:

        a: `numpy.ndarray`

        healpix_order: `str`
            One of ``'nested'``, ``'ring'``, or ``'nuniq'``.

        refinement_level: `int` or `None`, optional
            For a ``'nested'`` or ``'ring'`` ordered grid, the
            refinement level of the grid within the HEALPix hierarchy,
            starting at 0 for the base tesselation with 12 cells
            globally. Ignored for a ``'nuniq'`` ordered grid.

        lat: `bool`, optional

        lon: `bool`, optional
    
    :Returns:

        `numpy.ndarray`

    """        
    if lat:
        pos = 1
    elif lon:               
        pos = 0

    if healpix_order == "nuniq":
        # nuniq
        c = np.empty(a.shape, dtype='float64')
        
        nsides, a =  healpix.uniq2pix(a, nest=False)
        nsides, index, inverse = np.unique(nsides,
                                           return_index=True,
                                           return_inverse=True)
        for nside, i in zip(nsides, index):
            level = np.where(inverse == inverse[i])[0]
            c[level] = healpix.pix2ang(
                nside=nside, ipix=a[level], nest=False, lonlat=True
            )[pos]
    else:
        # nested or ring
        c = healpix.pix2ang(
            nside=healpix.order2nside(refinement_level),
            ipix=a,
            nest=healpix_order=="nested",
            lonlat=True
        )[pos]

    return c
                  
             
def cf_HEALPix_bounds(a, healpix_order, refinement_level=None, lat=False, lon=False):
    """Calculate HEALPix cell bounds.

    .. versionadded:: NEXTVERSION
    
    :Parameters:

        a: `numpy.ndarray`

        healpix_order: `str`
            One of ``'nested'``, ``'ring'``, or ``'nuniq'``.

        refinement_level: `int` or `None`, optional
            For a ``'nested'`` or ``'ring'`` ordered grid, the
            refinement level of the grid within the HEALPix hierarchy,
            starting at 0 for the base tesselation with 12 cells
            globally. Ignored for a ``'nuniq'`` ordered grid.

        lat: `bool`, optional

        lon: `bool`, optional
    
    :Returns:

        `numpy.ndarray`

    """
    # Keep an eye on https://github.com/ntessore/healpix/issues/66

    if lat:
        pos = 1
    elif lon:               
        pos = 0

    if healpix_order == "nested":
        bounds_func = healpix._chp.nest2ang_uv
    else:
        bounds_func = healpix._chp.ring2ang_uv        

 
    # Define the cell vertices in an anticlockwise direction, seen
    # from above.
    #
    # Vertex position  vertex (u, v)
    # ---------------  -------------
    # right            (1, 0)
    # top              (1, 1)
    # left             (0, 1)
    # bottom           (0, 0)
    vertices = ((1, 0), (1,1), (0, 1), (0,0))

    b = np.empty((a.size, 4), dtype='float64')
    
    if healpix_order == "nuniq":
        # nuniq
        nsides, a = healpix.uniq2pix(a, nest=False)
        nsides, index, inverse = np.unique(nsides,
                                           return_index=True,
                                           return_inverse=True)
        for nside, i in zip(nsides, index):
            level = np.where(inverse == inverse[i])[0]
            for j, (u, v) in enumerate(vertices):
                b[level, j] = bounds_func(nside, a[level], u, v)[pos]
    else:
        # nested or ring
        nside = healpix.order2nside(refinement_level)
        for j, (u, v) in enumerate(vertices):
            b[:,j] = bounds_func(nside, a, u, v)[pos]

    # Convert to degrees
    np.rad2deg(b, out=b)

    return b


def cf_HEALPix_nuniq_weights(a, measure=False, radius=None):
    """Calculate HEALPix cell weights for 'nuniq' indices.

    .. versionadded:: NEXTVERSION
    
    :Parameters:

        a: `numpy.ndarray`

        measure: `bool`, optional
            If True then create weights that are actual cell sizes, in
            units of the square of the radius units.

        radius: number, optional
            The radius of the sphere, in units of length. Must be set
            if *measure * is True, otherwise ignored.
    
    :Returns:

        `numpy.ndarray`
            The weights.

    """
    if measure:
        f = 4.0 * np.pi * (radius**2) / 12
    else:
        f = 1.0
        
    nsides = healpix.uniq2pix(a)[0]
    nsides, index, inverse = np.unique(nsides,
                                       return_index=True,
                                       return_inverse=True)
    
    w = np.empty(a.shape, dtype='float64')

    for nside, i in zip(nsides, index):
        w = np.where(inverse == inverse[i], f / (nside**2), w)
        
    return w
