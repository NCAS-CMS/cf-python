"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

from functools import partial

import numpy as np
from cfdm.data.dask_utils import cfdm_to_memory
from scipy.ndimage import convolve1d

from ..cfdatetime import dt, dt2rt, rt2dt
from ..units import Units


def cf_contains(a, value):
    """Whether or not an array contains a value.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.__contains__`

    :Parameters:

        a: array_like
            The array.

        value: array_like
            The value.

    :Returns:

        `numpy.ndarray`
            A size 1 Boolean array, with the same number of dimensions
            as *a*, that indicates whether or not *a* contains the
            value.

    """
    a = cfdm_to_memory(a)
    value = cfdm_to_memory(value)
    return np.array(value in a).reshape((1,) * a.ndim)


def cf_convolve1d(a, window=None, axis=-1, origin=0):
    """Calculate a 1-d convolution along the given axis.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.convolution_filter`

    :Parameters:

        a: `numpy.ndarray`
            The float array to be filtered.

        window: 1-d sequence of numbers
            The window of weights to use for the filter.

        axis: `int`, optional
            The axis of input along which to calculate. Default is -1.

        origin: `int`, optional
            Controls the placement of the filter on the input array’s
            pixels. A value of 0 (the default) centers the filter over
            the pixel, with positive values shifting the filter to the
            left, and negative ones to the right.

    :Returns:

        `numpy.ndarray`
            Convolved float array with same shape as input.

    """
    a = cfdm_to_memory(a)

    # Cast to float to ensure that NaNs can be stored
    if a.dtype != float:
        a = a.astype(float, copy=False)

    masked = np.ma.is_masked(a)
    if masked:
        # convolve1d does not deal with masked arrays, so uses NaNs
        # instead.
        a = a.filled(np.nan)

    c = convolve1d(
        a, window, axis=axis, mode="constant", cval=0.0, origin=origin
    )

    if masked or np.isnan(c).any():
        with np.errstate(invalid="ignore"):
            c = np.ma.masked_invalid(c)

    return c


def cf_percentile(a, q, axis, method, keepdims=False, mtol=1):
    """Compute percentiles of the data along the specified axes.

    See `cf.Data.percentile` for further details.

    .. note:: This function correctly sets the mask hardness of the
              output array.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.percentile`

    :Parameters:

        a: array_like
            Input array.

        q: `numpy.ndarray`
            Percentile or sequence of percentiles to compute, which
            must be between 0 and 100 inclusive.

        axis: `tuple` of `int`
            Axes along which the percentiles are computed.

        method: `str`
            Specifies the interpolation method to use when the desired
            percentile lies between two data points ``i < j``.

        keepdims: `bool`, optional
            If this is set to True, the axes which are reduced are
            left in the result as dimensions with size one. With this
            option, the result will broadcast correctly against the
            original array *a*.

        mtol: number, optional
            The sample size threshold below which collapsed values are
            set to missing data. It is defined as a fraction (between
            0 and 1 inclusive) of the contributing input data values.

            The default of *mtol* is 1, meaning that a missing datum
            in the output array occurs whenever all of its
            contributing input array elements are missing data.

            For other values, a missing datum in the output array
            occurs whenever more than ``100*mtol%`` of its
            contributing input array elements are missing data.

            Note that for non-zero values of *mtol*, different
            collapsed elements may have different sample sizes,
            depending on the distribution of missing data in the input
            data.

    :Returns:

        `numpy.ndarray`

    """
    from math import prod

    a = cfdm_to_memory(a)

    if np.ma.isMA(a) and not np.ma.is_masked(a):
        # Masked array with no masked elements
        a = a.data

    if np.ma.isMA(a):
        # ------------------------------------------------------------
        # Input array is masked: Replace missing values with NaNs and
        # remask later.
        # ------------------------------------------------------------
        if a.dtype != float:
            # Can't assign NaNs to integer arrays
            a = a.astype(float, copy=True)

        mask = None
        if mtol < 1:
            # Count the number of missing values that contribute to
            # each output percentile value and make a corresponding
            # mask
            full_size = prod(
                [size for i, size in enumerate(a.shape) if i in axis]
            )
            n_missing = full_size - np.ma.count(
                a, axis=axis, keepdims=keepdims
            )
            if n_missing.any():
                mask = np.where(n_missing > mtol * full_size, True, False)
                if q.ndim:
                    mask = np.expand_dims(mask, 0)

        a = np.ma.filled(a, np.nan)

        with np.testing.suppress_warnings() as sup:
            sup.filter(
                category=RuntimeWarning,
                message=".*All-NaN slice encountered.*",
            )
            p = np.nanpercentile(
                a,
                q,
                axis=axis,
                method=method,
                keepdims=keepdims,
                overwrite_input=True,
            )

        # Update the mask for NaN points
        nan_mask = np.isnan(p)
        if nan_mask.any():
            if mask is None:
                mask = nan_mask
            else:
                mask = np.ma.where(nan_mask, True, mask)

        # Mask any NaNs and elements below the mtol threshold
        if mask is not None:
            p = np.ma.where(mask, np.ma.masked, p)

    else:
        # ------------------------------------------------------------
        # Input array is not masked
        # ------------------------------------------------------------
        p = np.percentile(
            a,
            q,
            axis=axis,
            method=method,
            keepdims=keepdims,
            overwrite_input=False,
        )

    return p


def _getattr(x, attr):
    return getattr(x, attr, False)


_array_getattr = np.vectorize(_getattr, excluded="attr")


def cf_YMDhms(a, attr):
    """Return a date-time component from an array of date-time objects.

    Only applicable for data with reference time units. The returned
    array will have the same mask hardness as the original array.

    .. versionadded:: 3.14.0

    .. seealso:: `~cf.Data.year`, ~cf.Data.month`, `~cf.Data.day`,
                 `~cf.Data.hour`, `~cf.Data.minute`, `~cf.Data.second`

    :Parameters:

        a: `numpy.ndarray`
            The array from which to extract date-time component.

        attr: `str`
            The name of the date-time component, one of ``'year'``,
            ``'month'``, ``'day'``, ``'hour'``, ``'minute'``,
            ``'second'``.

    :Returns:

        `numpy.ndarray`
            The date-time component.

    **Examples**

    >>> import numpy as np
    >>> a = np.array([
    ...  cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False)
    ...  cftime.DatetimeGregorian(2000, 1, 2, 0, 0, 0, 0, has_year_zero=False)
    ... ])
    >>> cf_YMDmhs(a, 'day')
    array([1, 2])

    """
    a = cfdm_to_memory(a)
    return _array_getattr(a, attr=attr)


def cf_rt2dt(a, units):
    """Convert an array of reference times to date-time objects.

    .. versionadded:: 3.14.0

    .. seealso:: `cf._dt2rt`, `cf.Data._asdatetime`

    :Parameters:

        a: `numpy.ndarray`
            An array of numeric reference times.

        units: `Units`
            The units for the reference times


    :Returns:

        `numpy.ndarray`
            A array containing date-time objects.

    **Examples**

    >>> import numpy as np
    >>> print(cf_rt2dt(np.array([0, 1]), cf.Units('days since 2000-01-01')))
    [cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False)
     cftime.DatetimeGregorian(2000, 1, 2, 0, 0, 0, 0, has_year_zero=False)]

    """
    a = cfdm_to_memory(a)

    if not units.iscalendartime:
        return rt2dt(a, units_in=units)

    # Calendar month/year units
    from ..timeduration import TimeDuration

    def _convert(x, units, reftime):
        t = TimeDuration(x, units=units)
        if x > 0:
            return t.interval(reftime, end=False)[1]
        else:
            return t.interval(reftime, end=True)[0]

    return np.vectorize(
        partial(
            _convert,
            units=units._units_since_reftime,
            reftime=dt(units.reftime, calendar=units._calendar),
        ),
        otypes=[object],
    )(a)


def cf_dt2rt(a, units):
    """Convert an array of date-time objects to reference times.

    .. versionadded:: 3.14.0

    .. seealso:: `cf._rt2dt`, `cf.Data._asreftime`

    :Parameters:

        a: `numpy.ndarray`
            An array of date-time objects.

        units: `Units`
            The units for the reference times

    :Returns:

        `numpy.ndarray`
            An array containing numeric reference times

    **Examples**

    >>> import numpy as np
    >>> a = np.array([
    ...  cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False)
    ...  cftime.DatetimeGregorian(2000, 1, 2, 0, 0, 0, 0, has_year_zero=False)
    ... ])
    >>> print(cf_dt2rt(a, cf.Units('days since 1999-01-01')))
    [365 366]

    """
    a = cfdm_to_memory(a)
    return dt2rt(a, units_out=units, units_in=None)


def cf_units(a, from_units, to_units):
    """Convert array values to have different equivalent units.

    .. versionadded:: 3.14.0

    .. seealso:: `cf.Data.Units`

    :Parameters:

        a: `numpy.ndarray`
            The array.

        from_units: `Units`
            The existing units of the array.

        to_units: `Units`
            The units that the array should be converted to. Must be
            equivalent to *from_units*.

    :Returns:

        `numpy.ndarray`
            An array containing values in the new units. In order to
            represent the new units, the returned data type may be
            different from that of the input array. For instance, if
            *a* has an integer data type, *from_units* are kilometres,
            and *to_units* are ``'miles'`` then the returned array
            will have a float data type.

    **Examples**

    >>> import numpy as np
    >>> a = np.array([1, 2])
    >>> print(cf.data.dask_utils.cf_units(a, cf.Units('km'), cf.Units('m')))
    [1000. 2000.]

    """
    a = cfdm_to_memory(a)
    return Units.conform(
        a, from_units=from_units, to_units=to_units, inplace=False
    )


def cf_is_masked(a):
    """Determine whether an array has masked values.

    .. versionadded:: 3.16.3

    :Parameters:

        a: array_like
            The array.

    :Returns:

        `numpy.ndarray`
            A size 1 Boolean array with the same number of dimensions
            as *a*, for which `True` indicates that there are masked
            values.

    """
    a = cfdm_to_memory(a)
    out = np.ma.is_masked(a)
    return np.array(out).reshape((1,) * a.ndim)


def cf_filled(a, fill_value=None):
    """Replace masked elements with a fill value.

    .. versionadded:: 3.16.3

    :Parameters:

        a: array_like
            The array.

        fill_value: scalar
            The fill value.

    :Returns:

        `numpy.ndarray`
            The filled array.

    **Examples**

    >>> a = np.array([[1, 2, 3]])
    >>> print(cf.data.dask_utils.cf_filled(a, -999))
    [[1 2 3]]
    >>> a = np.ma.array([[1, 2, 3]], mask=[[True, False, False]])
    >>> print(cf.data.dask_utils.cf_filled(a, -999))
    [[-999    2    3]]

    """
    a = cfdm_to_memory(a)
    return np.ma.filled(a, fill_value=fill_value)


def cf_healpix_func(a, ncells, iaxis, conserve_integral):
    """Calculate HEALPix cell bounds.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    """
    if conserve_integral:
        a = a / ncells

    iaxis = iaxis + 1
    a = np.expand_dims(a, iaxis)

    shape = list(a.shape)
    shape[iaxis] = ncells

    a = np.broadcast_to(a, shape, subok=True)
    a = a.reshape(
        shape[: iaxis - 1]
        + [shape[iaxis - 1] * shape[iaxis]]
        + shape[iaxis + 1 :]
    )

    return a


def cf_healpix_funcy(a, ncells):
    """Calculate HEALPix cell bounds.

    For instance, when going from refinement level 1 to refinement
    level 2, if *a* is ``(2, 23, 17)`` then it will transformed to
    ``(8, 9, 10, 11, 92, 93, 94, 95, 68, 69, 70, 71)`` where
    ``8=2*ncells, 9=2*ncells+1, ..., 71=17*ncells+3``, and where
    ``ncells`` is the number of cells at refinement level 2 that lie
    inside one cell at refinement level 1, i.e. ``ncells=4**(2-1)=4``.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    """
    # PERFORMANCE: This function can use a lot of memory when 'a'
    #              and/or 'ncells' are large.

    a = a * ncells
    a = np.expand_dims(a, -1)
    a = np.broadcast_to(a, (a.size, ncells), subok=True).copy()
    a += np.arange(ncells)
    a = a.flatten()

    return a


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
    from HEALPix indices. Each cell has four bounds which are returned
    in an anticlockwise direction, as seen from above, starting with
    the northern-most vertex.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    M. Reinecke and E. Hivon: Efficient data structures for masks on
    2D grids. A&A, 580 (2015)
    A132. https://doi.org/10.1051/0004-6361/201526549

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
            or ``'ring'``, but is ignored for *indexing_scheme*
            ``'nested_unique'`` (in which case *refinement_level* may
            be `None`).

        lat: `bool`, optional
            If True then return latitude bounds.

        lon: `bool`, optional
            If True then return longitude bounds.

        pole_longitude: `None` or number
            The longitude of coordinate bounds that lie exactly on the
            north (south) pole. If `None` then the longitude of such a
            vertex will be the same as the south (north) vertex of the
            same cell. If set to a number, then the longitudes of such
            vertices will all be given that value.

    :Returns:

        `numpy.ndarray`
            A 2-d array containing the HEALPix cell bounds.

    **Examples**

    >>> cf.data.dask_utils.cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, lat=True
    )
    array([[41.8103149 , 19.47122063,  0.        , 19.47122063],
           [66.44353569, 41.8103149 , 19.47122063, 41.8103149 ],
           [66.44353569, 41.8103149 , 19.47122063, 41.8103149 ],
           [90.        , 66.44353569, 41.8103149 , 66.44353569]])
    >>> cf.data.dask_utils.cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, lon=True
    )
    array([[45. , 22.5, 45. , 67.5],
           [90. , 45. , 67.5, 90. ],
           [ 0. ,  0. , 22.5, 45. ],
           [45. ,  0. , 45. , 90. ]])
    >>> cf.data.dask_utils.cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, lon=True,
    ...     pole_longitude=3.14159
    )
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

    THe coordinates are the cell centres.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    M. Reinecke and E. Hivon: Efficient data structures for masks on
    2D grids. A&A, 580 (2015)
    A132. https://doi.org/10.1051/0004-6361/201526549

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
            or ``'ring'``, but is ignored for *indexing_scheme*
            ``'nested_unique'`` (in which case *refinement_level* may
            be `None`).

        lat: `bool`, optional
            If True then return latitude coordinates.

        lon: `bool`, optional
            If True then return longitude coordinates.

    :Returns:

        `numpy.ndarray`
            A 1-d array containing the HEALPix cell coordinates.

    **Examples**

    >>> cf.data.dask_utils.cf_healpix_coordinates(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, lat=True
    )
    array([19.47122063, 41.8103149 , 41.8103149 , 66.44353569])
    >>> cf.data.dask_utils.cf_healpix_coordinates(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, lon=True
    )
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

        nest = True
        orders, a = healpix.uniq2pix(a, nest=nest)
        for order in np.unique(orders):
            nside = healpix.order2nside(order)
            indices = np.where(orders == order)[0]
            c[indices] = healpix.pix2ang(
                nside=nside, ipix=a[indices], nest=nest, lonlat=True
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

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    M. Reinecke and E. Hivon: Efficient data structures for masks on
    2D grids. A&A, 580 (2015)
    A132. https://doi.org/10.1051/0004-6361/201526549

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
            or ``'ring'``, but is ignored for *indexing_scheme*
            ``'nested_unique'`` (in which case *refinement_level* may
            be `None`).

    :Returns:

        `numpy.ndarray`
            An array containing the new HEALPix indices.

    **Examples**

    >>> cf.data.dask_utils.cf_healpix_indexing_scheme(
    ...     [0, 1, 2, 3], 'nested', 'ring', 1
    ... )
    array([13,  5,  4,  0])
    >>> cf.data.dask_utils.cf_healpix_indexing_scheme(
    ...     [0, 1, 2, 3], 'nested', 'nested_unique', 1
    )
    array([16, 17, 18, 19])
    >>> cf.data.dask_utils.cf_healpix_indexing_scheme(
    ...     [16, 17, 18, 19], 'nested_unique', 'nest', None
    )
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

    match indexing_scheme:
        case "nested":
            match new_indexing_scheme:
                case "ring":
                    nside = healpix.order2nside(refinement_level)
                    return healpix.nest2ring(nside, a)

                case "nested_unique":
                    return healpix.pix2uniq(refinement_level, a, nest=True)

        case "ring":
            match new_indexing_scheme:
                case "nested":
                    nside = healpix.order2nside(refinement_level)
                    return healpix.ring2nest(nside, a)

                case "nested_unique":
                    return healpix.pix2uniq(refinement_level, a, nest=False)

        case "nested_unique":
            match new_indexing_scheme:
                case "nested" | "ring":
                    nest = new_indexing_scheme == "nested"
                    order, a = healpix.uniq2pix(a, nest=nest)

                    refinement_levels = np.unique(order)
                    if refinement_levels.size > 1:
                        raise ValueError(
                            "Can't change HEALPix indexing scheme from "
                            f"'nested_unique' to {new_indexing_scheme!r} "
                            "when the HEALPix indices span multiple "
                            "refinement levels (at least levels "
                            f"{refinement_levels.tolist()})"
                        )

                    return a

    raise RuntimeError(
        "cf_healpix_indexing_scheme: Failed during Dask computation"
    )


def cf_healpix_weights(a, indexing_scheme, measure=False, radius=None):
    """Calculate HEALPix cell area weights.

    K. Gorski, Eric Hivon, A. Banday, B. Wandelt, M. Bartelmann, et
    al.. HEALPix: A Framework for High-Resolution Discretization and
    Fast Analysis of Data Distributed on the Sphere. The Astrophysical
    Journal, 2005, 622 (2), pp.759-771.
    https://dx.doi.org/10.1086/427976

    M. Reinecke and E. Hivon: Efficient data structures for masks on
    2D grids. A&A, 580 (2015)
    A132. https://doi.org/10.1051/0004-6361/201526549

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

    >>> cf.data.dask_utils.cf_healpix_weights(
    ...     [76, 77, 78, 79, 20, 21], 'nested_unique'
    )
    array([0.0625, 0.0625, 0.0625, 0.0625, 0.25  , 0.25  ])
    >>> cf.data.dask_utils.cf_healpix_weights(
    ...     [76, 77, 78, 79, 20, 21], 'nested_unique',
    ...     measure=True, radius=6371000
    )
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

    if a.ndim != 1:
        raise ValueError(
            "Can only calculate HEALPix cell area weights when the "
            f"healpix_index array has one dimension. Got shape {a.shape}"
        )

    if measure:
        # Surface area of sphere is 4*pi*(r**2)
        # Number of HEALPix cells at refinement level N is 12*(4**N)
        # => Area of one cell is pi*(r**2)/(3* (4**N))
        x = np.pi * (radius**2) / 3.0
    else:
        x = 1.0

    orders = healpix.uniq2pix(a, nest=True)[0]
    orders, index, inverse = np.unique(
        orders, return_index=True, return_inverse=True
    )

    # Initialise the output weights array
    w = np.empty(a.shape, dtype="float64")

    # For each refinement level N, put the weights (= x/4**N) into 'w'
    # at the correct locations
    for order, i in zip(orders, index):
        w = np.where(inverse == inverse[i], x / (4**order), w)

    return w
