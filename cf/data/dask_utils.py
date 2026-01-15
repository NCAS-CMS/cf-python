"""Functions intended to be passed to be dask.

These will typically be functions that operate on dask chunks. For
instance, as would be passed to `dask.array.map_blocks`.

"""

from functools import partial

import numpy as np
from cfdm.data.dask_utils import cfdm_to_memory

from ..cfdatetime import dt, dt2rt, rt2dt
from ..units import Units


def _zuniq2pix(a, nest=False):
    """Convert from zuniq to ring or nested pixel scheme.

    This function emulates the as-yet-nonexistent function
    `healpix.zuniq2pix`.

    See
    https://github.com/cds-astro/cds-healpix-rust/blob/v0.7.3/src/nested/mod.rs#L188-L194
    for details.

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
            levels and a `numpy` array of HEALPix pixel indices. The
            indices will follow the rng indexing scheme if *nest* is
            True, otherwise the nested indiexing sheme.

    """
    import healpix

    if not nest:
        raise NotImplementedError(
            "Can't yet convert from zuniq to ring indices"
        )

    a = a.astype("int64", copy=False)

    depth_max = healpix.nside2order(healpix._chp.NSIDE_MAX)

    n_trailing_zeros = np.bitwise_count((a & -a) - 1)
    n_trailing_zeros = n_trailing_zeros.astype("int8", copy=False)

    delta_depth = n_trailing_zeros >> 1
    depth = depth_max - delta_depth

    a = a >> (n_trailing_zeros + 1)

    return depth, a


def _zuniq2uniq(a):
    """Convert from zuniq to nuniq pixel scheme.

    This function emulates the as-yet-nonexistent function
    `healpix.zuniq2uniq`.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: array_like
            An array of zuniq indices.

    :Returns:

        `numpy.ndarray`
            Returns a `numpy` array HEALPix pixel indices following
            the nuniq indexing scheme.

    """
    order, a = _zuniq2pix(a, nest=True)
    order = order.astype("int64", copy=False)

    order += 1
    a += 4**order

    return a


def _uniq2zuniq(a):
    """Convert from nuniq to zuniq pixel scheme.

    This function emulates (in a non-optimal way!) the
    as-yet-nonexistent function `healpix.uniq2zuniq`.

    .. versionadded:: NEXTVERSION

    :Parameters:

        a: array_like
            An array of nuniq indices.

    :Returns:

        `numpy.ndarray`
            Returns a `numpy` array HEALPix pixel indices following
            the zuniq indexing scheme.

    """
    import healpix

    depth_max = healpix.nside2order(healpix._chp.NSIDE_MAX)

    order, a = healpix.uniq2pix(a, nest=True)
    order = order.astype("int64", copy=False)

    order = 4 ** (depth_max - order)
    a *= 2
    a += 1
    a *= order

    return a


def _pix2zuniq(refinement_level, a, nest=False):
    """Convert ring or nested to zuniq pixel scheme.

    This function emulates (in a non-optimal way!) the
    as-yet-nonexistent function `healpix.pix2zuniq`.

    .. versionadded:: NEXTVERSION

    :Parameters:

        refinement_level: `int`
            The refinement level of the indices.

        a: array_like
            An array of nested or ring indices at the given refinement
            level.

        nest: `bool`, optional
            True if for nested indices, or False (the default) for
            ring indices.

    :Returns:

        `numpy.ndarray`
            Returns a `numpy` array HEALPix pixel indices following
            the zuniq indexing scheme.

    """
    import healpix

    if not nest:
        # Convert ring to nest
        nside = healpix.order2nside(refinement_level)
        a = healpix.ring2nest(nside, a)
    else:
        a = a.astype("int64", copy=True)

    # Convert nest to zuniq
    depth_max = healpix.nside2order(healpix._chp.NSIDE_MAX)

    a *= 2
    a += 1
    a *= 4 ** (depth_max - refinement_level)

    return a


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
            pixels. A value of 0 (the default) centres the filter over
            the pixel, with positive values shifting the filter to the
            left, and negative ones to the right.

    :Returns:

        `numpy.ndarray`
            Convolved float array with same shape as input.

    """
    from scipy.ndimage import convolve1d

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
        # re-mask later.
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

    .. seealso:: `cf.Field.create_latlon_coordinates`

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

    >>> cf.data.dask_utils.cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, latitude=True
    )
    array([[41.8103149 , 19.47122063,  0.        , 19.47122063],
           [66.44353569, 41.8103149 , 19.47122063, 41.8103149 ],
           [66.44353569, 41.8103149 , 19.47122063, 41.8103149 ],
           [90.        , 66.44353569, 41.8103149 , 66.44353569]])
    >>> cf.data.dask_utils.cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, longitude=True
    )
    array([[45. , 22.5, 45. , 67.5],
           [90. , 45. , 67.5, 90. ],
           [ 0. ,  0. , 22.5, 45. ],
           [45. ,  0. , 45. , 90. ]])
    >>> cf.data.dask_utils.cf_healpix_bounds(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, longitude=True,
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

    if latitude:
        pos = 1
    elif longitude:
        pos = 0

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

    .. seealso:: `cf.Field.create_latlon_coordinates`

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

    >>> cf.data.dask_utils.cf_healpix_coordinates(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, latitude=True
    )
    array([19.47122063, 41.8103149 , 41.8103149 , 66.44353569])
    >>> cf.data.dask_utils.cf_healpix_coordinates(
    ...     np.array([0, 1, 2, 3]), 'nested', 1, longitude=True
    )
    array([45. , 67.5, 22.5, 45. ])

    """
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

    if latitude == longitude:
        raise ValueError(
            "Can't calculate HEALPix cell coordinates: "
            f"latitude={latitude!r} and longitude={longitude!r}"
        )

    if latitude:
        pos = 1
    elif longitude:
        pos = 0

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


def cf_healpix_indexing_scheme(
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

    .. seealso:: `cf.Field.healpix_indexing_scheme`

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
            When changing from an nuniq or zuniq MOC indexing scheme
            to a ring or nested indexing scheme, *moc_refinment_level*
            must be set to the unique refinement level represented by
            the MOC indices, and if this is not the case then an
            Eexception is raised.

    :Returns:

        `numpy.ndarray`
            An array containing the new HEALPix indices.

    **Examples**

    >>> cf.data.dask_utils.cf_healpix_indexing_scheme(
    ...     [0, 1, 2, 3], 'nested', 'ring', 1
    ... )
    array([13,  5,  4,  0])
    >>> cf.data.dask_utils.cf_healpix_indexing_scheme(
    ...     [0, 1, 2, 3], 'nested', 'nuniq', 1
    )
    array([16, 17, 18, 19])
    >>> cf.data.dask_utils.cf_healpix_indexing_scheme(
    ...     [16, 17, 18, 19], 'nuniq', 'nested'
    )
    array([0, 1, 2, 3])

    """
    if new_indexing_scheme == indexing_scheme:
        # Null operation
        return a

    from ..constants import healpix_indexing_schemes

    if new_indexing_scheme not in healpix_indexing_schemes:
        raise ValueError(
            "Can't change HEALPix indexing scheme: Unknown "
            "'new_indexing_scheme' in cf_healpix_indexing_scheme: "
            f"{new_indexing_scheme!r}"
        )

    a = cfdm_to_memory(a)

    if indexing_scheme == new_indexing_scheme:
        # Null operation
        return a

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
                    nest = new_indexing_scheme == "nested"
                    order, a = healpix.uniq2pix(a, nest=nest)

                    order = np.unique(order)
                    if order.size > 1:
                        raise ValueError(
                            "Can't change HEALPix indexing scheme from "
                            f"{indexing_scheme!r} to {new_indexing_scheme!r}: "
                            "HEALPix indices span multiple refinement levels "
                            f"(at least levels {order.tolist()})"
                        )

                    if (
                        moc_refinement_level is None
                        or order != moc_refinement_level
                    ):
                        raise ValueError(
                            "TODOHEALPIX Can't change HEALPix indexing scheme from "
                            f"{indexing_scheme!r} to {new_indexing_scheme!r}: "
                            "HEALPix indices span multiple refinement levels "
                            f"(at least levels {order.tolist()})"
                        )

                case "zuniq":
                    a = _uniq2zuniq(a)

        case "zuniq":
            match new_indexing_scheme:
                case "nested" | "ring":
                    # Convert to zuniq to nested
                    order, a = _zuniq2pix(a, nest=True)

                    order = np.unique(order)
                    if order.size > 1:
                        raise ValueError(
                            "Can't change HEALPix indexing scheme from "
                            f"{indexing_scheme!r} to {new_indexing_scheme!r}: "
                            "HEALPix indices span multiple refinement levels "
                            f"(at least levels {order.tolist()})"
                        )

                    if (
                        moc_refinement_level is None
                        or order != moc_refinement_level
                    ):
                        raise ValueError(
                            "TODOHEALPIX Can't change HEALPix indexing scheme from "
                            f"{indexing_scheme!r} to {new_indexing_scheme!r}: "
                            "HEALPix indices span multiple refinement levels "
                            f"(at least levels {order.tolist()})"
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
                "'indexing_scheme' in cf_healpix_indexing_scheme: "
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

    .. seealso:: `cf.weights.Weights.healpix_area`

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

    >>> cf.data.dask_utils.cf_healpix_weights(
    ...     [76, 77, 78, 79, 20, 21], 'nuniq'
    )
    array([0.0625, 0.0625, 0.0625, 0.0625, 0.25  , 0.25  ])
    >>> cf.data.dask_utils.cf_healpix_weights(
    ...     [76, 77, 78, 79, 20, 21], 'nuniq',
    ...     measure=True, radius=6371000
    )
    array([2.65658579e+12, 2.65658579e+12, 2.65658579e+12, 2.65658579e+12,
           1.06263432e+13, 1.06263432e+13])

    """
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
