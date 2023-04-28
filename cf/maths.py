from . import Units
from .functions import _DEPRECATION_ERROR_FUNCTION


def relative_vorticity(
    u, v, wrap=None, one_sided_at_boundary=False, radius=6371229.0, cyclic=None
):
    """Calculate the relative vorticity using centred finite
    differences.

    Deprecated at version 3.15.1 and is no longer available. Use
    function `cf.curl_xy` instead. Note that `cf.curl_xy` uses a more
    accurate spherical polar coordinates formula.

    The relative vorticity of wind defined on a Cartesian domain (such
    as a plane projection) is defined as

      ζcartesian = δv/δx − δu/δy

    where x and y are points on along the 'X' and 'Y' Cartesian
    dimensions respectively; and u and v denote the 'X' and 'Y'
    components of the horizontal winds.

    If the wind field field is defined on a spherical
    latitude-longitude domain then a correction factor is included:

      ζspherical = δv/δx − δu/δy + (u/a)tan(ϕ)

    where u and v denote the longitudinal and latitudinal components
    of the horizontal wind field; a is the radius of the Earth; and ϕ
    is the latitude at each point.

    The relative vorticity is calculated using centred finite
    differences (see the *one_sided_at_boundary* parameter).

    The grid may be global or limited area. If missing values are
    present then missing values will be returned at points where the
    centred finite difference could not be calculated. The boundary
    conditions may be cyclic in longitude. The non-cyclic boundaries
    may either be filled with missing values or calculated with
    off-centre finite differences.

    Reference: H.B. Bluestein, Synoptic-Dynamic Meteorology in
    Midlatitudes, 1992, Oxford Univ. Press p113-114

    :Parameters:

        u: `Field`
            A field containing the x-wind. Must be on the same grid as
            the y-wind.

        v: `Field`
            A field containing the y-wind. Must be on the same grid as
            the x-wind.

        radius: optional
            The radius of the sphere when the winds are on a spherical
            polar coordinate domain. May be any numeric scalar object
            that can be converted to a `Data` object (which includes
            numpy array and `Data` objects). By default *radius* has a
            value of 6371229.0 metres, representing the Earth's
            radius. If units are not specified then units of metres
            are assumed.

            *Parameter example:*
              Five equivalent ways to set a radius of 6371200 metres:
              ``radius=6371200``, ``radius=numpy.array(6371200)``,
              ``radius=cf.Data(6371200)``, ``radius=cf.Data(6371200,
              'm')``, ``radius=cf.Data(6371.2, 'km')``.

        wrap: `bool`, optional
            Whether the longitude is cyclic or not. By default this is
            autodetected.

        one_sided_at_boundary: `bool`, optional
            If True then if the field is not cyclic off-centre finite
            differences are calculated at the boundaries, otherwise
            missing values are used at the boundaries.

    :Returns:

        `Field`
            The relative vorticity calculated with centred finite
            differences.

    """
    _DEPRECATION_ERROR_FUNCTION(
        "relative_vorticity",
        message="Use function 'cf.curl_xy` instead. Note that 'cf.curl_xy' uses a more accurate spherical polar coordinates formula.",
        version="3.15.1",
        removed_at="5.0.0",
    )  # pragman: no cover


def histogram(*digitized):
    """Return the distribution of a set of variables in the form of an
    N-dimensional histogram.

    The number of dimensions of the histogram is equal to the number
    of field constructs provided by the *digitized* argument. Each
    such field construct defines a sequence of bins and provides
    indices to the bins that each value of one of the variables
    belongs. There is no upper limit to the number of dimensions of
    the histogram.

    The output histogram bins are defined by the exterior product of
    the one-dimensional bins of each digitized field construct. For
    example, if only one digitized field construct is provided then
    the histogram bins simply comprise its one-dimensional bins; if
    there are two digitized field constructs then the histogram bins
    comprise the two-dimensional matrix formed by all possible
    combinations of the two sets of one-dimensional bins; etc.

    An output value for an histogram bin is formed by counting the
    number cells for which the digitized field constructs, taken
    together, index that bin. Note that it may be the case that not
    all output bins are indexed by the digitized field constructs, and
    for these bins missing data is returned.

    The returned field construct will have a domain axis construct for
    each dimension of the histogram, with a corresponding dimension
    coordinate construct that defines the bin boundaries.

    .. versionadded:: 3.0.2

    .. seealso:: `cf.Field.bin`, `cf.Field.collapse`,
                 `cf.Field.digitize`, `cf.Field.percentile`,
                 `cf.Field.where`

    :Parameters:

        digitized: one or more `Field`
            One or more field constructs that contain digitized data
            with corresponding metadata, as would be output by
            `cf.Field.digitize`. Each field construct contains indices
            to the one-dimensional bins to which each value of an
            original field construct belongs; and there must be
            ``bin_count`` and ``bin_bounds`` properties as defined by
            the `cf.Field.digitize` method (and any of the extra
            properties defined by that method are also recommended).

            The bins defined by the ``bin_count`` and ``bin_bounds``
            properties are used to create a dimension coordinate
            construct for the output field construct.

            Each digitized field construct must be transformable so
            that its data is broadcastable to any other digitized
            field contruct's data. This is done by using the metadata
            constructs of the to create a mapping of physically
            compatible dimensions between the fields, and then
            manipulating the dimensions of the digitized field
            construct's data to ensure that broadcasting can occur.

    :Returns:

        `Field`
            The field construct containing the histogram.

    **Examples**

    Create a one-dimensional histogram based on 10 equally-sized bins
    that exactly span the data range:

    >>> f = cf.example_field(0)
    >>> print(f)
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(latitude(5), longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]
    >>> print(f.array)
    [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
     [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
     [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
     [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
     [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
    >>> indices, bins = f.digitize(10, return_bins=True)
    >>> print(indices)
    Field: long_name=Bin index to which each 'specific_humidity' value belongs (ncvar%q)
    ------------------------------------------------------------------------------------
    Data            : long_name=Bin index to which each 'specific_humidity' value belongs(latitude(5), longitude(8))
    Cell methods    : area: mean
    Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]
    >>> print(bins.array)
    [[0.003  0.0173]
     [0.0173 0.0316]
     [0.0316 0.0459]
     [0.0459 0.0602]
     [0.0602 0.0745]
     [0.0745 0.0888]
     [0.0888 0.1031]
     [0.1031 0.1174]
     [0.1174 0.1317]
     [0.1317 0.146 ]]
    >>> h = cf.histogram(indices)
    >>> rint(h)
    Field: number_of_observations
    -----------------------------
    Data            : number_of_observations(specific_humidity(10)) 1
    Cell methods    : latitude: longitude: point
    Dimension coords: specific_humidity(10) = [0.01015, ..., 0.13885] 1
    >>> print(h.array)
    [9 7 9 4 5 1 1 1 2 1]
    >>> print(h.coordinate('specific_humidity').bounds.array)
    [[0.003  0.0173]
     [0.0173 0.0316]
     [0.0316 0.0459]
     [0.0459 0.0602]
     [0.0602 0.0745]
     [0.0745 0.0888]
     [0.0888 0.1031]
     [0.1031 0.1174]
     [0.1174 0.1317]
     [0.1317 0.146 ]]


    Create a two-dimensional histogram based on specific humidity and
    temperature bins. The temperature bins in this example are derived
    from a dummy temperature field construct with the same shape as
    the specific humidity field construct already in use:

    >>> g = f.copy()
    >>> g.standard_name = 'air_temperature'
    >>> import numpy
    >>> g[...] = numpy.random.normal(loc=290, scale=10, size=40).reshape(5, 8)
    >>> g.override_units('K', inplace=True)
    >>> print(g)
    Field: air_temperature (ncvar%q)
    --------------------------------
    Data            : air_temperature(latitude(5), longitude(8)) K
    Cell methods    : area: mean
    Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]

    >>> indices_t = g.digitize(5)
    >>> h = cf.histogram(indices, indices_t)
    >>> print(h)
    Field: number_of_observations
    -----------------------------
    Data            : number_of_observations(air_temperature(5), specific_humidity(10)) 1
    Cell methods    : latitude: longitude: point
    Dimension coords: air_temperature(5) = [281.1054839143287, ..., 313.9741786365939] K
                    : specific_humidity(10) = [0.01015, ..., 0.13885] 1
    >>> print(h.array)
    [[2  1  5  3  2 -- -- -- -- --]
     [1  1  2 --  1 --  1  1 -- --]
     [4  4  2  1  1  1 -- --  1  1]
     [1  1 -- --  1 -- -- --  1 --]
     [1 -- -- -- -- -- -- -- -- --]]
    >>> h.sum()
    <CF Data(): 40 1>

    """
    if not digitized:
        raise ValueError(
            "Must provide at least one 'digitized' field construct"
        )

    f = digitized[0].copy()
    f.clear_properties()

    return f.bin("sample_size", digitized=digitized)


def curl_xy(fx, fy, x_wrap=None, one_sided_at_boundary=False, radius=None):
    r"""Calculate the horizontal curl of an (X, Y) vector.

    The horizontal curl is calculated from orthogonal vector component
    fields which have dimension coordinates of X and Y, in either
    Cartesian (e.g. plane projection) or spherical polar coordinate
    systems.

    Note that the curl of the horizontal wind field is the relative
    vorticity.

    The horizontal curl of the :math:`(f_x, f_y)` vector in Cartesian
    coordinates is given by:

    .. math:: \nabla \times (f_{x}(x,y), f_{y}(x,y)) =
                \frac{\partial f_y}{\partial x}
                -
                \frac{\partial f_x}{\partial y}

    The horizontal curl of the :math:`(f_\theta, f_\phi)` vector in
    spherical polar coordinates is given by:

    .. math:: \nabla \times (f_\theta(\theta,\phi), f_\phi(\theta,\phi)) =
                \frac{1}{r \sin\theta}
                \left(
                \frac{\partial (f_\phi \sin\theta)}{\partial \theta}
                -
                \frac{\partial f_\theta}{\partial \phi}
                \right)

    where *r* is radial distance to the origin, :math:`\theta` is the
    polar angle with respect to polar axis, and :math:`\phi` is the
    azimuthal angle.

    The curl is calculated using centred finite differences apart from
    at the boundaries (see the *x_wrap* and *one_sided_at_boundary*
    parameters). If missing values are present then missing values
    will be returned at all points where a centred finite difference
    could not be calculated.

    .. versionadded:: 3.12.0

    .. seealso:: `cf.div_xy`, `cf.Field.derivative`,
                 `cf.Field.grad_xy`, `cf.Field.iscyclic`,
                 `cf.Field.laplacian_xy`

    :Parameters:

        fx, fy: `Field`
            The fields containing the X and Y vector components.

        x_wrap: `bool`, optional
            Whether the X axis is cyclic or not. By default *x_wrap*
            is set to the result of this call to the *fx* field
            construct's `~cf.Field.iscyclic`
            method:``fy.iscyclic('X')``. If the X axis is cyclic then
            centred differences at one X boundary will always use
            values from the other, regardless of the setting of
            *one_sided_at_boundary*.

            The cyclicity of the Y axis is always set to the result of
            ``fx.iscyclic('Y')``.

        one_sided_at_boundary: `bool`, optional
            If True then one-sided finite differences are calculated
            at the non-cyclic boundaries. By default missing values
            are set at non-cyclic boundaries.

        radius: optional
            Specify the radius of the latitude-longitude plane defined
            in spherical polar coordinates. The radius is that which
            would be returned by this call of the *fx* field
            construct's `cf.Field.radius` method:
            ``fx.radius(default=radius)``. The radius is defined by
            the datum of a coordinate reference construct, and if and
            only if no such radius is found then the default value
            given by the *radius* parameter is used instead. A value
            of ``'earth'`` is equivalent to a default value of 6371229
            metres.

    :Returns:

        `Field`
            The horizontal curl of the (X, Y) fields.

    **Examples**

    >>> f = cf.example_field(0)
    >>> print(f)
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(latitude(5), longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]
    >>> f[...] = 0.1
    >>> print(f.array)
    [[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
     [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
     [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
     [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
     [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]]
    >>> fx, fy = f.grad_xy(radius='earth', one_sided_at_boundary=True)
    >>> fx, fy
    (<CF Field: long_name=X gradient of specific_humidity(latitude(5), longitude(8)) m-1.rad-1>,
     <CF Field: long_name=Y gradient of specific_humidity(latitude(5), longitude(8)) m-1.rad-1>)
    >>> c = cf.curl_xy(fx, fy, radius='earth')
    >>> c
    <CF Field: long_name=Divergence of (long_name=X gradient of specific_humidity, long_name=Y gradient of specific_humidity)(latitude(5), longitude(8)) m-2.rad-2>
    >>> print(c.array)
    [[-- -- -- -- -- -- -- --]
     [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
     [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
     [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
     [-- -- -- -- -- -- -- --]]
    >>> c = cf.curl_xy(fx, fy, radius='earth', one_sided_at_boundary=True)
    >>> print(c.array)
    [[0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]]

    """
    from numpy import pi

    fx = fx.copy()
    fy = fy.copy()

    fx_x_key, fx_x_coord = fx.dimension_coordinate(
        "X", item=True, default=(None, None)
    )
    fx_y_key, fx_y_coord = fx.dimension_coordinate(
        "Y", item=True, default=(None, None)
    )
    fy_x_key, fy_x_coord = fy.dimension_coordinate(
        "X", item=True, default=(None, None)
    )
    fy_y_key, fy_y_coord = fy.dimension_coordinate(
        "Y", item=True, default=(None, None)
    )

    if fx_x_coord is None:
        raise ValueError("'fx' field has no unique 'X' dimension coordinate")

    if fx_y_coord is None:
        raise ValueError("'fx' field has no unique 'Y' dimension coordinate")

    if fy_x_coord is None:
        raise ValueError("'fy' field has no unique 'X' dimension coordinate")

    if fy_y_coord is None:
        raise ValueError("'fy' field has no unique 'Y' dimension coordinate")

    if x_wrap is None:
        x_wrap = fx.iscyclic(fy_x_key)

    x_units = fy_x_coord.Units
    y_units = fx_y_coord.Units

    # Check for spherical polar coordinates
    latlon = (x_units.islongitude and y_units.islatitude) or (
        x_units.units == "degrees" and y_units.units == "degrees"
    )

    if latlon:
        # --------------------------------------------------------
        # Spherical polar coordinates
        # --------------------------------------------------------
        # Convert latitude and longitude units to radians, so that the
        # units of the result are nice.
        radians = Units("radians")
        fx_x_coord.Units = radians
        fx_y_coord.Units = radians
        fy_x_coord.Units = radians
        fy_y_coord.Units = radians

        # Ensure that the lat and lon dimension coordinates have
        # standard names, so that metadata-aware broadcasting works as
        # expected when all of their units are radians.
        fx_x_coord.standard_name = "longitude"
        fx_y_coord.standard_name = "latitude"
        fy_x_coord.standard_name = "longitude"
        fy_y_coord.standard_name = "latitude"

        # Get theta as a field that will broadcast to f, and adjust
        # its values so that theta=0 is at the north pole.
        theta = pi / 2 - fx.convert(fx_y_key, full_domain=True)
        sin_theta = theta.sin()

        r = fx.radius(default=radius)

        term1 = (fx * sin_theta).derivative(
            fx_y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
        )
        term2 = fy.derivative(
            fy_x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
        )

        c = (term1 - term2) / (sin_theta * r)

        # Set output latitude and longitude coordinate units
        c.dimension_coordinate("longitude").Units = x_units
        c.dimension_coordinate("latitude").Units = y_units
    else:
        # --------------------------------------------------------
        # Cartesian coordinates
        # --------------------------------------------------------
        dfy_dx = fy.derivative(
            fy_x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
        )

        dfx_dy = fx.derivative(
            fx_y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
        )

        c = dfy_dx - dfx_dy

    # Set the standard name and long name
    c.set_property(
        "long_name", f"Horizontal curl of ({fx.identity()}, {fy.identity()})"
    )
    c.del_property("standard_name", None)

    return c


def div_xy(fx, fy, x_wrap=None, one_sided_at_boundary=False, radius=None):
    r"""Calculate the horizontal divergence of an (X, Y) vector.

    The horizontal divergence is calculated from orthogonal vector
    component fields which have dimension coordinates of X and Y, in
    either Cartesian (e.g. plane projection) or spherical polar
    coordinate systems.

    The horizontal divergence of the :math:`(f_x, f_y)` vector in
    Cartesian coordinates is given by:

    .. math:: \nabla \cdot (f_{x}(x,y), f_{y}(x,y)) =
                \frac{\partial f_x}{\partial x}
                +
                \frac{\partial f_y}{\partial y}

    The horizontal divergence of the :math:`(f_\theta, f_\phi)` vector
    in spherical polar coordinates is given by:

    .. math:: \nabla \cdot (f_\theta(\theta,\phi), f_\phi(\theta,\phi)) =
                \frac{1}{r \sin\theta}
                \left(
                \frac{\partial (f_\theta \sin\theta)}{\partial \theta}
                +
                \frac{\partial f_\phi}{\partial \phi}
                \right)

    where *r* is radial distance to the origin, :math:`\theta` is the
    polar angle with respect to polar axis, and :math:`\phi` is the
    azimuthal angle.

    The divergence is calculated using centred finite differences
    apart from at the boundaries (see the *x_wrap* and
    *one_sided_at_boundary* parameters). If missing values are present
    then missing values will be returned at all points where a centred
    finite difference could not be calculated.

    .. versionadded:: 3.12.0

    .. seealso:: `cf.curl_xy`, `cf.Field.derivative`,
                 `cf.Field.grad_xy`, `cf.Field.iscyclic`,
                 `cf.Field.laplacian_xy`

    :Parameters:

        fx, fy: `Field`
            The fields containing the X and Y vector components.

        x_wrap: `bool`, optional
            Whether the X axis is cyclic or not. By default *x_wrap*
            is set to the result of this call to the *fx* field
            construct's `~cf.Field.iscyclic`
            method:``fx.iscyclic('X')``. If the X axis is cyclic then
            centred differences at one X boundary will always use
            values from the other, regardless of the setting of
            *one_sided_at_boundary*.

            The cyclicity of the Y axis is always set to the result of
            ``fy.iscyclic('Y')``.

        one_sided_at_boundary: `bool`, optional
            If True then one-sided finite differences are calculated
            at the non-cyclic boundaries. By default missing values
            are set at non-cyclic boundaries.

        radius: optional
            Specify the radius of the latitude-longitude plane defined
            in spherical polar coordinates. The radius is that which
            would be returned by this call of the *fx* field
            construct's `cf.Field.radius` method:
            ``fx.radius(default=radius)``. The radius is defined by
            the datum of a coordinate reference construct, and if and
            only if no such radius is found then the default value
            given by the *radius* parameter is used instead. A value
            of ``'earth'`` is equivalent to a default value of 6371229
            metres.

    :Returns:

        `Field`
            The horizontal curl of the (X, Y) fields.

    **Examples**

    >>> f = cf.example_field(0)
    >>> print(f)
    Field: specific_humidity (ncvar%q)
    ----------------------------------
    Data            : specific_humidity(latitude(5), longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : time(1) = [2019-01-01 00:00:00]
    >>> f[...] = 0.1
    >>> print(f.array)
    [[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
     [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
     [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
     [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
     [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]]
    >>> fx, fy = f.grad_xy(radius='earth', one_sided_at_boundary=True)
    >>> fx, fy
    (<CF Field: long_name=X gradient of specific_humidity(latitude(5), longitude(8)) m-1.rad-1>,
     <CF Field: long_name=Y gradient of specific_humidity(latitude(5), longitude(8)) m-1.rad-1>)
    >>> d = cf.div_xy(fx, fy, radius='earth')
    >>> d
    <CF Field: long_name=Divergence of (long_name=X gradient of specific_humidity, long_name=Y gradient of specific_humidity)(latitude(5), longitude(8)) m-2.rad-2>
    >>> print(d.array)
    [[-- -- -- -- -- -- -- --]
     [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
     [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
     [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
     [-- -- -- -- -- -- -- --]]
    >>> d = cf.div_xy(fx, fy, radius='earth', one_sided_at_boundary=True)
    >>> print(d.array)
    [[0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]]

    """
    from numpy import pi

    fx = fx.copy()
    fy = fy.copy()

    fx_x_key, fx_x_coord = fx.dimension_coordinate(
        "X", item=True, default=(None, None)
    )
    fx_y_key, fx_y_coord = fx.dimension_coordinate(
        "Y", item=True, default=(None, None)
    )
    fy_x_key, fy_x_coord = fy.dimension_coordinate(
        "X", item=True, default=(None, None)
    )
    fy_y_key, fy_y_coord = fy.dimension_coordinate(
        "Y", item=True, default=(None, None)
    )

    if fx_x_coord is None:
        raise ValueError("'fx' field has no unique 'X' dimension coordinate")

    if fx_y_coord is None:
        raise ValueError("'fx' field has no unique 'Y' dimension coordinate")

    if fy_x_coord is None:
        raise ValueError("'fy' field has no unique 'X' dimension coordinate")

    if fy_y_coord is None:
        raise ValueError("'fy' field has no unique 'Y' dimension coordinate")

    if x_wrap is None:
        x_wrap = fx.iscyclic(fx_x_key)

    x_units = fx_x_coord.Units
    y_units = fy_y_coord.Units

    # Check for spherical polar coordinates
    latlon = (x_units.islongitude and y_units.islatitude) or (
        x_units.units == "degrees" and y_units.units == "degrees"
    )

    if latlon:
        # ------------------------------------------------------------
        # Spherical polar coordinates
        # ------------------------------------------------------------
        # Convert latitude and longitude units to radians, so that the
        # units of the result are nice.
        radians = Units("radians")
        fx_x_coord.Units = radians
        fx_y_coord.Units = radians
        fy_x_coord.Units = radians
        fy_y_coord.Units = radians

        # Ensure that the lat and lon dimension coordinates have
        # standard names, so that metadata-aware broadcasting works as
        # expected when all of their units are radians.
        fx_x_coord.standard_name = "longitude"
        fx_y_coord.standard_name = "latitude"
        fy_x_coord.standard_name = "longitude"
        fy_y_coord.standard_name = "latitude"

        # Get theta as a field that will broadcast to f, and adjust
        # its values so that theta=0 is at the north pole.
        theta = pi / 2 - fy.convert(fy_y_key, full_domain=True)
        sin_theta = theta.sin()

        r = fx.radius(default=radius)

        term1 = fx.derivative(
            fx_x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
        )

        term2 = (fy * sin_theta).derivative(
            fy_y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
        )

        d = (term1 + term2) / (sin_theta * r)

        # Set output latitude and longitude coordinate units
        d.dimension_coordinate("longitude").Units = x_units
        d.dimension_coordinate("latitude").Units = y_units
    else:
        # ------------------------------------------------------------
        # Cartesian coordinates
        # ------------------------------------------------------------
        term1 = fx.derivative(
            fx_x_key, wrap=x_wrap, one_sided_at_boundary=one_sided_at_boundary
        )

        term2 = fy.derivative(
            fy_y_key, wrap=None, one_sided_at_boundary=one_sided_at_boundary
        )

        d = term1 + term2

    # Set the standard name and long name
    d.set_property(
        "long_name",
        f"Horizontal divergence of ({fx.identity()}, {fy.identity()})",
    )
    d.del_property("standard_name", None)

    return d
