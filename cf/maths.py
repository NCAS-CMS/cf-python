from .functions import _DEPRECATION_ERROR_FUNCTION_KWARGS
from . import Units

from .data.data import Data


def relative_vorticity(u, v, wrap=None, one_sided_at_boundary=False,
                       radius=6371229.0, cyclic=None):
    '''Calculate the relative vorticity using centred finite differences.

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

    '''
    if cyclic:
        _DEPRECATION_ERROR_FUNCTION_KWARGS(
            'relative_vorticity', {'cyclic': cyclic},
            "Use the 'wrap' keyword instead"
        )  # pragma: no cover

    # Get the standard names of u and v
    u_std_name = u.get_property('standard_name', None)
    v_std_name = v.get_property('standard_name', None)

    # Copy u and v
    u = u.copy()
    v = v.copy()

    # Get the X and Y coordinates
    (u_x_key, u_y_key), (u_x, u_y) = u._regrid_get_cartesian_coords(
        'u', ('X', 'Y'))
    (v_x_key, v_y_key), (v_x, v_y) = v._regrid_get_cartesian_coords(
        'v', ('X', 'Y'))

    if not u_x.equals(v_x) or not u_y.equals(v_y):
        raise ValueError('u and v must be on the same grid.')

    # Check for lat/long
    is_latlong = ((u_x.Units.islongitude and u_y.Units.islatitude) or
                  (u_x.units == 'degrees' and u_y.units == 'degrees'))

    # Check for cyclicity
    if wrap is None:
        if is_latlong:
            wrap = u.iscyclic(u_x_key)
        else:
            wrap = False
    # --- End: if

    # Find the relative vorticity
    if is_latlong:
        # Save the units of the X and Y coordinates
        x_units = u_x.Units
        y_units = u_y.Units

        # Change the units of the lat/longs to radians
        u_x.Units = Units('radians')
        u_y.Units = Units('radians')
        v_x.Units = Units('radians')
        v_y.Units = Units('radians')

        # Find cos and tan of latitude
        cos_lat = u_y.cos()
        tan_lat = u_y.tan()

        # Reshape for broadcasting
        u_shape = [1] * u.ndim
        u_y_index = u.get_data_axes().index(u_y_key)
        u_shape[u_y_index] = u_y.size

        v_shape = [1] * v.ndim
        v_y_index = v.get_data_axes().index(v_y_key)
        v_shape[v_y_index] = v_y.size

        # Calculate the correction term
        corr = u.copy()
        corr *= tan_lat.array.reshape(u_shape)

        # Calculate the derivatives
        v.derivative(v_x_key, wrap=wrap,
                     one_sided_at_boundary=one_sided_at_boundary, inplace=True)
        v.data /= cos_lat.array.reshape(v_shape)
        u.derivative(u_y_key, one_sided_at_boundary=one_sided_at_boundary,
                     inplace=True)

        radius = Data.asdata(radius).squeeze()
        radius.dtype = float
        if radius.size != 1:
            raise ValueError("Multiple radii: radius={!r}".format(radius))

        if not radius.Units:
            radius.override_units(Units('metres'), inplace=True)
        elif not radius.Units.equivalent(Units('metres')):
            raise ValueError(
                "Invalid units for radius: {!r}".format(radius.Units))

        # Calculate the relative vorticity. Do v-(u-corr) rather than
        # v-u+corr to be nice with coordinate reference corner cases.
        rv = v - (u - corr)
        rv.data /= radius

        # Convert the units of latitude and longitude to canonical units
        rv.dim('X').Units = x_units
        rv.dim('Y').Units = y_units

    else:
        v.derivative(
            v_x_key, one_sided_at_boundary=one_sided_at_boundary, inplace=True)
        u.derivative(
            u_y_key, one_sided_at_boundary=one_sided_at_boundary, inplace=True)

        rv = v - u

    # Convert the units of relative vorticity to canonical units
    rv.Units = Units('s-1')

    # Set the standard name if appropriate and delete the long_name
    if ((u_std_name == 'eastward_wind' and v_std_name == 'northward_wind') or
            (u_std_name == 'x_wind' and v_std_name == 'y_wind')):
        rv.standard_name = 'atmosphere_relative_vorticity'
    else:
        rv.del_property('standard_name', None)

    rv.del_property('long_name', None)

    return rv


def histogram(*digitized):
    '''Return the distribution of a set of variables in the form of an
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

    **Examples:**

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

    '''
    if not digitized:
        raise ValueError(
            "Must provide at least one 'digitized' field construct")

    f = digitized[0].copy()
    f.clear_properties()

    return f.bin('sample_size', digitized=digitized)
