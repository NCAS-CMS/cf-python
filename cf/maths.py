#from cfunits import Units

from .functions import _DEPRECATION_ERROR_FUNCTION_KWARGS
from . import Units

from .data.data import Data


#def radius_of_earth():
#    '''
#
#Return a radius of the Earth.
#
#:Returns:
#
#    radius_of_earth: cf.Data
#
#    '''
#    return Data(6371229.0, 'meters')

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
        _DEPRECATION_ERROR_FUNCTION_KWARGS('relative_vorticity', {'cyclic': cyclic},
                                           "Use the 'wrap' keyword instead") # pragma: no cover
        
    # Get the standard names of u and v
    u_std_name = u.get_property('standard_name', None)
    v_std_name = v.get_property('standard_name', None)

    # Copy u and v
    u = u.copy()
    v = v.copy()
    
    # Get the X and Y coordinates
    (u_x_key, u_y_key), (u_x, u_y) = u._regrid_get_cartesian_coords('u', ('X', 'Y'))
    (v_x_key, v_y_key), (v_x, v_y) = v._regrid_get_cartesian_coords('v', ('X', 'Y'))

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
    #--- End: if

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
        v.derivative(v_x_key, one_sided_at_boundary=one_sided_at_boundary, inplace=True)
        u.derivative(u_y_key, one_sided_at_boundary=one_sided_at_boundary, inplace=True)

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

