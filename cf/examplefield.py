from . import AuxiliaryCoordinate
from . import Bounds
from . import CellMeasure
from . import CellMethod
from . import CoordinateReference
from . import DimensionCoordinate
from . import DomainAncillary
from . import DomainAxis
from . import FieldAncillary
from . import Field

from .data import Data

from .constants import masked


def example_field(n):
    '''Return an example field construct.
    
    .. versionadded:: 3.0.5

    .. seealso:: `cf.Field.creation_commands`

    :Parameters:

        n: `int`
            Select the example field construct to return, one of:

            =====  ===================================================
            *n*    Description
            =====  ===================================================
            ``0``  A field construct with properties as well as a
                   cell method constuct and dimension coordinate
                   constructs with bounds.

            ``1``  A field construct with properties as well as at
                   least one of every type of metadata construct.

            ``2``  A field construct that contains a monthly time
                   series at each latitude-longitude location.

            ``3``  A field construct that contains discrete sampling
                   geometry (DSG) "timeSeries" features.

            ``4``  A field construct that contains discrete sampling
                   geometry (DSG) "timeSeriesProfile" features.

            ``5``  A field construct that contains a 12 hourly time
                   series at each latitude-longitude location.
            =====  ===================================================

            See the examples for details.

    :Returns:

        `Field`
            The example field construct.

    **Examples:**

    >>> f = cf.example_field(0)
    >>> print(f)
    Field: specific_humidity(ncvar%q)
    ---------------------------------
    Data            : specific_humidity(latitude(5), longitude(8)) 1
    Cell methods    : area: mean
    Dimension coords: time(1) = [2019-01-01 00:00:00]
                    : latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
    >>> print(f.array)
    [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
     [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
     [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
     [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
     [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]

    >>> f = cf.example_field(1)
    >>> print(f)
    Field: air_temperature (ncvar%ta)
    ---------------------------------
    Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K
    Cell methods    : grid_latitude(10): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
    Field ancils    : air_temperature standard_error(grid_latitude(10), grid_longitude(9)) = [[0.76, ..., 0.32]] K
    Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                    : grid_latitude(10) = [2.2, ..., -1.76] degrees
                    : grid_longitude(9) = [-4.7, ..., -1.18] degrees
                    : time(1) = [2019-01-01 00:00:00]
    Auxiliary coords: latitude(grid_latitude(10), grid_longitude(9)) = [[53.941, ..., 50.225]] degrees_N
                    : longitude(grid_longitude(9), grid_latitude(10)) = [[2.004, ..., 8.156]] degrees_E
                    : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., b'kappa']
    Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
    Coord references: grid_mapping_name:rotated_latitude_longitude
                    : standard_name:atmosphere_hybrid_height_coordinate
    Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                    : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                    : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m

    >>> f = cf.example_field(2)
    >>> print(f)
    Field: air_potential_temperature (ncvar%air_potential_temperature)
    ------------------------------------------------------------------
    Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
    Cell methods    : area: mean
    Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                    : latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : air_pressure(1) = [850.0] hPa

    >>> f = cf.example_field(3)
    >>> print(f)     
    Field: precipitation_flux (ncvar%p)
    -----------------------------------
    Data            : precipitation_flux(cf_role=timeseries_id(4), ncdim%timeseries(9)) kg m-2 day-1
    Auxiliary coords: time(cf_role=timeseries_id(4), ncdim%timeseries(9)) = [[1969-12-29 00:00:00, ..., 1970-01-07 00:00:00]]
                    : latitude(cf_role=timeseries_id(4)) = [-9.0, ..., 78.0] degrees_north
                    : longitude(cf_role=timeseries_id(4)) = [-23.0, ..., 178.0] degrees_east
                    : height(cf_role=timeseries_id(4)) = [0.5, ..., 345.0] m
                    : cf_role=timeseries_id(cf_role=timeseries_id(4)) = [b'station1', ..., b'station4']
                    : long_name=station information(cf_role=timeseries_id(4)) = [-10, ..., -7]

    >>> f = cf.example_field(4)
    >>> print(f)
    Field: air_temperature (ncvar%ta)
    ---------------------------------
    Data            : air_temperature(cf_role=timeseries_id(3), ncdim%timeseries(26), ncdim%profile_1(4)) K
    Auxiliary coords: time(cf_role=timeseries_id(3), ncdim%timeseries(26)) = [[1970-01-04 00:00:00, ..., --]]
                    : latitude(cf_role=timeseries_id(3)) = [-9.0, 2.0, 34.0] degrees_north
                    : longitude(cf_role=timeseries_id(3)) = [-23.0, 0.0, 67.0] degrees_east
                    : height(cf_role=timeseries_id(3)) = [0.5, 12.6, 23.7] m
                    : altitude(cf_role=timeseries_id(3), ncdim%timeseries(26), ncdim%profile_1(4)) = [[[2.07, ..., --]]] km
                    : cf_role=timeseries_id(cf_role=timeseries_id(3)) = [b'station1', b'station2', b'station3']
                    : long_name=station information(cf_role=timeseries_id(3)) = [-10, -9, -8]
                    : cf_role=profile_id(cf_role=timeseries_id(3), ncdim%timeseries(26)) = [[102, ..., --]]

    >>> f = cf.example_field(5)
    >>> print(f)
    Field: air_potential_temperature (ncvar%air_potential_temperature)
    ------------------------------------------------------------------
    Data            : air_potential_temperature(time(118), latitude(5), longitude(8)) K
    Cell methods    : area: mean
    Dimension coords: time(118) = [1959-01-01 06:00:00, ..., 1959-02-28 18:00:00]
                    : latitude(5) = [-75.0, ..., 75.0] degrees_north
                    : longitude(8) = [22.5, ..., 337.5] degrees_east
                    : air_pressure(1) = [850.0] hPa

    '''
    if n == 0:
        f = Field()

        f.set_properties({'Conventions': 'CF-1.7', 'project': 'research', 'standard_name': 'specific_humidity', 'units': '1'})
        f.nc_set_variable('q')
        
        c = DomainAxis(size=5)
        c.nc_set_dimension('lat')
        f.set_construct(c, key='domainaxis0')
        c = DomainAxis(size=8)
        c.nc_set_dimension('lon')
        f.set_construct(c, key='domainaxis1')
        c = DomainAxis(size=1)
        f.set_construct(c, key='domainaxis2')
        
        data = Data([[0.007, 0.034, 0.003, 0.014, 0.018, 0.037, 0.024, 0.029], [0.023, 0.036, 0.045, 0.062, 0.046, 0.073, 0.006, 0.066], [0.11, 0.131, 0.124, 0.146, 0.087, 0.103, 0.057, 0.011], [0.029, 0.059, 0.039, 0.07, 0.058, 0.072, 0.009, 0.017], [0.006, 0.036, 0.019, 0.035, 0.018, 0.037, 0.034, 0.013]], units='1', dtype='f8')
        f.set_data(data, axes=('domainaxis0', 'domainaxis1'))
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'degrees_north', 'standard_name': 'latitude'})
        c.nc_set_variable('lat')
        data = Data([-75.0, -45.0, 0.0, 45.0, 75.0], units='degrees_north', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'degrees_north'})
        b.nc_set_variable('lat_bnds')
        data = Data([[-90.0, -60.0], [-60.0, -30.0], [-30.0, 30.0], [30.0, 60.0], [60.0, 90.0]], units='degrees_north', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis0',), key='dimensioncoordinate0', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'degrees_east', 'standard_name': 'longitude'})
        c.nc_set_variable('lon')
        data = Data([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5], units='degrees_east', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'degrees_east'})
        b.nc_set_variable('lon_bnds')
        data = Data([[0.0, 45.0], [45.0, 90.0], [90.0, 135.0], [135.0, 180.0], [180.0, 225.0], [225.0, 270.0], [270.0, 315.0], [315.0, 360.0]], units='degrees_east', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis1',), key='dimensioncoordinate1', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'days since 2018-12-01', 'standard_name': 'time'})
        c.nc_set_variable('time')
        data = Data([31.0], units='days since 2018-12-01', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis2',), key='dimensioncoordinate2', copy=False)
        
        # cell_method
        c = CellMethod()
        c.method = 'mean'
        c.axes = ('area',)
        f.set_construct(c)
        
    elif n == 1:
        f = Field()
        
        f.set_properties({'Conventions': 'CF-1.7', 'project': 'research', 'standard_name': 'air_temperature', 'units': 'K'})
        f.nc_set_variable('ta')
        
        c = DomainAxis(size=1)
        c.nc_set_dimension('atmosphere_hybrid_height_coordinate')
        f.set_construct(c, key='domainaxis0')
        c = DomainAxis(size=10)
        c.nc_set_dimension('y')
        f.set_construct(c, key='domainaxis1')
        c = DomainAxis(size=9)
        c.nc_set_dimension('x')
        f.set_construct(c, key='domainaxis2')
        c = DomainAxis(size=1)
        f.set_construct(c, key='domainaxis3')
        
        data = Data([[[262.8, 270.5, 279.8, 269.5, 260.9, 265.0, 263.5, 278.9, 269.2], [272.7, 268.4, 279.5, 278.9, 263.8, 263.3, 274.2, 265.7, 279.5], [269.7, 279.1, 273.4, 274.2, 279.6, 270.2, 280.0, 272.5, 263.7], [261.7, 260.6, 270.8, 260.3, 265.6, 279.4, 276.9, 267.6, 260.6], [264.2, 275.9, 262.5, 264.9, 264.7, 270.2, 270.4, 268.6, 275.3], [263.9, 263.8, 272.1, 263.7, 272.2, 264.2, 260.0, 263.5, 270.2], [273.8, 273.1, 268.5, 272.3, 264.3, 278.7, 270.6, 273.0, 270.6], [267.9, 273.5, 279.8, 260.3, 261.2, 275.3, 271.2, 260.8, 268.9], [270.9, 278.7, 273.2, 261.7, 271.6, 265.8, 273.0, 278.5, 266.4], [276.4, 264.2, 276.3, 266.1, 276.1, 268.1, 277.0, 273.4, 269.7]]], units='K', dtype='f8')
        f.set_data(data, axes=('domainaxis0', 'domainaxis1', 'domainaxis2'))
        
        # domain_ancillary
        c = DomainAncillary()
        c.set_properties({'units': 'm'})
        c.nc_set_variable('a')
        data = Data([10.0], units='m', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'm'})
        b.nc_set_variable('a_bounds')
        data = Data([[5.0, 15.0]], units='m', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis0',), key='domainancillary0', copy=False)
        
        # domain_ancillary
        c = DomainAncillary()
        c.nc_set_variable('b')
        data = Data([20.0], dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.nc_set_variable('b_bounds')
        data = Data([[14.0, 26.0]], dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis0',), key='domainancillary1', copy=False)
        
        # domain_ancillary
        c = DomainAncillary()
        c.set_properties({'units': 'm', 'standard_name': 'surface_altitude'})
        c.nc_set_variable('surface_altitude')
        data = Data([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 10.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 52.0, 40.0], [0.0, 0.0, 0.0, 7.0, 12.0, 8.0, 37.0, 73.0, 107.0], [0.0, 0.0, 28.0, 30.0, 30.0, 30.0, 83.0, 102.0, 164.0], [34.0, 38.0, 34.0, 32.0, 30.0, 31.0, 105.0, 281.0, 370.0], [91.0, 89.0, 95.0, 94.0, 132.0, 194.0, 154.0, 318.0, 357.0], [93.0, 114.0, 116.0, 178.0, 323.0, 365.0, 307.0, 289.0, 270.0]], units='m', dtype='f4')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis1', 'domainaxis2'), key='domainancillary2', copy=False)
        
        # cell_measure
        c = CellMeasure()
        c.set_properties({'units': 'km2'})
        c.nc_set_variable('cell_measure')
        data = Data([[2391.9657, 2391.9657, 2391.9657, 2391.9657, 2391.9657, 2391.9657, 2391.9657, 2391.9657, 2391.9657, 2392.6009], [2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2393.0949, 2393.0949], [2393.0949, 2393.0949, 2393.0949, 2393.0949, 2393.0949, 2393.0949, 2393.0949, 2393.4478, 2393.4478, 2393.4478], [2393.4478, 2393.4478, 2393.4478, 2393.4478, 2393.4478, 2393.4478, 2393.6595, 2393.6595, 2393.6595, 2393.6595], [2393.6595, 2393.6595, 2393.6595, 2393.6595, 2393.6595, 2393.7301, 2393.7301, 2393.7301, 2393.7301, 2393.7301], [2393.7301, 2393.7301, 2393.7301, 2393.7301, 2393.6595, 2393.6595, 2393.6595, 2393.6595, 2393.6595, 2393.6595], [2393.6595, 2393.6595, 2393.6595, 2393.4478, 2393.4478, 2393.4478, 2393.4478, 2393.4478, 2393.4478, 2393.4478], [2393.4478, 2393.4478, 2393.0949, 2393.0949, 2393.0949, 2393.0949, 2393.0949, 2393.0949, 2393.0949, 2393.0949], [2393.0949, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009, 2392.6009]], units='km2', dtype='f8')
        c.set_data(data)
        c.set_measure('area')
        f.set_construct(c, axes=('domainaxis2', 'domainaxis1'), key='cellmeasure0', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'units': 'degrees_N', 'standard_name': 'latitude'})
        c.nc_set_variable('latitude_1')
        data = Data([[53.941, 53.987, 54.029, 54.066, 54.099, 54.127, 54.15, 54.169, 54.184], [53.504, 53.55, 53.591, 53.627, 53.66, 53.687, 53.711, 53.729, 53.744], [53.067, 53.112, 53.152, 53.189, 53.221, 53.248, 53.271, 53.29, 53.304], [52.629, 52.674, 52.714, 52.75, 52.782, 52.809, 52.832, 52.85, 52.864], [52.192, 52.236, 52.276, 52.311, 52.343, 52.37, 52.392, 52.41, 52.424], [51.754, 51.798, 51.837, 51.873, 51.904, 51.93, 51.953, 51.971, 51.984], [51.316, 51.36, 51.399, 51.434, 51.465, 51.491, 51.513, 51.531, 51.545], [50.879, 50.922, 50.96, 50.995, 51.025, 51.052, 51.074, 51.091, 51.105], [50.441, 50.484, 50.522, 50.556, 50.586, 50.612, 50.634, 50.652, 50.665], [50.003, 50.045, 50.083, 50.117, 50.147, 50.173, 50.194, 50.212, 50.225]], units='degrees_N', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis1', 'domainaxis2'), key='auxiliarycoordinate0', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'units': 'degrees_E', 'standard_name': 'longitude'})
        c.nc_set_variable('longitude_1')
        data = Data([[2.004, 2.747, 3.492, 4.238, 4.986, 5.734, 6.484, 7.234, 7.985, 2.085], [2.821, 3.558, 4.297, 5.037, 5.778, 6.52, 7.262, 8.005, 2.165, 2.893], [3.623, 4.355, 5.087, 5.821, 6.555, 7.29, 8.026, 2.243, 2.964, 3.687], [4.411, 5.136, 5.862, 6.589, 7.317, 8.045, 2.319, 3.033, 3.749, 4.466], [5.184, 5.903, 6.623, 7.344, 8.065, 2.394, 3.101, 3.81, 4.52, 5.231], [5.944, 6.656, 7.37, 8.084, 2.467, 3.168, 3.87, 4.573, 5.278, 5.983], [6.689, 7.395, 8.102, 2.539, 3.233, 3.929, 4.626, 5.323, 6.022, 6.721], [7.42, 8.121, 2.61, 3.298, 3.987, 4.677, 5.368, 6.059, 6.752, 7.445], [8.139, 2.679, 3.361, 4.043, 4.727, 5.411, 6.097, 6.783, 7.469, 8.156]], units='degrees_E', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis2', 'domainaxis1'), key='auxiliarycoordinate1', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'long_name': 'Grid latitude name'})
        c.nc_set_variable('auxiliary')
        data = Data([b'', b'beta', b'gamma', b'delta', b'epsilon', b'zeta', b'eta', b'theta', b'iota', b'kappa'], dtype='S7')
        data_mask = Data([True, False, False, False, False, False, False, False, False, False], dtype='b1')
        data.where(data_mask, masked, inplace=True)
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis1',), key='auxiliarycoordinate2', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'computed_standard_name': 'altitude', 'standard_name': 'atmosphere_hybrid_height_coordinate'})
        c.nc_set_variable('atmosphere_hybrid_height_coordinate')
        data = Data([1.5], dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.nc_set_variable('atmosphere_hybrid_height_coordinate_bounds')
        data = Data([[1.0, 2.0]], dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis0',), key='dimensioncoordinate0', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'degrees', 'standard_name': 'grid_latitude'})
        c.nc_set_variable('y')
        data = Data([2.2, 1.76, 1.32, 0.88, 0.44, 0.0, -0.44, -0.88, -1.32, -1.76], units='degrees', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'degrees'})
        b.nc_set_variable('y_bnds')
        data = Data([[2.42, 1.98], [1.98, 1.54], [1.54, 1.1], [1.1, 0.66], [0.66, 0.22], [0.22, -0.22], [-0.22, -0.66], [-0.66, -1.1], [-1.1, -1.54], [-1.54, -1.98]], units='degrees', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis1',), key='dimensioncoordinate1', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'degrees', 'standard_name': 'grid_longitude'})
        c.nc_set_variable('x')
        data = Data([-4.7, -4.26, -3.82, -3.38, -2.94, -2.5, -2.06, -1.62, -1.18], units='degrees', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'degrees'})
        b.nc_set_variable('x_bnds')
        data = Data([[-4.92, -4.48], [-4.48, -4.04], [-4.04, -3.6], [-3.6, -3.16], [-3.16, -2.72], [-2.72, -2.28], [-2.28, -1.84], [-1.84, -1.4], [-1.4, -0.96]], units='degrees', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis2',), key='dimensioncoordinate2', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'days since 2018-12-01', 'standard_name': 'time'})
        c.nc_set_variable('time')
        data = Data([31.0], units='days since 2018-12-01', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis3',), key='dimensioncoordinate3', copy=False)
        
        # field_ancillary
        c = FieldAncillary()
        c.set_properties({'units': 'K', 'standard_name': 'air_temperature standard_error'})
        c.nc_set_variable('air_temperature_standard_error')
        data = Data([[0.76, 0.38, 0.68, 0.19, 0.14, 0.52, 0.57, 0.19, 0.81], [0.59, 0.68, 0.25, 0.13, 0.37, 0.12, 0.26, 0.45, 0.36], [0.88, 0.4, 0.35, 0.87, 0.24, 0.64, 0.78, 0.28, 0.11], [0.73, 0.49, 0.69, 0.54, 0.17, 0.6, 0.82, 0.89, 0.71], [0.43, 0.39, 0.45, 0.74, 0.85, 0.47, 0.37, 0.87, 0.46], [0.47, 0.31, 0.76, 0.69, 0.61, 0.26, 0.43, 0.75, 0.23], [0.43, 0.26, 0.5, 0.79, 0.25, 0.63, 0.25, 0.24, 0.74], [0.33, 0.26, 0.89, 0.48, 0.79, 0.88, 0.41, 0.89, 0.47], [0.25, 0.42, 0.61, 0.87, 0.58, 0.89, 0.58, 0.8, 0.32], [0.49, 0.48, 0.49, 0.16, 0.65, 0.66, 0.86, 0.74, 0.32]], units='K', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis1', 'domainaxis2'), key='fieldancillary0', copy=False)
        
        # cell_method
        c = CellMethod()
        c.method = 'mean'
        c.axes = ('domainaxis1', 'domainaxis2')
        c.set_qualifier('where', 'land')
        interval0 = Data(0.1, units='degrees', dtype='f8')
        c.set_qualifier('interval', [interval0])
        f.set_construct(c)
        
        # cell_method
        c = CellMethod()
        c.method = 'maximum'
        c.axes = ('domainaxis3',)
        f.set_construct(c)
        
        # coordinate_reference
        c = CoordinateReference()
        c.set_coordinates({'dimensioncoordinate0'})
        c.datum.set_parameter('earth_radius', 6371007)
        c.coordinate_conversion.set_parameter('standard_name', 'atmosphere_hybrid_height_coordinate')
        c.coordinate_conversion.set_parameter('computed_standard_name', 'altitude')
        c.coordinate_conversion.set_domain_ancillaries({'a': 'domainancillary0', 'b': 'domainancillary1', 'orog': 'domainancillary2'})
        f.set_construct(c)
        
        # coordinate_reference
        c = CoordinateReference()
        c.nc_set_variable('rotated_latitude_longitude')
        c.set_coordinates({'dimensioncoordinate2', 'auxiliarycoordinate1', 'dimensioncoordinate1', 'auxiliarycoordinate0'})
        c.datum.set_parameter('earth_radius', 6371007)
        c.coordinate_conversion.set_parameter('grid_north_pole_latitude', 38.0)
        c.coordinate_conversion.set_parameter('grid_north_pole_longitude', 190.0)
        c.coordinate_conversion.set_parameter('grid_mapping_name', 'rotated_latitude_longitude')
        f.set_construct(c)

    elif n == 3:
                                          
        f = Field()
        
        f.set_properties({'Conventions': 'CF-1.7', 'featureType': 'timeSeries', '_FillValue': -999.9, 'standard_name': 'precipitation_flux', 'units': 'kg m-2 day-1'})
        f.nc_set_variable('p')
        f.nc_set_global_attributes({'Conventions': None, 'featureType': None})
        
        # domain_axis
        c = DomainAxis(size=4)
        c.nc_set_dimension('station')
        f.set_construct(c, key='domainaxis0')
        
        # domain_axis
        c = DomainAxis(size=9)
        c.nc_set_dimension('timeseries')
        f.set_construct(c, key='domainaxis1')
        
        # field data
        data_mask = Data([[False, False, False, True, True, True, True, True, True], [False, False, False, False, False, False, False, True, True], [False, False, False, False, False, True, True, True, True], [False, False, False, False, False, False, False, False, False]], dtype='b1')
        data = Data([[3.98, 0.0, 0.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.0, 0.0, 0.0, 3.4, 0.0, 0.0, 4.61, 9.969209968386869e+36, 9.969209968386869e+36], [0.86, 0.8, 0.75, 0.0, 4.56, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.0, 0.09, 0.0, 0.91, 2.96, 1.14, 3.86, 0.0, 0.0]], units='kg m-2 day-1', dtype='f8', mask=data_mask)
        f.set_data(data, axes=('domainaxis0', 'domainaxis1'))
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'standard_name': 'time', 'long_name': 'time of measurement', 'units': 'days since 1970-01-01 00:00:00'})
        c.nc_set_variable('time')
        data_mask = Data([[False, False, False, True, True, True, True, True, True], [False, False, False, False, False, False, False, True, True], [False, False, False, False, False, True, True, True, True], [False, False, False, False, False, False, False, False, False]], dtype='b1')
        data = Data([[-3.0, -2.0, -1.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.969209968386869e+36, 9.969209968386869e+36], [0.5, 1.5, 2.5, 3.5, 4.5, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], units='days since 1970-01-01 00:00:00', dtype='f8', mask=data_mask)
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'days since 1970-01-01 00:00:00'})
        data_mask = Data([[[False, False], [False, False], [False, False], [True, True], [True, True], [True, True], [True, True], [True, True], [True, True]], [[False, False], [False, False], [False, False], [False, False], [False, False], [False, False], [False, False], [True, True], [True, True]], [[False, False], [False, False], [False, False], [False, False], [False, False], [True, True], [True, True], [True, True], [True, True]], [[False, False], [False, False], [False, False], [False, False], [False, False], [False, False], [False, False], [False, False], [False, False]]], dtype='b1')
        data = Data([[[-3.5, -2.5], [-2.5, -1.5], [-1.5, -0.5], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36]], [[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5], [5.5, 6.5], [6.5, 7.5], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36]], [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36]], [[-2.5, -1.5], [-1.5, -0.5], [-0.5, 0.5], [0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5], [5.5, 6.5]]], units='days since 1970-01-01 00:00:00', dtype='f8', mask=data_mask)
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis0', 'domainaxis1'), key='auxiliarycoordinate0', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'standard_name': 'latitude', 'long_name': 'station latitude', 'units': 'degrees_north'})
        c.nc_set_variable('lat')
        data = Data([-9.0, 2.0, 34.0, 78.0], units='degrees_north', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate1', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'standard_name': 'longitude', 'long_name': 'station longitude', 'units': 'degrees_east'})
        c.nc_set_variable('lon')
        data = Data([-23.0, 0.0, 67.0, 178.0], units='degrees_east', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate2', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'long_name': 'vertical distance above the surface', 'standard_name': 'height', 'units': 'm', 'positive': 'up', 'axis': 'Z'})
        c.nc_set_variable('alt')
        data = Data([0.5, 12.6, 23.7, 345.0], units='m', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate3', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'long_name': 'station name', 'cf_role': 'timeseries_id'})
        c.nc_set_variable('station_name')
        data = Data([b'station1', b'station2', b'station3', b'station4'], dtype='S8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate4', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'long_name': 'station information'})
        c.nc_set_variable('station_info')
        data = Data([-10, -9, -8, -7], dtype='i4')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate5', copy=False)

    elif n == 4:
        f = Field()
        
        f.set_properties({'Conventions': 'CF-1.7', 'featureType': 'timeSeriesProfile', '_FillValue': -999.9, 'standard_name': 'air_temperature', 'units': 'K'})
        f.nc_set_variable('ta')
        f.nc_set_global_attribute('Conventions', None)
        f.nc_set_global_attribute('featureType', None)
        
        # domain_axis
        c = DomainAxis(size=3)
        c.nc_set_dimension('station')
        f.set_construct(c, key='domainaxis0')
        
        # domain_axis
        c = DomainAxis(size=26)
        c.nc_set_dimension('timeseries')
        f.set_construct(c, key='domainaxis1')
        
        # domain_axis
        c = DomainAxis(size=4)
        c.nc_set_dimension('profile_1')
        f.set_construct(c, key='domainaxis2')
        
        # field data
        data = Data([[[290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [293.15, 288.84, 280.0, 9.969209968386869e+36], [291.65, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [290.45, 286.14, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [291.65, 288.57, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [293.27, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [293.36, 285.99, 285.46, 9.969209968386869e+36], [291.2, 285.96, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36]], [[291.74, 285.72, 283.21, 275.0], [290.0, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [290.15, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [291.08, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [291.32, 288.66, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 294.18, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 286.05, 280.0, 9.969209968386869e+36], [291.23, 285.0, 281.11, 9.969209968386869e+36], [295.88, 286.83, 285.01, 9.969209968386869e+36], [292.37, 285.6, 280.0, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [300.11, 285.0, 280.0, 9.969209968386869e+36], [290.0, 287.4, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [291.5, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [294.98, 290.64, 9.969209968386869e+36, 9.969209968386869e+36], [290.66, 292.92, 280.0, 9.969209968386869e+36], [290.24, 285.36, 280.36, 9.969209968386869e+36], [290.0, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [292.79, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 287.22, 280.0, 9.969209968386869e+36], [290.0, 286.14, 280.0, 9.969209968386869e+36]], [[291.74, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [291.44, 287.25, 280.0, 9.969209968386869e+36], [292.76, 285.0, 280.0, 9.969209968386869e+36], [291.59, 286.71, 284.47, 9.969209968386869e+36], [292.19, 286.35, 9.969209968386869e+36, 9.969209968386869e+36], [295.67, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [290.45, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [293.69, 285.9, 280.03, 9.969209968386869e+36], [290.0, 285.27, 280.87, 9.969209968386869e+36], [290.0, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [290.12, 286.44, 282.01, 9.969209968386869e+36], [291.23, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [292.97, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [290.0, 286.71, 9.969209968386869e+36, 9.969209968386869e+36], [292.01, 285.0, 9.969209968386869e+36, 9.969209968386869e+36], [294.62, 285.33, 282.01, 9.969209968386869e+36], [290.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [292.64, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36]]], units='K', dtype='f8')
        data_mask = Data([[[False, True, True, True], [False, False, False, True], [False, False, True, True], [False, False, True, True], [False, False, True, True], [False, True, True, True], [False, False, True, True], [False, True, True, True], [False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True]], [[False, False, False, False], [False, False, True, True], [False, False, True, True], [False, True, True, True], [False, False, True, True], [False, False, True, True], [False, False, True, True], [False, True, True, True], [False, False, False, True], [False, False, False, True], [False, False, False, True], [False, False, False, True], [False, True, True, True], [False, False, False, True], [False, False, True, True], [False, True, True, True], [False, True, True, True], [False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, False, True], [False, False, True, True], [False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, False, True]], [[False, True, True, True], [False, False, False, True], [False, False, False, True], [False, False, False, True], [False, False, True, True], [False, False, True, True], [False, True, True, True], [False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, False, True], [False, False, True, True], [False, False, False, True], [False, True, True, True], [False, True, True, True], [False, False, True, True], [False, False, True, True], [False, False, False, True], [False, True, True, True], [False, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True]]], dtype='b1')
        data.where(data_mask, masked, inplace=True)
        f.set_data(data, axes=('domainaxis0', 'domainaxis1', 'domainaxis2'))
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'standard_name': 'time', 'long_name': 'time', 'units': 'days since 1970-01-01 00:00:00'})
        c.nc_set_variable('time')
        data = Data([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0], [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36]], units='days since 1970-01-01 00:00:00', dtype='f8')
        data_mask = Data([[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]], dtype='b1')
        data.where(data_mask, masked, inplace=True)
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0', 'domainaxis1'), key='auxiliarycoordinate0', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'standard_name': 'latitude', 'long_name': 'station latitude', 'units': 'degrees_north'})
        c.nc_set_variable('lat')
        data = Data([-9.0, 2.0, 34.0], units='degrees_north', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate1', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'standard_name': 'longitude', 'long_name': 'station longitude', 'units': 'degrees_east'})
        c.nc_set_variable('lon')
        data = Data([-23.0, 0.0, 67.0], units='degrees_east', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate2', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'long_name': 'vertical distance above the surface', 'standard_name': 'height', 'units': 'm', 'positive': 'up', 'axis': 'Z'})
        c.nc_set_variable('alt')
        data = Data([0.5, 12.6, 23.7], units='m', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate3', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'standard_name': 'altitude', 'long_name': 'height above mean sea level', 'units': 'km', 'axis': 'Z', 'positive': 'up'})
        c.nc_set_variable('z')
        data = Data([[[2.07, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [1.01, 1.18, 1.82, 9.969209968386869e+36], [1.1, 1.18, 9.969209968386869e+36, 9.969209968386869e+36], [1.63, 2.0, 9.969209968386869e+36, 9.969209968386869e+36], [1.38, 1.83, 9.969209968386869e+36, 9.969209968386869e+36], [1.59, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [1.57, 2.12, 9.969209968386869e+36, 9.969209968386869e+36], [2.25, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [1.8, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [1.26, 2.17, 9.969209968386869e+36, 9.969209968386869e+36], [1.05, 1.29, 2.1, 9.969209968386869e+36], [1.6, 1.97, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36]], [[0.52, 0.58, 1.08, 1.38], [0.26, 0.92, 9.969209968386869e+36, 9.969209968386869e+36], [0.07, 0.4, 9.969209968386869e+36, 9.969209968386869e+36], [1.57, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.25, 1.6, 9.969209968386869e+36, 9.969209968386869e+36], [0.46, 0.98, 9.969209968386869e+36, 9.969209968386869e+36], [0.06, 0.31, 9.969209968386869e+36, 9.969209968386869e+36], [0.38, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.57, 1.29, 1.81, 9.969209968386869e+36], [0.39, 0.69, 1.69, 9.969209968386869e+36], [0.73, 1.38, 1.6, 9.969209968386869e+36], [0.45, 0.98, 1.13, 9.969209968386869e+36], [0.15, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.09, 0.43, 0.62, 9.969209968386869e+36], [0.17, 0.99, 9.969209968386869e+36, 9.969209968386869e+36], [0.93, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.07, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [1.57, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.07, 0.12, 9.969209968386869e+36, 9.969209968386869e+36], [0.45, 1.24, 1.3, 9.969209968386869e+36], [0.35, 0.68, 0.79, 9.969209968386869e+36], [0.81, 1.22, 9.969209968386869e+36, 9.969209968386869e+36], [0.59, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [0.1, 0.96, 9.969209968386869e+36, 9.969209968386869e+36], [0.56, 0.78, 0.91, 9.969209968386869e+36], [0.71, 0.9, 1.04, 9.969209968386869e+36]], [[3.52, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [3.47, 3.89, 4.81, 9.969209968386869e+36], [3.52, 3.93, 3.96, 9.969209968386869e+36], [4.03, 4.04, 4.8, 9.969209968386869e+36], [3.0, 3.65, 9.969209968386869e+36, 9.969209968386869e+36], [3.33, 4.33, 9.969209968386869e+36, 9.969209968386869e+36], [3.77, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [3.35, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [3.19, 3.37, 9.969209968386869e+36, 9.969209968386869e+36], [3.41, 3.54, 4.1, 9.969209968386869e+36], [3.02, 3.37, 3.87, 9.969209968386869e+36], [3.24, 4.24, 9.969209968386869e+36, 9.969209968386869e+36], [3.32, 3.49, 3.97, 9.969209968386869e+36], [3.32, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [3.85, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [3.73, 3.99, 9.969209968386869e+36, 9.969209968386869e+36], [3.0, 3.91, 9.969209968386869e+36, 9.969209968386869e+36], [3.64, 3.91, 4.56, 9.969209968386869e+36], [4.1, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [3.11, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36], [9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36, 9.969209968386869e+36]]], units='km', dtype='f8')
        data_mask = Data([[[False, True, True, True], [False, False, False, True], [False, False, True, True], [False, False, True, True], [False, False, True, True], [False, True, True, True], [False, False, True, True], [False, True, True, True], [False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True]], [[False, False, False, False], [False, False, True, True], [False, False, True, True], [False, True, True, True], [False, False, True, True], [False, False, True, True], [False, False, True, True], [False, True, True, True], [False, False, False, True], [False, False, False, True], [False, False, False, True], [False, False, False, True], [False, True, True, True], [False, False, False, True], [False, False, True, True], [False, True, True, True], [False, True, True, True], [False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, False, True], [False, False, True, True], [False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, False, True]], [[False, True, True, True], [False, False, False, True], [False, False, False, True], [False, False, False, True], [False, False, True, True], [False, False, True, True], [False, True, True, True], [False, True, True, True], [False, False, True, True], [False, False, False, True], [False, False, False, True], [False, False, True, True], [False, False, False, True], [False, True, True, True], [False, True, True, True], [False, False, True, True], [False, False, True, True], [False, False, False, True], [False, True, True, True], [False, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True], [True, True, True, True]]], dtype='b1')
        data.where(data_mask, masked, inplace=True)
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0', 'domainaxis1', 'domainaxis2'), key='auxiliarycoordinate4', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'long_name': 'station name', 'cf_role': 'timeseries_id'})
        c.nc_set_variable('station_name')
        data = Data([b'station1', b'station2', b'station3'], dtype='S8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate5', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'long_name': 'station information'})
        c.nc_set_variable('station_info')
        data = Data([-10, -9, -8], dtype='i4')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0',), key='auxiliarycoordinate6', copy=False)
        
        # auxiliary_coordinate
        c = AuxiliaryCoordinate()
        c.set_properties({'cf_role': 'profile_id'})
        c.nc_set_variable('profile')
        data = Data([[102, 106, 109, 117, 121, 124, 132, 136, 139, 147, 151, 154, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647], [101, 104, 105, 108, 110, 113, 114, 116, 119, 120, 123, 125, 128, 129, 131, 134, 135, 138, 140, 143, 144, 146, 149, 150, 153, 155], [100, 103, 107, 111, 112, 115, 118, 122, 126, 127, 130, 133, 137, 141, 142, 145, 148, 152, 156, 157, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647, -2147483647]], dtype='i4')
        data_mask = Data([[False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]], dtype='b1')
        data.where(data_mask, masked, inplace=True)
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis0', 'domainaxis1'), key='auxiliarycoordinate7', copy=False)

    elif n == 2:
        f = Field()
        
        f.set_properties({'Conventions': 'CF-1.7', 'standard_name': 'air_potential_temperature', 'units': 'K'})
        f.nc_set_variable('air_potential_temperature')
        f.nc_set_global_attribute('Conventions', None)
        
        # domain_axis
        c = DomainAxis(size=36)
        c.nc_set_dimension('time')
        f.set_construct(c, key='domainaxis0')
        
        # domain_axis
        c = DomainAxis(size=5)
        c.nc_set_dimension('lat')
        f.set_construct(c, key='domainaxis1')
        
        # domain_axis
        c = DomainAxis(size=8)
        c.nc_set_dimension('lon')
        f.set_construct(c, key='domainaxis2')
        
        # domain_axis
        c = DomainAxis(size=1)
        f.set_construct(c, key='domainaxis3')
        
        # field data
        data = Data([[[210.7, 212.9, 282.7, 293.9, 264.0, 228.0, 211.6, 266.5], [212.8, 224.3, 301.9, 308.2, 237.1, 233.9, 200.1, 208.2], [234.9, 273.0, 245.1, 233.7, 262.9, 242.0, 254.7, 299.3], [204.1, 233.4, 278.1, 293.7, 232.8, 246.2, 268.0, 298.4], [309.5, 294.7, 209.8, 204.3, 224.8, 271.1, 292.2, 293.9]], [[305.3, 271.6, 300.7, 267.8, 226.0, 204.9, 202.6, 294.3], [291.7, 251.6, 311.1, 252.2, 297.1, 205.4, 289.5, 258.4], [300.9, 304.2, 206.3, 259.2, 223.6, 236.2, 275.6, 284.9], [226.4, 309.3, 223.7, 305.3, 283.4, 219.1, 303.8, 271.3], [234.6, 301.8, 261.1, 226.6, 261.9, 267.0, 297.5, 241.3]], [[249.4, 278.9, 308.2, 283.3, 283.2, 227.7, 288.3, 221.9], [270.7, 268.0, 306.4, 294.9, 242.0, 276.5, 218.0, 247.2], [229.8, 291.0, 246.1, 205.2, 232.7, 244.7, 235.4, 264.5], [239.1, 278.2, 299.1, 203.2, 308.9, 303.2, 309.0, 228.9], [203.2, 279.8, 248.5, 270.8, 234.0, 307.7, 236.4, 237.1]], [[288.9, 235.0, 279.9, 218.1, 241.4, 233.6, 238.0, 221.9], [235.0, 289.2, 283.8, 306.0, 219.7, 240.4, 199.0, 238.8], [281.6, 262.3, 251.7, 290.0, 279.7, 267.1, 304.2, 244.6], [298.3, 201.7, 251.8, 233.3, 253.5, 229.2, 228.7, 277.6], [213.5, 290.2, 202.6, 217.9, 244.9, 224.1, 286.4, 268.1]], [[231.1, 243.5, 199.9, 264.0, 282.2, 288.8, 269.4, 291.2], [216.6, 256.6, 201.9, 279.4, 211.0, 223.4, 205.8, 207.7], [230.2, 218.6, 248.6, 292.3, 253.3, 212.5, 205.9, 208.9], [273.5, 249.9, 257.5, 237.6, 250.7, 233.4, 255.6, 255.2], [274.0, 208.6, 283.6, 259.4, 242.0, 307.4, 251.2, 233.7]], [[200.0, 265.0, 273.4, 235.5, 297.4, 232.2, 266.0, 277.8], [260.3, 221.3, 243.1, 217.2, 205.8, 225.9, 227.8, 264.1], [231.1, 210.2, 258.6, 215.7, 221.1, 303.4, 283.0, 209.3], [260.5, 231.8, 212.0, 249.7, 202.5, 256.0, 291.2, 232.0], [282.3, 266.8, 207.4, 262.4, 303.2, 277.5, 263.5, 294.7]], [[234.4, 304.3, 226.2, 231.7, 261.9, 204.3, 287.5, 229.5], [264.4, 291.3, 289.0, 295.4, 250.5, 252.1, 275.0, 244.1], [245.2, 228.1, 227.2, 252.0, 307.9, 296.9, 247.9, 219.6], [302.0, 256.5, 298.4, 222.3, 285.8, 308.4, 225.5, 202.6], [308.8, 274.1, 215.4, 288.5, 230.5, 213.0, 310.2, 205.8]], [[289.2, 275.2, 241.5, 231.5, 261.6, 310.1, 235.5, 280.6], [273.7, 201.4, 290.0, 287.6, 220.2, 215.2, 215.1, 266.6], [290.8, 309.8, 278.5, 286.3, 278.6, 203.6, 231.7, 263.9], [231.1, 299.2, 301.8, 217.8, 286.1, 206.8, 254.8, 234.1], [238.3, 301.8, 244.9, 263.8, 202.2, 257.2, 245.8, 199.4]], [[204.3, 301.8, 247.5, 279.3, 276.3, 258.3, 252.2, 297.9], [261.3, 230.3, 277.2, 255.9, 286.6, 203.5, 288.3, 246.3], [281.8, 309.0, 241.4, 307.3, 261.0, 199.4, 311.1, 278.5], [259.2, 302.6, 283.3, 206.4, 206.5, 250.3, 249.0, 271.0], [274.2, 304.6, 252.5, 236.2, 244.3, 229.8, 221.1, 289.0]], [[203.6, 292.0, 201.0, 280.9, 238.8, 199.9, 200.3, 244.4], [244.9, 232.6, 204.9, 257.3, 265.7, 230.9, 231.6, 295.7], [281.6, 282.9, 271.7, 250.3, 217.4, 269.5, 219.5, 262.1], [308.2, 283.4, 259.5, 234.6, 248.6, 212.6, 262.7, 237.0], [258.0, 257.9, 272.7, 310.4, 291.0, 265.9, 205.4, 256.9]], [[261.8, 308.8, 303.2, 210.1, 281.0, 275.7, 200.6, 285.6], [199.8, 219.7, 248.1, 231.8, 217.9, 199.1, 272.8, 282.2], [264.1, 301.2, 269.9, 243.0, 223.8, 281.5, 247.8, 222.6], [273.6, 199.5, 256.0, 199.9, 234.7, 273.7, 285.0, 288.2], [306.7, 275.5, 301.5, 207.4, 278.4, 228.9, 245.3, 266.9]], [[256.2, 229.4, 296.5, 305.5, 202.0, 247.2, 254.9, 306.1], [251.3, 279.2, 215.4, 250.5, 204.3, 253.1, 275.8, 210.1], [306.9, 208.4, 267.3, 284.5, 226.5, 280.0, 252.6, 286.8], [293.9, 261.8, 262.9, 218.9, 238.7, 298.4, 311.2, 288.7], [277.6, 223.3, 224.4, 202.2, 274.1, 203.7, 225.3, 229.6]], [[212.3, 253.2, 257.3, 261.4, 268.9, 260.4, 255.5, 224.1], [208.3, 227.6, 296.5, 307.7, 297.6, 230.3, 300.7, 273.5], [268.9, 255.9, 220.3, 307.2, 274.5, 249.9, 284.0, 217.6], [285.4, 306.5, 203.9, 232.4, 306.1, 219.9, 272.1, 222.3], [220.6, 258.6, 307.9, 280.9, 310.4, 202.0, 237.3, 294.9]], [[231.7, 299.5, 217.4, 267.3, 278.6, 204.4, 234.7, 233.5], [266.0, 302.9, 215.8, 281.6, 254.0, 223.6, 248.1, 310.0], [281.4, 257.0, 269.8, 207.8, 286.4, 221.4, 239.3, 251.8], [237.9, 228.6, 289.9, 245.3, 232.9, 302.9, 278.2, 248.4], [252.2, 249.6, 290.7, 203.2, 293.1, 205.7, 302.5, 217.6]], [[255.1, 200.6, 268.4, 216.3, 246.5, 250.2, 292.9, 226.8], [297.8, 280.7, 271.8, 251.7, 298.1, 218.0, 295.2, 234.5], [231.8, 281.5, 305.0, 261.4, 222.9, 217.0, 211.6, 275.2], [218.0, 308.1, 221.3, 251.8, 252.1, 254.3, 270.6, 294.9], [299.6, 237.6, 216.5, 300.0, 286.2, 277.1, 242.6, 284.3]], [[213.9, 219.2, 212.0, 241.9, 276.3, 269.1, 298.9, 200.9], [274.2, 236.4, 218.4, 241.8, 208.6, 287.3, 219.0, 232.8], [254.0, 266.1, 307.4, 239.1, 252.1, 284.1, 210.7, 291.2], [200.4, 266.3, 298.3, 205.6, 305.1, 247.9, 285.4, 219.6], [284.4, 274.0, 216.4, 210.8, 201.0, 223.1, 279.1, 224.8]], [[255.8, 229.6, 292.6, 243.5, 304.0, 264.1, 285.4, 256.0], [250.5, 262.1, 263.1, 281.3, 299.9, 289.5, 289.3, 235.6], [226.9, 226.0, 218.7, 287.3, 227.2, 199.7, 283.5, 281.4], [258.2, 237.4, 223.9, 214.6, 292.1, 280.1, 278.4, 233.0], [309.7, 203.1, 299.0, 296.1, 250.3, 234.5, 231.0, 214.5]], [[301.2, 216.9, 214.8, 310.8, 246.6, 201.3, 303.0, 306.4], [284.8, 275.3, 303.3, 221.7, 262.8, 300.3, 264.8, 292.1], [288.9, 219.7, 294.3, 206.1, 213.5, 234.4, 209.6, 269.4], [282.5, 230.5, 248.4, 279.0, 249.4, 242.6, 286.0, 238.3], [275.5, 236.7, 210.9, 296.1, 210.4, 209.1, 246.9, 298.5]], [[213.3, 277.8, 289.6, 213.5, 242.6, 292.9, 273.9, 293.0], [268.7, 300.8, 310.2, 274.6, 228.1, 248.0, 245.3, 214.7], [234.9, 279.7, 306.4, 306.1, 301.8, 210.3, 297.3, 310.7], [263.2, 293.6, 225.8, 311.1, 277.1, 248.0, 220.4, 308.1], [243.4, 285.4, 290.6, 235.2, 211.5, 229.2, 250.9, 262.8]], [[200.1, 290.2, 222.1, 274.7, 291.9, 226.3, 227.9, 210.4], [217.9, 270.3, 238.3, 246.0, 285.9, 213.6, 310.6, 299.0], [239.6, 309.7, 261.7, 273.4, 305.2, 243.0, 274.1, 255.3], [245.9, 292.1, 216.8, 199.5, 309.2, 286.8, 289.9, 299.7], [210.3, 208.9, 211.2, 245.7, 240.7, 249.1, 219.0, 256.6]], [[204.6, 266.5, 294.7, 242.1, 282.9, 204.9, 241.7, 303.7], [251.1, 220.4, 263.1, 211.7, 219.9, 240.0, 278.6, 240.3], [308.8, 255.9, 258.2, 253.4, 279.9, 308.5, 229.5, 254.0], [270.8, 278.9, 269.2, 272.7, 285.4, 206.3, 216.8, 238.3], [305.4, 205.9, 306.8, 272.7, 234.2, 244.4, 277.6, 295.4]], [[203.2, 246.8, 305.1, 289.9, 260.8, 274.0, 310.7, 299.0], [292.7, 241.5, 255.5, 205.8, 212.6, 243.9, 287.4, 232.5], [200.3, 301.1, 221.0, 311.2, 246.9, 290.8, 309.0, 286.5], [214.0, 206.0, 254.6, 227.0, 217.5, 236.1, 213.1, 260.2], [302.5, 230.8, 294.0, 235.9, 250.7, 209.4, 218.7, 266.0]], [[244.6, 287.8, 273.8, 267.2, 237.6, 224.2, 206.1, 242.4], [201.9, 243.0, 270.5, 308.8, 241.6, 243.9, 271.9, 250.5], [216.1, 305.7, 257.5, 311.2, 223.2, 276.0, 213.0, 252.5], [233.4, 221.0, 262.4, 257.7, 234.0, 225.8, 219.6, 308.1], [282.1, 223.3, 284.9, 238.4, 235.8, 305.6, 308.6, 219.4]], [[238.4, 201.7, 229.7, 224.8, 209.0, 280.5, 293.8, 260.4], [273.7, 253.7, 299.4, 241.4, 229.3, 230.3, 265.6, 287.3], [283.9, 265.2, 289.2, 284.3, 221.0, 306.3, 253.9, 246.3], [241.3, 289.5, 212.4, 217.2, 201.7, 238.0, 265.2, 257.7], [269.9, 213.4, 256.6, 290.1, 266.8, 278.9, 247.5, 286.8]], [[304.5, 275.8, 216.5, 273.4, 220.4, 251.1, 255.9, 282.2], [300.2, 274.3, 297.8, 229.3, 207.1, 297.7, 280.1, 216.4], [287.9, 308.4, 283.0, 281.3, 222.6, 228.0, 257.0, 222.2], [310.1, 263.2, 248.8, 243.0, 241.8, 219.4, 293.1, 277.2], [299.7, 249.8, 241.3, 267.5, 290.6, 258.1, 261.7, 293.5]], [[269.8, 297.4, 264.3, 253.0, 249.8, 228.4, 259.8, 278.6], [288.0, 274.6, 299.8, 298.8, 248.1, 267.0, 287.7, 206.5], [221.2, 235.6, 235.3, 252.1, 220.6, 215.5, 284.1, 237.9], [292.9, 264.2, 297.6, 284.0, 304.3, 211.0, 271.5, 199.7], [245.3, 293.9, 243.8, 268.9, 260.6, 262.5, 264.8, 211.0]], [[267.9, 244.2, 269.1, 215.6, 284.3, 229.4, 307.6, 255.2], [296.3, 280.6, 302.1, 302.1, 215.1, 206.6, 227.5, 263.2], [253.7, 287.9, 280.9, 299.6, 206.1, 300.4, 307.1, 211.6], [260.0, 276.3, 296.1, 285.9, 270.2, 243.4, 231.6, 267.1], [303.5, 199.4, 307.1, 213.2, 236.5, 265.4, 249.6, 268.8]], [[282.4, 298.8, 306.8, 311.1, 263.3, 239.8, 205.8, 199.4], [247.0, 255.0, 220.5, 263.8, 254.0, 257.5, 299.2, 271.9], [295.1, 253.6, 241.7, 214.4, 246.8, 293.7, 230.0, 285.2], [298.6, 241.6, 217.5, 296.1, 265.1, 215.2, 249.0, 237.3], [261.4, 235.1, 298.9, 248.9, 211.0, 235.1, 273.1, 255.3]], [[215.0, 214.4, 204.8, 304.0, 235.6, 300.1, 234.4, 272.1], [274.2, 209.0, 306.8, 229.5, 303.7, 284.1, 223.7, 272.6], [266.5, 259.3, 264.1, 311.2, 305.4, 261.1, 262.7, 309.6], [310.0, 308.1, 273.8, 250.9, 233.3, 209.8, 249.8, 273.8], [200.0, 221.3, 294.8, 216.5, 206.1, 297.6, 211.8, 275.1]], [[288.7, 234.6, 235.5, 205.8, 205.4, 214.5, 288.4, 254.0], [200.5, 228.3, 244.9, 238.5, 263.1, 285.6, 292.7, 295.0], [291.0, 246.3, 268.1, 208.8, 215.3, 278.1, 286.1, 290.0], [258.5, 283.0, 279.5, 257.4, 234.2, 269.8, 256.3, 209.2], [303.9, 206.7, 293.6, 272.0, 290.2, 288.6, 236.9, 268.3]], [[217.3, 264.2, 249.4, 296.9, 208.4, 232.0, 288.0, 299.4], [258.0, 218.9, 205.5, 279.4, 293.8, 260.4, 228.3, 224.0], [210.6, 217.2, 241.7, 201.7, 215.0, 255.9, 241.0, 240.8], [256.2, 305.1, 293.5, 253.9, 271.0, 248.8, 206.2, 305.7], [275.0, 301.1, 284.7, 227.8, 252.3, 231.2, 214.9, 243.8]], [[307.1, 206.8, 207.1, 260.3, 257.4, 310.1, 287.4, 242.8], [291.5, 266.9, 302.8, 232.3, 283.1, 207.8, 249.3, 252.4], [207.4, 222.4, 218.9, 266.7, 214.4, 227.9, 254.5, 310.7], [232.6, 248.7, 257.5, 243.6, 261.9, 220.7, 294.0, 286.5], [286.3, 262.3, 202.2, 279.2, 257.1, 230.2, 250.6, 225.3]], [[299.3, 268.7, 296.3, 199.9, 254.3, 295.7, 275.3, 271.8], [250.6, 226.6, 301.3, 207.4, 242.9, 273.1, 216.1, 252.0], [275.8, 291.3, 270.6, 282.9, 250.5, 291.3, 260.6, 310.1], [253.2, 221.3, 281.1, 283.0, 268.0, 263.9, 224.3, 284.0], [236.5, 218.9, 229.2, 227.9, 226.2, 247.3, 298.1, 226.8]], [[215.9, 289.9, 222.7, 270.5, 247.7, 200.7, 219.0, 252.4], [202.8, 278.9, 259.1, 207.2, 299.8, 249.2, 259.8, 200.7], [249.3, 205.9, 303.5, 304.2, 216.8, 308.1, 201.5, 241.9], [256.9, 264.6, 227.4, 229.5, 294.2, 271.0, 254.5, 274.6], [268.1, 199.3, 275.7, 289.0, 205.0, 218.2, 270.6, 280.4]], [[290.2, 274.0, 281.7, 263.1, 202.1, 199.7, 228.1, 260.0], [248.7, 305.0, 306.2, 255.3, 298.0, 254.6, 276.0, 249.4], [217.2, 272.4, 278.8, 252.1, 236.4, 223.6, 201.8, 300.9], [302.4, 305.0, 273.1, 261.9, 241.4, 285.0, 275.1, 210.2], [242.1, 208.1, 258.0, 222.2, 244.7, 236.9, 216.0, 260.5]], [[239.9, 220.7, 246.1, 209.0, 247.9, 247.4, 227.1, 291.7], [205.5, 287.2, 305.5, 238.8, 291.1, 250.0, 202.0, 234.0], [275.4, 210.0, 276.8, 287.3, 281.2, 279.6, 306.0, 228.3], [301.9, 295.9, 298.4, 304.0, 227.9, 301.7, 296.2, 247.4], [210.1, 212.0, 275.1, 271.8, 254.0, 274.8, 283.8, 286.6]]], units='K', dtype='f8')
        f.set_data(data, axes=('domainaxis0', 'domainaxis1', 'domainaxis2'))
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'standard_name': 'time', 'units': 'days since 1959-01-01'})
        c.nc_set_variable('time')
        data = Data([349.5, 380.5, 410.5, 440.5, 471.0, 501.5, 532.0, 562.5, 593.5, 624.0, 654.5, 685.0, 715.5, 746.5, 776.0, 805.5, 836.0, 866.5, 897.0, 927.5, 958.5, 989.0, 1019.5, 1050.0, 1080.5, 1111.5, 1141.0, 1170.5, 1201.0, 1231.5, 1262.0, 1292.5, 1323.5, 1354.0, 1384.5, 1415.0], units='days since 1959-01-01', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'days since 1959-01-01'})
        b.nc_set_variable('bounds')
        data = Data([[334.0, 365.0], [365.0, 396.0], [396.0, 425.0], [425.0, 456.0], [456.0, 486.0], [486.0, 517.0], [517.0, 547.0], [547.0, 578.0], [578.0, 609.0], [609.0, 639.0], [639.0, 670.0], [670.0, 700.0], [700.0, 731.0], [731.0, 762.0], [762.0, 790.0], [790.0, 821.0], [821.0, 851.0], [851.0, 882.0], [882.0, 912.0], [912.0, 943.0], [943.0, 974.0], [974.0, 1004.0], [1004.0, 1035.0], [1035.0, 1065.0], [1065.0, 1096.0], [1096.0, 1127.0], [1127.0, 1155.0], [1155.0, 1186.0], [1186.0, 1216.0], [1216.0, 1247.0], [1247.0, 1277.0], [1277.0, 1308.0], [1308.0, 1339.0], [1339.0, 1369.0], [1369.0, 1400.0], [1400.0, 1430.0]], units='days since 1959-01-01', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis0',), key='dimensioncoordinate0', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'degrees_north', 'standard_name': 'latitude'})
        c.nc_set_variable('lat')
        data = Data([-75.0, -45.0, 0.0, 45.0, 75.0], units='degrees_north', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'degrees_north'})
        b.nc_set_variable('lat_bnds')
        data = Data([[-90.0, -60.0], [-60.0, -30.0], [-30.0, 30.0], [30.0, 60.0], [60.0, 90.0]], units='degrees_north', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis1',), key='dimensioncoordinate1', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'degrees_east', 'standard_name': 'longitude'})
        c.nc_set_variable('lon')
        data = Data([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5], units='degrees_east', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'degrees_east'})
        b.nc_set_variable('lon_bnds')
        data = Data([[0.0, 45.0], [45.0, 90.0], [90.0, 135.0], [135.0, 180.0], [180.0, 225.0], [225.0, 270.0], [270.0, 315.0], [315.0, 360.0]], units='degrees_east', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis2',), key='dimensioncoordinate2', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'standard_name': 'air_pressure', 'units': 'hPa'})
        c.nc_set_variable('air_pressure')
        data = Data([850.0], units='hPa', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis3',), key='dimensioncoordinate3', copy=False)
        
        # cell_method
        c = CellMethod()
        c.method = 'mean'
        c.axes = ('area',)
        f.set_construct(c)

    elif n == 5:
        f = Field()
        
        f.set_properties({'Conventions': 'CF-1.7', 'standard_name': 'air_potential_temperature', 'units': 'K'})
        f.nc_set_variable('air_potential_temperature')
        f.nc_set_global_attributes({'Conventions': None})
        
        # domain_axis
        c = DomainAxis(size=118)
        c.nc_set_dimension('time')
        f.set_construct(c, key='domainaxis0')
        
        # domain_axis
        c = DomainAxis(size=5)
        c.nc_set_dimension('lat')
        f.set_construct(c, key='domainaxis1')
        
        # domain_axis
        c = DomainAxis(size=8)
        c.nc_set_dimension('lon')
        f.set_construct(c, key='domainaxis2')
        
        # domain_axis
        c = DomainAxis(size=1)
        f.set_construct(c, key='domainaxis3')
        
        # field data
        data = Data([[[274.0, 282.2, 267.4, 275.3, 274.3, 280.0, 281.9, 266.7], [263.9, 268.6, 276.9, 275.8, 271.1, 277.3, 270.2, 278.0], [266.5, 267.9, 267.8, 268.2, 257.7, 263.5, 273.9, 281.0], [274.9, 265.2, 284.2, 275.5, 254.3, 277.4, 258.1, 273.3], [279.9, 291.9, 273.9, 285.8, 277.1, 271.3, 273.4, 261.8]], [[292.2, 275.7, 286.9, 287.2, 269.2, 266.1, 286.4, 272.4], [276.9, 269.2, 290.9, 283.1, 268.2, 274.0, 267.8, 253.2], [285.5, 288.9, 261.7, 278.6, 287.6, 269.0, 277.2, 279.1], [264.1, 256.4, 278.1, 272.1, 271.4, 277.2, 271.2, 274.5], [277.5, 290.1, 282.4, 264.7, 277.8, 257.5, 261.3, 284.6]], [[277.2, 279.8, 259.0, 267.0, 277.9, 282.0, 268.7, 277.7], [280.2, 256.7, 264.0, 272.8, 284.2, 262.6, 292.9, 273.4], [256.3, 276.7, 280.7, 258.6, 267.7, 260.7, 273.3, 273.7], [278.9, 279.4, 276.5, 272.7, 271.3, 260.1, 287.4, 278.9], [288.7, 278.6, 284.2, 277.1, 283.8, 283.5, 262.4, 268.1]], [[266.4, 277.7, 279.8, 271.1, 257.1, 286.9, 277.9, 261.5], [277.6, 273.5, 261.1, 280.8, 280.1, 266.0, 270.8, 256.0], [281.0, 290.1, 263.1, 274.8, 288.2, 277.2, 278.8, 260.4], [248.1, 285.8, 274.2, 268.1, 279.6, 278.1, 262.3, 286.0], [274.3, 272.8, 276.4, 281.7, 258.1, 275.2, 259.4, 279.5]], [[266.7, 259.6, 257.7, 265.6, 259.3, 256.6, 255.7, 285.6], [283.4, 274.8, 268.7, 277.2, 265.2, 281.5, 282.5, 258.1], [284.2, 291.0, 268.9, 260.0, 281.3, 266.9, 274.6, 289.2], [279.4, 284.7, 266.6, 285.6, 275.1, 284.8, 286.4, 284.3], [269.1, 273.3, 272.8, 279.9, 283.2, 285.5, 258.1, 261.7]], [[296.4, 281.1, 278.6, 273.2, 288.7, 281.3, 265.6, 284.2], [276.8, 271.7, 274.4, 271.1, 279.9, 265.4, 292.5, 259.7], [278.7, 279.6, 277.0, 270.7, 266.1, 265.2, 272.5, 278.1], [284.0, 254.0, 279.7, 291.1, 282.4, 279.3, 257.6, 285.1], [272.2, 283.6, 270.0, 271.9, 294.8, 260.4, 275.9, 275.5]], [[273.5, 273.9, 298.2, 275.1, 268.2, 260.2, 260.5, 272.5], [264.7, 241.2, 261.1, 260.3, 272.8, 285.2, 283.7, 275.5], [273.2, 256.0, 282.9, 272.0, 253.9, 291.0, 267.5, 272.1], [259.6, 262.7, 278.5, 271.6, 260.1, 273.3, 286.1, 267.7], [266.3, 262.5, 273.4, 278.9, 274.9, 267.1, 274.6, 286.1]], [[278.1, 269.1, 271.4, 266.1, 258.6, 281.9, 256.9, 281.7], [275.0, 281.3, 256.7, 252.4, 281.5, 273.6, 259.1, 266.1], [264.1, 266.0, 278.9, 267.4, 286.4, 281.7, 270.2, 266.1], [274.2, 261.9, 270.1, 291.9, 292.1, 277.6, 283.6, 279.4], [281.4, 270.6, 255.5, 269.4, 264.8, 262.4, 275.3, 286.9]], [[269.6, 269.8, 270.8, 270.1, 277.6, 271.0, 263.6, 274.5], [259.0, 251.2, 295.4, 262.1, 262.6, 283.3, 269.0, 268.1], [286.0, 275.6, 264.2, 265.3, 263.7, 268.6, 269.9, 281.2], [269.6, 274.9, 256.7, 284.1, 271.1, 254.8, 258.1, 273.0], [269.2, 271.2, 275.3, 279.8, 278.7, 278.8, 280.5, 285.8]], [[285.9, 246.7, 255.7, 282.1, 258.9, 262.6, 300.0, 288.3], [279.2, 266.6, 271.0, 258.5, 257.9, 274.8, 264.4, 267.4], [282.0, 266.8, 280.1, 289.6, 278.2, 282.2, 288.7, 273.6], [279.2, 273.1, 266.7, 271.5, 275.4, 278.2, 269.7, 291.6], [281.1, 282.1, 271.3, 260.0, 275.7, 291.4, 266.6, 265.2]], [[278.5, 263.3, 285.9, 255.2, 283.8, 288.7, 282.8, 262.9], [272.6, 288.2, 257.5, 269.8, 273.6, 266.7, 276.2, 275.5], [261.8, 274.2, 278.3, 273.3, 268.8, 278.5, 273.0, 276.7], [276.2, 271.2, 284.8, 272.2, 274.7, 253.6, 268.7, 273.1], [285.6, 270.2, 271.4, 285.5, 248.1, 273.1, 294.9, 272.9]], [[269.3, 252.5, 271.3, 268.2, 270.8, 282.2, 275.0, 274.2], [257.5, 295.6, 278.0, 284.6, 277.3, 277.3, 273.1, 278.8], [279.8, 286.5, 259.5, 287.2, 276.8, 282.0, 265.9, 283.8], [276.2, 282.4, 252.7, 265.5, 252.9, 274.6, 265.5, 274.1], [269.8, 267.4, 292.5, 256.7, 274.3, 278.9, 270.3, 252.5]], [[287.4, 275.6, 287.9, 284.6, 281.7, 280.5, 267.8, 283.9], [292.0, 286.4, 276.1, 277.8, 280.8, 268.4, 281.6, 262.4], [260.7, 265.9, 274.9, 275.9, 277.3, 286.2, 296.4, 280.1], [251.1, 283.6, 265.0, 280.6, 254.8, 251.6, 275.2, 279.2], [273.3, 272.0, 254.3, 287.5, 275.3, 282.1, 272.6, 266.8]], [[279.6, 268.3, 280.6, 267.5, 260.8, 268.2, 276.2, 247.4], [268.1, 275.0, 278.7, 265.8, 283.8, 271.6, 284.5, 276.6], [269.7, 270.1, 274.9, 252.2, 285.5, 254.3, 266.2, 270.6], [274.1, 273.7, 269.4, 262.9, 281.7, 282.7, 270.0, 264.8], [280.7, 265.3, 291.6, 281.2, 273.1, 273.5, 291.3, 274.4]], [[270.4, 273.8, 260.8, 262.9, 268.9, 278.1, 261.7, 257.3], [262.4, 261.2, 265.5, 276.6, 264.4, 271.6, 272.9, 273.3], [247.1, 271.7, 272.0, 279.3, 269.3, 255.2, 279.9, 272.8], [291.6, 279.5, 263.2, 285.0, 263.5, 257.0, 274.2, 270.1], [261.5, 270.7, 280.2, 264.5, 267.0, 260.3, 277.4, 288.1]], [[261.5, 285.4, 275.3, 276.7, 279.4, 269.1, 264.1, 254.2], [262.1, 272.5, 262.2, 275.6, 276.1, 269.9, 263.3, 281.2], [287.1, 276.5, 285.9, 267.1, 274.2, 269.0, 265.3, 281.2], [265.7, 278.5, 251.2, 269.6, 263.7, 260.4, 264.8, 280.3], [269.0, 269.8, 262.3, 277.5, 269.8, 269.3, 262.7, 266.8]], [[287.6, 279.1, 268.2, 285.1, 286.4, 260.3, 267.4, 275.6], [263.2, 280.9, 271.3, 271.5, 263.9, 280.2, 265.9, 283.9], [272.8, 280.7, 272.9, 263.4, 264.2, 280.2, 289.7, 279.0], [279.0, 279.7, 285.9, 293.0, 299.2, 268.6, 274.9, 279.1], [261.8, 277.4, 275.8, 262.5, 275.9, 265.9, 284.6, 264.7]], [[265.2, 284.8, 277.7, 267.9, 265.4, 283.8, 276.7, 274.6], [274.6, 273.5, 285.1, 268.8, 272.6, 268.6, 284.2, 288.1], [268.9, 271.7, 274.2, 280.7, 288.0, 280.6, 269.8, 278.0], [263.6, 261.7, 279.2, 271.7, 280.5, 246.8, 263.4, 266.8], [268.9, 266.0, 269.0, 273.5, 287.7, 272.5, 278.3, 269.5]], [[265.0, 269.2, 275.2, 263.1, 293.9, 275.4, 256.5, 262.7], [272.2, 276.3, 286.5, 255.0, 264.8, 287.4, 277.6, 249.9], [283.1, 278.2, 285.0, 268.0, 277.9, 259.3, 261.4, 279.2], [278.7, 277.0, 254.3, 281.4, 283.8, 266.5, 260.9, 267.0], [282.3, 273.5, 278.7, 290.6, 274.6, 284.2, 281.4, 284.3]], [[277.0, 268.1, 269.3, 272.3, 286.1, 266.7, 294.1, 274.9], [261.3, 265.1, 283.2, 258.9, 278.1, 275.1, 284.4, 287.3], [264.6, 274.7, 270.3, 261.1, 274.0, 287.4, 267.2, 277.8], [276.2, 284.9, 281.6, 265.8, 253.0, 270.2, 276.5, 282.8], [244.8, 282.7, 265.0, 261.4, 277.6, 283.7, 285.7, 267.3]], [[276.9, 275.1, 256.0, 270.9, 285.0, 271.6, 272.0, 268.9], [244.3, 266.3, 285.2, 266.5, 279.4, 271.0, 267.2, 274.7], [275.0, 264.0, 280.7, 286.7, 275.2, 280.2, 282.1, 271.0], [272.8, 265.3, 278.5, 283.8, 291.0, 280.9, 266.6, 269.2], [287.0, 263.0, 269.0, 263.4, 267.3, 256.1, 280.7, 263.3]], [[264.8, 281.0, 286.5, 260.8, 285.7, 262.2, 260.5, 267.2], [277.2, 285.5, 263.2, 259.4, 266.5, 271.3, 272.8, 259.7], [273.8, 286.1, 271.7, 263.6, 286.8, 270.7, 277.0, 262.0], [279.0, 262.3, 283.9, 257.4, 279.4, 261.5, 276.1, 264.8], [266.3, 268.7, 264.2, 279.7, 287.5, 271.9, 262.9, 272.0]], [[288.8, 263.8, 278.6, 276.3, 269.9, 247.4, 258.2, 286.5], [284.4, 262.1, 262.1, 265.7, 275.7, 265.7, 269.6, 281.1], [259.3, 288.6, 267.9, 268.0, 274.9, 274.8, 255.8, 270.7], [270.7, 276.4, 288.7, 276.5, 279.7, 251.3, 274.0, 281.2], [258.4, 277.9, 262.8, 272.4, 286.9, 277.0, 279.6, 291.4]], [[282.9, 289.5, 275.3, 281.6, 271.8, 275.7, 282.4, 265.0], [250.2, 288.4, 278.6, 269.3, 264.3, 285.2, 259.2, 263.9], [271.0, 280.5, 272.9, 269.5, 277.2, 276.4, 247.4, 302.6], [276.1, 254.1, 300.1, 279.7, 265.7, 270.9, 281.0, 264.6], [260.8, 258.2, 287.8, 277.3, 276.3, 258.8, 283.3, 273.4]], [[287.8, 276.5, 281.3, 290.4, 248.3, 279.4, 265.3, 277.0], [262.0, 266.8, 274.4, 273.1, 287.7, 268.5, 276.0, 275.8], [280.5, 281.0, 258.2, 268.6, 266.0, 274.2, 260.6, 257.3], [274.1, 265.9, 263.6, 267.9, 277.5, 277.1, 264.6, 275.6], [284.1, 267.1, 272.9, 262.4, 264.9, 283.6, 270.7, 274.8]], [[262.7, 275.7, 276.7, 269.0, 274.4, 266.2, 270.8, 258.2], [262.2, 271.5, 276.8, 283.7, 287.6, 278.3, 268.9, 262.6], [280.8, 269.7, 273.0, 275.2, 260.9, 246.2, 288.0, 274.2], [264.8, 288.0, 258.0, 270.1, 276.8, 280.9, 271.2, 274.1], [259.9, 284.3, 292.6, 273.7, 270.2, 290.0, 272.4, 278.5]], [[255.0, 268.8, 260.8, 280.4, 269.7, 275.9, 282.3, 279.8], [276.1, 284.3, 275.5, 263.6, 254.1, 265.5, 271.2, 274.5], [281.3, 269.2, 281.2, 292.6, 284.0, 251.0, 271.4, 276.0], [278.0, 266.5, 282.5, 272.1, 283.1, 270.0, 281.2, 250.7], [279.9, 271.5, 286.0, 289.7, 259.0, 279.6, 264.3, 286.0]], [[278.7, 270.2, 269.1, 263.8, 285.6, 285.3, 252.7, 271.3], [282.5, 278.2, 285.7, 267.2, 271.0, 278.2, 278.1, 270.0], [282.5, 261.9, 268.3, 278.6, 279.8, 276.1, 275.5, 272.7], [273.3, 286.7, 297.1, 277.4, 272.2, 268.7, 289.2, 274.8], [270.0, 278.6, 277.3, 269.1, 255.0, 277.0, 271.3, 263.4]], [[265.3, 270.9, 271.1, 276.2, 260.0, 277.5, 267.1, 259.3], [259.2, 271.4, 287.1, 288.8, 265.3, 274.7, 277.0, 261.0], [265.4, 283.4, 272.0, 272.8, 276.2, 271.8, 270.6, 280.9], [283.6, 270.2, 271.2, 276.5, 268.7, 274.7, 275.7, 276.1], [277.6, 279.7, 251.2, 285.2, 268.9, 275.1, 284.2, 271.5]], [[267.4, 261.0, 283.3, 269.1, 272.5, 263.9, 290.3, 257.6], [264.5, 277.0, 282.8, 267.8, 286.2, 290.0, 283.1, 291.9], [279.4, 261.9, 279.5, 262.4, 278.8, 278.7, 267.4, 272.1], [274.6, 269.4, 290.0, 282.7, 274.5, 271.4, 268.4, 275.8], [256.1, 264.2, 284.7, 267.2, 271.0, 263.2, 276.9, 277.7]], [[262.4, 266.1, 272.0, 279.0, 286.3, 259.2, 278.4, 264.0], [283.9, 267.5, 276.3, 264.9, 262.6, 277.1, 268.6, 279.2], [271.2, 257.6, 266.2, 299.8, 279.7, 270.3, 293.7, 256.1], [293.1, 290.4, 282.9, 283.2, 278.9, 255.9, 266.3, 272.9], [271.2, 255.9, 273.1, 254.2, 267.5, 272.5, 270.5, 281.1]], [[267.8, 271.9, 276.9, 265.0, 260.1, 291.9, 287.7, 272.5], [270.0, 292.6, 282.1, 275.1, 274.0, 270.6, 280.5, 285.6], [273.4, 270.6, 258.3, 267.3, 260.4, 284.7, 259.4, 296.0], [280.9, 270.6, 273.0, 268.5, 282.0, 295.1, 283.0, 259.2], [286.5, 267.2, 260.9, 260.6, 276.0, 276.3, 260.6, 273.8]], [[248.9, 284.4, 284.9, 272.2, 288.7, 266.3, 272.7, 279.8], [274.7, 278.5, 275.1, 260.0, 277.1, 276.1, 278.0, 280.2], [267.5, 253.5, 266.4, 267.0, 283.0, 269.9, 265.7, 248.3], [270.6, 266.6, 260.1, 278.7, 293.7, 274.4, 273.6, 284.3], [291.1, 287.7, 293.3, 272.9, 289.3, 271.5, 264.8, 272.1]], [[275.5, 271.5, 265.2, 298.2, 274.5, 263.3, 275.7, 285.8], [284.6, 281.6, 271.0, 274.6, 262.3, 266.5, 268.1, 272.8], [256.3, 262.6, 256.6, 292.2, 273.1, 287.6, 284.1, 263.4], [292.0, 275.0, 260.0, 270.3, 258.7, 266.5, 273.1, 265.2], [270.5, 273.5, 260.0, 280.1, 283.2, 282.2, 286.3, 280.6]], [[261.6, 286.0, 288.0, 286.7, 267.7, 265.3, 272.0, 256.4], [271.8, 266.4, 260.4, 272.6, 268.7, 262.2, 269.6, 269.9], [275.9, 281.0, 285.7, 271.3, 265.1, 263.6, 272.1, 264.9], [298.8, 266.1, 269.2, 267.4, 277.4, 264.1, 262.9, 260.6], [279.5, 263.5, 267.1, 259.3, 286.7, 267.9, 268.7, 276.2]], [[275.1, 275.0, 256.0, 261.4, 273.9, 273.7, 263.5, 266.9], [263.9, 284.4, 282.9, 261.3, 275.6, 270.6, 258.6, 271.9], [257.9, 272.3, 275.2, 272.2, 275.0, 289.1, 266.7, 276.6], [283.9, 272.3, 270.0, 260.1, 270.5, 275.9, 275.6, 273.8], [273.5, 269.5, 261.0, 278.1, 284.3, 262.6, 266.0, 271.4]], [[280.2, 272.6, 271.9, 283.6, 258.8, 274.2, 260.2, 270.7], [276.4, 279.2, 282.1, 276.6, 278.5, 263.4, 274.7, 278.0], [286.4, 265.1, 279.5, 243.6, 279.8, 281.4, 266.6, 276.4], [265.8, 274.9, 263.3, 278.6, 289.6, 260.3, 267.0, 265.7], [266.1, 263.7, 278.3, 267.4, 270.1, 253.0, 280.0, 275.1]], [[266.6, 265.7, 272.9, 273.6, 260.6, 271.8, 284.2, 285.6], [268.4, 277.9, 268.6, 265.6, 259.7, 264.3, 272.2, 265.5], [280.3, 283.1, 276.5, 261.6, 264.8, 267.9, 270.3, 282.4], [270.6, 268.4, 254.2, 288.3, 281.0, 258.9, 273.9, 283.1], [270.0, 267.0, 260.7, 275.1, 285.3, 279.1, 261.6, 270.3]], [[275.1, 262.0, 269.0, 280.9, 275.9, 271.9, 276.4, 275.0], [285.3, 280.8, 269.3, 275.2, 276.8, 280.6, 290.2, 255.7], [264.8, 280.8, 279.2, 275.2, 261.6, 280.8, 267.7, 282.6], [259.4, 280.5, 264.9, 254.4, 269.2, 287.8, 278.7, 266.8], [290.6, 278.2, 276.7, 261.8, 277.6, 264.4, 279.3, 258.9]], [[265.9, 287.0, 267.1, 267.3, 252.9, 269.2, 271.2, 252.7], [262.0, 293.8, 276.6, 264.1, 270.0, 276.3, 277.2, 269.3], [292.2, 270.0, 266.5, 260.7, 276.7, 284.4, 276.2, 269.0], [260.5, 273.0, 269.1, 282.8, 261.4, 271.9, 275.2, 274.6], [285.1, 272.6, 263.9, 270.1, 287.4, 282.0, 279.4, 264.7]], [[272.7, 257.4, 285.7, 260.9, 262.0, 287.7, 277.6, 277.3], [279.9, 280.4, 278.3, 272.9, 278.8, 272.5, 282.9, 260.9], [267.3, 274.3, 265.2, 287.9, 286.6, 269.5, 283.9, 265.8], [274.3, 265.8, 276.6, 271.2, 276.8, 287.1, 257.2, 269.3], [283.4, 269.9, 278.7, 272.3, 272.6, 268.4, 280.7, 266.7]], [[262.4, 278.1, 257.7, 280.3, 274.5, 285.5, 272.5, 283.3], [268.8, 269.6, 278.0, 269.5, 277.7, 266.9, 285.6, 268.4], [269.3, 287.8, 270.0, 271.4, 288.3, 285.3, 275.5, 300.0], [273.6, 261.2, 284.5, 276.2, 271.4, 276.6, 266.2, 263.0], [274.5, 275.4, 269.7, 274.6, 271.2, 260.3, 275.3, 265.1]], [[281.5, 288.1, 283.9, 275.4, 265.8, 271.4, 261.6, 260.1], [257.7, 278.1, 286.9, 276.0, 271.3, 286.2, 277.2, 275.2], [266.6, 278.6, 256.8, 264.8, 278.1, 261.7, 269.6, 278.0], [283.8, 290.7, 274.1, 267.2, 258.9, 282.4, 292.4, 310.1], [275.8, 280.5, 250.5, 276.1, 269.2, 294.5, 279.6, 268.5]], [[280.2, 268.9, 273.2, 270.7, 285.5, 276.7, 263.0, 255.3], [276.8, 263.3, 266.9, 292.5, 260.9, 278.0, 278.0, 263.6], [283.3, 294.0, 282.1, 277.8, 282.4, 270.7, 268.2, 285.4], [266.9, 272.5, 288.0, 266.4, 279.6, 257.7, 288.5, 267.1], [267.6, 267.1, 276.3, 268.3, 266.4, 269.1, 270.1, 274.4]], [[294.5, 269.9, 284.3, 276.0, 278.7, 289.9, 272.1, 273.0], [274.5, 260.5, 283.4, 261.7, 281.3, 272.5, 268.9, 268.8], [271.3, 277.3, 279.4, 269.1, 274.3, 274.2, 281.9, 269.7], [280.2, 267.1, 284.0, 270.8, 276.6, 271.6, 271.5, 267.7], [266.9, 272.4, 267.8, 279.6, 262.5, 253.3, 265.8, 256.1]], [[269.9, 282.7, 277.3, 270.7, 280.0, 276.0, 278.2, 257.2], [271.8, 265.5, 271.6, 274.9, 272.9, 289.5, 258.8, 260.8], [287.4, 266.8, 285.1, 275.6, 292.4, 262.5, 269.3, 279.0], [269.9, 268.1, 259.4, 274.3, 290.3, 271.1, 290.3, 276.5], [272.8, 264.2, 264.4, 277.6, 264.3, 290.3, 265.4, 266.4]], [[268.4, 277.8, 287.6, 273.6, 281.8, 262.4, 265.5, 289.1], [289.5, 264.5, 275.2, 256.3, 259.8, 290.0, 260.7, 272.7], [281.6, 278.4, 266.6, 264.4, 264.6, 264.2, 281.8, 271.2], [285.1, 282.4, 289.6, 262.2, 285.0, 273.2, 257.9, 280.2], [290.3, 284.4, 299.6, 266.6, 261.0, 273.5, 274.5, 284.6]], [[270.8, 282.5, 290.8, 266.6, 285.0, 275.8, 290.5, 268.7], [281.1, 283.5, 279.0, 272.2, 276.8, 280.3, 272.9, 275.6], [283.8, 269.0, 276.2, 265.1, 283.9, 285.1, 280.4, 273.9], [271.5, 280.2, 280.5, 278.4, 265.4, 271.7, 287.2, 261.4], [290.3, 251.7, 269.1, 279.9, 281.1, 270.9, 259.6, 284.7]], [[279.0, 264.6, 274.8, 282.1, 271.7, 254.4, 268.8, 271.1], [293.9, 283.5, 265.1, 263.8, 278.4, 263.5, 270.8, 270.8], [266.6, 257.7, 277.7, 275.2, 257.4, 269.6, 289.5, 269.2], [274.4, 287.4, 277.3, 257.5, 269.0, 271.2, 272.6, 272.8], [272.5, 271.5, 260.6, 274.3, 274.7, 262.7, 260.6, 253.6]], [[278.7, 267.4, 279.0, 271.9, 269.8, 260.8, 284.9, 282.4], [288.6, 262.9, 260.4, 272.2, 271.1, 280.6, 273.7, 282.8], [272.1, 264.7, 284.6, 299.6, 258.7, 265.3, 269.5, 276.7], [286.5, 271.9, 282.3, 266.2, 277.7, 260.4, 267.9, 287.9], [269.8, 255.4, 276.4, 281.8, 266.6, 275.7, 288.3, 265.8]], [[261.3, 245.6, 265.9, 267.4, 266.7, 276.5, 272.7, 256.9], [264.1, 285.6, 278.5, 269.2, 268.6, 259.6, 253.3, 260.1], [272.9, 266.8, 278.3, 280.0, 283.0, 281.2, 276.7, 275.0], [273.1, 261.5, 276.6, 272.7, 280.9, 287.7, 273.2, 274.7], [285.0, 271.5, 271.9, 264.1, 278.7, 273.1, 271.5, 255.9]], [[264.6, 288.9, 278.1, 253.0, 281.4, 294.3, 252.1, 260.7], [273.0, 275.1, 283.2, 256.1, 284.4, 283.8, 274.2, 288.1], [260.1, 269.9, 277.9, 281.7, 282.4, 280.8, 278.3, 278.7], [275.0, 274.1, 281.0, 269.8, 276.6, 276.2, 263.7, 264.0], [280.4, 280.4, 257.8, 249.8, 275.1, 265.2, 261.9, 285.8]], [[269.3, 274.1, 277.9, 265.4, 272.4, 274.9, 272.8, 270.7], [276.4, 280.4, 294.0, 260.7, 281.6, 271.0, 283.6, 277.4], [278.9, 257.9, 268.4, 279.0, 278.0, 276.4, 260.1, 260.9], [282.2, 272.1, 249.6, 289.8, 269.7, 280.0, 280.9, 266.1], [251.9, 269.4, 270.0, 278.7, 265.4, 271.9, 282.9, 256.7]], [[258.5, 291.3, 274.2, 273.1, 276.9, 280.7, 275.0, 259.9], [262.6, 266.9, 261.4, 274.7, 267.8, 296.9, 271.9, 261.0], [266.9, 273.3, 274.6, 274.2, 264.5, 271.5, 288.2, 289.1], [270.4, 288.9, 276.0, 268.6, 277.1, 277.8, 277.2, 284.2], [279.6, 265.8, 280.9, 295.2, 255.6, 269.8, 265.8, 259.3]], [[261.9, 275.9, 262.0, 273.1, 268.1, 277.6, 265.8, 285.0], [260.2, 280.5, 262.2, 263.6, 264.0, 275.7, 262.7, 286.0], [250.8, 284.8, 260.6, 272.9, 290.0, 264.2, 266.3, 264.6], [278.6, 292.3, 272.7, 284.3, 285.9, 278.1, 273.1, 272.0], [248.7, 268.6, 280.3, 274.9, 272.8, 298.1, 272.7, 281.5]], [[282.3, 279.1, 265.4, 269.3, 258.5, 264.1, 272.3, 279.0], [268.5, 274.1, 265.4, 256.7, 279.8, 275.6, 270.7, 285.5], [269.1, 297.5, 283.9, 244.9, 258.4, 272.7, 265.2, 265.2], [289.8, 281.7, 278.2, 299.5, 281.1, 270.7, 269.4, 275.4], [273.0, 287.2, 272.5, 274.0, 287.7, 275.6, 278.4, 266.4]], [[287.8, 302.7, 261.9, 270.8, 285.9, 285.5, 262.4, 274.5], [281.9, 273.0, 268.4, 265.0, 279.0, 258.6, 266.1, 280.5], [274.5, 277.5, 283.3, 266.4, 287.0, 270.0, 265.0, 269.0], [287.4, 257.5, 269.6, 275.0, 278.6, 287.2, 279.4, 282.2], [261.1, 265.7, 281.6, 271.7, 278.0, 272.4, 278.9, 264.4]], [[280.2, 269.2, 247.1, 286.4, 273.2, 270.5, 284.3, 264.1], [280.4, 264.9, 274.1, 275.1, 273.8, 286.6, 286.8, 276.2], [264.1, 274.5, 276.3, 263.8, 277.7, 265.6, 269.1, 271.3], [269.9, 287.5, 283.1, 267.8, 272.5, 272.0, 268.1, 291.4], [267.6, 290.6, 277.0, 283.0, 285.9, 261.4, 274.7, 275.9]], [[295.2, 298.9, 273.2, 274.7, 268.2, 274.9, 273.0, 277.7], [261.1, 283.1, 261.0, 295.8, 284.1, 276.2, 280.9, 281.2], [265.5, 277.7, 270.3, 260.5, 252.6, 273.5, 271.3, 278.4], [286.9, 266.0, 277.0, 277.8, 280.6, 260.7, 277.7, 272.2], [272.3, 291.7, 260.2, 272.0, 286.3, 276.4, 285.2, 260.8]], [[285.7, 275.2, 279.7, 266.4, 269.5, 273.6, 272.7, 274.0], [276.0, 265.2, 278.8, 271.5, 288.0, 273.8, 269.9, 254.7], [262.7, 258.9, 279.5, 265.9, 276.1, 283.3, 286.1, 286.4], [268.8, 272.1, 281.3, 269.7, 255.5, 273.0, 273.2, 275.1], [255.8, 282.3, 262.0, 276.3, 289.3, 270.5, 265.6, 267.7]], [[293.2, 280.3, 295.2, 273.1, 263.9, 266.4, 278.2, 279.8], [283.8, 280.8, 280.5, 263.9, 279.7, 269.4, 246.4, 263.9], [271.1, 257.8, 266.7, 263.8, 264.6, 256.0, 273.8, 298.3], [278.6, 282.3, 267.9, 265.2, 277.1, 273.2, 283.6, 277.0], [290.1, 275.6, 265.0, 267.6, 265.7, 263.7, 277.6, 290.6]], [[275.3, 294.0, 267.9, 268.1, 268.5, 286.1, 289.5, 261.5], [263.4, 276.0, 257.0, 289.8, 280.5, 262.4, 279.0, 272.5], [280.0, 272.6, 279.6, 258.1, 271.4, 271.1, 290.0, 241.2], [268.4, 290.5, 281.8, 276.8, 277.6, 282.8, 274.1, 267.6], [275.6, 272.5, 260.1, 261.6, 264.9, 266.8, 277.0, 256.7]], [[263.9, 274.4, 279.7, 260.2, 271.7, 270.0, 272.9, 271.1], [281.6, 293.4, 286.1, 277.4, 275.0, 286.5, 279.0, 267.1], [282.3, 272.2, 283.8, 277.5, 292.2, 287.3, 275.5, 274.4], [278.2, 267.3, 276.3, 268.8, 264.1, 257.0, 278.8, 282.5], [294.8, 280.0, 276.5, 266.6, 278.3, 256.7, 264.4, 291.7]], [[274.2, 259.3, 267.9, 268.5, 279.0, 264.4, 263.2, 269.1], [280.5, 272.5, 264.1, 290.7, 288.6, 263.6, 279.9, 278.5], [276.4, 277.5, 275.6, 283.6, 288.1, 270.3, 286.0, 281.5], [255.3, 259.7, 261.7, 290.6, 295.0, 280.1, 254.9, 262.2], [255.4, 257.5, 273.1, 257.7, 258.4, 294.7, 279.3, 282.0]], [[270.4, 273.0, 256.5, 259.9, 268.2, 258.4, 275.5, 294.0], [280.1, 263.8, 258.3, 274.2, 273.0, 259.8, 253.7, 267.3], [278.0, 260.8, 263.1, 269.8, 291.2, 279.7, 261.4, 288.6], [272.1, 292.0, 287.7, 272.4, 273.5, 275.2, 270.0, 268.1], [286.6, 289.0, 268.9, 277.8, 276.3, 278.3, 262.5, 280.1]], [[275.3, 283.6, 274.4, 272.1, 272.1, 271.1, 273.2, 288.4], [287.5, 258.1, 286.5, 277.6, 279.5, 288.8, 261.1, 281.0], [272.5, 264.2, 258.5, 260.8, 267.7, 263.1, 277.2, 280.8], [265.9, 280.6, 273.1, 244.5, 266.9, 270.7, 277.0, 266.7], [262.6, 271.8, 270.4, 264.7, 277.9, 267.0, 281.2, 270.1]], [[269.0, 265.1, 271.9, 260.6, 266.8, 255.9, 289.6, 257.0], [263.2, 271.7, 279.0, 296.9, 271.8, 280.4, 289.9, 277.7], [251.9, 272.5, 265.7, 272.6, 260.0, 289.5, 262.7, 267.6], [272.1, 261.7, 265.0, 272.5, 260.6, 283.2, 273.3, 274.1], [264.0, 266.1, 278.9, 261.3, 266.3, 277.6, 281.9, 284.8]], [[258.6, 265.0, 272.3, 275.7, 279.0, 265.1, 270.8, 258.5], [266.0, 267.3, 272.9, 274.0, 267.3, 260.6, 280.1, 268.4], [271.9, 263.6, 266.4, 270.7, 249.5, 286.0, 283.6, 279.2], [286.9, 288.0, 271.0, 272.5, 271.0, 268.6, 274.6, 267.2], [269.9, 285.8, 287.2, 277.3, 263.6, 273.8, 281.6, 264.7]], [[269.3, 262.2, 271.6, 265.8, 277.2, 276.9, 273.2, 255.7], [257.7, 266.9, 269.7, 255.2, 265.2, 301.2, 284.5, 284.7], [292.6, 268.3, 267.8, 283.6, 262.1, 276.8, 257.8, 271.6], [259.7, 289.9, 268.4, 277.1, 281.3, 280.5, 265.2, 266.4], [261.3, 270.0, 266.3, 271.8, 266.7, 254.9, 281.9, 268.6]], [[284.2, 272.6, 278.2, 288.0, 277.0, 261.2, 263.4, 277.3], [292.5, 270.3, 273.6, 280.3, 261.1, 275.7, 287.1, 278.1], [295.6, 289.6, 259.1, 266.5, 272.6, 263.2, 272.3, 273.0], [277.6, 265.0, 267.4, 286.5, 276.2, 276.7, 284.1, 272.1], [268.4, 273.3, 279.4, 271.9, 261.0, 258.6, 254.6, 269.2]], [[283.8, 265.7, 276.6, 273.9, 268.4, 273.4, 253.4, 271.6], [276.3, 267.8, 261.1, 267.5, 264.4, 272.4, 291.2, 278.9], [264.5, 288.0, 272.0, 275.1, 272.2, 275.9, 273.7, 276.4], [261.7, 252.8, 263.5, 279.2, 285.4, 278.1, 257.6, 264.5], [267.9, 271.1, 273.4, 276.0, 270.4, 280.8, 272.3, 271.2]], [[272.1, 283.6, 274.5, 271.8, 260.5, 254.9, 280.2, 257.2], [274.4, 273.8, 263.3, 272.1, 268.6, 279.6, 268.9, 255.7], [288.6, 288.7, 260.7, 260.6, 273.0, 270.3, 260.6, 281.4], [256.1, 279.3, 273.7, 250.0, 264.3, 279.6, 277.0, 282.7], [278.4, 265.3, 272.3, 274.0, 270.8, 272.3, 275.2, 285.1]], [[279.1, 268.2, 268.7, 275.2, 280.5, 274.3, 285.5, 249.8], [281.2, 267.2, 269.4, 254.8, 284.3, 265.8, 275.3, 260.3], [282.4, 281.2, 274.6, 277.3, 272.2, 256.8, 273.4, 284.1], [281.1, 288.6, 269.8, 281.0, 264.7, 258.8, 281.7, 255.5], [278.4, 280.5, 273.5, 272.0, 279.4, 278.1, 295.4, 246.6]], [[270.0, 259.1, 266.0, 288.3, 268.3, 258.3, 270.1, 286.4], [268.0, 266.1, 276.0, 253.1, 272.1, 271.2, 270.4, 270.7], [279.8, 282.8, 264.2, 257.1, 292.4, 276.8, 270.5, 256.9], [281.7, 280.6, 276.0, 266.0, 278.0, 278.5, 282.8, 268.0], [275.3, 262.2, 251.9, 270.9, 273.2, 287.4, 285.4, 263.5]], [[282.1, 283.6, 269.3, 276.5, 274.8, 271.5, 276.9, 274.5], [270.7, 274.9, 286.1, 285.4, 277.8, 269.6, 269.9, 276.3], [273.1, 278.9, 264.2, 277.1, 257.9, 271.7, 278.9, 262.2], [292.8, 262.7, 276.5, 274.8, 266.3, 278.1, 277.9, 267.2], [284.3, 276.9, 282.4, 292.9, 269.5, 269.0, 270.2, 284.6]], [[273.7, 285.3, 272.8, 255.1, 275.0, 269.7, 255.4, 277.4], [278.6, 272.9, 273.1, 299.3, 271.3, 274.8, 262.7, 272.2], [272.7, 278.3, 260.2, 264.1, 288.9, 283.2, 259.9, 271.9], [278.5, 270.0, 265.9, 276.4, 270.0, 255.6, 263.7, 260.0], [273.4, 266.5, 267.4, 286.9, 268.0, 269.7, 275.1, 269.9]], [[250.5, 267.5, 277.8, 287.8, 276.0, 272.6, 274.8, 292.1], [263.8, 276.1, 265.4, 266.5, 262.2, 257.9, 275.3, 267.0], [276.2, 277.0, 276.8, 295.3, 285.2, 257.5, 259.0, 287.8], [291.8, 257.9, 271.5, 253.3, 270.8, 273.7, 270.7, 280.3], [270.4, 262.5, 272.5, 268.6, 282.5, 254.3, 272.1, 280.2]], [[274.4, 290.9, 276.8, 267.3, 274.7, 289.6, 267.5, 292.8], [260.7, 281.5, 264.6, 272.4, 259.0, 274.3, 279.7, 272.6], [273.2, 262.4, 269.2, 280.0, 275.7, 270.5, 286.6, 293.1], [267.2, 277.4, 274.5, 274.4, 274.2, 298.3, 286.3, 265.0], [285.1, 272.3, 263.6, 282.3, 248.3, 275.0, 254.5, 288.2]], [[273.3, 268.5, 282.8, 259.8, 251.3, 270.6, 259.7, 269.6], [269.6, 278.6, 281.7, 286.3, 283.3, 255.9, 267.4, 283.3], [256.9, 272.4, 273.2, 263.5, 269.9, 277.0, 259.4, 289.0], [249.0, 271.5, 279.6, 264.4, 264.4, 263.8, 269.5, 266.3], [274.4, 278.7, 262.8, 286.0, 282.3, 282.2, 267.3, 273.7]], [[266.9, 259.2, 262.7, 286.4, 257.2, 265.0, 261.5, 283.8], [275.8, 285.6, 284.7, 258.0, 277.1, 281.5, 266.8, 256.5], [279.2, 273.6, 267.9, 266.6, 278.4, 275.8, 257.4, 258.7], [269.9, 247.3, 288.3, 265.3, 269.2, 268.7, 267.9, 280.9], [252.7, 276.1, 267.5, 279.1, 276.5, 279.8, 257.0, 264.3]], [[284.5, 284.9, 267.2, 274.0, 284.2, 285.8, 274.6, 279.3], [274.4, 281.6, 260.7, 253.7, 264.9, 261.2, 279.6, 284.1], [260.1, 276.5, 278.0, 264.0, 264.1, 271.5, 265.2, 290.4], [274.2, 262.7, 284.7, 290.1, 279.3, 263.6, 279.3, 270.4], [271.5, 273.7, 264.5, 271.8, 286.5, 263.7, 272.3, 273.6]], [[274.6, 255.3, 290.2, 273.7, 268.6, 267.0, 284.3, 257.5], [260.7, 248.0, 271.0, 279.5, 279.1, 278.6, 258.2, 290.7], [264.4, 281.6, 271.5, 283.0, 276.7, 292.1, 274.3, 267.2], [280.3, 266.8, 272.8, 267.6, 271.0, 272.1, 285.2, 262.9], [267.5, 256.5, 283.1, 280.3, 263.4, 270.6, 274.9, 268.2]], [[277.8, 270.0, 288.3, 276.6, 276.9, 267.6, 286.9, 275.1], [282.3, 259.3, 273.5, 295.0, 282.6, 273.9, 275.9, 288.2], [269.0, 270.2, 292.6, 254.1, 271.9, 271.7, 271.7, 279.6], [279.5, 252.0, 285.7, 260.6, 252.7, 275.4, 290.4, 277.3], [276.0, 281.8, 270.6, 272.5, 259.3, 274.4, 290.2, 278.3]], [[256.6, 274.6, 274.0, 285.3, 263.0, 266.5, 264.6, 277.0], [276.0, 279.5, 272.4, 259.3, 273.0, 276.7, 286.7, 284.0], [279.8, 265.5, 272.2, 276.8, 283.9, 272.7, 287.7, 263.0], [267.9, 266.4, 274.4, 251.5, 279.1, 274.1, 258.6, 284.7], [279.9, 274.3, 268.2, 273.2, 284.7, 291.7, 257.4, 281.0]], [[264.8, 274.6, 275.8, 269.6, 263.1, 254.5, 286.9, 260.2], [279.8, 281.9, 277.0, 256.9, 268.3, 277.3, 258.9, 268.9], [255.6, 269.4, 290.6, 276.8, 261.0, 261.6, 286.6, 279.8], [295.8, 270.1, 259.1, 286.5, 282.2, 269.1, 274.1, 293.4], [281.4, 264.5, 258.0, 283.1, 272.5, 263.5, 269.8, 266.1]], [[281.0, 287.9, 289.3, 251.7, 284.6, 273.3, 269.2, 274.5], [268.7, 266.3, 265.5, 271.6, 258.9, 270.1, 277.6, 279.0], [250.8, 277.2, 260.2, 285.0, 263.0, 279.1, 278.1, 251.8], [267.4, 272.5, 250.2, 283.1, 269.6, 275.8, 257.6, 275.4], [273.7, 261.9, 258.7, 267.0, 277.0, 272.1, 280.8, 265.4]], [[281.4, 274.2, 274.0, 278.7, 282.4, 285.4, 262.9, 266.7], [294.3, 283.0, 282.1, 269.5, 266.3, 292.0, 258.3, 273.1], [274.7, 257.9, 264.8, 261.9, 247.3, 284.5, 283.9, 274.9], [287.1, 273.3, 286.9, 283.1, 288.3, 265.8, 272.0, 295.6], [276.9, 267.3, 271.6, 275.2, 279.6, 277.0, 262.8, 265.3]], [[277.1, 289.1, 284.3, 279.8, 276.7, 290.2, 265.7, 260.0], [260.8, 251.6, 263.8, 267.6, 266.2, 259.9, 276.7, 267.7], [273.2, 265.0, 281.7, 270.2, 281.7, 259.8, 264.0, 260.8], [275.8, 274.9, 279.4, 283.3, 272.6, 273.0, 279.1, 288.0], [266.9, 261.1, 270.0, 289.2, 250.5, 276.0, 289.7, 281.9]], [[274.8, 273.8, 277.1, 288.6, 277.1, 257.3, 277.7, 279.4], [278.3, 269.0, 278.2, 277.3, 275.1, 266.6, 280.7, 276.7], [270.3, 257.8, 264.6, 271.6, 287.7, 274.0, 272.5, 265.8], [259.1, 269.2, 290.7, 281.9, 270.4, 287.1, 290.8, 290.0], [285.0, 273.4, 283.4, 278.1, 286.7, 259.6, 249.6, 273.6]], [[269.2, 291.3, 282.4, 279.8, 269.8, 272.5, 271.7, 261.8], [261.1, 273.4, 287.5, 282.1, 262.1, 275.4, 271.9, 269.1], [266.1, 272.4, 270.5, 281.2, 271.9, 285.6, 259.6, 290.2], [270.5, 275.9, 255.8, 275.6, 277.8, 272.3, 271.8, 278.8], [268.7, 277.8, 267.6, 264.7, 279.0, 258.8, 288.6, 277.3]], [[272.1, 267.0, 251.9, 277.8, 263.8, 275.8, 275.8, 293.0], [264.4, 268.3, 261.4, 266.6, 283.4, 282.3, 255.5, 272.3], [262.7, 285.2, 276.2, 283.8, 275.3, 274.8, 290.9, 280.4], [275.2, 263.9, 275.8, 267.1, 267.2, 267.4, 269.2, 270.3], [289.4, 259.3, 275.2, 274.0, 268.4, 280.8, 278.5, 266.6]], [[261.1, 264.3, 275.0, 282.6, 286.0, 271.9, 276.3, 263.2], [280.9, 268.7, 274.8, 280.0, 263.5, 284.9, 279.8, 256.3], [269.5, 274.8, 271.5, 273.5, 265.1, 283.4, 271.6, 269.9], [293.7, 278.1, 267.6, 265.9, 261.6, 269.8, 272.4, 277.1], [267.8, 292.5, 271.5, 279.8, 256.7, 271.9, 273.7, 275.1]], [[290.6, 269.7, 282.3, 277.8, 289.0, 284.4, 274.0, 275.3], [261.0, 276.2, 282.4, 260.1, 274.1, 279.1, 280.7, 266.0], [275.6, 265.3, 287.5, 272.9, 278.9, 258.9, 273.2, 264.0], [287.0, 280.2, 268.2, 277.5, 277.2, 258.7, 263.0, 262.2], [277.2, 293.3, 270.3, 265.9, 264.7, 262.0, 281.5, 275.9]], [[277.8, 282.3, 278.4, 263.6, 270.7, 275.2, 277.3, 281.0], [275.2, 263.9, 285.6, 269.5, 272.8, 279.1, 269.4, 268.2], [261.5, 267.2, 260.3, 293.5, 281.6, 280.1, 282.5, 274.3], [279.7, 264.8, 258.8, 279.2, 278.6, 261.3, 261.7, 268.0], [281.1, 258.5, 290.3, 268.9, 275.5, 272.2, 267.3, 276.0]], [[257.4, 261.4, 271.5, 273.8, 272.6, 254.5, 282.5, 262.8], [260.2, 260.9, 280.4, 279.8, 268.2, 273.2, 275.6, 274.7], [275.1, 277.5, 285.3, 280.8, 273.3, 263.0, 263.2, 297.7], [274.7, 275.9, 265.6, 266.9, 273.4, 269.6, 279.8, 276.2], [279.6, 268.9, 270.3, 269.1, 267.5, 282.4, 279.1, 252.3]], [[277.5, 272.1, 261.2, 266.4, 291.3, 273.9, 270.5, 270.9], [275.4, 270.1, 260.8, 270.1, 277.3, 271.5, 280.9, 268.7], [277.8, 277.3, 274.3, 269.1, 280.6, 283.8, 268.4, 278.3], [291.0, 267.1, 261.3, 269.0, 271.5, 280.5, 274.5, 290.5], [280.9, 268.4, 283.4, 284.3, 269.5, 261.1, 279.7, 288.4]], [[274.7, 258.8, 271.0, 272.1, 263.9, 268.4, 264.6, 288.0], [280.5, 254.7, 272.6, 278.1, 250.7, 280.2, 285.2, 275.4], [286.7, 281.5, 254.7, 276.5, 263.9, 281.3, 278.1, 273.0], [259.4, 277.5, 271.9, 273.2, 264.0, 273.4, 286.5, 268.5], [277.3, 270.3, 286.2, 273.9, 268.8, 282.7, 272.1, 274.1]], [[273.2, 257.3, 287.3, 268.4, 278.7, 272.9, 249.7, 272.8], [272.3, 266.7, 284.1, 280.8, 265.1, 275.5, 282.5, 249.9], [270.0, 272.4, 276.6, 269.2, 262.0, 289.0, 278.1, 282.9], [277.6, 246.5, 264.1, 285.4, 279.1, 279.4, 279.2, 288.6], [276.9, 259.9, 276.0, 252.1, 272.8, 277.6, 277.4, 281.5]], [[280.5, 277.4, 273.8, 284.9, 263.1, 262.4, 297.5, 274.8], [265.6, 271.3, 294.9, 290.9, 275.9, 275.0, 279.2, 256.7], [275.8, 266.7, 260.6, 273.5, 270.9, 266.2, 253.6, 270.1], [282.8, 270.1, 269.4, 278.6, 276.0, 270.8, 279.3, 272.3], [277.4, 282.5, 257.7, 267.2, 271.4, 267.1, 269.9, 253.6]], [[275.5, 272.7, 268.2, 253.0, 272.0, 272.8, 282.0, 274.8], [271.3, 295.3, 275.4, 276.9, 270.7, 268.7, 281.0, 294.1], [285.9, 254.7, 274.8, 259.6, 262.1, 279.0, 280.7, 275.5], [276.9, 276.7, 258.6, 274.9, 281.7, 279.6, 267.7, 269.1], [289.2, 256.6, 274.6, 254.9, 257.0, 280.3, 270.2, 272.6]], [[269.6, 270.3, 269.7, 279.4, 253.0, 257.1, 295.4, 287.3], [264.8, 277.6, 271.6, 271.1, 268.8, 275.6, 262.6, 267.5], [272.7, 256.1, 259.7, 273.3, 265.6, 287.5, 270.7, 283.3], [262.8, 276.3, 263.5, 281.9, 264.6, 267.2, 260.5, 263.5], [286.6, 267.0, 273.4, 277.2, 276.6, 287.9, 262.0, 261.8]], [[277.3, 277.7, 269.9, 263.9, 282.3, 281.7, 280.3, 269.8], [275.0, 285.4, 290.3, 280.3, 265.4, 262.6, 251.7, 273.8], [275.0, 276.3, 262.7, 266.6, 273.2, 269.0, 269.6, 256.1], [271.4, 291.2, 275.5, 265.8, 274.0, 280.6, 291.5, 260.2], [258.3, 281.9, 264.1, 278.0, 279.1, 269.8, 278.9, 257.5]], [[276.7, 280.6, 262.5, 260.8, 276.9, 284.7, 289.2, 245.8], [266.1, 251.6, 267.1, 272.0, 275.1, 261.7, 272.0, 269.1], [276.2, 267.4, 264.4, 279.1, 278.0, 261.8, 291.5, 268.1], [281.6, 282.3, 267.2, 285.8, 267.3, 287.0, 262.5, 282.3], [276.1, 276.5, 253.5, 274.9, 264.8, 262.6, 268.3, 276.9]], [[281.0, 290.1, 285.8, 279.5, 275.5, 286.3, 277.5, 268.3], [284.4, 279.7, 290.9, 273.3, 275.1, 276.8, 270.6, 271.8], [270.2, 271.8, 266.6, 269.5, 287.0, 281.4, 261.6, 264.1], [276.7, 269.5, 278.1, 275.2, 281.0, 264.9, 275.4, 272.4], [267.5, 287.3, 262.6, 270.7, 273.2, 280.5, 264.8, 263.9]], [[287.5, 290.3, 280.7, 271.0, 265.0, 274.3, 277.4, 265.4], [271.6, 263.7, 294.4, 289.2, 273.0, 267.7, 276.1, 279.3], [277.1, 273.6, 269.9, 258.6, 252.7, 276.9, 286.8, 266.5], [277.3, 271.8, 269.1, 280.1, 291.2, 262.5, 263.6, 261.4], [276.4, 260.8, 259.8, 265.0, 296.3, 269.9, 276.8, 278.7]], [[261.1, 260.2, 270.9, 269.2, 279.8, 243.0, 270.1, 265.4], [275.1, 271.3, 271.6, 297.3, 275.4, 264.1, 275.1, 268.9], [274.9, 277.0, 274.7, 280.4, 283.5, 269.7, 253.7, 272.1], [271.8, 262.5, 259.3, 266.2, 279.2, 275.5, 276.8, 272.5], [269.8, 277.3, 275.8, 283.6, 261.1, 268.3, 248.6, 262.8]], [[270.5, 274.3, 270.4, 269.5, 271.5, 273.9, 281.5, 276.6], [277.3, 271.4, 294.4, 281.4, 269.7, 287.9, 260.6, 276.1], [270.9, 271.2, 269.0, 269.8, 263.8, 263.1, 281.3, 288.1], [263.8, 277.5, 270.3, 264.8, 270.0, 290.4, 265.1, 273.1], [251.5, 282.6, 265.7, 262.8, 267.1, 269.4, 285.3, 269.2]], [[270.1, 281.4, 284.2, 263.7, 278.9, 266.2, 282.9, 273.6], [286.6, 275.3, 281.8, 272.7, 265.7, 253.1, 269.1, 264.6], [293.7, 278.5, 272.9, 275.7, 279.7, 281.6, 270.3, 260.9], [277.6, 289.1, 253.7, 282.5, 285.1, 276.3, 260.7, 289.7], [286.6, 289.8, 244.6, 286.1, 265.8, 289.2, 284.5, 284.1]], [[280.4, 269.5, 270.3, 274.6, 272.1, 284.9, 272.3, 276.4], [265.1, 285.5, 267.2, 275.9, 277.2, 266.5, 280.4, 275.9], [291.2, 265.5, 294.6, 258.7, 267.8, 269.9, 283.3, 275.3], [270.5, 267.0, 263.4, 270.2, 279.5, 270.6, 265.6, 286.5], [274.4, 277.7, 272.3, 267.6, 278.8, 267.4, 275.6, 265.9]], [[279.7, 272.4, 275.7, 276.4, 267.8, 263.7, 289.2, 296.3], [279.9, 305.1, 272.7, 269.6, 273.8, 282.6, 277.3, 267.1], [276.4, 274.6, 273.0, 265.4, 268.2, 283.6, 273.2, 260.1], [285.8, 275.2, 253.9, 261.6, 268.5, 284.8, 268.9, 263.2], [279.8, 281.4, 286.1, 275.0, 271.4, 280.0, 282.9, 279.9]], [[286.3, 290.2, 283.0, 250.6, 274.3, 257.9, 273.7, 285.2], [286.2, 277.6, 266.2, 271.3, 262.4, 262.6, 273.9, 275.3], [273.4, 285.1, 282.6, 289.2, 294.0, 265.3, 275.9, 257.5], [283.3, 272.9, 255.5, 264.4, 280.0, 261.6, 283.3, 271.8], [291.1, 279.2, 266.4, 264.4, 267.3, 284.5, 276.7, 265.7]], [[262.9, 284.6, 268.7, 277.1, 280.8, 267.9, 268.4, 273.8], [257.7, 284.1, 292.8, 284.7, 271.5, 257.8, 271.2, 276.5], [293.1, 285.6, 279.3, 266.0, 291.1, 273.8, 278.9, 272.9], [258.5, 271.1, 271.4, 266.5, 261.7, 254.0, 280.3, 277.9], [284.2, 257.3, 273.4, 258.4, 280.2, 279.0, 268.1, 264.2]], [[282.2, 266.1, 282.7, 296.2, 258.5, 269.3, 267.5, 286.1], [264.6, 282.6, 255.9, 278.9, 267.8, 265.5, 270.1, 280.6], [273.7, 274.2, 272.4, 275.9, 260.4, 289.8, 264.6, 277.4], [272.6, 278.7, 283.1, 277.4, 278.2, 289.5, 271.6, 263.3], [268.1, 268.8, 285.6, 259.7, 248.9, 282.8, 279.1, 273.2]], [[281.6, 276.5, 258.1, 265.5, 256.5, 266.2, 285.0, 276.8], [278.5, 273.0, 279.2, 260.2, 268.4, 271.8, 266.4, 258.7], [274.8, 258.4, 258.4, 279.9, 267.1, 279.5, 281.9, 281.7], [275.0, 279.3, 267.4, 267.1, 275.9, 251.9, 273.9, 259.5], [291.5, 262.5, 276.4, 281.7, 275.2, 260.9, 272.7, 277.5]], [[255.6, 257.6, 273.1, 272.6, 293.2, 260.9, 277.7, 280.5], [254.3, 283.5, 275.1, 268.0, 293.5, 271.7, 283.8, 280.2], [264.8, 261.5, 270.9, 278.2, 268.8, 278.5, 278.8, 270.3], [279.5, 276.8, 271.4, 262.3, 273.4, 259.1, 280.3, 265.6], [256.2, 278.8, 260.5, 262.3, 251.4, 286.6, 281.1, 259.6]], [[272.7, 274.6, 263.7, 276.5, 269.4, 287.5, 259.3, 281.9], [284.4, 257.9, 245.8, 258.5, 258.6, 269.5, 262.1, 268.8], [275.2, 263.3, 270.5, 285.4, 256.8, 280.1, 267.2, 276.7], [272.6, 292.5, 281.8, 265.6, 270.6, 280.1, 272.6, 261.0], [269.2, 272.2, 275.0, 250.4, 281.1, 269.8, 273.7, 277.4]], [[261.9, 289.6, 262.0, 286.8, 266.3, 263.4, 263.0, 274.8], [284.5, 273.8, 272.9, 273.6, 264.3, 278.5, 269.5, 293.6], [272.4, 268.8, 262.8, 270.0, 285.0, 264.0, 286.0, 294.4], [279.9, 275.6, 256.9, 285.8, 268.5, 270.1, 268.0, 263.0], [270.8, 279.4, 276.6, 274.5, 264.0, 277.9, 281.6, 269.4]], [[277.3, 271.4, 268.6, 262.2, 259.6, 262.5, 264.0, 278.7], [273.8, 280.7, 275.7, 254.3, 270.0, 268.9, 278.2, 269.7], [284.6, 270.7, 284.1, 277.3, 271.7, 276.2, 276.2, 270.5], [288.2, 264.5, 262.5, 279.8, 272.5, 266.7, 260.6, 287.1], [276.0, 286.9, 272.8, 279.3, 266.5, 281.7, 275.4, 274.1]]], units='K', dtype='f8')
        f.set_data(data, axes=('domainaxis0', 'domainaxis1', 'domainaxis2'))
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'standard_name': 'time', 'units': 'days since 1959-01-01'})
        c.nc_set_variable('time')
        data = Data([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25, 19.75, 20.25, 20.75, 21.25, 21.75, 22.25, 22.75, 23.25, 23.75, 24.25, 24.75, 25.25, 25.75, 26.25, 26.75, 27.25, 27.75, 28.25, 28.75, 29.25, 29.75, 30.25, 30.75, 31.25, 31.75, 32.25, 32.75, 33.25, 33.75, 34.25, 34.75, 35.25, 35.75, 36.25, 36.75, 37.25, 37.75, 38.25, 38.75, 39.25, 39.75, 40.25, 40.75, 41.25, 41.75, 42.25, 42.75, 43.25, 43.75, 44.25, 44.75, 45.25, 45.75, 46.25, 46.75, 47.25, 47.75, 48.25, 48.75, 49.25, 49.75, 50.25, 50.75, 51.25, 51.75, 52.25, 52.75, 53.25, 53.75, 54.25, 54.75, 55.25, 55.75, 56.25, 56.75, 57.25, 57.75, 58.25, 58.75], units='days since 1959-01-01', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'days since 1959-01-01'})
        b.nc_set_variable('bounds')
        data = Data([[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5], [2.5, 3.0], [3.0, 3.5], [3.5, 4.0], [4.0, 4.5], [4.5, 5.0], [5.0, 5.5], [5.5, 6.0], [6.0, 6.5], [6.5, 7.0], [7.0, 7.5], [7.5, 8.0], [8.0, 8.5], [8.5, 9.0], [9.0, 9.5], [9.5, 10.0], [10.0, 10.5], [10.5, 11.0], [11.0, 11.5], [11.5, 12.0], [12.0, 12.5], [12.5, 13.0], [13.0, 13.5], [13.5, 14.0], [14.0, 14.5], [14.5, 15.0], [15.0, 15.5], [15.5, 16.0], [16.0, 16.5], [16.5, 17.0], [17.0, 17.5], [17.5, 18.0], [18.0, 18.5], [18.5, 19.0], [19.0, 19.5], [19.5, 20.0], [20.0, 20.5], [20.5, 21.0], [21.0, 21.5], [21.5, 22.0], [22.0, 22.5], [22.5, 23.0], [23.0, 23.5], [23.5, 24.0], [24.0, 24.5], [24.5, 25.0], [25.0, 25.5], [25.5, 26.0], [26.0, 26.5], [26.5, 27.0], [27.0, 27.5], [27.5, 28.0], [28.0, 28.5], [28.5, 29.0], [29.0, 29.5], [29.5, 30.0], [30.0, 30.5], [30.5, 31.0], [31.0, 31.5], [31.5, 32.0], [32.0, 32.5], [32.5, 33.0], [33.0, 33.5], [33.5, 34.0], [34.0, 34.5], [34.5, 35.0], [35.0, 35.5], [35.5, 36.0], [36.0, 36.5], [36.5, 37.0], [37.0, 37.5], [37.5, 38.0], [38.0, 38.5], [38.5, 39.0], [39.0, 39.5], [39.5, 40.0], [40.0, 40.5], [40.5, 41.0], [41.0, 41.5], [41.5, 42.0], [42.0, 42.5], [42.5, 43.0], [43.0, 43.5], [43.5, 44.0], [44.0, 44.5], [44.5, 45.0], [45.0, 45.5], [45.5, 46.0], [46.0, 46.5], [46.5, 47.0], [47.0, 47.5], [47.5, 48.0], [48.0, 48.5], [48.5, 49.0], [49.0, 49.5], [49.5, 50.0], [50.0, 50.5], [50.5, 51.0], [51.0, 51.5], [51.5, 52.0], [52.0, 52.5], [52.5, 53.0], [53.0, 53.5], [53.5, 54.0], [54.0, 54.5], [54.5, 55.0], [55.0, 55.5], [55.5, 56.0], [56.0, 56.5], [56.5, 57.0], [57.0, 57.5], [57.5, 58.0], [58.0, 58.5], [58.5, 59.0]], units='days since 1959-01-01', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis0',), key='dimensioncoordinate0', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'degrees_north', 'standard_name': 'latitude'})
        c.nc_set_variable('lat')
        data = Data([-75.0, -45.0, 0.0, 45.0, 75.0], units='degrees_north', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'degrees_north'})
        b.nc_set_variable('lat_bnds')
        data = Data([[-90.0, -60.0], [-60.0, -30.0], [-30.0, 30.0], [30.0, 60.0], [60.0, 90.0]], units='degrees_north', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis1',), key='dimensioncoordinate1', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'units': 'degrees_east', 'standard_name': 'longitude'})
        c.nc_set_variable('lon')
        data = Data([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5], units='degrees_east', dtype='f8')
        c.set_data(data)
        b = Bounds()
        b.set_properties({'units': 'degrees_east'})
        b.nc_set_variable('lon_bnds')
        data = Data([[0.0, 45.0], [45.0, 90.0], [90.0, 135.0], [135.0, 180.0], [180.0, 225.0], [225.0, 270.0], [270.0, 315.0], [315.0, 360.0]], units='degrees_east', dtype='f8')
        b.set_data(data)
        c.set_bounds(b)
        f.set_construct(c, axes=('domainaxis2',), key='dimensioncoordinate2', copy=False)
        
        # dimension_coordinate
        c = DimensionCoordinate()
        c.set_properties({'standard_name': 'air_pressure', 'units': 'hPa'})
        c.nc_set_variable('air_pressure')
        data = Data([850.0], units='hPa', dtype='f8')
        c.set_data(data)
        f.set_construct(c, axes=('domainaxis3',), key='dimensioncoordinate3', copy=False)
        
        # cell_method
        c = CellMethod()
        c.method = 'mean'
        c.axes = ('area',)
        f.set_construct(c)

    else:
        raise ValueError(
            "Must select an example field construct with an argument of 1, 2, 3 or 4. Got {!r}".format(n))
    
    return f
    
