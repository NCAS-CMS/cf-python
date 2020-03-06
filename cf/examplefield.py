from functools import partial as functools_partial

import cfdm

from .cfimplementation import implementation

example_field = functools_partial(cfdm.example_field,
                                  _implementation=implementation())
example_field.__doc__ = cfdm.example_field.__doc__
#_implementation = implementation()
#
#def example_field(n):
#    '''Return an example field construct.
#
#    .. versionadded:: 3.0.5
#
#    .. seealso:: `cf.Field.creation_commands`
#
#    :Parameters:
#
#        n: `int`
#            Select the example field construct to return, one of:
#
#            =====  ===================================================
#            *n*    Description
#            =====  ===================================================
#            ``0``  A field construct with properties as well as a
#                   cell method constuct and dimension coordinate
#                   constructs with bounds.
#
#            ``1``  A field construct with properties as well as at
#                   least one of every type of metadata construct.
#
#            ``2``  A field construct that contains a monthly time
#                   series at each latitude-longitude location.
#
#            ``3``  A field construct that contains discrete sampling
#                   geometry (DSG) "timeSeries" features.
#
#            ``4``  A field construct that contains discrete sampling
#                   geometry (DSG) "timeSeriesProfile" features.
#
#            ``5``  A field construct that contains a 12 hourly time
#                   series at each latitude-longitude location.
#            =====  ===================================================
#
#            See the examples for details.
#
#    :Returns:
#
#        `Field`
#            The example field construct.
#
#    **Examples:**
#
#    >>> f = cf.example_field(0)
#    >>> print(f)
#    Field: specific_humidity(ncvar%q)
#    ---------------------------------
#    Data            : specific_humidity(latitude(5), longitude(8)) 1
#    Cell methods    : area: mean
#    Dimension coords: time(1) = [2019-01-01 00:00:00]
#                    : latitude(5) = [-75.0, ..., 75.0] degrees_north
#                    : longitude(8) = [22.5, ..., 337.5] degrees_east
#    >>> print(f.array)
#    [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
#     [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
#     [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
#     [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
#     [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
#
#    >>> f = cf.example_field(1)
#    >>> print(f)
#    Field: air_temperature (ncvar%ta)
#    ---------------------------------
#    Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K
#    Cell methods    : grid_latitude(10): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
#    Field ancils    : air_temperature standard_error(grid_latitude(10), grid_longitude(9)) = [[0.76, ..., 0.32]] K
#    Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
#                    : grid_latitude(10) = [2.2, ..., -1.76] degrees
#                    : grid_longitude(9) = [-4.7, ..., -1.18] degrees
#                    : time(1) = [2019-01-01 00:00:00]
#    Auxiliary coords: latitude(grid_latitude(10), grid_longitude(9)) = [[53.941, ..., 50.225]] degrees_N
#                    : longitude(grid_longitude(9), grid_latitude(10)) = [[2.004, ..., 8.156]] degrees_E
#                    : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., b'kappa']
#    Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
#    Coord references: grid_mapping_name:rotated_latitude_longitude
#                    : standard_name:atmosphere_hybrid_height_coordinate
#    Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
#                    : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
#                    : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
#
#    >>> f = cf.example_field(2)
#    >>> print(f)
#    Field: air_potential_temperature (ncvar%air_potential_temperature)
#    ------------------------------------------------------------------
#    Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
#    Cell methods    : area: mean
#    Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
#                    : latitude(5) = [-75.0, ..., 75.0] degrees_north
#                    : longitude(8) = [22.5, ..., 337.5] degrees_east
#                    : air_pressure(1) = [850.0] hPa
#
#    >>> f = cf.example_field(3)
#    >>> print(f)
#    Field: precipitation_flux (ncvar%p)
#    -----------------------------------
#    Data            : precipitation_flux(cf_role=timeseries_id(4), ncdim%timeseries(9)) kg m-2 day-1
#    Auxiliary coords: time(cf_role=timeseries_id(4), ncdim%timeseries(9)) = [[1969-12-29 00:00:00, ..., 1970-01-07 00:00:00]]
#                    : latitude(cf_role=timeseries_id(4)) = [-9.0, ..., 78.0] degrees_north
#                    : longitude(cf_role=timeseries_id(4)) = [-23.0, ..., 178.0] degrees_east
#                    : height(cf_role=timeseries_id(4)) = [0.5, ..., 345.0] m
#                    : cf_role=timeseries_id(cf_role=timeseries_id(4)) = [b'station1', ..., b'station4']
#                    : long_name=station information(cf_role=timeseries_id(4)) = [-10, ..., -7]
#
#    >>> f = cf.example_field(4)
#    >>> print(f)
#    Field: air_temperature (ncvar%ta)
#    ---------------------------------
#    Data            : air_temperature(cf_role=timeseries_id(3), ncdim%timeseries(26), ncdim%profile_1(4)) K
#    Auxiliary coords: time(cf_role=timeseries_id(3), ncdim%timeseries(26)) = [[1970-01-04 00:00:00, ..., --]]
#                    : latitude(cf_role=timeseries_id(3)) = [-9.0, 2.0, 34.0] degrees_north
#                    : longitude(cf_role=timeseries_id(3)) = [-23.0, 0.0, 67.0] degrees_east
#                    : height(cf_role=timeseries_id(3)) = [0.5, 12.6, 23.7] m
#                    : altitude(cf_role=timeseries_id(3), ncdim%timeseries(26), ncdim%profile_1(4)) = [[[2.07, ..., --]]] km
#                    : cf_role=timeseries_id(cf_role=timeseries_id(3)) = [b'station1', b'station2', b'station3']
#                    : long_name=station information(cf_role=timeseries_id(3)) = [-10, -9, -8]
#                    : cf_role=profile_id(cf_role=timeseries_id(3), ncdim%timeseries(26)) = [[102, ..., --]]
#
#    >>> f = cf.example_field(5)
#    >>> print(f)
#    Field: air_potential_temperature (ncvar%air_potential_temperature)
#    ------------------------------------------------------------------
#    Data            : air_potential_temperature(time(118), latitude(5), longitude(8)) K
#    Cell methods    : area: mean
#    Dimension coords: time(118) = [1959-01-01 06:00:00, ..., 1959-02-28 18:00:00]
#                    : latitude(5) = [-75.0, ..., 75.0] degrees_north
#                    : longitude(8) = [22.5, ..., 337.5] degrees_east
#                    : air_pressure(1) = [850.0] hPa
#
#    '''
#    return cfdm.example_field(n, _implementation=_implementation)
