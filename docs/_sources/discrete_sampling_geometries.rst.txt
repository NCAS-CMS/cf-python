.. currentmodule:: cf
.. default-role:: obj

.. _DSG:

Discrete sampling geometries
============================

`Discrete sampling geometries (DSG)
<http://cfconventions.org/cf-conventions/cf-conventions.html#discrete-sampling-geometries>`_
may be read into `cf.Field` objects and also written to disk.


Ragged arrays
-------------

Contiguous and indexed ragged arrays are stored in memory as
incomplete multidimensional arrays (still using :ref:`LAMA <LAMA>`
functionality) and may only be written to disk in the latter form.

Contiguous
^^^^^^^^^^

.. highlight:: pypy
	       
The following CF-netCDF file contains `contiguous ragged array
representations
<http://cfconventions.org/cf-conventions/cf-conventions.html#_contiguous_ragged_array_representation>`_
of time series, each of which is has a different length::

   $ ncdump DSG_timeSeries_contiguous.nc
   netcdf DSG_timeSeries_contiguous {
   dimensions:
   	station = 4 ;
   	obs = 24 ;
   	name_strlen = 8 ;
   
   variables:
   	double lon(station) ;
   		lon:standard_name = "longitude" ;
   		lon:long_name = "station longitude" ;
   		lon:units = "degrees_east" ;
   	double lat(station) ;
   		lat:standard_name = "latitude" ;
   		lat:long_name = "station latitude" ;
   		lat:units = "degrees_north" ;
   	double alt(station) ;
   		alt:long_name = "vertical distance above the surface" ;
   		alt:standard_name = "height" ;
   		alt:units = "m" ;
   		alt:positive = "up" ;
   		alt:axis = "Z" ;
   	char station_name(station, name_strlen) ;
   		station_name:long_name = "station name" ;
   		station_name:cf_role = "timeseries_id" ;
   	long station_info(station) ;
   		station_info:long_name = "some kind of station info" ;
   	long row_size(station) ;
   		row_size:long_name = "number of observations for this station" ;
   		row_size:sample_dimension = "obs" ;
   	double time(obs) ;
   		time:standard_name = "time" ;
   		time:long_name = "time of measurement" ;
   		time:units = "days since 1970-01-01 00:00:00" ;
   	double humidity(obs) ;
   		humidity:_FillValue = -999.9 ;
   		humidity:standard_name = "specific_humidity" ;
   		humidity:coordinates = "time lat lon alt station_name station_info" ;
   	double temp(obs) ;
   		temp:_FillValue = -999.9 ;
   		temp:standard_name = "air_temperature" ;
   		temp:units = "Celsius" ;
   		temp:coordinates = "time lat lon alt station_name station_info" ;
   
   // global attributes:
   		:Conventions = "CF-1.6" ;
   		:featureType = "timeSeries" ;
   
   data:
   
    lon = -23, 0, 67, 178 ;
   
    lat = -9, 2, 34, 78 ;
   
    alt = 0.5, 12.6, 23.7, 345 ;
   
    station_name =
     "station1",
     "station2",
     "station3",
     "station4" ;
   
    station_info = 4294967286, 4294967287, 4294967288, 4294967289 ;
   
    row_size = 3, 7, 5, 9 ;
   
    time = -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 0.5, 1.5, 2.5, 3.5, 4.5, -2, -1, 0, 
       1, 2, 3, 4, 5, 6 ;
   
    humidity = 0, 1, 2, 1, 11, 21, 31, 41, 51, 61, 2, 102, 202, 302, 402, 3, 
       1003, 2003, 3003, 4003, 5003, 6003, 7003, 8003 ;
   
    temp = 273.15, 274.15, 275.15, 274.15, 284.15, 294.15, 304.15, 314.15, 
       324.15, 334.15, 275.15, 375.15, 475.15, 575.15, 675.15, 276.15, 1276.15, 
       2276.15, 3276.15, 4276.15, 5276.15, 6276.15, 7276.15, 8276.15 ;
   }

.. highlight:: python
	       
When read by cf-python, the contiguous ragged arrays are converted to
`incomplete multidimensional arrays
<http://cfconventions.org/cf-conventions/cf-conventions.html#_incomplete_multidimensional_array_representation>`_::
  
   >>> f=cf.read('DSG_timeSeries_contiguous.nc')
   >>> print f
   Field: air_temperature (ncvar%temp)
   -----------------------------------
   Data           : air_temperature(ncdim%station(4), ncdim%timeseries(9)) Celsius
   Axes           : ncdim%station(4)
                  : ncdim%timeseries(9)
   Aux coords     : time(ncdim%station(4), ncdim%timeseries(9)) = [[1969-12-29T00:00:00Z, ..., 1970-01-07T00:00:00Z]]
                  : latitude(ncdim%station(4)) = [-9.0, ..., 78.0] degrees_north
                  : longitude(ncdim%station(4)) = [-23.0, ..., 178.0] degrees_east
                  : height(ncdim%station(4)) = [0.5, ..., 345.0] m
                  : long_name:station name(ncdim%station(4)) = [station1, ..., station4]
                  : long_name:some kind of station info(ncdim%station(4)) = [-10, ..., -7]
   
   Field: specific_humidity (ncvar%humidity)
   -----------------------------------------
   Data           : specific_humidity(ncdim%station(4), ncdim%timeseries(9)) 
   Axes           : ncdim%station(4)
                  : ncdim%timeseries(9)
   Aux coords     : time(ncdim%station(4), ncdim%timeseries(9)) = [[1969-12-29T00:00:00Z, ..., 1970-01-07T00:00:00Z]]
                  : latitude(ncdim%station(4)) = [-9.0, ..., 78.0] degrees_north
                  : longitude(ncdim%station(4)) = [-23.0, ..., 178.0] degrees_east
                  : height(ncdim%station(4)) = [0.5, ..., 345.0] m
                  : long_name:station name(ncdim%station(4)) = [station1, ..., station4]
                  : long_name:some kind of station info(ncdim%station(4)) = [-10, ..., -7]
   
   
   >>> print f[0][2].array
   [[275.15 375.15 475.15 575.15 675.15 -- -- -- --]]
   >>> print f[0].coord('time')[2].array
   [[0.5 1.5 2.5 3.5 4.5 -- -- -- --]]
   

Indexed
^^^^^^^

.. highlight:: pypy
	       
The following CF-netCDF file contains `indexed ragged array
representations
<http://cfconventions.org/cf-conventions/cf-conventions.html#_indexed_ragged_array_representation>`_
of time series, each of which is has a different length::

   $ ncdump DSG_timeSeries_indexed.nc
   netcdf DSG_timeSeries_indexed {
   dimensions:
   	station = 4 ;
   	obs = UNLIMITED ; // (24 currently)
   	name_strlen = 8 ;
   
   variables:
   	double lon(station) ;
   		lon:standard_name = "longitude" ;
   		lon:long_name = "station longitude" ;
   		lon:units = "degrees_east" ;
   	double lat(station) ;
   		lat:standard_name = "latitude" ;
   		lat:long_name = "station latitude" ;
   		lat:units = "degrees_north" ;
   	double alt(station) ;
   		alt:long_name = "vertical distance above the surface" ;
   		alt:standard_name = "height" ;
   		alt:units = "m" ;
    		alt:positive = "up" ;
   		alt:axis = "Z" ;
   	char station_name(station, name_strlen) ;
   		station_name:long_name = "station name" ;
   		station_name:cf_role = "timeseries_id" ;
   	long station_info(station) ;
   		station_info:long_name = "some kind of station info" ;
   	long stationIndex(obs) ;
   		stationIndex:long_name = "which station this obs is for" ;
   		stationIndex:instance_dimension = "station" ;
   	double time(obs) ;
   		time:standard_name = "time" ;
   		time:long_name = "time of measurement" ;
   		time:units = "days since 1970-01-01 00:00:00" ;
   	double humidity(obs) ;
   		humidity:_FillValue = -999.9 ;
   		humidity:standard_name = "specific_humidity" ;
   		humidity:coordinates = "time lat lon alt station_name station_info" ;
   	double temp(obs) ;
   		temp:_FillValue = -999.9 ;
   		temp:standard_name = "air_temperature" ;
   		temp:units = "Celsius" ;
   		temp:coordinates = "time lat lon alt station_name station_info" ;
   
   // global attributes:
   		:Conventions = "CF-1.6" ;
   		:featureType = "timeSeries" ;
   
   data:
   
    lon = -23, 0, 67, 178 ;
   
    lat = -9, 2, 34, 78 ;
   
    alt = 0.5, 12.6, 23.7, 345 ;
   
    station_name =
     "station1",
     "station2",
     "station3",
     "station4" ;
   
    station_info = 4294967286, 4294967287, 4294967288, 4294967289 ;
   
    stationIndex = 3, 2, 1, 0, 2, 3, 3, 3, 1, 1, 0, 2, 3, 1, 0, 1, 2, 3, 2, 3, 
       3, 3, 1, 1 ;
   
    time = -2, 0.5, 1, -3, 1.5, -1, 0, 1, 2, 3, -2, 2.5, 2, 4, -1, 5, 3.5, 3, 
       4.5, 4, 5, 6, 6, 7 ;
   
    humidity = 3, 2, 1, 0, 102, 1003, 2003, 3003, 11, 21, 1, 202, 4003, 31, 2, 
       41, 302, 5003, 402, 6003, 7003, 8003, 51, 61 ;
   
    temp = 276.15, 275.15, 274.15, 273.15, 375.15, 1276.15, 2276.15, 3276.15, 
       284.15, 294.15, 274.15, 475.15, 4276.15, 304.15, 275.15, 314.15, 575.15, 
       5276.15, 675.15, 6276.15, 7276.15, 8276.15, 324.15, 334.15 ;

.. highlight:: python
	       
When read by cf-python, the indexed ragged arrays are converted to
`incomplete multidimensional arrays
<http://cfconventions.org/cf-conventions/cf-conventions.html#_incomplete_multidimensional_array_representation>`_::
  
   >>> f = cf.read('netcdf DSG_timeSeries_indexed.nc')
   >>> print f
   Field: air_temperature (ncvar%temp)
   -----------------------------------
   Data           : air_temperature(ncdim%station(4), ncdim%timeseries(9)) Celsius
   Axes           : ncdim%station(4)
                  : ncdim%timeseries(9)
   Aux coords     : time(ncdim%station(4), ncdim%timeseries(9)) = [[1969-12-29T00:00:00Z, ..., 1970-01-07T00:00:00Z]]
                  : latitude(ncdim%station(4)) = [-9.0, ..., 78.0] degrees_north
                  : longitude(ncdim%station(4)) = [-23.0, ..., 178.0] degrees_east
                  : height(ncdim%station(4)) = [0.5, ..., 345.0] m
                  : long_name:station name(ncdim%station(4)) = [station1, ..., station4]
                  : long_name:some kind of station info(ncdim%station(4)) = [-10, ..., -7]
   
   Field: specific_humidity (ncvar%humidity)
   -----------------------------------------
   Data           : specific_humidity(ncdim%station(4), ncdim%timeseries(9)) 
   Axes           : ncdim%station(4)
                  : ncdim%timeseries(9)
   Aux coords     : time(ncdim%station(4), ncdim%timeseries(9)) = [[1969-12-29T00:00:00Z, ..., 1970-01-07T00:00:00Z]]
                  : latitude(ncdim%station(4)) = [-9.0, ..., 78.0] degrees_north
                  : longitude(ncdim%station(4)) = [-23.0, ..., 178.0] degrees_east
                  : height(ncdim%station(4)) = [0.5, ..., 345.0] m
                  : long_name:station name(ncdim%station(4)) = [station1, ..., station4]
                  : long_name:some kind of station info(ncdim%station(4)) = [-10, ..., -7]

   >>> print f[0][2].array
   [[275.15 375.15 475.15 575.15 675.15 -- -- -- --]]	      
   >>> print f[0].coord('time')[2].array
   [[0.5 1.5 2.5 3.5 4.5 -- -- -- --]]

Indexed contiguous
^^^^^^^^^^^^^^^^^^

.. highlight:: pypy
	       
The following CF-netCDF file contains `indexed contiguous ragged array
representations
<http://cfconventions.org/cf-conventions/cf-conventions.html#_ragged_array_representation_of_time_series_profiles>`_
of time series profiles. Each profile uses the contiguous ragged array
representation, and the indexed ragged array representation is used to
organise the profiles into time series::

   $ ncdump -h DSG_timeSeriesProfile_indexed_contiguous.nc
   netcdf DSG_timeSeriesProfile_indexed_contiguous {
   dimensions:
   	station = 3 ;
   	profile = 58 ;
   	obs = UNLIMITED ; // (118 currently)
   	name_strlen = 8 ;
   
   variables:
   	double lon(station) ;
   		lon:standard_name = "longitude" ;
   		lon:long_name = "station longitude" ;
   		lon:units = "degrees_east" ;
   	double lat(station) ;
   		lat:standard_name = "latitude" ;
   		lat:long_name = "station latitude" ;
   		lat:units = "degrees_north" ;
   	double alt(station) ;
   		alt:long_name = "vertical distance above the surface" ;
   		alt:standard_name = "height" ;
   		alt:units = "m" ;
   		alt:positive = "up" ;
   		alt:axis = "Z" ;
   	char station_name(station, name_strlen) ;
   		station_name:long_name = "station name" ;
   		station_name:cf_role = "timeseries_id" ;
   	long profile(profile) ;
   		profile:cf_role = "profile_id" ;
   	long station_info(station) ;
   		station_info:long_name = "some kind of station info" ;
   	long stationIndex(profile) ;
   		stationIndex:long_name = "which station this profile is for" ;
   		stationIndex:instance_dimension = "station" ;
   	long row_size(profile) ;
   		row_size:long_name = "number of observations for this profile" ;
   		row_size:sample_dimension = "obs" ;
   	double time(profile) ;
   		time:standard_name = "time" ;
   		time:long_name = "time" ;
   		time:units = "days since 1970-01-01 00:00:00" ;
   	double z(obs) ;
   		z:standard_name = "altitude" ;
   		z:long_name = "height above mean sea level" ;
   		z:units = "km" ;
   		z:axis = "Z" ;
   		z:positive = "up" ;
   	double humidity(obs) ;
   		humidity:_FillValue = -999.9 ;
   		humidity:standard_name = "specific_humidity" ;
   		humidity:coordinates = "time lat lon alt z station_name station_info profile" ;
   	double temp(obs) ;
   		temp:_FillValue = -999.9 ;
   		temp:standard_name = "air_temperature" ;
   		temp:units = "Celsius" ;
   		temp:coordinates = "time lat lon alt z station_name station_info profile" ;
   
   // global attributes:
   		:Conventions = "CF-1.6" ;
   		:featureType = "timeSeriesProfile" ;
   }

.. highlight:: python
	       
When read by cf-python, the indexed contiguous ragged arrays are
converted to `incomplete multidimensional arrays
<http://cfconventions.org/cf-conventions/cf-conventions.html#_incomplete_multidimensional_array_representation>`_::

   >>> f = cf.read('DSG_timeSeriesProfile_indexed_contiguous.nc')
   >>> print f
   Field: air_temperature (ncvar%temp)
   -----------------------------------
   Data           : air_temperature(ncdim%station(3), ncdim%timeseries(26), ncdim%profile_1(4)) Celsius
   Axes           : ncdim%station(3)
                  : ncdim%timeseries(26)
                  : ncdim%profile_1(4)
   Aux coords     : time(ncdim%station(3), ncdim%timeseries(26)) = [[1970-01-04T00:00:00Z, ..., --]]
                  : latitude(ncdim%station(3)) = [-9.0, ..., 34.0] degrees_north
                  : longitude(ncdim%station(3)) = [-23.0, ..., 67.0] degrees_east
                  : height(ncdim%station(3)) = [0.5, ..., 23.7] m
                  : altitude(ncdim%station(3), ncdim%timeseries(26), ncdim%profile_1(4)) = [[[2.07123455612, ..., --]]] km
                  : long_name:station name(ncdim%station(3)) = [station1, ..., station3]
                  : long_name:some kind of station info(ncdim%station(3)) = [-10, ..., -8]
                  : ncvar%profile(ncdim%station(3), ncdim%timeseries(26)) = [[102, ..., --]]
   
   Field: specific_humidity (ncvar%humidity)
   -----------------------------------------
   Data           : specific_humidity(ncdim%station(3), ncdim%timeseries(26), ncdim%profile_1(4)) 
   Axes           : ncdim%station(3)
                  : ncdim%timeseries(26)
                  : ncdim%profile_1(4)
   Aux coords     : time(ncdim%station(3), ncdim%timeseries(26)) = [[1970-01-04T00:00:00Z, ..., --]]
                  : latitude(ncdim%station(3)) = [-9.0, ..., 34.0] degrees_north
                  : longitude(ncdim%station(3)) = [-23.0, ..., 67.0] degrees_east
                  : height(ncdim%station(3)) = [0.5, ..., 23.7] m
                  : altitude(ncdim%station(3), ncdim%timeseries(26), ncdim%profile_1(4)) = [[[2.07123455612, ..., --]]] km
                  : long_name:station name(ncdim%station(3)) = [station1, ..., station3]
                  : long_name:some kind of station info(ncdim%station(3)) = [-10, ..., -8]
                  : ncvar%profile(ncdim%station(3), ncdim%timeseries(26)) = [[102, ..., --]]


