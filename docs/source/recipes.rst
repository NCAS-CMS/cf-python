.. currentmodule:: cf
.. default-role:: obj

.. TODODASK - review this entire section

**Recipes using cf-python**
===============

----

Version |release| for version |version| of the CF conventions.

.. contents::
   :local:
   :backlinks: entry

**Calculating global mean temperature timeseries:**
----------

In this recipe, we will calculate the global mean temperature timeseries and plot it.

1. Import cf-python and cf-plot:

   .. code-block:: python

      >>> import cf
      >>> import cfplot as cfp

2. Read the field constructs using `~cf.read` function:

   .. code-block:: python

      >>> f = cf.read('file.nc')
      >>> print(f)
      [<CF Field: ncvar%stn(long_name=time(120), long_name=latitude(360), long_name=longitude(720))>,
      <CF Field: long_name=near-surface temperature(long_name=time(120), long_name=latitude(360), long_name=longitude(720)) degrees Celsius>]

3. Select near surface temperature by index and look at its contents:

   .. code-block:: python

      >>> temp = f[1]
      >>> print(temp)
      Field: long_name=near-surface temperature (ncvar%tmp)
      -----------------------------------------------------
      Data            : long_name=near-surface temperature(long_name=time(1452), long_name=latitude(360), long_name=longitude(720)) degrees Celsius
      Dimension coords: long_name=time(1452) = [1901-01-16 00:00:00, ..., 2021-12-16 00:00:00] gregorian
                      : long_name=latitude(360) = [-89.75, ..., 89.75] degrees_north
                      : long_name=longitude(720) = [-179.75, ..., 179.75] degrees_east

4. Select latitude and longitude dimensions by identities using the `~cf.Field.coordinate` method:

   .. code-block:: python

      >>> lon = temp.coordinate('long_name=longitude')
      >>> lat = temp.coordinate('Y')

5. Print the desciption of near surface temperature using the `~cf.Field.dump` method to show properties of all constructs:

   .. code-block:: python

      >>> temp.dump()
      -----------------------------------------------------
      Field: long_name=near-surface temperature (ncvar%tmp)
      -----------------------------------------------------
      Conventions = 'CF-1.4'
      _FillValue = 9.96921e+36
      comment = 'Access to these data is available to any registered CEDA user.'
      contact = 'support@ceda.ac.uk'
      correlation_decay_distance = 1200.0
      history = 'Fri 29 Apr 14:35:01 BST 2022 : User f098 : Program makegridsauto.for
                 called by update.for'
      institution = 'Data held at British Atmospheric Data Centre, RAL, UK.'
      long_name = 'near-surface temperature'
      missing_value = 9.96921e+36
      references = 'Information on the data is available at
                    http://badc.nerc.ac.uk/data/cru/'
      source = 'Run ID = 2204291347. Data generated from:tmp.2204291209.dtb'
      title = 'CRU TS4.06 Mean Temperature'
      units = 'degrees Celsius'

      Data(long_name=time(1452), long_name=latitude(360), long_name=longitude(720)) = [[[--, ..., --]]] degrees Celsius

      Domain Axis: long_name=latitude(360)
      Domain Axis: long_name=longitude(720)
      Domain Axis: long_name=time(1452)

      Dimension coordinate: long_name=time
          calendar = 'gregorian'
          long_name = 'time'
          units = 'days since 1900-1-1'
          Data(long_name=time(1452)) = [1901-01-16 00:00:00, ..., 2021-12-16 00:00:00] gregorian

      Dimension coordinate: long_name=latitude
          long_name = 'latitude'
          units = 'degrees_north'
          Data(long_name=latitude(360)) = [-89.75, ..., 89.75] degrees_north

      Dimension coordinate: long_name=longitude
          long_name = 'longitude'
          units = 'degrees_east'
          Data(long_name=longitude(720)) = [-179.75, ..., 179.75] degrees_east

6. Cell bounds are absent in the dimension coordinates which are created using `~cf.DimensionCoordinate.create_bounds` and set using `~cf.DimensionCoordinate.set_bounds` for latitude and longitude:

   .. code-block:: python

      >>> a = lat.create_bounds()
      >>> lat.set_bounds(a)
      >>> lat.dump()
      Dimension coordinate: long_name=latitude
          long_name = 'latitude'
          units = 'degrees_north'
          Data(360) = [-89.75, ..., 89.75] degrees_north
          Bounds:units = 'degrees_north'
          Bounds:Data(360, 2) = [[-90.0, ..., 90.0]] degrees_north

      >>> b = lon.create_bounds()
      >>> lon.set_bounds(b)
      >>> lon.dump()
      Dimension coordinate: long_name=longitude
          long_name = 'longitude'
          units = 'degrees_east'
          Data(720) = [-179.75, ..., 179.75] degrees_east
          Bounds:units = 'degrees_east'
          Bounds:Data(720, 2) = [[-180.0, ..., 180.0]] degrees_east

      >>> print(b.array)
      [[-180.  -179.5]
       [-179.5 -179. ]
       [-179.  -178.5]
       ...
       [ 178.5  179. ]
       [ 179.   179.5]
       [ 179.5  180. ]]

7. Cell bounds are similarly created and set for the dimension coordinate time with cell size of one calendar month using using `~cf.DimensionCoordinate.create_bounds` and `~cf.DimensionCoordinate.set_bounds` respectively:

   .. code-block:: python

      >>> time = temp.coordinate('long_name=time')
      >>> c = time.create_bounds(cellsize=cf.M())
      >>> time.set_bounds(c)
      >>> time.dump()
      Dimension coordinate: long_name=time
          calendar = 'gregorian'
          long_name = 'time'
          units = 'days since 1900-1-1'
          Data(1452) = [1901-01-16 00:00:00, ..., 2021-12-16 00:00:00] gregorian
          Bounds:calendar = 'gregorian'
          Bounds:units = 'days since 1900-1-1'
          Bounds:Data(1452, 2) = [[1901-01-01 00:00:00, ..., 2022-01-01 00:00:00]] gregorian

8. Calculate and plot the area weighted mean surface temperature using the `~cf.Field.collapse` method:

   .. code-block:: python

      >>> global_avg = temp.collapse('area: mean', weights=True)
      >>> cfp.lineplot(global_avg, color='red', title='Global mean surface temperature')

   .. figure:: images/global_mean_temp.png

9. Calculate and plot the annual global mean surface temperature using `lineplot
<http://ajheaps.github.io/cf-plot/lineplot.html>`:

   .. code-block:: python

      >>> annual_global_avg = global_avg.collapse('T: mean', group=cf.Y())
      >>> cfp.lineplot(annual_global_avg, color='red', title='Annual global mean surface temperature')

   .. figure:: images/annual_mean_temp.png

**Plotting global mean temperatures spatially:**
----------

In this recipe, we will plot the global mean temperature spatially.

1. Import cf-python and cf-plot:

   .. code-block:: python

      >>> import cf
      >>> import cfplot as cfp

2. Read the field constructs using `~cf.read` function:

   .. code-block:: python

      >>> f = cf.read('file.nc')
      >>> print(f)
      [<CF Field: ncvar%stn(long_name=time(1452), long_name=latitude(360), long_name=longitude(720))>,
      <CF Field: long_name=near-surface temperature(long_name=time(1452), long_name=latitude(360), long_name=longitude(720)) degrees Celsius>]


3. Select near surface temperature by index and look at its contents:

   .. code-block:: python

      >>> temp = f[1]
      >>> print(temp)
      Field: long_name=near-surface temperature (ncvar%tmp)
      -----------------------------------------------------
      Data            : long_name=near-surface temperature(long_name=time(1452), long_name=latitude(360), long_name=longitude(720)) degrees Celsius
      Dimension coords: long_name=time(1452) = [1901-01-16 00:00:00, ..., 2021-12-16 00:00:00] gregorian
                      : long_name=latitude(360) = [-89.75, ..., 89.75] degrees_north
                      : long_name=longitude(720) = [-179.75, ..., 179.75] degrees_east

4. Average the monthly mean surface temperature values by the time axis using the `~cf.Field.collapse` method and check the array's dimension size using `~cf.Data.shape`:

   .. code-block:: python

      >>> global_avg = temp.collapse('mean',  axes='long_name=time')
      >>> global_avg.shape
      (1, 360, 720)

5.  As the global_avg data array is 3-dimensional, the time axis is removed using `~cf.Data.squeeze` method so that it could be plottled on a map:

   .. code-block:: python

      >>> global_avg_2d = global_avg.squeeze((0,))
      >>> global_avg_2d.shape
      (360, 720)

6. Plot the global mean surface temperatures using using `con
<http://ajheaps.github.io/cf-plot/con.html>`:

   .. code-block:: python

      >>> cfp.con(global_avg_2d, lines=False, title='Global mean surface temperature')
   .. figure:: images/global_mean_map.png



**Calculating global average temperature anomalies:**
----------


**Comparing two datasets with different resolutions using regridding:**
----------

In this recipe, we will regrid two different datasets with different resolutions. An example use case could be one where the observational dataset with a higher resolution needs to be regridded to that of the model dataset so that they can be compared with each other.

1. Import cf-python:

   .. code-block:: python

      >>> import cf

2. Read the field constructs using `~cf.read` function:

   .. code-block:: python

      >>> obs = cf.read('observation.nc')
      >>> print(obs)
      [<CF Field: ncvar%stn(long_name=time(120), long_name=latitude(360), long_name=longitude(720))>,
      <CF Field: long_name=near-surface temperature(long_name=time(120), long_name=latitude(360), long_name=longitude(720)) degrees Celsius>]

      >>> model = cf.read('model.nc')
      >>> print(model)
      [<CF Field: air_temperature(time(1980), latitude(144), longitude(192)) K>]

3. Select observation and model temperature by index and look at the contents:

   .. code-block:: python

      >>> obs_temp = obs[1]
      >>> print(obs_temp)
      Field: long_name=near-surface temperature (ncvar%tmp)
      -----------------------------------------------------
      Data            : long_name=near-surface temperature(long_name=time(1452), long_name=latitude(360), long_name=longitude(720)) degrees Celsius
      Dimension coords: long_name=time(1452) = [1901-01-16 00:00:00, ..., 2021-12-16 00:00:00] gregorian
                      : long_name=latitude(360) = [-89.75, ..., 89.75] degrees_north
                      : long_name=longitude(720) = [-179.75, ..., 179.75] degrees_east

      >>> model_temp = model[0]
      >>> print(model_temp)
      Field: air_temperature (ncvar%tasmax)
      -------------------------------------
      Data            : air_temperature(time(1980), latitude(144), longitude(192)) K
      Cell methods    : time(1980): maximum (interval: 1 hour)
      Dimension coords: time(1980) = [1850-01-16 00:00:00, ..., 2014-12-16 00:00:00] 360_day
                      : latitude(144) = [-89.375, ..., 89.375] degrees_north
                      : longitude(192) = [0.9375, ..., 359.0625] degrees_east
                      : height(1) = [1.5] m
      Coord references: grid_mapping_name:latitude_longitude

4. Regrid observational data to that of the model data (`~cf.Field.regrids` method) and create a new low resolution observational data using bilinear interpolation:

   .. code-block:: python

      >>> obs_temp_regrid = obs_temp.regrids(model_temp, method='bilinear')
      >>> print(obs_temp_regrid)
      Field: long_name=near-surface temperature (ncvar%tmp)
      -----------------------------------------------------
      Data            : long_name=near-surface temperature(long_name=time(1452), latitude(144), longitude(192)) degrees Celsius
      Dimension coords: long_name=time(1452) = [1901-01-16 00:00:00, ..., 2021-12-16 00:00:00] gregorian
                      : latitude(144) = [-89.375, ..., 89.375] degrees_north
                      : longitude(192) = [0.9375, ..., 359.0625] degrees_east
      Coord references: grid_mapping_name:latitude_longitude


**Plotting an overlay of wind over precipitation data:**
----------


