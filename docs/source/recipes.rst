.. currentmodule:: cf
.. default-role:: obj

.. TODODASK - review this entire section

**Recipes using cf**
===============

----

Version |release| for version |version| of the CF conventions.

.. contents::
   :local:
   :backlinks: entry

**Calculating global mean temperature timeseries**
----------

In this recipe we will calculate and plot monthly and annual global mean temperature timeseries.

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

4. Select latitude and longitude dimensions by identities, with two different techniques, using the `~cf.Field.coordinate` method:

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

6. Latitude and longitude dimension coordinate cell bounds are absent, but they can be created using `~cf.DimensionCoordinate.create_bounds` and set using `~cf.DimensionCoordinate.set_bounds`:

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

7. Time dimension coordinate cell bounds are similarly created and set for cell sizes of one calendar month:

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

8. Calculate and plot the area weighted mean surface temperature for each time using the `~cf.Field.collapse` method:

   .. code-block:: python

      >>> global_avg = temp.collapse('area: mean', weights=True)
      >>> cfp.lineplot(global_avg, color='red', title='Global mean surface temperature')

   .. figure:: images/global_mean_temp.png

9. Calculate and plot the annual global mean surface temperature using `cfplot.lineplot
<http://ajheaps.github.io/cf-plot/lineplot.html>`_:

   .. code-block:: python

      >>> annual_global_avg = global_avg.collapse('T: mean', group=cf.Y())
      >>> cfp.lineplot(annual_global_avg,
      ...     color='red', title='Annual global mean surface temperature')

   .. figure:: images/annual_mean_temp.png

Calculating and plotting the global average temperature anomalies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. The temperature values are averaged for the climatological period of 1961-1990 by defining a subspace within these years using `cf.wi` query instance over `~cf.Field.subspace` and doing a statistical collapse with the `~cf.Field.collapse` method:

   .. code-block:: python

      >>> annual_global_avg_61_90 = annual_global_avg.subspace(T=cf.year(cf.wi(1961, 1990)))
      >>> print(annual_global_avg_61_90)
      Field: long_name=near-surface temperature (ncvar%tmp)
      -----------------------------------------------------
      Data            : long_name=near-surface temperature(long_name=time(30), long_name=latitude(1), long_name=longitude(1)) degrees Celsius
      Cell methods    : area: mean long_name=time(30): mean
      Dimension coords: long_name=time(30) = [1961-07-02 12:00:00, ..., 1990-07-02 12:00:00] gregorian
                      : long_name=latitude(1) = [0.0] degrees_north
                      : long_name=longitude(1) = [0.0] degrees_east

      >>> temp_clim = annual_global_avg_61_90.collapse('T: mean')
      >>> print(temp_clim)
      Field: long_name=near-surface temperature (ncvar%tmp)
      -----------------------------------------------------
      Data            : long_name=near-surface temperature(long_name=time(1), long_name=latitude(1), long_name=longitude(1)) degrees Celsius
      Cell methods    : area: mean long_name=time(1): mean
      Dimension coords: long_name=time(1) = [1976-01-01 12:00:00] gregorian
                      : long_name=latitude(1) = [0.0] degrees_north
                      : long_name=longitude(1) = [0.0] degrees_east

2. The temperature anomaly is then calculated by subtracting these climatological temperature values from the annual global average temperatures and plotting them using `lineplot
<http://ajheaps.github.io/cf-plot/lineplot.html>`_:

   .. code-block:: python

      >>> temp_anomaly = annual_global_avg - temp_clim
      >>> cfp.lineplot(temp_anomaly, 
      ...     color='red',
      ...     title='Global Average Temperature Anomaly (1901-2021)',
      ...     ylabel='1961-1990 climatology difference ',
      ...     yunits='degree Celcius')

   .. figure:: images/anomaly.png

----
----

**Plotting global mean temperatures spatially**
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

4. Average the monthly mean surface temperature values by the time axis using the `~cf.Field.collapse` method:

   .. code-block:: python

      >>> global_avg = temp.collapse('mean',  axes='long_name=time')


5. Plot the global mean surface temperatures using using `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_:

   .. code-block:: python

      >>> cfp.con(global_avg, lines=False, title='Global mean surface temperature')
   .. figure:: images/global_mean_map.png

----

**Comparing two datasets with different resolutions using regridding**
----------

In this recipe, we will regrid two different datasets with different resolutions. An example use case could be one where the observational dataset with a higher resolution needs to be regridded to that of the model dataset so that they can be compared with each other.

1. Import cf-python:

   .. code-block:: python

      >>> import cf

2. Read the field constructs using `~cf.read` function:

   .. code-block:: python

      >>> obs = cf.read('observation.nc')
      >>> print(obs)
      [<CF Field: ncvar%stn(long_name=time(1452), long_name=latitude(360), long_name=longitude(720))>,
      <CF Field: long_name=near-surface temperature(long_name=time(1452), long_name=latitude(360), long_name=longitude(720)) degrees Celsius>]

      >>> model = cf.read('model.nc')
      >>> print(model)
      [<CF Field: air_temperature(time(1980), latitude(144), longitude(192)) K>]

3. Select observation and model temperature fields by identity and index respectively, and look at their contents:

   .. code-block:: python

      >>> obs_temp = obs.select_field('long_name=near-surface temperature')
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

      >>> obs_temp_regrid = obs_temp.regrids(model_temp, method='linear')
      >>> print(obs_temp_regrid)
      Field: long_name=near-surface temperature (ncvar%tmp)
      -----------------------------------------------------
      Data            : long_name=near-surface temperature(long_name=time(1452), latitude(144), longitude(192)) degrees Celsius
      Dimension coords: long_name=time(1452) = [1901-01-16 00:00:00, ..., 2021-12-16 00:00:00] gregorian
                      : latitude(144) = [-89.375, ..., 89.375] degrees_north
                      : longitude(192) = [0.9375, ..., 359.0625] degrees_east
      Coord references: grid_mapping_name:latitude_longitude

----

**Plotting wind vectors overlaid on precipitation data**
----------

In this recipe we will plot wind vectors, derived from northward and eastward wind components, over precipitation data.

1. Import cf-python and cf-plot:

   .. code-block:: python

      >>> import cf
      >>> import cfplot as cfp

2. Read the field constructs using `cf.read` function:

   .. code-block:: python

      >>> f1 = cf.read('northward.nc')
      >>> print(f1)
      [<CF Field: northward_wind(time(1980), latitude(144), longitude(192)) m s-1>]
      
      >>> f2 = cf.read('eastward.nc')
      >>> print(f2)
      [<CF Field: eastward_wind(time(1980), latitude(144), longitude(192)) m s-1>]

      >>> f3 = cf.read('monthly_precipitation.nc')
      >>> print(f3)
      [<CF Field: long_name=precipitation(long_name=time(1452), latitude(144), longitude(192)) mm/month>]

3. Select wind vectors and precipitation data by index and look at their contents:

   .. code-block:: python

      >>> v = f1[0]
      >>> print(v)
      Field: northward_wind (ncvar%vas)
      ---------------------------------
      Data            : northward_wind(time(1980), latitude(144), longitude(192)) m s-1
      Cell methods    : area: time(1980): mean
      Dimension coords: time(1980) = [1850-01-16 00:00:00, ..., 2014-12-16 00:00:00] 360_day
                      : latitude(144) = [-89.375, ..., 89.375] degrees_north
                      : longitude(192) = [0.0, ..., 358.125] degrees_east
                      : height(1) = [10.0] m

      >>> u = f2[0]
      >>> print(u)
      Field: eastward_wind (ncvar%uas)
      --------------------------------
      Data            : eastward_wind(time(1980), latitude(144), longitude(192)) m s-1
      Cell methods    : area: time(1980): mean
      Dimension coords: time(1980) = [1850-01-16 00:00:00, ..., 2014-12-16 00:00:00] 360_day
                      : latitude(144) = [-89.375, ..., 89.375] degrees_north
                      : longitude(192) = [0.0, ..., 358.125] degrees_east
                      : height(1) = [10.0] m

      >>> pre = f3[0]
      >>> print(pre)
      Field: long_name=precipitation (ncvar%pre)
      ------------------------------------------
      Data            : long_name=precipitation(long_name=time(1452), latitude(144), longitude(192)) mm/month
      Dimension coords: long_name=time(1452) = [1901-01-16 00:00:00, ..., 2021-12-16 00:00:00] gregorian
                      : latitude(144) = [-89.375, ..., 89.375] degrees_north
                      : longitude(192) = [0.0, ..., 358.125] degrees_east

4. Plot the wind vectors on top of precipitation data for June 1995 by creating a subspace (`~cf.Field.subspace`) with a date-time object (`cf.dt`) and using `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_. Here `cfplot.gopen <http://ajheaps.github.io/cf-plot/gopen.html>`_ is used to define the parts of the plot area, which is closed by `cfplot.gclose <http://ajheaps.github.io/cf-plot/gclose.html>`_; `cfplot.cscale <http://ajheaps.github.io/cf-plot/cscale.html>`_ is used to choose one of the colour maps amongst many available; `cfplot.levs <http://ajheaps.github.io/cf-plot/levs.html>`_ is used to set the contour levels for precipitation data; and `cfplot.vect <http://ajheaps.github.io/cf-plot/vect.html>`_ is used to plot the wind vectors for June 1995:

   .. code-block:: python

      >>> june_95 = cf.year(1995) & cf.month(6)
      >>> cfp.gopen()
      >>> cfp.cscale('precip4_11lev')
      >>> cfp.levs(step=100)
      >>> cfp.con(pre.subspace(T=june_95),
      ...         lines=False, title = 'June 1995 monthly global precipitation')
      >>> cfp.vect(u=u.subspace(T=june_95), v=v.subspace(T=june_95),
      ...          key_length=10, scale=35, stride=5)
      >>> cfp.gclose()
   .. figure:: images/june1995_preci.png

----

**Converting from rotated latitude-longitude to regular latitude-longitude**
----------

In this recipe, we will be regridding from a rotated latitude-longitude source domain to a regular latitude-longitude destination domain.

1. Import cf-python, cf-plot and numpy:

   .. code-block:: python

      >>> import cf
      >>> import cfplot as cfp
      >>> import numpy as np

2. Read the field constructs of the UK Met Office PP format file using `~cf.read` function:

   .. code-block:: python

      >>> f = cf.read('file.pp')
      >>> print(f)
      [<CF Field: id%UM_m01s03i463_vn1006(time(8), grid_latitude(432), grid_longitude(444))>]

3. Select the field by index and print its desciption using the `~cf.Field.dump` method to show properties of all constructs:

   .. code-block:: python

      >>> gust = f[0]
      >>> gust.dump()
      -----------------------------------------------------------
      Field: id%UM_m01s03i463_vn1006 (ncvar%UM_m01s03i463_vn1006)
      -----------------------------------------------------------
      Conventions = 'CF-1.10'
      _FillValue = -1073741824.0
      history = 'Converted from UM/PP by cf-python v3.14.0'
      lbproc = '8192'
      lbtim = '122'
      long_name = 'WIND GUST'
      runid = 'aaaaa'
      source = 'UM vn1006'
      stash_code = '3463'
      submodel = '1'
      um_stash_source = 'm01s03i463'

      Data(time(8), grid_latitude(432), grid_longitude(444)) = [[[5.587890625, ..., 5.1376953125]]]

      Cell Method: time(8): maximum

      Domain Axis: grid_latitude(432)
      Domain Axis: grid_longitude(444)
      Domain Axis: height(1)
      Domain Axis: time(8)

      Dimension coordinate: time
          axis = 'T'
          calendar = '360_day'
          standard_name = 'time'
          units = 'days since 2051-1-1'
          Data(time(8)) = [2051-04-14 01:30:00, ..., 2051-04-14 22:30:00] 360_day
          Bounds:calendar = '360_day'
          Bounds:units = 'days since 2051-1-1'
          Bounds:Data(time(8), 2) = [[2051-04-14 00:00:00, ..., 2051-04-15 00:00:00]] 360_day

      Dimension coordinate: height
          axis = 'Z'
          positive = 'up'
          standard_name = 'height'
          units = 'm'
          Data(height(1)) = [-1.0] m

      Dimension coordinate: grid_latitude
          axis = 'Y'
          standard_name = 'grid_latitude'
          units = 'degrees'
          Data(grid_latitude(432)) = [-24.474999085068703, ..., 22.93500065803528] degrees
          Bounds:units = 'degrees'
          Bounds:Data(grid_latitude(432), 2) = [[-24.52999908477068, ..., 22.990000657737255]] degrees

      Dimension coordinate: grid_longitude
          axis = 'X'
          standard_name = 'grid_longitude'
          units = 'degrees'
          Data(grid_longitude(444)) = [-29.47499145567417, ..., 19.255008280277252] degrees
          Bounds:units = 'degrees'
          Bounds:Data(grid_longitude(444), 2) = [[-29.52999145537615, ..., 19.31000827997923]] degrees

      Auxiliary coordinate: latitude
          standard_name = 'latitude'
          units = 'degrees_north'
          Data(grid_latitude(432), grid_longitude(444)) = [[20.576467692711244, ..., 66.9022518505943]] degrees_north
          Bounds:units = 'degrees_north'
          Bounds:Data(grid_latitude(432), grid_longitude(444), 4) = [[[20.505853650744182, ..., 66.82752183591477]]] degrees_north

      Auxiliary coordinate: longitude
          standard_name = 'longitude'
          units = 'degrees_east'
          Data(grid_latitude(432), grid_longitude(444)) = [[-10.577446822867152, ..., 68.72895292160312]] degrees_east
          Bounds:units = 'degrees_east'
          Bounds:Data(grid_latitude(432), grid_longitude(444), 4) = [[[-10.602339269012656, ..., 68.73573608505069]]] degrees_east

      Coordinate reference: grid_mapping_name:rotated_latitude_longitude
          Coordinate conversion:grid_mapping_name = rotated_latitude_longitude
          Coordinate conversion:grid_north_pole_latitude = 39.25
          Coordinate conversion:grid_north_pole_longitude = 198.0
          Dimension Coordinate: grid_longitude
          Dimension Coordinate: grid_latitude
          Auxiliary Coordinate: longitude
          Auxiliary Coordinate: latitude


4. Access the time coordinate of the gust field using `~cf.Field.coordinate` method and retrieve the datetime values of the time coordinate using `~cf.DimensionCoordinate.datetime_array`:

   .. code-block:: python

      >>> print(gust.coordinate('time').datetime_array)
      [cftime.Datetime360Day(2051, 4, 14, 1, 30, 0, 0, has_year_zero=True)
       cftime.Datetime360Day(2051, 4, 14, 4, 30, 0, 0, has_year_zero=True)
       cftime.Datetime360Day(2051, 4, 14, 7, 30, 0, 0, has_year_zero=True)
       cftime.Datetime360Day(2051, 4, 14, 10, 30, 0, 0, has_year_zero=True)
       cftime.Datetime360Day(2051, 4, 14, 13, 30, 0, 0, has_year_zero=True)
       cftime.Datetime360Day(2051, 4, 14, 16, 30, 0, 0, has_year_zero=True)
       cftime.Datetime360Day(2051, 4, 14, 19, 30, 0, 0, has_year_zero=True)
       cftime.Datetime360Day(2051, 4, 14, 22, 30, 0, 0, has_year_zero=True)]

5. Create a new instance of the `cf.dt` class with a specified year, month, day, hour, minute, second and microsecond. Then store the result in the variable test:

   .. code-block:: python

      >>> test = cf.dt(2051, 4, 14, 1, 30, 0, 0)
      >>> print(test)
      2051-04-14 01:30:00

6. Plot the wind gust by creating a `~cf.Field.subspace` for the specified variable ``test`` using `cfplot.con <http://ajheaps.github.io/cf-plot/con.html>`_. Here `cfplot.mapset <http://ajheaps.github.io/cf-plot/mapset.html>`_ is used to set the mapping parameters like setting the map resolution to 50m:

   .. code-block:: python

      >>> cfp.mapset(resolution='50m')
      >>> cfp.con(gust.subspace(T=test), lines=False)
   .. figure:: images/windgust.png

7. To see the rotated pole data on the native grid, the above steps are repeated and projection is set to rotated in `cfplot.mapset <http://ajheaps.github.io/cf-plot/mapset.html>`_:

   .. code-block:: python

      >>> cfp.mapset(resolution='50m', proj='rotated')
      >>> cfp.con(gust.subspace(T=test), lines=False)
   .. figure:: images/windgust_rotated.png

8. Create dimension coordinates for the destination grid with the latitude and longitude values for Europe using `~cf.DimensionCoordinate` class. `np.linspace <https://numpy.org/doc/stable/reference/generated/numpy.linspace.html>`_ generates evenly spaced values between the specified latitude and longitude range. Bounds of the target longitude and target latitude are created using `~cf.DimensionCoordinate.create_bounds` method. Spherical regridding is then performed on the gust variable by calling the `~cf.Field.regrids` method and passing the target latitude and target longitude as arguments. The method also takes an argument ``'linear'`` which specifies the type of regridding method to use. The desciption of the ``regridded_data`` is finally printed using the `~cf.Field.dump` method to show properties of all its constructs:

   .. code-block:: python

      >>> target_latitude = cf.DimensionCoordinate(data=cf.Data(np.linspace(34, 72, num=30), 'degrees_north'))
      >>> target_longitude = cf.DimensionCoordinate(data=cf.Data(np.linspace(-25, 45, num=30), 'degrees_east'))

      >>> lon_bounds = target_longitude.create_bounds()
      >>> lat_bounds = target_latitude.create_bounds()

      >>> target_longitude.set_bounds(lon_bounds)
      >>> target_latitude.set_bounds(lat_bounds)

      >>> regridded_data = gust.regrids((target_latitude, target_longitude), 'linear')
      >>> regridded_data.dump()
      -----------------------------------------------------------
      Field: id%UM_m01s03i463_vn1006 (ncvar%UM_m01s03i463_vn1006)
      -----------------------------------------------------------
      Conventions = 'CF-1.10'
      _FillValue = -1073741824.0
      history = 'Converted from UM/PP by cf-python v3.14.0'
      lbproc = '8192'
      lbtim = '122'
      long_name = 'WIND GUST'
      runid = 'aaaaa'
      source = 'UM vn1006'
      stash_code = '3463'
      submodel = '1'
      um_stash_source = 'm01s03i463'

      Data(time(8), latitude(30), longitude(30)) = [[[--, ..., 6.108851101534697]]]

      Cell Method: time(8): maximum

      Domain Axis: height(1)
      Domain Axis: latitude(30)
      Domain Axis: longitude(30)
      Domain Axis: time(8)

      Dimension coordinate: time
          axis = 'T'
          calendar = '360_day'
          standard_name = 'time'
          units = 'days since 2051-1-1'
          Data(time(8)) = [2051-04-14 01:30:00, ..., 2051-04-14 22:30:00] 360_day
          Bounds:calendar = '360_day'
          Bounds:units = 'days since 2051-1-1'
          Bounds:Data(time(8), 2) = [[2051-04-14 00:00:00, ..., 2051-04-15 00:00:00]] 360_day

      Dimension coordinate: height
          axis = 'Z'
          positive = 'up'
          standard_name = 'height'
          units = 'm'
          Data(height(1)) = [-1.0] m

      Dimension coordinate: latitude
          standard_name = 'latitude'
          units = 'degrees_north'
          Data(latitude(30)) = [34.0, ..., 72.0] degrees_north
          Bounds:units = 'degrees_north'
          Bounds:Data(latitude(30), 2) = [[33.3448275862069, ..., 72.65517241379311]] degrees_north

      Dimension coordinate: longitude
          standard_name = 'longitude'
          units = 'degrees_east'
          Data(longitude(30)) = [-25.0, ..., 45.0] degrees_east
          Bounds:units = 'degrees_east'
          Bounds:Data(longitude(30), 2) = [[-26.20689655172414, ..., 46.20689655172414]] degrees_east

9. Step 6 is similarly repeated for the ``regridded_data`` to plot the wind gust on a regular latitude-longitude domain:

   .. code-block:: python

      >>> cfp.mapset(resolution='50m')
      >>> cfp.con(regridded_data.subspace(T=test), lines=False)
   .. figure:: images/windgust_regridded.png

----