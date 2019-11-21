import datetime
import os
import unittest

import numpy
import netCDF4

import cf


def _make_contiguous_file(filename):        
    n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
    
    n.Conventions = 'CF-1.7'
    n.featureType = 'timeSeries'

    station = n.createDimension('station', 4)
    obs     = n.createDimension('obs'    , 24)
    name_strlen = n.createDimension('name_strlen', 8)
    bounds  = n.createDimension('bounds', 2)
  
    lon = n.createVariable('lon', 'f8', ('station',))
    lon.standard_name = "longitude"
    lon.long_name = "station longitude"
    lon.units = "degrees_east"
    lon.bounds = "lon_bounds"
    lon[...] = [-23, 0, 67, 178]
    
    lon_bounds = n.createVariable('lon_bounds', 'f8', ('station', 'bounds'))
    lon_bounds[...] = [[-24, -22],
                       [ -1,   1],
                       [ 66,  68],
                       [177, 179]]
    
    lat = n.createVariable('lat', 'f8', ('station',))
    lat.standard_name = "latitude"
    lat.long_name = "station latitude" 
    lat.units = "degrees_north"
    lat[...] = [-9, 2, 34, 78]
    
    alt = n.createVariable('alt', 'f8', ('station',))
    alt.long_name = "vertical distance above the surface" 
    alt.standard_name = "height" 
    alt.units = "m"
    alt.positive = "up"
    alt.axis = "Z"
    alt[...] = [0.5, 12.6, 23.7, 345]
    
    station_name = n.createVariable('station_name', 'S1',
                                    ('station', 'name_strlen'))
    station_name.long_name = "station name"
    station_name.cf_role = "timeseries_id"
    station_name[...] = numpy.array([[x for x in 'station1'],
                                     [x for x in 'station2'],
                                     [x for x in 'station3'],
                                     [x for x in 'station4']])
    
    station_info = n.createVariable('station_info', 'i4', ('station',))
    station_info.long_name = "some kind of station info"
    station_info[...] = [-10, -9, -8, -7]
    
    row_size = n.createVariable('row_size', 'i4', ('station',))
    row_size.long_name = "number of observations for this station"
    row_size.sample_dimension = "obs"
    row_size[...] = [3, 7, 5, 9]
    
    time = n.createVariable('time', 'f8', ('obs',))
    time.standard_name = "time"
    time.long_name = "time of measurement" 
    time.units = "days since 1970-01-01 00:00:00"
    time.bounds = "time_bounds"
    time[ 0: 3] = [-3, -2, -1]
    time[ 3:10] = [1, 2, 3, 4, 5, 6, 7]
    time[10:15] = [0.5, 1.5, 2.5, 3.5, 4.5]
    time[15:24] = range(-2, 7)
    
    time_bounds = n.createVariable('time_bounds', 'f8', ('obs', 'bounds'))
    time_bounds[..., 0] = time[...] - 0.5
    time_bounds[..., 1] = time[...] + 0.5
     
    humidity = n.createVariable('humidity', 'f8', ('obs',), fill_value=-999.9)
    humidity.standard_name = "specific_humidity"
    humidity.coordinates = "time lat lon alt station_name station_info"
    humidity[ 0: 3] = numpy.arange(0, 3)
    humidity[ 3:10] = numpy.arange(1, 71, 10)
    humidity[10:15] = numpy.arange(2, 502, 100)
    humidity[15:24] = numpy.arange(3, 9003, 1000)
    
    temp = n.createVariable('temp', 'f8', ('obs',), fill_value=-999.9)
    temp.standard_name = "air_temperature"
    temp.units = "Celsius"
    temp.coordinates = "time lat lon alt station_name station_info"
    temp[...] = humidity[...] + 273.15
    
    n.close()
    
    return filename


def _make_indexed_file(filename):        
    n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
    
    n.Conventions = 'CF-1.7'
    n.featureType = 'timeSeries'
    
    station = n.createDimension('station', 4)
    obs     = n.createDimension('obs'    , None)
    name_strlen = n.createDimension('name_strlen', 8)
    bounds  = n.createDimension('bounds', 2)
    
    lon = n.createVariable('lon', 'f8', ('station',))
    lon.standard_name = "longitude"
    lon.long_name = "station longitude"
    lon.units = "degrees_east"
    lon.bounds = "lon_bounds"
    lon[...] = [-23, 0, 67, 178]
    
    lon_bounds = n.createVariable('lon_bounds', 'f8', ('station', 'bounds'))
    lon_bounds[...] = [[-24, -22],
                       [ -1,   1],
                       [ 66,  68],
                       [177, 179]]
    
    lat = n.createVariable('lat', 'f8', ('station',))
    lat.standard_name = "latitude"
    lat.long_name = "station latitude" 
    lat.units = "degrees_north"
    lat[...] = [-9, 2, 34, 78]
    
    alt = n.createVariable('alt', 'f8', ('station',))
    alt.long_name = "vertical distance above the surface" 
    alt.standard_name = "height" 
    alt.units = "m"
    alt.positive = "up"
    alt.axis = "Z"
    alt[...] = [0.5, 12.6, 23.7, 345]
    
    station_name = n.createVariable('station_name', 'S1',
                                    ('station', 'name_strlen'))
    station_name.long_name = "station name"
    station_name.cf_role = "timeseries_id"
    station_name[...] = numpy.array([[x for x in 'station1'],
                                     [x for x in 'station2'],
                                     [x for x in 'station3'],
                                     [x for x in 'station4']])
    
    station_info = n.createVariable('station_info', 'i4', ('station',))
    station_info.long_name = "some kind of station info"
    station_info[...] = [-10, -9, -8, -7]
    
    #row_size[...] = [3, 7, 5, 9]
    stationIndex = n.createVariable('stationIndex', 'i4', ('obs',))
    stationIndex.long_name = "which station this obs is for"
    stationIndex.instance_dimension= "station"
    stationIndex[...] = [3, 2, 1, 0, 2, 3, 3, 3, 1, 1, 0, 2, 
                         3, 1, 0, 1, 2, 3, 2, 3, 3, 3, 1, 1]
    
    t = [[-3, -2, -1],
         [1, 2, 3, 4, 5, 6, 7],
         [0.5, 1.5, 2.5, 3.5, 4.5],
         range(-2, 7)]
    
    time = n.createVariable('time', 'f8', ('obs',))
    time.standard_name = "time";
    time.long_name = "time of measurement" 
    time.units = "days since 1970-01-01 00:00:00"
    time.bounds = "time_bounds"
    ssi = [0, 0, 0, 0]
    for i, si in enumerate(stationIndex[...]):
        time[i] = t[si][ssi[si]]
        ssi[si] += 1

    time_bounds = n.createVariable('time_bounds', 'f8', ('obs', 'bounds'))
    time_bounds[..., 0] = time[...] - 0.5
    time_bounds[..., 1] = time[...] + 0.5
    
    humidity = n.createVariable('humidity', 'f8', ('obs',), fill_value=-999.9)
    humidity.standard_name = "specific_humidity"
    humidity.coordinates = "time lat lon alt station_name station_info"
    
    h = [numpy.arange(0, 3),
         numpy.arange(1, 71, 10),
         numpy.arange(2, 502, 100),
         numpy.arange(3, 9003, 1000)]
    
    ssi = [0, 0, 0, 0]
    for i, si in enumerate(stationIndex[...]):
        humidity[i] = h[si][ssi[si]]
        ssi[si] += 1
        
    temp = n.createVariable('temp', 'f8', ('obs',), fill_value=-999.9)
    temp.standard_name = "air_temperature"
    temp.units = "Celsius"
    temp.coordinates = "time lat lon alt station_name station_info"
    temp[...] = humidity[...] + 273.15
    
    n.close()
    
    return filename


def _make_indexed_contiguous_file(filename):        
    n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
    
    n.Conventions = 'CF-1.6'
    n.featureType = "timeSeriesProfile"

    # 3 stations
    station     = n.createDimension('station', 3)
    # 58 profiles spreadover 4 stations, each at a different time
    profile     = n.createDimension('profile', 58) 
    obs         = n.createDimension('obs'    , None)
    name_strlen = n.createDimension('name_strlen', 8)
    bounds      = n.createDimension('bounds', 2)
    
    lon = n.createVariable('lon', 'f8', ('station',))
    lon.standard_name = "longitude"
    lon.long_name = "station longitude"
    lon.units = "degrees_east"
    lon.bounds = "lon_bounds"
    lon[...] = [-23, 0, 67]
    
    lon_bounds = n.createVariable('lon_bounds', 'f8', ('station', 'bounds'))
    lon_bounds[...] = [[-24, -22],
                       [ -1,   1],
                       [ 66,  68]]
    
    lat = n.createVariable('lat', 'f8', ('station',))
    lat.standard_name = "latitude"
    lat.long_name = "station latitude" 
    lat.units = "degrees_north"
    lat[...] = [-9, 2, 34]
    
    alt = n.createVariable('alt', 'f8', ('station',))
    alt.long_name = "vertical distance above the surface" 
    alt.standard_name = "height" 
    alt.units = "m"
    alt.positive = "up"
    alt.axis = "Z"
    alt[...] = [0.5, 12.6, 23.7]
    
    station_name = n.createVariable('station_name', 'S1',
                                    ('station', 'name_strlen'))
    station_name.long_name = "station name"
    station_name.cf_role = "timeseries_id"
    station_name[...] = numpy.array([[x for x in 'station1'],
                                     [x for x in 'station2'],
                                     [x for x in 'station3']])
    
    profile = n.createVariable('profile', 'i4', ('profile'))
    profile.cf_role = "profile_id"
    profile[...] = numpy.arange(58) + 100
    
    station_info = n.createVariable('station_info', 'i4', ('station',))
    station_info.long_name = "some kind of station info"
    station_info[...] = [-10, -9, -8]
    
    stationIndex = n.createVariable('stationIndex', 'i4', ('profile',))
    stationIndex.long_name = "which station this profile is for"
    stationIndex.instance_dimension= "station"
    stationIndex[...] = [2, 1, 0, 2, 1, 1, 0, 2, 
                         1, 0, 1, 2, 2, 1, 1,                     
                         2, 1, 0, 2, 1, 1, 0, 2, 
                         1, 0, 1, 2, 2, 1, 1,
                         2, 1, 0, 2, 1, 1, 0, 2, 
                         1, 0, 1, 2, 2, 1, 1,
                         2, 1, 0, 2, 1, 1, 0, 2, 
                         1, 0, 1, 2, 2]
    # station N has list(stationIndex[...]).count(N) profiles
    
    row_size = n.createVariable('row_size', 'i4', ('profile',))
    row_size.long_name = "number of observations for this profile"
    row_size.sample_dimension = "obs"
    row_size[...] = [1, 4, 1, 3, 2, 2, 3, 3, 1, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 3, 3, 2, 1,
                     3, 1, 3, 2, 3, 1, 3, 3, 2, 2, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1, 3, 3, 2,
                     2, 2, 2, 1, 2, 3, 3, 3, 2, 3, 1, 1] # sum = 118
    
    time = n.createVariable('time', 'f8', ('profile',))
    time.standard_name = "time"
    time.long_name = "time"
    time.units = "days since 1970-01-01 00:00:00"
    time.bounds = "time_bounds"
    t0 = [3, 0, -3]
    ssi = [0, 0, 0]
    for i, si in enumerate(stationIndex[...]):
        time[i] = t0[si] + ssi[si]
        ssi[si] += 1
        
    time_bounds = n.createVariable('time_bounds', 'f8', ('profile', 'bounds'))
    time_bounds[..., 0] = time[...] - 0.5
    time_bounds[..., 1] = time[...] + 0.5

    z = n.createVariable('z', 'f8', ('obs',))
    z.standard_name = "altitude"
    z.long_name = "height above mean sea level"
    z.units = "km"
    z.axis = "Z"  
    z.positive = "up"
    z.bounds = "z_bounds"

#        z0 = [1, 0, 3]
#        i = 0
#        for s, r in zip(stationIndex[...], row_size[...]):
#            z[i:i+r] = z0[s] + numpy.sort(numpy.random.uniform(0, numpy.random.uniform(1, 2), r))
#            i += r

    data =  [3.51977705293769, 0.521185292100177, 0.575154265863394, 
             1.08495843717095, 1.37710968624395, 2.07123455611723, 3.47064474274781, 
             3.88569849023813, 4.81069254279537, 0.264339600625496, 0.915704970094182, 
             0.0701532210336895, 0.395517651420933, 1.00657582854276, 
             1.17721374303641, 1.82189345615046, 3.52424307197668, 3.93200473199559, 
             3.95715099603671, 1.57047493027102, 1.09938982652955, 1.17768722826975, 
             0.251803399458277, 1.59673486865804, 4.02868944763605, 4.03749228832264, 
             4.79858281590985, 3.00019933315412, 3.65124061660449, 0.458463542157766, 
             0.978678197083262, 0.0561560792556281, 0.31182013232255, 
             3.33350065357286, 4.33143904011861, 0.377894196412131, 1.63020681064712, 
             2.00097025264771, 3.76948048424458, 0.572927165845568, 1.29408313557905, 
             1.81296270533192, 0.387142669131077, 0.693459187515738, 1.69261930636298, 
             1.38258797228361, 1.82590759889566, 3.34993297710761, 0.725250730922501, 
             1.38221693486728, 1.59828555215646, 1.59281225554253, 0.452340646918555, 
             0.976663373825433, 1.12640496317618, 3.19366847375422, 3.37209133117904, 
             3.40665008236976, 3.53525896684001, 4.10444186715724, 0.14920937817654, 
             0.0907197953552753, 0.42527916794473, 0.618685137936187, 
             3.01900591447357, 3.37205542289986, 3.86957342976163, 0.17175098751914, 
             0.990040375014957, 1.57011428605984, 2.12140567043994, 3.24374743730506, 
             4.24042441581785, 0.929509749153725, 0.0711997786817564, 
             2.25090028461898, 3.31520955860746, 3.49482624434274, 3.96812568493549, 
             1.5681807261767, 1.79993011515465, 0.068325990211909, 0.124469638352167, 
             3.31990436971169, 3.84766748039389, 0.451973490541035, 1.24303219956085, 
             1.30478004656262, 0.351892459787624, 0.683685812990457, 
             0.788883736575568, 3.73033428872491, 3.99479807507392, 0.811582011950481, 
             1.2241242448019, 1.25563109687369, 2.16603674712822, 3.00010622131408, 
             3.90637137662453, 0.589586644805982, 0.104656387266266, 
             0.961185900148304, 1.05120351477824, 1.29460917520233, 2.10139985693684, 
             3.64252693587415, 3.91197236350995, 4.56466622863717, 0.556476687600461, 
             0.783717448678148, 0.910917550635007, 1.59750076220451, 1.97101264162631, 
             0.714693043642084, 0.904381625638779, 1.03767817888021, 4.10124675852254, 
             3.1059214185543]        
    data = numpy.around(data, 2)
    z[...] = data
       
    z_bounds = n.createVariable('z_bounds', 'f8', ('obs', 'bounds'))
    z_bounds[..., 0] = z[...] - 0.01
    z_bounds[..., 1] = z[...] + 0.01

    humidity = n.createVariable('humidity', 'f8', ('obs',), fill_value=-999.9)
    humidity.standard_name = "specific_humidity"
    humidity.coordinates = "time lat lon alt z station_name station_info profile"
    
    data *= 10
    data = numpy.around(data, 2)
    humidity[...] = data
    
    temp = n.createVariable('temp', 'f8', ('obs',), fill_value=-999.9)
    temp.standard_name = "air_temperature"
    temp.units = "Celsius"
    temp.coordinates = "time lat lon alt z station_name station_info profile"
    
    data += 2731.5
    data = numpy.around(data, 2)
    temp[...] = data
    
    n.close()
    
    return filename


contiguous_file = _make_contiguous_file('DSG_timeSeries_contiguous.nc')
indexed_file    = _make_indexed_file('DSG_timeSeries_indexed.nc')
indexed_contiguous_file = _make_indexed_contiguous_file('DSG_timeSeriesProfile_indexed_contiguous.nc')


def _make_external_files():
    '''
    '''
    def _pp(filename, parent=False, external=False, combined=False, external_missing=False):
        '''
        '''
        nc = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
        
        nc.createDimension('grid_latitude', 10)
        nc.createDimension('grid_longitude', 9)
        
        nc.Conventions = 'CF-1.7'
        if parent:
            nc.external_variables = 'areacella'

        if parent or combined or external_missing:
            grid_latitude = nc.createVariable(dimensions=('grid_latitude',),
                                              datatype='f8',
                                              varname='grid_latitude')
            grid_latitude.setncatts({'units': 'degrees', 'standard_name': 'grid_latitude'})
            grid_latitude[...] = range(10)
            
            grid_longitude = nc.createVariable(dimensions=('grid_longitude',),
                                               datatype='f8',
                                               varname='grid_longitude')
            grid_longitude.setncatts({'units': 'degrees', 'standard_name': 'grid_longitude'})
            grid_longitude[...] = range(9)
            
            latitude = nc.createVariable(dimensions=('grid_latitude', 'grid_longitude'),
                                         datatype='i4',
                                         varname='latitude')
            latitude.setncatts({'units': 'degree_N', 'standard_name': 'latitude'})
            
            latitude[...] = numpy.arange(90).reshape(10, 9)
            
            longitude = nc.createVariable(dimensions=('grid_longitude', 'grid_latitude'),
                                          datatype='i4',
                                          varname='longitude')
            longitude.setncatts({'units': 'degreeE', 'standard_name': 'longitude'})
            longitude[...] = numpy.arange(90).reshape(9, 10)
            
            eastward_wind = nc.createVariable(dimensions=('grid_latitude', 'grid_longitude'),
                                              datatype='f8',
                                              varname=u'eastward_wind')
            eastward_wind.coordinates = u'latitude longitude'
            eastward_wind.standard_name = 'eastward_wind'
            eastward_wind.cell_methods = 'grid_longitude: mean (interval: 1 day comment: ok) grid_latitude: maximum where sea'
            eastward_wind.cell_measures = 'area: areacella'
            eastward_wind.units = 'm s-1'
            eastward_wind[...] = numpy.arange(90).reshape(10, 9) - 45.5

        if external or combined:                
            areacella = nc.createVariable(dimensions=('grid_longitude', 'grid_latitude'),
                                          datatype='f8',
                                          varname='areacella')
            areacella.setncatts({'units': 'm2', 'standard_name': 'cell_area'})
            areacella[...] = numpy.arange(90).reshape(9, 10) + 100000.5
            
        nc.close()
    #--- End: def

    parent_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'parent.nc')        
    external_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'external.nc')            
    combined_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'combined.nc')        
    external_missing_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'external_missing.nc')            
    
    _pp(parent_file          , parent=True)
    _pp(external_file        , external=True)
    _pp(combined_file        , combined=True)
    _pp(external_missing_file, external_missing=True)

    return parent_file, external_file, combined_file, external_missing_file


(parent_file,
 external_file,
 combined_file,
 external_missing_file) = _make_external_files()


def _make_gathered_file(filename):
    '''
    '''
    def _jj(shape, list_values):
        array = numpy.ma.masked_all(shape)
        for i, (index, x) in enumerate(numpy.ndenumerate(array)):
            if i in list_values:
                array[index] = i
        return array
    #--- End: def    

    n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
    
    n.Conventions = 'CF-1.6'
    
    time    = n.createDimension('time'   ,  2)
    height  = n.createDimension('height' ,  3)
    lat     = n.createDimension('lat'    ,  4)
    lon     = n.createDimension('lon'    ,  5)
    p       = n.createDimension('p'      ,  6)
    
    list1  = n.createDimension('list1',  4)
    list2  = n.createDimension('list2',  9)
    list3  = n.createDimension('list3', 14)
    
    # Dimension coordinate variables
    time = n.createVariable('time', 'f8', ('time',))
    time.standard_name = "time"
    time.units = "days since 2000-1-1"
    time[...] = [31, 60]
    
    height = n.createVariable('height', 'f8', ('height',))
    height.standard_name = "height"
    height.units = "metres"
    height.positive = "up"
    height[...] = [0.5, 1.5, 2.5]
    
    lat = n.createVariable('lat', 'f8', ('lat',))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat[...] = [-90, -85, -80, -75]
    
    p = n.createVariable('p', 'i4', ('p',))
    p.long_name = "pseudolevel"
    p[...] = [1, 2, 3, 4, 5, 6]
    
    # Auxiliary coordinate variables
    
    aux0 = n.createVariable('aux0', 'f8', ('list1',))
    aux0.standard_name = "longitude"
    aux0.units = "degrees_east"
    aux0[...] = numpy.arange(list1.size)
    
    aux1 = n.createVariable('aux1', 'f8', ('list3',))
    aux1[...] = numpy.arange(list3.size)
    
    aux2 = n.createVariable('aux2', 'f8', ('time', 'list3', 'p'))
    aux2[...] = numpy.arange(time.size * list3.size * p.size).reshape(time.size, list3.size, p.size)
    
    aux3 = n.createVariable('aux3', 'f8', ('p', 'list3', 'time'))
    aux3[...] = numpy.arange(p.size * list3.size * time.size).reshape(p.size, list3.size, time.size)
    
    aux4 = n.createVariable('aux4', 'f8', ('p', 'time', 'list3'))
    aux4[...] = numpy.arange(p.size * time.size * list3.size).reshape(p.size, time.size, list3.size)
    
    aux5 = n.createVariable('aux5', 'f8', ('list3', 'p', 'time'))
    aux5[...] = numpy.arange(list3.size * p.size * time.size).reshape(list3.size, p.size, time.size)
    
    aux6 = n.createVariable('aux6', 'f8', ('list3', 'time'))
    aux6[...] = numpy.arange(list3.size * time.size).reshape(list3.size, time.size)
    
    aux7 = n.createVariable('aux7', 'f8', ('lat',))
    aux7[...] = numpy.arange(lat.size)
    
    aux8 = n.createVariable('aux8', 'f8', ('lon', 'lat',))
    aux8[...] = numpy.arange(lon.size * lat.size).reshape(lon.size, lat.size)
    
    aux9 = n.createVariable('aux9', 'f8', ('time', 'height'))
    aux9[...] = numpy.arange(time.size * height.size).reshape(time.size, height.size)
    
    # List variables
    list1 = n.createVariable('list1', 'i', ('list1',))
    list1.compress = "lon"
    list1[...] = [0, 1, 3, 4]
    
    list2 = n.createVariable('list2', 'i', ('list2',))
    list2.compress = "lat lon"
    list2[...] = [0,  1,  5,  6, 13, 14, 17, 18, 19]
    
    list3 = n.createVariable('list3', 'i', ('list3',))
    list3.compress = "height lat lon"
    array = _jj((3, 4, 5),
                [0, 1, 5, 6, 13, 14, 25, 26, 37, 38, 48, 49, 58, 59])
    list3[...] = array.compressed()
    
    # Data variables
    temp1 = n.createVariable('temp1', 'f8', ('time', 'height', 'lat', 'list1', 'p'))
    temp1.long_name = "temp1"
    temp1.units = "K"
    temp1.coordinates = "aux0 aux7 aux8 aux9"
    temp1[...] = numpy.arange(2*3*4*4*6).reshape(2, 3, 4, 4, 6)
    
    temp2 = n.createVariable('temp2', 'f8', ('time', 'height', 'list2', 'p'))
    temp2.long_name = "temp2"
    temp2.units = "K"
    temp2.coordinates = "aux7 aux8 aux9"
    temp2[...] = numpy.arange(2*3*9*6).reshape(2, 3, 9, 6)
    
    temp3 = n.createVariable('temp3', 'f8', ('time', 'list3', 'p'))
    temp3.long_name = "temp3"
    temp3.units = "K"
    temp3.coordinates = "aux0 aux1 aux2 aux3 aux4 aux5 aux6 aux7 aux8 aux9"
    temp3[...] = numpy.arange(2*14*6).reshape(2, 14, 6)
    
    n.close()

    return filename

gathered = _make_gathered_file('gathered.nc')
    

if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    print(cf.environment(display=False))
    print()
    unittest.main(verbosity=2)
