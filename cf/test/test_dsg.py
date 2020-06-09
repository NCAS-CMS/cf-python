import datetime
import inspect
import os
import tempfile
import unittest

import numpy

import cf


# def _make_contiguous_file_with_bounds(filename):
#     n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
#
#     n.Conventions = 'CF-1.7'
#     n.featureType = 'timeSeries'
#
#     station = n.createDimension('station', 4)
#     obs     = n.createDimension('obs'    , 24)
#     name_strlen = n.createDimension('name_strlen', 8)
#     bounds  = n.createDimension('bounds', 2)
#
#     lon = n.createVariable('lon', 'f8', ('station',))
#     lon.standard_name = "longitude"
#     lon.long_name = "station longitude"
#     lon.units = "degrees_east"
#     lon.bounds = "lon_bounds"
#     lon[...] = [-23, 0, 67, 178]
#
#     lon_bounds = n.createVariable('lon_bounds', 'f8', ('station', 'bounds'))
#     lon_bounds[...] = [[-24, -22],
#                        [ -1,   1],
#                        [ 66,  68],
#                        [177, 179]]
#
#     lat = n.createVariable('lat', 'f8', ('station',))
#     lat.standard_name = "latitude"
#     lat.long_name = "station latitude"
#     lat.units = "degrees_north"
#     lat[...] = [-9, 2, 34, 78]
#
#     alt = n.createVariable('alt', 'f8', ('station',))
#     alt.long_name = "vertical distance above the surface"
#     alt.standard_name = "height"
#     alt.units = "m"
#     alt.positive = "up"
#     alt.axis = "Z"
#     alt[...] = [0.5, 12.6, 23.7, 345]
#
#     station_name = n.createVariable('station_name', 'S1',
#                                     ('station', 'name_strlen'))
#     station_name.long_name = "station name"
#     station_name.cf_role = "timeseries_id"
#     station_name[...] = numpy.array([[x for x in 'station1'],
#                                      [x for x in 'station2'],
#                                      [x for x in 'station3'],
#                                      [x for x in 'station4']])
#
#     station_info = n.createVariable('station_info', 'i4', ('station',))
#     station_info.long_name = "some kind of station info"
#     station_info[...] = [-10, -9, -8, -7]
#
#     row_size = n.createVariable('row_size', 'i4', ('station',))
#     row_size.long_name = "number of observations for this station"
#     row_size.sample_dimension = "obs"
#     row_size[...] = [3, 7, 5, 9]
#
#     time = n.createVariable('time', 'f8', ('obs',))
#     time.standard_name = "time"
#     time.long_name = "time of measurement"
#     time.units = "days since 1970-01-01 00:00:00"
#     time.bounds = "time_bounds"
#     time[ 0: 3] = [-3, -2, -1]
#     time[ 3:10] = [1, 2, 3, 4, 5, 6, 7]
#     time[10:15] = [0.5, 1.5, 2.5, 3.5, 4.5]
#     time[15:24] = range(-2, 7)
#
#     time_bounds = n.createVariable('time_bounds', 'f8', ('obs', 'bounds'))
#     time_bounds[..., 0] = time[...] - 0.5
#     time_bounds[..., 1] = time[...] + 0.5
#
#     humidity = n.createVariable(
#         'humidity', 'f8', ('obs',), fill_value=-999.9)
#     humidity.standard_name = "specific_humidity"
#     humidity.coordinates = "time lat lon alt station_name station_info"
#     humidity[ 0: 3] = numpy.arange(0, 3)
#     humidity[ 3:10] = numpy.arange(1, 71, 10)
#     humidity[10:15] = numpy.arange(2, 502, 100)
#     humidity[15:24] = numpy.arange(3, 9003, 1000)
#
#     temp = n.createVariable('temp', 'f8', ('obs',), fill_value=-999.9)
#     temp.standard_name = "air_temperature"
#     temp.units = "Celsius"
#     temp.coordinates = "time lat lon alt station_name station_info"
#     temp[...] = humidity[...] + 273.15
#
#     n.close()
#
#     return filename
#
#
# def _make_contiguous_file(filename):
#     n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
#
#     n.Conventions = 'CF-1.7'
#     n.featureType = 'timeSeries'
#
#     station = n.createDimension('station', 4)
#     obs     = n.createDimension('obs'    , 24)
#     name_strlen = n.createDimension('name_strlen', 8)
#
#     lon = n.createVariable('lon', 'f8', ('station',))
#     lon.standard_name = "longitude"
#     lon.long_name = "station longitude"
#     lon.units = "degrees_east"
#     lon[...] = [-23, 0, 67, 178]
#
#     lat = n.createVariable('lat', 'f8', ('station',))
#     lat.standard_name = "latitude"
#     lat.long_name = "station latitude"
#     lat.units = "degrees_north"
#     lat[...] = [-9, 2, 34, 78]
#
#     alt = n.createVariable('alt', 'f8', ('station',))
#     alt.long_name = "vertical distance above the surface"
#     alt.standard_name = "height"
#     alt.units = "m"
#     alt.positive = "up"
#     alt.axis = "Z"
#     alt[...] = [0.5, 12.6, 23.7, 345]
#
#     station_name = n.createVariable('station_name', 'S1',
#                                     ('station', 'name_strlen'))
#     station_name.long_name = "station name"
#     station_name.cf_role = "timeseries_id"
#     station_name[...] = numpy.array([[x for x in 'station1'],
#                                      [x for x in 'station2'],
#                                      [x for x in 'station3'],
#                                      [x for x in 'station4']])
#
#     station_info = n.createVariable('station_info', 'i4', ('station',))
#     station_info.long_name = "some kind of station info"
#     station_info[...] = [-10, -9, -8, -7]
#
#     row_size = n.createVariable('row_size', 'i4', ('station',))
#     row_size.long_name = "number of observations for this station"
#     row_size.sample_dimension = "obs"
#     row_size[...] = [3, 7, 5, 9]
#
#     time = n.createVariable('time', 'f8', ('obs',))
#     time.standard_name = "time"
#     time.long_name = "time of measurement"
#     time.units = "days since 1970-01-01 00:00:00"
#     time[ 0: 3] = [-3, -2, -1]
#     time[ 3:10] = [1, 2, 3, 4, 5, 6, 7]
#     time[10:15] = [0.5, 1.5, 2.5, 3.5, 4.5]
#     time[15:24] = range(-2, 7)
#
#     humidity = n.createVariable(
#         'humidity', 'f8', ('obs',), fill_value=-999.9)
#     humidity.standard_name = "specific_humidity"
#     humidity.coordinates = "time lat lon alt station_name station_info"
#     humidity[ 0: 3] = numpy.arange(0, 3)
#     humidity[ 3:10] = numpy.arange(1, 71, 10)
#     humidity[10:15] = numpy.arange(2, 502, 100)
#     humidity[15:24] = numpy.arange(3, 9003, 1000)
#
#     temp = n.createVariable('temp', 'f8', ('obs',), fill_value=-999.9)
#     temp.standard_name = "air_temperature"
#     temp.units = "Celsius"
#     temp.coordinates = "time lat lon alt station_name station_info"
#     temp[...] = humidity[...] + 273.15
#
#     n.close()
#
#     return filename
#
#
# def _make_indexed_file(filename):
#     n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
#
#     n.Conventions = 'CF-1.7'
#     n.featureType = 'timeSeries'
#
#     station = n.createDimension('station', 4)
#     obs     = n.createDimension('obs'    , None)
#     name_strlen = n.createDimension('name_strlen', 8)
#
#     lon = n.createVariable('lon', 'f8', ('station',))
#     lon.standard_name = "longitude"
#     lon.long_name = "station longitude"
#     lon.units = "degrees_east"
#     lon[...] = [-23, 0, 67, 178]
#
#     lat = n.createVariable('lat', 'f8', ('station',))
#     lat.standard_name = "latitude"
#     lat.long_name = "station latitude"
#     lat.units = "degrees_north"
#     lat[...] = [-9, 2, 34, 78]
#
#     alt = n.createVariable('alt', 'f8', ('station',))
#     alt.long_name = "vertical distance above the surface"
#     alt.standard_name = "height"
#     alt.units = "m"
#     alt.positive = "up"
#     alt.axis = "Z"
#     alt[...] = [0.5, 12.6, 23.7, 345]
#
#     station_name = n.createVariable('station_name', 'S1',
#                                     ('station', 'name_strlen'))
#     station_name.long_name = "station name"
#     station_name.cf_role = "timeseries_id"
#     station_name[...] = numpy.array([[x for x in 'station1'],
#                                      [x for x in 'station2'],
#                                      [x for x in 'station3'],
#                                      [x for x in 'station4']])
#
#     station_info = n.createVariable('station_info', 'i4', ('station',))
#     station_info.long_name = "some kind of station info"
#     station_info[...] = [-10, -9, -8, -7]
#
#     #row_size[...] = [3, 7, 5, 9]
#     stationIndex = n.createVariable('stationIndex', 'i4', ('obs',))
#     stationIndex.long_name = "which station this obs is for"
#     stationIndex.instance_dimension= "station"
#     stationIndex[...] = [3, 2, 1, 0, 2, 3, 3, 3, 1, 1, 0, 2,
#                          3, 1, 0, 1, 2, 3, 2, 3, 3, 3, 1, 1]
#
#     t = [[-3, -2, -1],
#          [1, 2, 3, 4, 5, 6, 7],
#          [0.5, 1.5, 2.5, 3.5, 4.5],
#          range(-2, 7)]
#
#     time = n.createVariable('time', 'f8', ('obs',))
#     time.standard_name = "time"
#     time.long_name = "time of measurement"
#     time.units = "days since 1970-01-01 00:00:00"
#     ssi = [0, 0, 0, 0]
#     for i, si in enumerate(stationIndex[...]):
#         time[i] = t[si][ssi[si]]
#         ssi[si] += 1
#
#     humidity = n.createVariable(
#         'humidity', 'f8', ('obs',), fill_value=-999.9)
#     humidity.standard_name = "specific_humidity"
#     humidity.coordinates = "time lat lon alt station_name station_info"
#
#     h = [numpy.arange(0, 3),
#          numpy.arange(1, 71, 10),
#          numpy.arange(2, 502, 100),
#          numpy.arange(3, 9003, 1000)]
#
#     ssi = [0, 0, 0, 0]
#     for i, si in enumerate(stationIndex[...]):
#         humidity[i] = h[si][ssi[si]]
#         ssi[si] += 1
#
#     temp = n.createVariable('temp', 'f8', ('obs',), fill_value=-999.9)
#     temp.standard_name = "air_temperature"
#     temp.units = "Celsius"
#     temp.coordinates = "time lat lon alt station_name station_info"
#     temp[...] = humidity[...] + 273.15
#
#     n.close()
#
#     return filename
#
#
# def _make_indexed_contiguous_file(filename):
#     n = netCDF4.Dataset(filename, 'w', format='NETCDF3_CLASSIC')
#
#     n.Conventions = 'CF-1.6'
#     n.featureType = "timeSeriesProfile"
#
#     # 3 stations
#     station = n.createDimension('station', 3)
#     # 58 profiles spreadover 4 stations, each at a different time
#     profile = n.createDimension('profile', 58)
#     obs      = n.createDimension('obs'    , None)
#     name_strlen = n.createDimension('name_strlen', 8)
#
#     lon = n.createVariable('lon', 'f8', ('station',))
#     lon.standard_name = "longitude"
#     lon.long_name = "station longitude"
#     lon.units = "degrees_east"
#     lon[...] = [-23, 0, 67]
#
#     lat = n.createVariable('lat', 'f8', ('station',))
#     lat.standard_name = "latitude"
#     lat.long_name = "station latitude"
#     lat.units = "degrees_north"
#     lat[...] = [-9, 2, 34]
#
#     alt = n.createVariable('alt', 'f8', ('station',))
#     alt.long_name = "vertical distance above the surface"
#     alt.standard_name = "height"
#     alt.units = "m"
#     alt.positive = "up"
#     alt.axis = "Z"
#     alt[...] = [0.5, 12.6, 23.7]
#
#     station_name = n.createVariable('station_name', 'S1',
#                                     ('station', 'name_strlen'))
#     station_name.long_name = "station name"
#     station_name.cf_role = "timeseries_id"
#     station_name[...] = numpy.array([[x for x in 'station1'],
#                                      [x for x in 'station2'],
#                                      [x for x in 'station3']])
#
#     profile = n.createVariable('profile', 'i4', ('profile'))
#     profile.cf_role = "profile_id"
#     profile[...] = numpy.arange(58) + 100
#
#     station_info = n.createVariable('station_info', 'i4', ('station',))
#     station_info.long_name = "some kind of station info"
#     station_info[...] = [-10, -9, -8]
#
#     stationIndex = n.createVariable('stationIndex', 'i4', ('profile',))
#     stationIndex.long_name = "which station this profile is for"
#     stationIndex.instance_dimension= "station"
#     stationIndex[...] = [2, 1, 0, 2, 1, 1, 0, 2,
#                          1, 0, 1, 2, 2, 1, 1,
#                          2, 1, 0, 2, 1, 1, 0, 2,
#                          1, 0, 1, 2, 2, 1, 1,
#                          2, 1, 0, 2, 1, 1, 0, 2,
#                          1, 0, 1, 2, 2, 1, 1,
#                          2, 1, 0, 2, 1, 1, 0, 2,
#                          1, 0, 1, 2, 2]
#     # station N has list(stationIndex[...]).count(N) profiles
#
#     row_size = n.createVariable('row_size', 'i4', ('profile',))
#     row_size.long_name = "number of observations for this profile"
#     row_size.sample_dimension = "obs"
#     row_size[...] = [
#         1, 4, 1, 3, 2, 2, 3, 3, 1, 2, 2, 3, 2, 2, 2, 2, 1, 2, 1, 3, 3, 2, 1,
#         3, 1, 3, 2, 3, 1, 3, 3, 2, 2, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1, 3, 3, 2,
#         2, 2, 2, 1, 2, 3, 3, 3, 2, 3, 1, 1
#     ] # sum = 118
#
#     time = n.createVariable('time', 'f8', ('profile',))
#     time.standard_name = "time"
#     time.long_name = "time"
#     time.units = "days since 1970-01-01 00:00:00"
#     t0 = [3, 0, -3]
#     ssi = [0, 0, 0]
#     for i, si in enumerate(stationIndex[...]):
#         time[i] = t0[si] + ssi[si]
#         ssi[si] += 1
#
#     z = n.createVariable('z', 'f8', ('obs',))
#     z.standard_name = "altitude"
#     z.long_name = "height above mean sea level"
#     z.units = "km"
#     z.axis = "Z"
#     z.positive = "up"
#
# #         z0 = [1, 0, 3]
# #         i = 0
# #         for s, r in zip(stationIndex[...], row_size[...]):
# #             z[i:i+r] = z0[s] + numpy.sort(
#                  numpy.random.uniform(0, numpy.random.uniform(1, 2), r))
# #            i += r
#
#     data = [
#         3.51977705293769, 0.521185292100177, 0.575154265863394,
#         1.08495843717095, 1.37710968624395, 2.07123455611723,
#         3.47064474274781, 3.88569849023813, 4.81069254279537,
#         0.264339600625496, 0.915704970094182, 0.0701532210336895,
#         0.395517651420933, 1.00657582854276, 1.17721374303641,
#         1.82189345615046, 3.52424307197668, 3.93200473199559,
#         3.95715099603671, 1.57047493027102, 1.09938982652955,
#         1.17768722826975, 0.251803399458277, 1.59673486865804,
#         4.02868944763605, 4.03749228832264, 4.79858281590985,
#         3.00019933315412, 3.65124061660449, 0.458463542157766,
#         0.978678197083262, 0.0561560792556281, 0.31182013232255,
#         3.33350065357286, 4.33143904011861, 0.377894196412131,
#         1.63020681064712, 2.00097025264771, 3.76948048424458,
#         0.572927165845568, 1.29408313557905, 1.81296270533192,
#         0.387142669131077, 0.693459187515738, 1.69261930636298,
#         1.38258797228361, 1.82590759889566, 3.34993297710761,
#         0.725250730922501, 1.38221693486728, 1.59828555215646,
#         1.59281225554253, 0.452340646918555, 0.976663373825433,
#         1.12640496317618, 3.19366847375422, 3.37209133117904,
#         3.40665008236976, 3.53525896684001, 4.10444186715724,
#         0.14920937817654, 0.0907197953552753, 0.42527916794473,
#         0.618685137936187, 3.01900591447357, 3.37205542289986,
#         3.86957342976163, 0.17175098751914, 0.990040375014957,
#         1.57011428605984, 2.12140567043994, 3.24374743730506,
#         4.24042441581785, 0.929509749153725, 0.0711997786817564,
#         2.25090028461898, 3.31520955860746, 3.49482624434274,
#         3.96812568493549, 1.5681807261767, 1.79993011515465,
#         0.068325990211909, 0.124469638352167, 3.31990436971169,
#         3.84766748039389, 0.451973490541035, 1.24303219956085,
#         1.30478004656262, 0.351892459787624, 0.683685812990457,
#         0.788883736575568, 3.73033428872491, 3.99479807507392,
#         0.811582011950481, 1.2241242448019, 1.25563109687369,
#         2.16603674712822, 3.00010622131408, 3.90637137662453,
#         0.589586644805982, 0.104656387266266, 0.961185900148304,
#         1.05120351477824, 1.29460917520233, 2.10139985693684,
#         3.64252693587415, 3.91197236350995, 4.56466622863717,
#         0.556476687600461, 0.783717448678148, 0.910917550635007,
#         1.59750076220451, 1.97101264162631, 0.714693043642084,
#         0.904381625638779, 1.03767817888021, 4.10124675852254,
#         3.1059214185543]
#     data = numpy.around(data, 2)
#     z[...] = data
#
#     humidity = n.createVariable(
#         'humidity', 'f8', ('obs',), fill_value=-999.9)
#     humidity.standard_name = "specific_humidity"
#     humidity.coordinates = (
#         "time lat lon alt z station_name station_info profile")
#
#     data *= 10
#     data = numpy.around(data, 2)
#     humidity[...] = data
#
#     temp = n.createVariable('temp', 'f8', ('obs',), fill_value=-999.9)
#     temp.standard_name = "air_temperature"
#     temp.units = "Celsius"
#     temp.coordinates = "time lat lon alt z station_name station_info profile"
#
#     data += 2731.5
#     data = numpy.around(data, 2)
#     temp[...] = data
#
#     n.close()
#
#     return filename
#
#
#  contiguous_file = _make_contiguous_file_with_bounds(
#      'DSG_timeSeries_contiguous.nc')
#  indexed_file    = _make_indexed_file('DSG_timeSeries_indexed.nc')
#  indexed_contiguous_file = _make_indexed_contiguous_file(
#      'DSG_timeSeriesProfile_indexed_contiguous.nc')


class DSGTest(unittest.TestCase):
    def setUp(self):
        self.contiguous = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'DSG_timeSeries_contiguous.nc'
        )
        self.indexed = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'DSG_timeSeries_indexed.nc'
        )
        self.indexed_contiguous = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'DSG_timeSeriesProfile_indexed_contiguous.nc'
        )

        (fd, self.tempfilename) = tempfile.mkstemp(
            suffix='.nc', prefix='cf_', dir='.')
        os.close(fd)

        a = numpy.ma.masked_all((4, 9), dtype=float)
        a[0, 0:3] = [0.0, 1.0, 2.0]
        a[1, 0:7] = [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0]
        a[2, 0:5] = [2.0, 102.0, 202.0, 302.0, 402.0]
        a[3, 0:9] = [3.0, 1003.0, 2003.0, 3003.0, 4003.0, 5003.0, 6003.0,
                     7003.0, 8003.0]
        self.a = a

        b = numpy.array([[[20.7, -99, -99, -99],
                          [10.1, 11.8, 18.2, -99],
                          [11.0, 11.8, -99, -99],
                          [16.3, 20.0, -99, -99],
                          [13.8, 18.3, -99, -99],
                          [15.9, -99, -99, -99],
                          [15.7, 21.2, -99, -99],
                          [22.5, -99, -99, -99],
                          [18.0, -99, -99, -99],
                          [12.6, 21.7, -99, -99],
                          [10.5, 12.9, 21.0, -99],
                          [16.0, 19.7, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99]],

                         [[5.2, 5.8, 10.8, 13.8],
                          [2.6, 9.2, -99, -99],
                          [0.7, 4.0, -99, -99],
                          [15.7, -99, -99, -99],
                          [2.5, 16.0, -99, -99],
                          [4.6, 9.8, -99, -99],
                          [0.6, 3.1, -99, -99],
                          [3.8, -99, -99, -99],
                          [5.7, 12.9, 18.1, -99],
                          [3.9, 6.9, 16.9, -99],
                          [7.3, 13.8, 16.0, -99],
                          [4.5, 9.8, 11.3, -99],
                          [1.5, -99, -99, -99],
                          [0.9, 4.3, 6.2, -99],
                          [1.7, 9.9, -99, -99],
                          [9.3, -99, -99, -99],
                          [0.7, -99, -99, -99],
                          [15.7, -99, -99, -99],
                          [0.7, 1.2, -99, -99],
                          [4.5, 12.4, 13.0, -99],
                          [3.5, 6.8, 7.9, -99],
                          [8.1, 12.2, -99, -99],
                          [5.9, -99, -99, -99],
                          [1.0, 9.6, -99, -99],
                          [5.6, 7.8, 9.1, -99],
                          [7.1, 9.0, 10.4, -99]],

                         [[35.2, -99, -99, -99],
                          [34.7, 38.9, 48.1, -99],
                          [35.2, 39.3, 39.6, -99],
                          [40.3, 40.4, 48.0, -99],
                          [30.0, 36.5, -99, -99],
                          [33.3, 43.3, -99, -99],
                          [37.7, -99, -99, -99],
                          [33.5, -99, -99, -99],
                          [31.9, 33.7, -99, -99],
                          [34.1, 35.4, 41.0, -99],
                          [30.2, 33.7, 38.7, -99],
                          [32.4, 42.4, -99, -99],
                          [33.2, 34.9, 39.7, -99],
                          [33.2, -99, -99, -99],
                          [38.5, -99, -99, -99],
                          [37.3, 39.9, -99, -99],
                          [30.0, 39.1, -99, -99],
                          [36.4, 39.1, 45.6, -99],
                          [41.0, -99, -99, -99],
                          [31.1, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99],
                          [-99, -99, -99, -99]]])

        b = numpy.ma.where(b == -99, numpy.ma.masked, b)
        self.b = b

        self.test_only = []
#        self.test_only = ['test_DSG_indexed']

    def tearDown(self):
        os.remove(self.tempfilename)

    def test_DSG_contiguous(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.contiguous, verbose=0)

        self.assertTrue(len(f) == 2)

#        print ('\nf\n')
#        for x in f:
#            x.dump()

        # Select the specific humidity field
        q = [g for g in f
             if g.get_property('standard_name') == 'specific_humidity'][0]
#        f[0].data.inspect()
#        f[1].data.inspect()

        self.assertTrue(q._equals(q.data.array.mask, self.a.mask))

        self.assertTrue(q._equals(self.a, q.data.array),
                        '\nself.a=\n'+str(self.a)+'\nq.array=\n'+str(q.array))

        cf.write(f, self.tempfilename, verbose=0)
        g = cf.read(self.tempfilename)

#        print ('\ng\n')
#        for x in g:
#            print(x)

        self.assertTrue(len(g) == len(f))

        for i in range(len(f)):
            self.assertTrue(g[i].equals(f[i], verbose=2))

        # ------------------------------------------------------------
        # Test creation
        # ------------------------------------------------------------
        # Define the ragged array values
        ragged_array = numpy.array([280, 282.5, 281, 279, 278, 279.5],
                                   dtype='float32')

        # Define the count array values
        count_array = [2, 4]

        # Create the count variable
        count_variable = cf.Count(data=cf.Data(count_array))
        count_variable.set_property(
            'long_name', 'number of obs for this timeseries')

        # Create the contiguous ragged array object
        array = cf.RaggedContiguousArray(
            compressed_array=cf.Data(ragged_array),
            shape=(2, 4), size=8, ndim=2,
            count_variable=count_variable)

        # Create the field construct with the domain axes and the ragged
        # array
        tas = cf.Field()
        tas.set_properties({'standard_name': 'air_temperature',
                            'units': 'K',
                            'featureType': 'timeSeries'})

        # Create the domain axis constructs for the uncompressed array
        X = tas.set_construct(cf.DomainAxis(4))
        Y = tas.set_construct(cf.DomainAxis(2))

        # Set the data for the field
        tas.set_data(cf.Data(array), axes=[Y, X])

        cf.write(tas, self.tempfilename)

    def test_DSG_indexed(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.indexed)

        self.assertTrue(len(f) == 2)

        # Select the specific humidity field
        q = [g for g in f
             if g.get_property('standard_name') == 'specific_humidity'][0]

        self.assertTrue(q._equals(q.data.array.mask, self.a.mask))

        self.assertTrue(
            q._equals(q.data.array, self.a),
            '\nself.a=\n' + str(self.a) + '\nq.array=\n' + str(q.array)
        )


#        print ('\nf\n')
#        for x in f:
#            print(x)

        cf.write(f, self.tempfilename, verbose=0)
        g = cf.read(self.tempfilename)

#        print ('\ng\n')
#        for x in g:
#            print(x)

        self.assertTrue(len(g) == len(f))

        for i in range(len(f)):
            self.assertTrue(g[i].equals(f[i], verbose=2))

    def test_DSG_indexed_contiguous(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        f = cf.read(self.indexed_contiguous, verbose=0)

        self.assertTrue(len(f) == 2)

        # Select the specific humidity field
        q = f.select('specific_humidity')[0]
#        q = [g for g in f
#             if g.get_property('standard_name') == 'specific_humidity'][0]

        qa = q.data.array

#        print (qa[0, 12])
#        print (self.b[0, 12])

        for n in range(qa.shape[0]):
            for m in range(qa.shape[1]):
                self.assertTrue(
                    q._equals(qa.mask[n, m], self.b.mask[n, m]),
                    str(n) + ' ' + str(m) + ' ' + str(qa[n, m]) + ' ' +
                    str(self.b[n, m])
                )

        message = repr(qa-self.b)
        # ... +'\n'+repr(qa[2,0])+'\n'+repr(self.b[2, 0])

        self.assertTrue(q._equals(qa, self.b), message)

#        print ('\nf\n')
#        for x in f:
#            print(x)

        cf.write(f, self.tempfilename, verbose=0)
        g = cf.read(self.tempfilename, verbose=0)

#        print ('\ng\n')
#        for x in g:
#            print(x)

        self.assertTrue(len(g) == len(f))

        for i in range(len(f)):
            self.assertTrue(g[i].equals(f[i], verbose=2))

    def test_DSG_create_contiguous(self):
        if self.test_only and inspect.stack()[0][3] not in self.test_only:
            return

        # Define the ragged array values
        ragged_array = numpy.array([1, 3, 4, 3, 6], dtype='float32')
        # Define the count array values
        count_array = [2, 3]

        # Initialise the count variable
        count_variable = cf.Count(data=cf.Data(count_array))
        count_variable.set_property(
            'long_name', 'number of obs for this timeseries')

        # Initialise the contiguous ragged array object
        array = cf.RaggedContiguousArray(
            compressed_array=cf.Data(ragged_array),
            shape=(2, 3), size=6, ndim=2,
            count_variable=count_variable)

        # Initialize the auxiliary coordinate construct with the ragged
        # array and set some properties
        z = cf.AuxiliaryCoordinate(
            data=cf.Data(array),
            properties={'standard_name': 'height',
                        'units': 'km',
                        'positive': 'up'})

        self.assertTrue((z.data.array == numpy.ma.masked_array(
            data=[[1.0, 3.0, 99],
                  [4.0, 3.0, 6.0]],
            mask=[[False, False,  True],
                  [False, False, False]],
            fill_value=1e+20,
            dtype='float32')).all())

        self.assertTrue(z.data.get_compression_type() == 'ragged contiguous')

        self.assertTrue((z.data.compressed_array == numpy.array(
            [1., 3., 4., 3., 6.], dtype='float32')).all())

        self.assertTrue((z.data.get_count().data.array == numpy.array(
            [2, 3])).all())


# --- End: class


if __name__ == '__main__':
    print('Run date:', datetime.datetime.utcnow())
    print(cf.environment(display=False))
    print()
    unittest.main(verbosity=2)
