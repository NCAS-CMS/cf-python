import datetime
import faulthandler
import os
import unittest

import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import cfdm
import netCDF4

VN = cfdm.CF()

# Load large arrays
filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "create_test_files.npz"
)
arrays = np.load(filename)


# --------------------------------------------------------------------
# DSG files
# --------------------------------------------------------------------
def _make_contiguous_file(filename):
    """Make a netCDF file with a contiguous ragged array DSG feature."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeries"

    n.createDimension("station", 4)
    n.createDimension("obs", 24)
    n.createDimension("name_strlen", 8)
    n.createDimension("bounds", 2)

    lon = n.createVariable("lon", "f8", ("station",))
    lon.standard_name = "longitude"
    lon.long_name = "station longitude"
    lon.units = "degrees_east"
    lon.bounds = "lon_bounds"
    lon[...] = [-23, 0, 67, 178]

    lon_bounds = n.createVariable("lon_bounds", "f8", ("station", "bounds"))
    lon_bounds[...] = [[-24, -22], [-1, 1], [66, 68], [177, 179]]

    lat = n.createVariable("lat", "f8", ("station",))
    lat.standard_name = "latitude"
    lat.long_name = "station latitude"
    lat.units = "degrees_north"
    lat[...] = [-9, 2, 34, 78]

    alt = n.createVariable("alt", "f8", ("station",))
    alt.long_name = "vertical distance above the surface"
    alt.standard_name = "height"
    alt.units = "m"
    alt.positive = "up"
    alt.axis = "Z"
    alt[...] = [0.5, 12.6, 23.7, 345]

    station_name = n.createVariable(
        "station_name", "S1", ("station", "name_strlen")
    )
    station_name.long_name = "station name"
    station_name.cf_role = "timeseries_id"
    station_name[...] = np.array(
        [
            [x for x in "station1"],
            [x for x in "station2"],
            [x for x in "station3"],
            [x for x in "station4"],
        ]
    )

    station_info = n.createVariable("station_info", "i4", ("station",))
    station_info.long_name = "some kind of station info"
    station_info[...] = [-10, -9, -8, -7]

    row_size = n.createVariable("row_size", "i4", ("station",))
    row_size.long_name = "number of observations for this station"
    row_size.sample_dimension = "obs"
    row_size[...] = [3, 7, 5, 9]

    time = n.createVariable("time", "f8", ("obs",))
    time.standard_name = "time"
    time.long_name = "time of measurement"
    time.units = "days since 1970-01-01 00:00:00"
    time.bounds = "time_bounds"
    time[0:3] = [-3, -2, -1]
    time[3:10] = [1, 2, 3, 4, 5, 6, 7]
    time[10:15] = [0.5, 1.5, 2.5, 3.5, 4.5]
    time[15:24] = range(-2, 7)

    time_bounds = n.createVariable("time_bounds", "f8", ("obs", "bounds"))
    time_bounds[..., 0] = time[...] - 0.5
    time_bounds[..., 1] = time[...] + 0.5

    humidity = n.createVariable("humidity", "f8", ("obs",), fill_value=-999.9)
    humidity.standard_name = "specific_humidity"
    humidity.coordinates = "time lat lon alt station_name station_info"
    humidity[0:3] = np.arange(0, 3)
    humidity[3:10] = np.arange(1, 71, 10)
    humidity[10:15] = np.arange(2, 502, 100)
    humidity[15:24] = np.arange(3, 9003, 1000)

    temp = n.createVariable("temp", "f8", ("obs",), fill_value=-999.9)
    temp.standard_name = "air_temperature"
    temp.units = "Celsius"
    temp.coordinates = "time lat lon alt station_name station_info"
    temp[...] = humidity[...] + 273.15

    n.close()

    return filename


def _make_indexed_file(filename):
    """Make a netCDF file with an indexed ragged array DSG feature."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeries"

    n.createDimension("station", 4)
    n.createDimension("obs", None)
    n.createDimension("name_strlen", 8)
    n.createDimension("bounds", 2)

    lon = n.createVariable("lon", "f8", ("station",))
    lon.standard_name = "longitude"
    lon.long_name = "station longitude"
    lon.units = "degrees_east"
    lon.bounds = "lon_bounds"
    lon[...] = [-23, 0, 67, 178]

    lon_bounds = n.createVariable("lon_bounds", "f8", ("station", "bounds"))
    lon_bounds[...] = [[-24, -22], [-1, 1], [66, 68], [177, 179]]

    lat = n.createVariable("lat", "f8", ("station",))
    lat.standard_name = "latitude"
    lat.long_name = "station latitude"
    lat.units = "degrees_north"
    lat[...] = [-9, 2, 34, 78]

    alt = n.createVariable("alt", "f8", ("station",))
    alt.long_name = "vertical distance above the surface"
    alt.standard_name = "height"
    alt.units = "m"
    alt.positive = "up"
    alt.axis = "Z"
    alt[...] = [0.5, 12.6, 23.7, 345]

    station_name = n.createVariable(
        "station_name", "S1", ("station", "name_strlen")
    )
    station_name.long_name = "station name"
    station_name.cf_role = "timeseries_id"
    station_name[...] = np.array(
        [
            [x for x in "station1"],
            [x for x in "station2"],
            [x for x in "station3"],
            [x for x in "station4"],
        ]
    )

    station_info = n.createVariable("station_info", "i4", ("station",))
    station_info.long_name = "some kind of station info"
    station_info[...] = [-10, -9, -8, -7]

    # row_size[...] = [3, 7, 5, 9]
    stationIndex = n.createVariable("stationIndex", "i4", ("obs",))
    stationIndex.long_name = "which station this obs is for"
    stationIndex.instance_dimension = "station"
    stationIndex[...] = [
        3,
        2,
        1,
        0,
        2,
        3,
        3,
        3,
        1,
        1,
        0,
        2,
        3,
        1,
        0,
        1,
        2,
        3,
        2,
        3,
        3,
        3,
        1,
        1,
    ]

    t = [
        [-3, -2, -1],
        [1, 2, 3, 4, 5, 6, 7],
        [0.5, 1.5, 2.5, 3.5, 4.5],
        range(-2, 7),
    ]

    time = n.createVariable("time", "f8", ("obs",))
    time.standard_name = "time"
    time.long_name = "time of measurement"
    time.units = "days since 1970-01-01 00:00:00"
    time.bounds = "time_bounds"
    ssi = [0, 0, 0, 0]
    for i, si in enumerate(stationIndex[...]):
        time[i] = t[si][ssi[si]]
        ssi[si] += 1

    time_bounds = n.createVariable("time_bounds", "f8", ("obs", "bounds"))
    time_bounds[..., 0] = time[...] - 0.5
    time_bounds[..., 1] = time[...] + 0.5

    humidity = n.createVariable("humidity", "f8", ("obs",), fill_value=-999.9)
    humidity.standard_name = "specific_humidity"
    humidity.coordinates = "time lat lon alt station_name station_info"

    h = [
        np.arange(0, 3),
        np.arange(1, 71, 10),
        np.arange(2, 502, 100),
        np.arange(3, 9003, 1000),
    ]

    ssi = [0, 0, 0, 0]
    for i, si in enumerate(stationIndex[...]):
        humidity[i] = h[si][ssi[si]]
        ssi[si] += 1

    temp = n.createVariable("temp", "f8", ("obs",), fill_value=-999.9)
    temp.standard_name = "air_temperature"
    temp.units = "Celsius"
    temp.coordinates = "time lat lon alt station_name station_info"
    temp[...] = humidity[...] + 273.15

    n.close()

    return filename


def _make_indexed_contiguous_file(filename):
    """Make a netCDF file with an indexed contiguous ragged array."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeriesProfile"

    # 3 stations
    n.createDimension("station", 3)
    # 58 profiles spreadover 4 stations, each at a different time
    n.createDimension("profile", 58)
    n.createDimension("obs", None)
    n.createDimension("name_strlen", 8)
    n.createDimension("bounds", 2)

    lon = n.createVariable("lon", "f8", ("station",))
    lon.standard_name = "longitude"
    lon.long_name = "station longitude"
    lon.units = "degrees_east"
    lon.bounds = "lon_bounds"
    lon[...] = [-23, 0, 67]

    lon_bounds = n.createVariable("lon_bounds", "f8", ("station", "bounds"))
    lon_bounds[...] = [[-24, -22], [-1, 1], [66, 68]]

    lat = n.createVariable("lat", "f8", ("station",))
    lat.standard_name = "latitude"
    lat.long_name = "station latitude"
    lat.units = "degrees_north"
    lat[...] = [-9, 2, 34]

    alt = n.createVariable("alt", "f8", ("station",))
    alt.long_name = "vertical distance above the surface"
    alt.standard_name = "height"
    alt.units = "m"
    alt.positive = "up"
    alt.axis = "Z"
    alt[...] = [0.5, 12.6, 23.7]

    station_name = n.createVariable(
        "station_name", "S1", ("station", "name_strlen")
    )
    station_name.long_name = "station name"
    station_name.cf_role = "timeseries_id"
    station_name[...] = np.array(
        [
            [x for x in "station1"],
            [x for x in "station2"],
            [x for x in "station3"],
        ]
    )

    profile = n.createVariable("profile", "i4", ("profile"))
    profile.cf_role = "profile_id"
    profile[...] = np.arange(58) + 100

    station_info = n.createVariable("station_info", "i4", ("station",))
    station_info.long_name = "some kind of station info"
    station_info[...] = [-10, -9, -8]

    stationIndex = n.createVariable("stationIndex", "i4", ("profile",))
    stationIndex.long_name = "which station this profile is for"
    stationIndex.instance_dimension = "station"
    stationIndex[...] = [
        2,
        1,
        0,
        2,
        1,
        1,
        0,
        2,
        1,
        0,
        1,
        2,
        2,
        1,
        1,
        2,
        1,
        0,
        2,
        1,
        1,
        0,
        2,
        1,
        0,
        1,
        2,
        2,
        1,
        1,
        2,
        1,
        0,
        2,
        1,
        1,
        0,
        2,
        1,
        0,
        1,
        2,
        2,
        1,
        1,
        2,
        1,
        0,
        2,
        1,
        1,
        0,
        2,
        1,
        0,
        1,
        2,
        2,
    ]
    # station N has list(stationIndex[...]).count(N) profiles

    row_size = n.createVariable("row_size", "i4", ("profile",))
    row_size.long_name = "number of observations for this profile"
    row_size.sample_dimension = "obs"
    row_size[...] = [
        1,
        4,
        1,
        3,
        2,
        2,
        3,
        3,
        1,
        2,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        2,
        1,
        3,
        3,
        2,
        1,
        3,
        1,
        3,
        2,
        3,
        1,
        3,
        3,
        2,
        2,
        2,
        1,
        1,
        1,
        3,
        1,
        1,
        2,
        1,
        1,
        3,
        3,
        2,
        2,
        2,
        2,
        1,
        2,
        3,
        3,
        3,
        2,
        3,
        1,
        1,
    ]  # sum = 118

    time = n.createVariable("time", "f8", ("profile",))
    time.standard_name = "time"
    time.long_name = "time"
    time.units = "days since 1970-01-01 00:00:00"
    time.bounds = "time_bounds"
    t0 = [3, 0, -3]
    ssi = [0, 0, 0]
    for i, si in enumerate(stationIndex[...]):
        time[i] = t0[si] + ssi[si]
        ssi[si] += 1

    time_bounds = n.createVariable("time_bounds", "f8", ("profile", "bounds"))
    time_bounds[..., 0] = time[...] - 0.5
    time_bounds[..., 1] = time[...] + 0.5

    z = n.createVariable("z", "f8", ("obs",))
    z.standard_name = "altitude"
    z.long_name = "height above mean sea level"
    z.units = "km"
    z.axis = "Z"
    z.positive = "up"
    z.bounds = "z_bounds"

    #        z0 = [1, 0, 3]
    #        i = 0
    #        for s, r in zip(stationIndex[...], row_size[...]):
    #            z[i:i+r] = z0[s] + np.sort(
    #                np.random.uniform(0, np.random.uniform(1, 2), r))
    #            i += r

    data = [
        3.51977705293769,
        0.521185292100177,
        0.575154265863394,
        1.08495843717095,
        1.37710968624395,
        2.07123455611723,
        3.47064474274781,
        3.88569849023813,
        4.81069254279537,
        0.264339600625496,
        0.915704970094182,
        0.0701532210336895,
        0.395517651420933,
        1.00657582854276,
        1.17721374303641,
        1.82189345615046,
        3.52424307197668,
        3.93200473199559,
        3.95715099603671,
        1.57047493027102,
        1.09938982652955,
        1.17768722826975,
        0.251803399458277,
        1.59673486865804,
        4.02868944763605,
        4.03749228832264,
        4.79858281590985,
        3.00019933315412,
        3.65124061660449,
        0.458463542157766,
        0.978678197083262,
        0.0561560792556281,
        0.31182013232255,
        3.33350065357286,
        4.33143904011861,
        0.377894196412131,
        1.63020681064712,
        2.00097025264771,
        3.76948048424458,
        0.572927165845568,
        1.29408313557905,
        1.81296270533192,
        0.387142669131077,
        0.693459187515738,
        1.69261930636298,
        1.38258797228361,
        1.82590759889566,
        3.34993297710761,
        0.725250730922501,
        1.38221693486728,
        1.59828555215646,
        1.59281225554253,
        0.452340646918555,
        0.976663373825433,
        1.12640496317618,
        3.19366847375422,
        3.37209133117904,
        3.40665008236976,
        3.53525896684001,
        4.10444186715724,
        0.14920937817654,
        0.0907197953552753,
        0.42527916794473,
        0.618685137936187,
        3.01900591447357,
        3.37205542289986,
        3.86957342976163,
        0.17175098751914,
        0.990040375014957,
        1.57011428605984,
        2.12140567043994,
        3.24374743730506,
        4.24042441581785,
        0.929509749153725,
        0.0711997786817564,
        2.25090028461898,
        3.31520955860746,
        3.49482624434274,
        3.96812568493549,
        1.5681807261767,
        1.79993011515465,
        0.068325990211909,
        0.124469638352167,
        3.31990436971169,
        3.84766748039389,
        0.451973490541035,
        1.24303219956085,
        1.30478004656262,
        0.351892459787624,
        0.683685812990457,
        0.788883736575568,
        3.73033428872491,
        3.99479807507392,
        0.811582011950481,
        1.2241242448019,
        1.25563109687369,
        2.16603674712822,
        3.00010622131408,
        3.90637137662453,
        0.589586644805982,
        0.104656387266266,
        0.961185900148304,
        1.05120351477824,
        1.29460917520233,
        2.10139985693684,
        3.64252693587415,
        3.91197236350995,
        4.56466622863717,
        0.556476687600461,
        0.783717448678148,
        0.910917550635007,
        1.59750076220451,
        1.97101264162631,
        0.714693043642084,
        0.904381625638779,
        1.03767817888021,
        4.10124675852254,
        3.1059214185543,
    ]
    data = np.around(data, 2)
    z[...] = data

    z_bounds = n.createVariable("z_bounds", "f8", ("obs", "bounds"))
    z_bounds[..., 0] = z[...] - 0.01
    z_bounds[..., 1] = z[...] + 0.01

    humidity = n.createVariable("humidity", "f8", ("obs",), fill_value=-999.9)
    humidity.standard_name = "specific_humidity"
    humidity.coordinates = (
        "time lat lon alt z station_name station_info profile"
    )

    data *= 10
    data = np.around(data, 2)
    humidity[...] = data

    temp = n.createVariable("temp", "f8", ("obs",), fill_value=-999.9)
    temp.standard_name = "air_temperature"
    temp.units = "Celsius"
    temp.coordinates = "time lat lon alt z station_name station_info profile"

    data += 2731.5
    data = np.around(data, 2)
    temp[...] = data

    n.close()

    return filename


# --------------------------------------------------------------------
# External variable files
# --------------------------------------------------------------------
def _make_external_files():
    """Make netCDF files with external variables."""

    def _pp(
        filename,
        parent=False,
        external=False,
        combined=False,
        external_missing=False,
    ):
        """Make a netCDF file with some external variables."""
        nc = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

        nc.createDimension("grid_latitude", 10)
        nc.createDimension("grid_longitude", 9)

        nc.Conventions = f"CF-{VN}"
        if parent:
            nc.external_variables = "areacella"

        if parent or combined or external_missing:
            grid_latitude = nc.createVariable(
                dimensions=("grid_latitude",),
                datatype="f8",
                varname="grid_latitude",
            )
            grid_latitude.setncatts(
                {"units": "degrees", "standard_name": "grid_latitude"}
            )
            grid_latitude[...] = range(10)

            grid_longitude = nc.createVariable(
                dimensions=("grid_longitude",),
                datatype="f8",
                varname="grid_longitude",
            )
            grid_longitude.setncatts(
                {"units": "degrees", "standard_name": "grid_longitude"}
            )
            grid_longitude[...] = range(9)

            latitude = nc.createVariable(
                dimensions=("grid_latitude", "grid_longitude"),
                datatype="i4",
                varname="latitude",
            )
            latitude.setncatts(
                {"units": "degree_N", "standard_name": "latitude"}
            )

            latitude[...] = np.arange(90).reshape(10, 9)

            longitude = nc.createVariable(
                dimensions=("grid_longitude", "grid_latitude"),
                datatype="i4",
                varname="longitude",
            )
            longitude.setncatts(
                {"units": "degreeE", "standard_name": "longitude"}
            )
            longitude[...] = np.arange(90).reshape(9, 10)

            eastward_wind = nc.createVariable(
                dimensions=("grid_latitude", "grid_longitude"),
                datatype="f8",
                varname="eastward_wind",
            )
            eastward_wind.coordinates = "latitude longitude"
            eastward_wind.standard_name = "eastward_wind"
            eastward_wind.cell_methods = (
                "grid_longitude: mean (interval: 1 day comment: ok) "
                "grid_latitude: maximum where sea"
            )
            eastward_wind.cell_measures = "area: areacella"
            eastward_wind.units = "m s-1"
            eastward_wind[...] = np.arange(90).reshape(10, 9) - 45.5

        if external or combined:
            areacella = nc.createVariable(
                dimensions=("grid_longitude", "grid_latitude"),
                datatype="f8",
                varname="areacella",
            )
            areacella.setncatts({"units": "m2", "standard_name": "cell_area"})
            areacella[...] = np.arange(90).reshape(9, 10) + 100000.5

        nc.close()

    parent_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "parent.nc"
    )
    external_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "external.nc"
    )
    combined_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "combined.nc"
    )
    external_missing_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "external_missing.nc"
    )

    _pp(parent_file, parent=True)
    _pp(external_file, external=True)
    _pp(combined_file, combined=True)
    _pp(external_missing_file, external_missing=True)

    return parent_file, external_file, combined_file, external_missing_file


# --------------------------------------------------------------------
# Gathered files
# --------------------------------------------------------------------
def _make_gathered_file(filename):
    """Make a netCDF file with a gathered array."""

    def _jj(shape, list_values):
        """Create and return a gathered array."""
        array = np.ma.masked_all(shape)
        for i, (index, x) in enumerate(np.ndenumerate(array)):
            if i in list_values:
                array[index] = i
        return array

    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"

    time = n.createDimension("time", 2)
    height = n.createDimension("height", 3)
    lat = n.createDimension("lat", 4)
    lon = n.createDimension("lon", 5)
    p = n.createDimension("p", 6)

    list1 = n.createDimension("list1", 4)
    list2 = n.createDimension("list2", 9)
    list3 = n.createDimension("list3", 14)

    # Dimension coordinate variables
    time = n.createVariable("time", "f8", ("time",))
    time.standard_name = "time"
    time.units = "days since 2000-1-1"
    time[...] = [31, 60]

    height = n.createVariable("height", "f8", ("height",))
    height.standard_name = "height"
    height.units = "metres"
    height.positive = "up"
    height[...] = [0.5, 1.5, 2.5]

    lat = n.createVariable("lat", "f8", ("lat",))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat[...] = [-90, -85, -80, -75]

    p = n.createVariable("p", "i4", ("p",))
    p.long_name = "pseudolevel"
    p[...] = [1, 2, 3, 4, 5, 6]

    # Auxiliary coordinate variables

    aux0 = n.createVariable("aux0", "f8", ("list1",))
    aux0.standard_name = "longitude"
    aux0.units = "degrees_east"
    aux0[...] = np.arange(list1.size)

    aux1 = n.createVariable("aux1", "f8", ("list3",))
    aux1[...] = np.arange(list3.size)

    aux2 = n.createVariable("aux2", "f8", ("time", "list3", "p"))
    aux2[...] = np.arange(time.size * list3.size * p.size).reshape(
        time.size, list3.size, p.size
    )

    aux3 = n.createVariable("aux3", "f8", ("p", "list3", "time"))
    aux3[...] = np.arange(p.size * list3.size * time.size).reshape(
        p.size, list3.size, time.size
    )

    aux4 = n.createVariable("aux4", "f8", ("p", "time", "list3"))
    aux4[...] = np.arange(p.size * time.size * list3.size).reshape(
        p.size, time.size, list3.size
    )

    aux5 = n.createVariable("aux5", "f8", ("list3", "p", "time"))
    aux5[...] = np.arange(list3.size * p.size * time.size).reshape(
        list3.size, p.size, time.size
    )

    aux6 = n.createVariable("aux6", "f8", ("list3", "time"))
    aux6[...] = np.arange(list3.size * time.size).reshape(
        list3.size, time.size
    )

    aux7 = n.createVariable("aux7", "f8", ("lat",))
    aux7[...] = np.arange(lat.size)

    aux8 = n.createVariable("aux8", "f8", ("lon", "lat"))
    aux8[...] = np.arange(lon.size * lat.size).reshape(lon.size, lat.size)

    aux9 = n.createVariable("aux9", "f8", ("time", "height"))
    aux9[...] = np.arange(time.size * height.size).reshape(
        time.size, height.size
    )

    # List variables
    list1 = n.createVariable("list1", "i", ("list1",))
    list1.compress = "lon"
    list1[...] = [0, 1, 3, 4]

    list2 = n.createVariable("list2", "i", ("list2",))
    list2.compress = "lat lon"
    list2[...] = [0, 1, 5, 6, 13, 14, 17, 18, 19]

    list3 = n.createVariable("list3", "i", ("list3",))
    list3.compress = "height lat lon"
    array = _jj(
        (3, 4, 5), [0, 1, 5, 6, 13, 14, 25, 26, 37, 38, 48, 49, 58, 59]
    )
    list3[...] = array.compressed()

    # Data variables
    temp1 = n.createVariable(
        "temp1", "f8", ("time", "height", "lat", "list1", "p")
    )
    temp1.long_name = "temp1"
    temp1.units = "K"
    temp1.coordinates = "aux0 aux7 aux8 aux9"
    temp1[...] = np.arange(2 * 3 * 4 * 4 * 6).reshape(2, 3, 4, 4, 6)

    temp2 = n.createVariable("temp2", "f8", ("time", "height", "list2", "p"))
    temp2.long_name = "temp2"
    temp2.units = "K"
    temp2.coordinates = "aux7 aux8 aux9"
    temp2[...] = np.arange(2 * 3 * 9 * 6).reshape(2, 3, 9, 6)

    temp3 = n.createVariable("temp3", "f8", ("time", "list3", "p"))
    temp3.long_name = "temp3"
    temp3.units = "K"
    temp3.coordinates = "aux0 aux1 aux2 aux3 aux4 aux5 aux6 aux7 aux8 aux9"
    temp3[...] = np.arange(2 * 14 * 6).reshape(2, 14, 6)

    n.close()

    return filename


gathered = _make_gathered_file("gathered.nc")


# --------------------------------------------------------------------
# Geometry files
# --------------------------------------------------------------------
def _make_geometry_1_file(filename):
    """See n.comment for details."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeries"
    n.comment = (
        "Make a netCDF file with 2 node coordinates variables, "
        "each of which has a corresponding auxiliary coordinate "
        "variable."
    )

    n.createDimension("time", 4)
    n.createDimension("instance", 2)
    n.createDimension("node", 5)

    t = n.createVariable("time", "i4", ("time",))
    t.units = "seconds since 2016-11-07 20:00 UTC"
    t[...] = [1, 2, 3, 4]

    lat = n.createVariable("lat", "f8", ("instance",))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat.nodes = "y"
    lat[...] = [30, 50]

    lon = n.createVariable("lon", "f8", ("instance",))
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    lon.nodes = "x"
    lon[...] = [10, 60]

    datum = n.createVariable("datum", "i4", ())
    datum.grid_mapping_name = "latitude_longitude"
    datum.longitude_of_prime_meridian = 0.0
    datum.semi_major_axis = 6378137.0
    datum.inverse_flattening = 298.257223563

    geometry_container = n.createVariable("geometry_container", "i4", ())
    geometry_container.geometry_type = "line"
    geometry_container.node_count = "node_count"
    geometry_container.node_coordinates = "x y"

    node_count = n.createVariable("node_count", "i4", ("instance",))
    node_count[...] = [3, 2]

    x = n.createVariable("x", "f8", ("node",))
    x.units = "degrees_east"
    x.standard_name = "longitude"
    x.axis = "X"
    x[...] = [30, 10, 40, 50, 50]

    y = n.createVariable("y", "f8", ("node",))
    y.units = "degrees_north"
    y.standard_name = "latitude"
    y.axis = "Y"
    y[...] = [10, 30, 40, 60, 50]

    pr = n.createVariable("pr", "f8", ("instance", "time"))
    pr.standard_name = "precipitation_amount"
    pr.units = "kg m-2"
    pr.coordinates = "time lat lon"
    pr.grid_mapping = "datum"
    pr.geometry = "geometry_container"
    pr[...] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    someData_2 = n.createVariable("someData_2", "f8", ("instance", "time"))
    someData_2.coordinates = "time lat lon"
    someData_2.grid_mapping = "datum"
    someData_2.geometry = "geometry_container"
    someData_2[...] = [[10, 20, 30, 40], [50, 60, 70, 80]]

    n.close()

    return filename


def _make_geometry_2_file(filename):
    """See n.comment for details."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeries"
    n.comment = (
        "A netCDF file with 3 node coordinates variables, only "
        "two of which have a corresponding auxiliary coordinate "
        "variable."
    )

    n.createDimension("time", 4)
    n.createDimension("instance", 2)
    n.createDimension("node", 5)

    t = n.createVariable("time", "i4", ("time",))
    t.units = "seconds since 2016-11-07 20:00 UTC"
    t[...] = [1, 2, 3, 4]

    lat = n.createVariable("lat", "f8", ("instance",))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat.nodes = "y"
    lat[...] = [30, 50]

    lon = n.createVariable("lon", "f8", ("instance",))
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    lon.nodes = "x"
    lon[...] = [10, 60]

    datum = n.createVariable("datum", "i4", ())
    datum.grid_mapping_name = "latitude_longitude"
    datum.longitude_of_prime_meridian = 0.0
    datum.semi_major_axis = 6378137.0
    datum.inverse_flattening = 298.257223563

    geometry_container = n.createVariable("geometry_container", "i4", ())
    geometry_container.geometry_type = "line"
    geometry_container.node_count = "node_count"
    geometry_container.node_coordinates = "x y z"

    node_count = n.createVariable("node_count", "i4", ("instance",))
    node_count[...] = [3, 2]

    x = n.createVariable("x", "f8", ("node",))
    x.units = "degrees_east"
    x.standard_name = "longitude"
    x.axis = "X"
    x[...] = [30, 10, 40, 50, 50]

    y = n.createVariable("y", "f8", ("node",))
    y.units = "degrees_north"
    y.standard_name = "latitude"
    y.axis = "Y"
    y[...] = [10, 30, 40, 60, 50]

    z = n.createVariable("z", "f8", ("node",))
    z.units = "m"
    z.standard_name = "altitude"
    z.axis = "Z"
    z[...] = [100, 150, 200, 125, 80]

    someData = n.createVariable("someData", "f8", ("instance", "time"))
    someData.coordinates = "time lat lon"
    someData.grid_mapping = "datum"
    someData.geometry = "geometry_container"
    someData[...] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    someData_2 = n.createVariable("someData_2", "f8", ("instance", "time"))
    someData_2.coordinates = "time lat lon"
    someData_2.grid_mapping = "datum"
    someData_2.geometry = "geometry_container"
    someData_2[...] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    n.close()

    return filename


def _make_geometry_3_file(filename):
    """See n.comment for details."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeries"
    n.comment = (
        "A netCDF file with 3 node coordinates variables, each of "
        "which contains only one point, only two of which have a "
        "corresponding auxiliary coordinate variables. There is no "
        "node count variable."
    )

    n.createDimension("time", 4)
    n.createDimension("instance", 3)
    #    node     = n.createDimension('node'    , 3)

    t = n.createVariable("time", "i4", ("time",))
    t.units = "seconds since 2016-11-07 20:00 UTC"
    t[...] = [1, 2, 3, 4]

    lat = n.createVariable("lat", "f8", ("instance",))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat.nodes = "y"
    lat[...] = [30, 50, 70]

    lon = n.createVariable("lon", "f8", ("instance",))
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    lon.nodes = "x"
    lon[...] = [10, 60, 80]

    datum = n.createVariable("datum", "i4", ())
    datum.grid_mapping_name = "latitude_longitude"
    datum.longitude_of_prime_meridian = 0.0
    datum.semi_major_axis = 6378137.0
    datum.inverse_flattening = 298.257223563

    geometry_container = n.createVariable("geometry_container", "i4", ())
    geometry_container.geometry_type = "point"
    geometry_container.node_coordinates = "x y z"

    x = n.createVariable("x", "f8", ("instance",))
    x.units = "degrees_east"
    x.standard_name = "longitude"
    x.axis = "X"
    x[...] = [30, 10, 40]

    y = n.createVariable("y", "f8", ("instance",))
    y.units = "degrees_north"
    y.standard_name = "latitude"
    y.axis = "Y"
    y[...] = [10, 30, 40]

    z = n.createVariable("z", "f8", ("instance",))
    z.units = "m"
    z.standard_name = "altitude"
    z.axis = "Z"
    z[...] = [100, 150, 200]

    someData_1 = n.createVariable("someData_1", "f8", ("instance", "time"))
    someData_1.coordinates = "lat lon"
    someData_1.grid_mapping = "datum"
    someData_1.geometry = "geometry_container"
    someData_1[...] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    someData_2 = n.createVariable("someData_2", "f8", ("instance", "time"))
    someData_2.coordinates = "lat lon"
    someData_2.grid_mapping = "datum"
    someData_2.geometry = "geometry_container"
    someData_2[...] = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]

    n.close()

    return filename


def _make_geometry_4_file(filename):
    """See n.comment for details."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeries"
    n.comment = (
        "A netCDF file with 2 node coordinates variables, none of "
        "which have a corresponding auxiliary coordinate variable."
    )

    n.createDimension("time", 4)
    n.createDimension("instance", 2)
    n.createDimension("node", 5)
    n.createDimension("strlen", 2)

    # Variables
    t = n.createVariable("time", "i4", ("time",))
    t.standard_name = "time"
    t.units = "days since 2000-01-01"
    t[...] = [1, 2, 3, 4]

    instance_id = n.createVariable("instance_id", "S1", ("instance", "strlen"))
    instance_id.cf_role = "timeseries_id"
    instance_id[...] = [["x", "1"], ["y", "2"]]

    datum = n.createVariable("datum", "i4", ())
    datum.grid_mapping_name = "latitude_longitude"
    datum.longitude_of_prime_meridian = 0.0
    datum.semi_major_axis = 6378137.0
    datum.inverse_flattening = 298.257223563

    geometry_container = n.createVariable("geometry_container", "i4", ())
    geometry_container.geometry_type = "line"
    geometry_container.node_count = "node_count"
    geometry_container.node_coordinates = "x y"

    node_count = n.createVariable("node_count", "i4", ("instance",))
    node_count[...] = [3, 2]

    x = n.createVariable("x", "f8", ("node",))
    x.units = "degrees_east"
    x.standard_name = "longitude"
    x.axis = "X"
    x[...] = [30, 10, 40, 50, 50]

    y = n.createVariable("y", "f8", ("node",))
    y.units = "degrees_north"
    y.standard_name = "latitude"
    y.axis = "Y"
    y[...] = [10, 30, 40, 60, 50]

    someData_1 = n.createVariable("someData_1", "f8", ("instance", "time"))
    someData_1.coordinates = "instance_id"
    someData_1.grid_mapping = "datum"
    someData_1.geometry = "geometry_container"
    someData_1[...] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    someData_2 = n.createVariable("someData_2", "f8", ("instance", "time"))
    someData_2.coordinates = "instance_id"
    someData_2.grid_mapping = "datum"
    someData_2.geometry = "geometry_container"
    someData_2[...] = [[10, 20, 30, 40], [50, 60, 70, 80]]

    n.close()

    return filename


def _make_interior_ring_file(filename):
    """See n.comment for details."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    # Global attributes
    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeries"
    n.comment = (
        "A netCDF file with an interior ring variable of geometry "
        "coordinates where x and y (but not z) are node "
        "coordinates."
    )

    # Dimensions
    n.createDimension("time", 4)
    n.createDimension("instance", 2)
    n.createDimension("node", 13)
    n.createDimension("part", 4)
    n.createDimension("strlen", 2)

    # Variables
    t = n.createVariable("time", "i4", ("time",))
    t.standard_name = "time"
    t.units = "days since 2000-01-01"
    t[...] = [1, 2, 3, 4]

    instance_id = n.createVariable("instance_id", "S1", ("instance", "strlen"))
    instance_id.cf_role = "timeseries_id"
    instance_id[...] = [["x", "1"], ["y", "2"]]

    x = n.createVariable("x", "f8", ("node",))
    x.units = "degrees_east"
    x.standard_name = "longitude"
    x.axis = "X"
    x[...] = [20, 10, 0, 5, 10, 15, 10, 20, 10, 0, 50, 40, 30]

    y = n.createVariable("y", "f8", ("node",))
    y.units = "degrees_north"
    y.standard_name = "latitude"
    y.axis = "Y"
    y[...] = [0, 15, 0, 5, 10, 5, 5, 20, 35, 20, 0, 15, 0]

    z = n.createVariable("z", "f8", ("instance",))
    z.units = "m"
    z.standard_name = "altitude"
    z.positive = "up"
    z.axis = "Z"
    z[...] = [5000, 20]

    lat = n.createVariable("lat", "f8", ("instance",))
    lat.units = "degrees_north"
    lat.standard_name = "latitude"
    lat.nodes = "y"
    lat[...] = [25, 7]

    lon = n.createVariable("lon", "f8", ("instance",))
    lon.units = "degrees_east"
    lon.standard_name = "longitude"
    lon.nodes = "x"
    lon[...] = [10, 40]

    geometry_container = n.createVariable("geometry_container", "i4", ())
    geometry_container.geometry_type = "polygon"
    geometry_container.node_count = "node_count"
    geometry_container.node_coordinates = "x y"
    geometry_container.grid_mapping = "datum"
    geometry_container.coordinates = "lat lon"
    geometry_container.part_node_count = "part_node_count"
    geometry_container.interior_ring = "interior_ring"

    node_count = n.createVariable("node_count", "i4", ("instance"))
    node_count[...] = [10, 3]

    part_node_count = n.createVariable("part_node_count", "i4", ("part"))
    part_node_count[...] = [3, 4, 3, 3]

    interior_ring = n.createVariable("interior_ring", "i4", ("part"))
    interior_ring[...] = [0, 1, 0, 0]

    datum = n.createVariable("datum", "f4", ())
    datum.grid_mapping_name = "latitude_longitude"
    datum.semi_major_axis = 6378137.0
    datum.inverse_flattening = 298.257223563
    datum.longitude_of_prime_meridian = 0.0

    pr = n.createVariable("pr", "f8", ("instance", "time"))
    pr.standard_name = "precipitation_amount"
    pr.standard_units = "kg m-2"
    pr.coordinates = "time lat lon z instance_id"
    pr.grid_mapping = "datum"
    pr.geometry = "geometry_container"
    pr[...] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    someData_2 = n.createVariable("someData_2", "f8", ("instance", "time"))
    someData_2.coordinates = "time lat lon z instance_id"
    someData_2.grid_mapping = "datum"
    someData_2.geometry = "geometry_container"
    someData_2[...] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    n.close()

    return filename


def _make_interior_ring_file_2(filename):
    """See n.comment for details."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    # Global attributes
    n.Conventions = f"CF-{VN}"
    n.featureType = "timeSeries"
    n.comment = (
        "A netCDF file with an interior ring variable of geometry "
        "coordinates where x, y and z are node coordinates."
    )

    # Dimensions
    n.createDimension("time", 4)
    n.createDimension("instance", 2)
    n.createDimension("node", 13)
    n.createDimension("part", 4)
    n.createDimension("strlen", 2)

    # Variables
    t = n.createVariable("time", "i4", ("time",))
    t.standard_name = "time"
    t.units = "days since 2000-01-01"
    t[...] = [1, 2, 3, 4]

    instance_id = n.createVariable("instance_id", "S1", ("instance", "strlen"))
    instance_id.cf_role = "timeseries_id"
    instance_id[...] = [["x", "1"], ["y", "2"]]

    x = n.createVariable("x", "f8", ("node",))
    x.units = "degrees_east"
    x.standard_name = "longitude"
    x.axis = "X"
    x[...] = [20, 10, 0, 5, 10, 15, 10, 20, 10, 0, 50, 40, 30]

    y = n.createVariable("y", "f8", ("node",))
    y.units = "degrees_north"
    y.standard_name = "latitude"
    y.axis = "Y"
    y[...] = [0, 15, 0, 5, 10, 5, 5, 20, 35, 20, 0, 15, 0]

    z = n.createVariable("z", "f8", ("node",))
    z.units = "m"
    z.standard_name = "altitude"
    z.axis = "Z"
    z[...] = [1, 2, 4, 2, 3, 4, 5, 5, 1, 4, 3, 2, 1]

    lat = n.createVariable("lat", "f8", ("instance",))
    lat.units = "degrees_north"
    lat.standard_name = "latitude"
    lat.nodes = "y"
    lat[...] = [25, 7]

    lon = n.createVariable("lon", "f8", ("instance",))
    lon.units = "degrees_east"
    lon.standard_name = "longitude"
    lon.nodes = "x"
    lon[...] = [10, 40]

    geometry_container = n.createVariable("geometry_container", "i4", ())
    geometry_container.geometry_type = "polygon"
    geometry_container.node_count = "node_count"
    geometry_container.node_coordinates = "x y z"
    geometry_container.grid_mapping = "datum"
    geometry_container.coordinates = "lat lon"
    geometry_container.part_node_count = "part_node_count"
    geometry_container.interior_ring = "interior_ring"

    node_count = n.createVariable("node_count", "i4", ("instance"))
    node_count[...] = [10, 3]

    part_node_count = n.createVariable("part_node_count", "i4", ("part"))
    part_node_count[...] = [3, 4, 3, 3]

    interior_ring = n.createVariable("interior_ring", "i4", ("part"))
    interior_ring[...] = [0, 1, 0, 0]

    datum = n.createVariable("datum", "f4", ())
    datum.grid_mapping_name = "latitude_longitude"
    datum.semi_major_axis = 6378137.0
    datum.inverse_flattening = 298.257223563
    datum.longitude_of_prime_meridian = 0.0

    pr = n.createVariable("pr", "f8", ("instance", "time"))
    pr.standard_name = "precipitation_amount"
    pr.standard_units = "kg m-2"
    pr.coordinates = "time lat lon z instance_id"
    pr.grid_mapping = "datum"
    pr.geometry = "geometry_container"
    pr[...] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    someData_2 = n.createVariable("someData_2", "f8", ("instance", "time"))
    someData_2.coordinates = "time lat lon z instance_id"
    someData_2.grid_mapping = "datum"
    someData_2.geometry = "geometry_container"
    someData_2[...] = [[1, 2, 3, 4], [5, 6, 7, 8]]

    n.close()

    return filename


def _make_string_char_file(filename):
    """See n.comment for details."""
    n = netCDF4.Dataset(filename, "w", format="NETCDF4")

    n.Conventions = f"CF-{VN}"
    n.comment = "A netCDF file with variables of string and char data types"

    n.createDimension("dim1", 1)
    n.createDimension("time", 4)
    n.createDimension("lat", 2)
    n.createDimension("lon", 3)
    n.createDimension("strlen8", 8)
    n.createDimension("strlen7", 7)
    n.createDimension("strlen5", 5)
    n.createDimension("strlen3", 3)

    months = np.array(["January", "February", "March", "April"], dtype="S8")

    months_m = np.ma.array(
        months, dtype="S7", mask=[0, 1, 0, 0], fill_value=b""
    )

    numbers = np.array(
        [["one", "two", "three"], ["four", "five", "six"]], dtype="S5"
    )

    s_months4 = n.createVariable("s_months4", str, ("time",))
    s_months4.long_name = "string: Four months"
    s_months4[:] = months

    s_months1 = n.createVariable("s_months1", str, ("dim1",))
    s_months1.long_name = "string: One month"
    s_months1[:] = np.array(["December"], dtype="S8")

    s_months0 = n.createVariable("s_months0", str, ())
    s_months0.long_name = "string: One month (scalar)"
    s_months0[:] = np.array(["May"], dtype="S3")

    s_numbers = n.createVariable("s_numbers", str, ("lat", "lon"))
    s_numbers.long_name = "string: Two dimensional"
    s_numbers[...] = numbers

    s_months4m = n.createVariable("s_months4m", str, ("time",))
    s_months4m.long_name = "string: Four months (masked)"
    array = months.copy()
    array[1] = ""
    s_months4m[...] = array

    c_months4 = n.createVariable("c_months4", "S1", ("time", "strlen8"))
    c_months4.long_name = "char: Four months"
    c_months4[:, :] = netCDF4.stringtochar(months)

    c_months1 = n.createVariable("c_months1", "S1", ("dim1", "strlen8"))
    c_months1.long_name = "char: One month"
    c_months1[:] = netCDF4.stringtochar(np.array(["December"], dtype="S8"))

    c_months0 = n.createVariable("c_months0", "S1", ("strlen3",))
    c_months0.long_name = "char: One month (scalar)"
    c_months0[:] = np.array(list("May"))

    c_numbers = n.createVariable("c_numbers", "S1", ("lat", "lon", "strlen5"))
    c_numbers.long_name = "char: Two dimensional"
    np.empty((2, 3, 5), dtype="S1")
    c_numbers[...] = netCDF4.stringtochar(numbers)

    c_months4m = n.createVariable("c_months4m", "S1", ("time", "strlen7"))
    c_months4m.long_name = "char: Four months (masked)"
    array = netCDF4.stringtochar(months_m)
    c_months4m[:, :] = array

    n.close()

    return filename


def _make_subsampled_1(filename):
    """Lossy compression by coordinate subsampling (1).

    Make a netCDF file with lossy compression by coordinate subsampling
    and reconstitution by linear, bilinear, and quadratic interpolation.

    """
    n = netCDF4.Dataset(filename, "w", format="NETCDF3_CLASSIC")

    n.Conventions = f"CF-{VN}"
    n.comment = (
        "A netCDF file with lossy compression by coordinate subsampling "
        "and reconstitution by linear, bilinear, and quadratic "
        "interpolation."
    )

    # Dimensions
    n.createDimension("time", 2)
    n.createDimension("lat", 18)
    n.createDimension("lon", 12)
    n.createDimension("tp_lat", 4)
    n.createDimension("tp_lon", 5)
    n.createDimension("subarea_lat", 2)
    n.createDimension("subarea_lon", 3)

    n.createDimension("bounds2", 2)
    n.createDimension("bounds4", 4)

    # Tie point index variables
    lat_indices = n.createVariable("lat_indices", "i4", ("tp_lat",))
    lat_indices[...] = [0, 8, 9, 17]

    lon_indices = n.createVariable("lon_indices", "i4", ("tp_lon",))
    lon_indices[...] = [0, 4, 7, 8, 11]

    # Dimension coordinates
    time = n.createVariable("time", "f4", ("time",))
    time.standard_name = "time"
    time.units = "days since 2000-01-01"
    time[...] = [0, 31]

    # Auxiliary coordinates
    reftime = n.createVariable("reftime", "f4", ("time",))
    reftime.standard_name = "forecast_reference_time"
    reftime.units = "days since 1900-01-01"
    reftime[...] = [31, 45]

    # Tie point coordinate variables
    lon = n.createVariable("lon", "f4", ("tp_lon",))
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    lon.bounds_tie_points = "lon_bounds"
    lon[...] = [15, 135, 225, 255, 345]

    lat = n.createVariable("lat", "f4", ("tp_lat",))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat.bounds_tie_points = "lat_bounds"
    lat[...] = [-85, -5, 5, 85]

    c = np.array(
        [
            [0, 4, 7, 8, 11],
            [96, 100, 103, 104, 107],
            [108, 112, 115, 116, 119],
            [204, 208, 211, 212, 215],
        ],
        dtype="float32",
    )

    a_2d = n.createVariable("a_2d", "f4", ("tp_lat", "tp_lon"))
    a_2d.units = "m"
    a_2d.bounds_tie_points = "a_2d_bounds"
    a_2d[...] = c

    b_2d = n.createVariable("b_2d", "f4", ("tp_lat", "tp_lon"))
    b_2d.units = "m"
    b_2d.bounds_tie_points = "b_2d_bounds"
    b_2d[...] = -c

    # Tie point bounds variables
    lat_bounds = n.createVariable("lat_bounds", "f4", ("tp_lat",))
    lat_bounds[...] = [-90, 0, 0, 90]

    lon_bounds = n.createVariable("lon_bounds", "f4", ("tp_lon",))
    lon_bounds[...] = [0, 150, 240, 240, 360]

    bounds_2d = np.array(
        [
            [0, 5, 8, 8, 12],
            [117, 122, 125, 125, 129],
            [117, 122, 125, 125, 129],
            [234, 239, 242, 242, 246],
        ],
        dtype="float32",
    )

    a_2d_bounds = n.createVariable("a_2d_bounds", "f4", ("tp_lat", "tp_lon"))
    a_2d_bounds[...] = bounds_2d

    b_2d_bounds = n.createVariable("b_2d_bounds", "f4", ("tp_lat", "tp_lon"))
    b_2d_bounds[...] = -bounds_2d

    # Interpolation variables
    linear_lat = n.createVariable("linear_lat", "i4", ())
    linear_lat.interpolation_name = "linear"
    linear_lat.computational_precision = "64"
    linear_lat.foo = "bar"
    linear_lat.tie_point_mapping = "lat: lat_indices tp_lat"

    linear_lon = n.createVariable("linear_lon", "i4", ())
    linear_lon.interpolation_name = "linear"
    linear_lon.computational_precision = "64"
    linear_lon.foo = "bar"
    linear_lon.tie_point_mapping = "lon: lon_indices tp_lon"

    bilinear = n.createVariable("bilinear", "i4", ())
    bilinear.interpolation_name = "bi_linear"
    bilinear.computational_precision = "64"
    bilinear.tie_point_mapping = (
        "lat: lat_indices tp_lat lon: lon_indices tp_lon"
    )

    quadratic_lat = n.createVariable("quadratic_lat", "i4", ())
    quadratic_lat.interpolation_name = "quadratic"
    quadratic_lat.computational_precision = "64"
    quadratic_lat.tie_point_mapping = "lat: lat_indices tp_lat subarea_lat"
    quadratic_lat.interpolation_parameters = "w: w_lat"

    quadratic_lon = n.createVariable("quadratic_lon", "i4", ())
    quadratic_lon.interpolation_name = "quadratic"
    quadratic_lon.computational_precision = "64"
    quadratic_lon.tie_point_mapping = "lon: lon_indices tp_lon subarea_lon"
    quadratic_lon.interpolation_parameters = "w: w_lon"

    general = n.createVariable("general", "i4", ())
    general.interpolation_description = "A new method"
    general.computational_precision = "64"
    general.tie_point_mapping = (
        "lat: lat_indices tp_lat lon: lon_indices tp_lon subarea_lon"
    )
    general.interpolation_parameters = "c: cp"

    # Interpolation parameters
    w_lat = n.createVariable("w_lat", "f8", ("subarea_lat",))
    w_lat.long_name = "quadratic interpolation coefficient (lat)"
    w_lat[...] = [1, 2]

    w_lon = n.createVariable("w_lon", "f8", ("subarea_lon",))
    w_lon.long_name = "quadratic interpolation coefficient (lon)"
    w_lon[...] = [10, 5, 15]

    cp = n.createVariable("cp", "f8", ("subarea_lon", "tp_lat"))
    cp.long_name = "interpolation coefficient (lon & lat)"
    cp[...] = np.arange(3 * 4).reshape(3, 4)

    # Data variables
    q = n.createVariable("q", "f4", ("lat", "lon"))
    q.standard_name = "specific_humidity"
    q.units = "1"
    q.coordinate_interpolation = (
        "lat: linear_lat " "lon: linear_lon " "a_2d: b_2d: bilinear"
    )
    q[...] = (np.arange(18 * 12).reshape(18, 12) / (18 * 12 + 1)).round(2)

    t = n.createVariable("t", "f4", ("time", "lat", "lon"))
    t.standard_name = "air_temperature"
    t.units = "K"
    t.coordinates = "reftime"
    t.coordinate_interpolation = (
        "lat: linear_lat " "lon: linear_lon " "a_2d: b_2d: bilinear"
    )
    t[...] = np.arange(2 * 18 * 12).reshape(2, 18, 12).round(0)

    t2 = n.createVariable("t2", "f4", ("time", "lat", "lon"))
    t2.standard_name = "air_temperature"
    t2.units = "K"
    t2.coordinates = "reftime"
    t2.coordinate_interpolation = (
        "lat: quadratic_lat " "lon: quadratic_lon " "a_2d: b_2d: bilinear"
    )
    t2[...] = np.arange(2 * 18 * 12).reshape(2, 18, 12).round(0)

    t3 = n.createVariable("t3", "f4", ("time", "lat", "lon"))
    t3.standard_name = "air_temperature"
    t3.units = "K"
    t3.coordinates = "reftime"
    t3.coordinate_interpolation = "a_2d: b_2d: general"
    t3[...] = np.arange(2 * 18 * 12).reshape(2, 18, 12).round(0)

    # Original coordinates
    rlon = n.createVariable("rlon", "f4", ("lon",))
    rlon.units = "degrees_east"
    rlon.bounds_tie_points = "rlon_bounds"
    rlon[...] = np.linspace(15, 345, 12)

    rlat = n.createVariable("rlat", "f4", ("lat",))
    rlat.units = "degrees_north"
    rlat.bounds_tie_points = "rlat_bounds"
    rlat[...] = np.linspace(-85, 85, 18)

    x = np.linspace(-90, 90, 19)

    rlat_bounds = n.createVariable("rlat_bounds", "f4", ("lat", "bounds2"))
    rlat_bounds.units = "degrees_north"
    rlat_bounds[...] = np.column_stack((x[:-1], x[1:]))

    x = np.linspace(0, 360, 13)

    rlon_bounds = n.createVariable("rlon_bounds", "f4", ("lon", "bounds2"))
    rlon_bounds.units = "degrees_east"
    rlon_bounds[...] = np.column_stack((x[:-1], x[1:]))

    ra_2d = n.createVariable("ra_2d", "f4", ("lat", "lon"))
    ra_2d.units = "m"
    ra_2d.bounds_tie_points = "ra_2d_bounds"
    ra_2d[...] = np.arange(18 * 12).reshape(18, 12)

    rb_2d = n.createVariable("rb_2d", "f4", ("lat", "lon"))
    rb_2d.units = "m"
    rb_2d.bounds_tie_points = "rb_2d_bounds"
    rb_2d[...] = -np.arange(18 * 12).reshape(18, 12)

    x = np.arange(19 * 13).reshape(19, 13)
    x = np.stack([x[:-1, :-1], x[:-1, 1:], x[1:, 1:], x[1:, :-1]], axis=2)

    ra_2d_bounds = n.createVariable(
        "ra_2d_bounds", "f4", ("lat", "lon", "bounds4")
    )
    ra_2d_bounds.units = "m"
    ra_2d_bounds[...] = x

    rb_2d_bounds = n.createVariable(
        "rb_2d_bounds", "f4", ("lat", "lon", "bounds4")
    )
    rb_2d_bounds.units = "m"
    rb_2d_bounds[...] = -x

    rlon_quadratic = n.createVariable("rlon_quadratic", "f4", ("lon",))
    rlon_quadratic.units = "degrees_east"
    rlon_quadratic.bounds_tie_points = "rlon_quadratic_bounds"
    rlon_quadratic[...] = np.array(
        [
            15.0,
            52.5,
            85.0,
            112.5,
            135.0,
            169.44444444,
            199.44444444,
            225.0,
            255.0,
            298.33333333,
            328.33333333,
            345.0,
        ]
    )

    rlat_quadratic = n.createVariable("rlat_quadratic", "f4", ("lat",))
    rlat_quadratic.units = "degrees_north"
    rlat_quadratic.bounds_tie_points = "rlat_quadratic_bounds"
    rlat_quadratic[...] = np.array(
        [
            -85.0,
            -74.5625,
            -64.25,
            -54.0625,
            -44.0,
            -34.0625,
            -24.25,
            -14.5625,
            -5.0,
            5.0,
            15.875,
            26.5,
            36.875,
            47.0,
            56.875,
            66.5,
            75.875,
            85.0,
        ]
    )

    x = np.array(
        [
            -90.0,
            -79.60493827,
            -69.30864198,
            -59.11111111,
            -49.01234568,
            -39.01234568,
            -29.11111111,
            -19.30864198,
            -9.60493827,
            0.0,
            10.79012346,
            21.38271605,
            31.77777778,
            41.97530864,
            51.97530864,
            61.77777778,
            71.38271605,
            80.79012346,
            90.0,
        ]
    )

    rlat_quadratic_bounds = n.createVariable(
        "rlat_quadratic_bounds", "f4", ("lat", "bounds2")
    )
    rlat_quadratic_bounds.units = "degrees_north"
    rlat_quadratic_bounds[...] = np.column_stack((x[:-1], x[1:]))

    x = np.array(
        [
            0.0,
            36.4,
            69.6,
            99.6,
            126.4,
            150.0,
            184.44444444,
            214.44444444,
            240.0,
            281.25,
            315.0,
            341.25,
            360.0,
        ]
    )

    rlon_quadratic_bounds = n.createVariable(
        "rlon_quadratic_bounds", "f4", ("lon", "bounds2")
    )
    rlon_quadratic_bounds.units = "degrees_east"
    rlon_quadratic_bounds[...] = np.column_stack((x[:-1], x[1:]))

    n.close()

    return filename


def _make_subsampled_2(filename):
    """Lossy compression by coordinate subsampling (2).

    Make a netCDF file with lossy compression by coordinate subsampling
    and reconstitution by bi_quadratic_latitude_longitude.

    """
    n = netCDF4.Dataset(filename, "w", format="NETCDF4")

    n.Conventions = f"CF-{VN}"
    n.comment = (
        "A netCDF file with lossy compression by coordinate subsampling "
        "and reconstitution by bi_quadratic_latitude_longitude."
    )

    # Dimensions
    n.createDimension("track", 48)
    n.createDimension("scan", 32)
    n.createDimension("tie_point_track", 6)
    n.createDimension("tie_point_scan", 3)
    n.createDimension("subarea_track", 4)
    n.createDimension("subarea_scan", 2)

    # Tie point index variables
    track_indices = n.createVariable(
        "track_indices", "i4", ("tie_point_track",)
    )
    track_indices[...] = [0, 15, 16, 31, 32, 47]

    scan_indices = n.createVariable("scan_indices", "i4", ("tie_point_scan",))
    scan_indices[...] = [0, 15, 31]

    # Tie point coordinate variables
    lon = n.createVariable("lon", "f4", ("tie_point_track", "tie_point_scan"))
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    lon[...] = [
        [-63.87722, -64.134476, -64.39908],
        [-63.883564, -64.14137, -64.40653],
        [-63.88726, -64.14484, -64.40984],
        [-63.893456, -64.15159, -64.41716],
        [-63.898563, -64.15655, -64.42192],
        [-63.90473, -64.163284, -64.42923],
    ]

    lat = n.createVariable("lat", "f4", ("tie_point_track", "tie_point_scan"))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat[...] = [
        [31.443592, 31.437656, 31.431015],
        [31.664017, 31.655293, 31.645786],
        [31.546421, 31.540571, 31.534006],
        [31.766857, 31.758223, 31.748795],
        [31.648563, 31.642809, 31.636333],
        [31.868998, 31.86045, 31.851114],
    ]

    # Tie point bounds variables

    # Reconstituded coordinates
    rec_lon = n.createVariable("rec_lon", "f4", ("track", "scan"))
    rec_lon.standard_name = "longitude"
    rec_lon.units = "degrees_east"
    rec_lon[...] = arrays["rec_lon"]

    rec_lat = n.createVariable("rec_lat", "f4", ("track", "scan"))
    rec_lat.standard_name = "latitude"
    rec_lat.units = "degrees_north"
    rec_lat[...] = arrays["rec_lat"]

    # Interpolation variables
    tp_interpolation = n.createVariable("tp_interpolation", "i4", ())
    tp_interpolation.interpolation_name = "bi_quadratic_latitude_longitude"
    tp_interpolation.computational_precision = "32"
    tp_interpolation.tie_point_mapping = (
        "track: track_indices tie_point_track subarea_track "
        "scan: scan_indices tie_point_scan subarea_scan"
    )
    tp_interpolation.interpolation_parameters = (
        "ce1: ce1 ca2: ca2 ca3: ca3 "
        "interpolation_subarea_flags: interpolation_subarea_flags"
    )

    # Interpolation parameters
    ce1 = n.createVariable("ce1", "f4", ("tie_point_track", "subarea_scan"))
    ce1[...] = [
        [-0.00446631, -0.00456698],
        [-0.00446898, -0.00459249],
        [-0.00447288, -0.00457435],
        [-0.00448335, -0.0045854],
        [-0.00448197, -0.00459688],
        [-0.00445641, -0.00456489],
    ]
    ca1 = n.createVariable("ca1", "f4", ("tie_point_track", "subarea_scan"))
    ca1[...] = [
        [-4.6104342e-06, 6.1736027e-06],
        [-2.8001858e-07, 2.6631827e-07],
        [-1.7692255e-06, 5.2904676e-07],
        [1.6754498e-06, -5.7874269e-07],
        [4.3095083e-06, -1.1395372e-06],
        [1.3514027e-06, 3.5284631e-06],
    ]
    ce2 = n.createVariable("ce2", "f4", ("subarea_track", "tie_point_scan"))
    ce2[...] = [
        [1.0699123e-05, 1.4358953e-05, 1.3666599e-05],
        [9.6899485e-06, 3.3324793e-06, 6.9370931e-06],
        [5.7393891e-06, 1.0187923e-05, 8.7080189e-06],
        [7.8894655e-06, 1.6178783e-05, 1.1387640e-05],
    ]
    ca2 = n.createVariable("ca2", "f4", ("subarea_track", "tie_point_scan"))
    ca2[...] = (
        [
            [0.00127299, 0.00128059, 0.00123599],
            [0.00127416, 0.0013045, 0.00124623],
            [0.00127661, 0.00127138, 0.00122882],
            [0.0012689, 0.00129565, 0.00122312],
        ],
    )
    ce3 = n.createVariable("ce3", "f4", ("subarea_track", "subarea_scan"))
    ce3[...] = [
        [1.31605511e-05, 1.18703929e-05],
        [7.31968385e-06, 1.04031105e-05],
        [8.58208659e-06, 8.13388488e-06],
        [1.54361387e-05, 5.58498641e-06],
    ]
    ca3 = n.createVariable("ca3", "f4", ("subarea_track", "subarea_scan"))
    ca3[...] = [
        [0.00129351, 0.00123733],
        [0.00128829, 0.00123154],
        [0.0012818, 0.0012121],
        [0.00127719, 0.00122236],
    ]

    interpolation_subarea_flags = n.createVariable(
        "interpolation_subarea_flags", "i1", ("subarea_track", "subarea_scan")
    )
    interpolation_subarea_flags.flag_meanings = (
        "location_use_3d_cartesian "
        "sensor_direction_use_3d_cartesian "
        "solar_direction_use_3d_cartesian"
    )
    interpolation_subarea_flags.valid_range = np.array([0, 7], dtype="int8")
    interpolation_subarea_flags.flag_masks = np.array([1, 2, 4], dtype="int8")
    interpolation_subarea_flags[...] = [[0, 0], [0, 0], [0, 0], [0, 0]]

    # Data variables
    r = n.createVariable("r", "f4", ("track", "scan"))
    r.long_name = "radiance"
    r.units = "W m-2 sr-1"
    r.coordinate_interpolation = "lat: lon: tp_interpolation"
    r[...] = np.arange(48 * 32).reshape(48, 32)

    n.close()
    return filename


def _make_ugrid_1(filename):
    """Create a UGRID file with a 2-d mesh topology."""
    n = netCDF4.Dataset(filename, "w")

    n.Conventions = f"CF-{VN} UGRID-1.0"

    n.createDimension("time", 2)
    n.createDimension("nMesh2_node", 7)
    n.createDimension("nMesh2_edge", 9)
    n.createDimension("nMesh2_face", 3)
    n.createDimension("Two", 2)
    n.createDimension("Four", 4)

    Mesh2 = n.createVariable("Mesh2", "i4", ())
    Mesh2.cf_role = "mesh_topology"
    Mesh2.topology_dimension = 2
    Mesh2.node_coordinates = "Mesh2_node_x Mesh2_node_y"
    Mesh2.face_node_connectivity = "Mesh2_face_nodes"
    Mesh2.edge_node_connectivity = "Mesh2_edge_nodes"
    Mesh2.edge_dimension = "nMesh2_edge"
    Mesh2.edge_coordinates = "Mesh2_edge_x Mesh2_edge_y"
    Mesh2.face_coordinates = "Mesh2_face_x Mesh2_face_y"
    Mesh2.face_edge_connectivity = "Mesh2_face_edges"
    Mesh2.face_face_connectivity = "Mesh2_face_links"
    Mesh2.edge_face_connectivity = "Mesh2_edge_face_links"

    Mesh2_face_nodes = n.createVariable(
        "Mesh2_face_nodes", "i4", ("nMesh2_face", "Four"), fill_value=-99
    )
    Mesh2_face_nodes.long_name = "Maps every face to its corner nodes"
    Mesh2_face_nodes[...] = [[2, 3, 1, 0], [4, 5, 3, 2], [1, 3, 6, -99]]

    Mesh2_edge_nodes = n.createVariable(
        "Mesh2_edge_nodes", "i4", ("Two", "nMesh2_edge")
    )
    Mesh2_edge_nodes.long_name = "Maps every edge to its two nodes"
    Mesh2_edge_nodes[...] = [
        [1, 3, 3, 0, 2, 2, 2, 5, 3],
        [6, 6, 1, 1, 0, 3, 4, 4, 5],
    ]

    # Optional mesh topology variables
    Mesh2_face_edges = n.createVariable(
        "Mesh2_face_edges", "i4", ("nMesh2_face", "Four"), fill_value=-99
    )
    Mesh2_face_edges.long_name = "Maps every face to its edges."

    Mesh2_face_links = n.createVariable(
        "Mesh2_face_links", "i4", ("nMesh2_face", "Four"), fill_value=-99
    )
    Mesh2_face_links.long_name = "neighbour faces for faces"
    Mesh2_face_links[...] = [
        [1, 2, -99, -99],
        [0, -99, -99, -99],
        [0, -99, -99, -99],
    ]

    Mesh2_edge_face_links = n.createVariable(
        "Mesh2_edge_face_links", "i4", ("nMesh2_edge", "Two"), fill_value=-99
    )
    Mesh2_edge_face_links.long_name = "neighbour faces for edges"

    # Mesh node coordinates
    Mesh2_node_x = n.createVariable("Mesh2_node_x", "f4", ("nMesh2_node",))
    Mesh2_node_x.standard_name = "longitude"
    Mesh2_node_x.units = "degrees_east"
    Mesh2_node_x[...] = [-45, -43, -45, -43, -45, -43, -40]

    Mesh2_node_y = n.createVariable("Mesh2_node_y", "f4", ("nMesh2_node",))
    Mesh2_node_y.standard_name = "latitude"
    Mesh2_node_y.units = "degrees_north"
    Mesh2_node_y[...] = [35, 35, 33, 33, 31, 31, 34]

    # Optional mesh face and edge coordinate variables
    Mesh2_face_x = n.createVariable("Mesh2_face_x", "f4", ("nMesh2_face",))
    Mesh2_face_x.standard_name = "longitude"
    Mesh2_face_x.units = "degrees_east"
    Mesh2_face_x[...] = [-44, -44, -42]

    Mesh2_face_y = n.createVariable("Mesh2_face_y", "f4", ("nMesh2_face",))
    Mesh2_face_y.standard_name = "latitude"
    Mesh2_face_y.units = "degrees_north"
    Mesh2_face_y[...] = [34, 32, 34]

    Mesh2_edge_x = n.createVariable("Mesh2_edge_x", "f4", ("nMesh2_edge",))
    Mesh2_edge_x.standard_name = "longitude"
    Mesh2_edge_x.units = "degrees_east"
    Mesh2_edge_x[...] = [-41.5, -41.5, -43, -44, -45, -44, -45, -44, -43]

    Mesh2_edge_y = n.createVariable("Mesh2_edge_y", "f4", ("nMesh2_edge",))
    Mesh2_edge_y.standard_name = "latitude"
    Mesh2_edge_y.units = "degrees_north"
    Mesh2_edge_y[...] = [34.5, 33.5, 34, 35, 34, 33, 32, 31, 32]

    # Non-mesh coordinates
    t = n.createVariable("time", "f8", ("time",))
    t.standard_name = "time"
    t.units = "seconds since 2016-01-01 00:00:00"
    t.bounds = "time_bounds"
    t[...] = [43200, 129600]

    t_bounds = n.createVariable("time_bounds", "f8", ("time", "Two"))
    t_bounds[...] = [[0, 86400], [86400, 172800]]

    # Data variables
    ta = n.createVariable("ta", "f4", ("time", "nMesh2_face"))
    ta.standard_name = "air_temperature"
    ta.units = "K"
    ta.mesh = "Mesh2"
    ta.location = "face"
    ta.coordinates = "Mesh2_face_x Mesh2_face_y"
    ta[...] = [[282.96, 282.69, 283.21], [281.53, 280.99, 281.23]]

    v = n.createVariable("v", "f4", ("time", "nMesh2_edge"))
    v.standard_name = "northward_wind"
    v.units = "ms-1"
    v.mesh = "Mesh2"
    v.location = "edge"
    v.coordinates = "Mesh2_edge_x Mesh2_edge_y"
    v[...] = [
        [10.2, 10.63, 8.74, 9.05, 8.15, 10.89, 8.44, 10.66, 8.93],
        [9.66, 10.74, 9.24, 10.58, 9.79, 10.27, 10.58, 11.68, 11.22],
    ]

    pa = n.createVariable("pa", "f4", ("time", "nMesh2_node"))
    pa.standard_name = "air_pressure"
    pa.units = "hPa"
    pa.mesh = "Mesh2"
    pa.location = "node"
    pa.coordinates = "Mesh2_node_x Mesh2_node_y"
    pa[...] = [
        [999.67, 1006.45, 999.85, 1006.55, 1006.14, 1005.68, 999.48],
        [
            1003.48,
            1006.42,
            1000.83,
            1002.98,
            1008.28,
            1002.97,
            1002.47,
        ],
    ]

    n.close()
    return filename


def _make_ugrid_2(filename):
    """Create a UGRID file with a 2-d mesh topology."""
    n = netCDF4.Dataset(filename, "w")

    n.Conventions = f"CF-{VN}"

    n.createDimension("time", 2)
    n.createDimension("nMesh2_node", 7)
    n.createDimension("nMesh2_edge", 9)
    n.createDimension("nMesh2_face", 3)
    n.createDimension("Two", 2)
    n.createDimension("Four", 4)

    Mesh2 = n.createVariable("Mesh2", "i4", ())
    Mesh2.cf_role = "mesh_topology"
    Mesh2.topology_dimension = 2
    Mesh2.node_coordinates = "Mesh2_node_x Mesh2_node_y"
    Mesh2.face_node_connectivity = "Mesh2_face_nodes"
    Mesh2.edge_node_connectivity = "Mesh2_edge_nodes"
    Mesh2.face_dimension = "nMesh2_face"
    Mesh2.edge_dimension = "nMesh2_edge"
    Mesh2.face_face_connectivity = "Mesh2_face_links"

    Mesh2_face_nodes = n.createVariable(
        "Mesh2_face_nodes", "i4", ("Four", "nMesh2_face"), fill_value=-99
    )
    Mesh2_face_nodes.long_name = "Maps every face to its corner nodes"
    Mesh2_face_nodes[...] = [[2, 4, 1], [3, 5, 3], [1, 3, 6], [0, 2, -99]]

    Mesh2_edge_nodes = n.createVariable(
        "Mesh2_edge_nodes", "i4", ("nMesh2_edge", "Two")
    )
    Mesh2_edge_nodes.long_name = "Maps every edge to its two nodes"
    Mesh2_edge_nodes[...] = [
        [1, 6],
        [3, 6],
        [3, 1],
        [0, 1],
        [2, 0],
        [2, 3],
        [2, 4],
        [5, 4],
        [3, 5],
    ]

    # Mesh node coordinates
    Mesh2_node_x = n.createVariable("Mesh2_node_x", "f4", ("nMesh2_node",))
    Mesh2_node_x.standard_name = "longitude"
    Mesh2_node_x.units = "degrees_east"
    Mesh2_node_x[...] = [-45, -43, -45, -43, -45, -43, -40]

    Mesh2_node_y = n.createVariable("Mesh2_node_y", "f4", ("nMesh2_node",))
    Mesh2_node_y.standard_name = "latitude"
    Mesh2_node_y.units = "degrees_north"
    Mesh2_node_y[...] = [35, 35, 33, 33, 31, 31, 34]

    # Optional mesh topology variables
    Mesh2_face_links = n.createVariable(
        "Mesh2_face_links", "i4", ("Four", "nMesh2_face"), fill_value=-99
    )
    Mesh2_face_links.long_name = "neighbour faces for faces"
    Mesh2_face_links[...] = [
        [1, 0, 0],
        [2, -99, -99],
        [-99, -99, -99],
        [-99, -99, -99],
    ]

    # Non-mesh coordinates
    t = n.createVariable("time", "f8", ("time",))
    t.standard_name = "time"
    t.units = "seconds since 2016-01-01 00:00:00"
    t.bounds = "time_bounds"
    t[...] = [43200, 129600]

    t_bounds = n.createVariable("time_bounds", "f8", ("time", "Two"))
    t_bounds[...] = [[0, 86400], [86400, 172800]]

    # Data variables
    ta = n.createVariable("ta", "f4", ("time", "nMesh2_face"))
    ta.standard_name = "air_temperature"
    ta.units = "K"
    ta.mesh = "Mesh2"
    ta.location = "face"
    ta[...] = [[282.96, 282.69, 283.21], [281.53, 280.99, 281.23]]

    v = n.createVariable("v", "f4", ("time", "nMesh2_edge"))
    v.standard_name = "northward_wind"
    v.units = "ms-1"
    v.mesh = "Mesh2"
    v.location = "edge"
    v[...] = [
        [10.2, 10.63, 8.74, 9.05, 8.15, 10.89, 8.44, 10.66, 8.93],
        [9.66, 10.74, 9.24, 10.58, 9.79, 10.27, 10.58, 11.68, 11.22],
    ]

    pa = n.createVariable("pa", "f4", ("time", "nMesh2_node"))
    pa.standard_name = "air_pressure"
    pa.units = "hPa"
    pa.mesh = "Mesh2"
    pa.location = "node"
    pa[...] = [
        [999.67, 1006.45, 999.85, 1006.55, 1006.14, 1005.68, 999.48],
        [
            1003.48,
            1006.42,
            1000.83,
            1002.98,
            1008.28,
            1002.97,
            1002.47,
        ],
    ]

    n.close()
    return filename


contiguous_file = _make_contiguous_file("DSG_timeSeries_contiguous.nc")
indexed_file = _make_indexed_file("DSG_timeSeries_indexed.nc")
indexed_contiguous_file = _make_indexed_contiguous_file(
    "DSG_timeSeriesProfile_indexed_contiguous.nc"
)

(
    parent_file,
    external_file,
    combined_file,
    external_missing_file,
) = _make_external_files()

geometry_1_file = _make_geometry_1_file("geometry_1.nc")
geometry_2_file = _make_geometry_2_file("geometry_2.nc")
geometry_3_file = _make_geometry_3_file("geometry_3.nc")
geometry_4_file = _make_geometry_4_file("geometry_4.nc")
interior_ring_file = _make_interior_ring_file("geometry_interior_ring.nc")
interior_ring_file_2 = _make_interior_ring_file_2(
    "geometry_interior_ring_2.nc"
)

string_char_file = _make_string_char_file("string_char.nc")

subsampled_file_1 = _make_subsampled_1("subsampled_1.nc")
subsampled_file_1 = _make_subsampled_2("subsampled_2.nc")

ugrid_1 = _make_ugrid_1("ugrid_1.nc")
ugrid_2 = _make_ugrid_2("ugrid_2.nc")

if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cfdm.environment()
    print()
    unittest.main(verbosity=2)
