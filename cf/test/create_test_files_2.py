import datetime
import faulthandler
import os
import unittest

import numpy as np

faulthandler.enable()  # to debug seg faults and timeouts

import netCDF4

import cf

VN = cf.CF()

# Load large arrays
filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "create_test_files_2.npz"
)
arrays = np.load(filename)


# --------------------------------------------------------------------
# Add a new array
# --------------------------------------------------------------------
# new_key = <new key>
# new_arrays = dict(arrays)
# new_arrays[new_key] = <new_array>
# np.savez("create_test_files_2", **new_arrays)


def _make_broken_bounds_cdl(filename):
    with open(filename, mode="w") as f:
        f.write(
            """netcdf broken_bounds {
dimensions:
      lat = 180 ;
      bnds = 2 ;
      lon = 288 ;
      time = UNLIMITED ; // (1825 currently)
variables:
      double lat(lat) ;
           lat:long_name = "latitude" ;
           lat:units = "degrees_north" ;
           lat:axis = "Y" ;
           lat:bounds = "lat_bnds" ;
           lat:standard_name = "latitude" ;
           lat:cell_methods = "time: point" ;
      double lat_bnds(lat, bnds) ;
           lat_bnds:long_name = "latitude bounds" ;
           lat_bnds:units = "degrees_north" ;
           lat_bnds:axis = "Y" ;
      double lon(lon) ;
           lon:long_name = "longitude" ;
           lon:units = "degrees_east" ;
           lon:axis = "X" ;
           lon:bounds = "lon_bnds" ;
           lon:standard_name = "longitude" ;
           lon:cell_methods = "time: point" ;
      double lon_bnds(lon, bnds) ;
           lon_bnds:long_name = "longitude bounds" ;
           lon_bnds:units = "m" ;
           lon_bnds:axis = "X" ;
      float pr(time, lat, lon) ;
           pr:long_name = "Precipitation" ;
           pr:units = "kg m-2 s-1" ;
           pr:missing_value = 1.e+20f ;
           pr:_FillValue = 1.e+20f ;
           pr:cell_methods = "area: time: mean" ;
           pr:cell_measures = "area: areacella" ;
           pr:standard_name = "precipitation_flux" ;
           pr:interp_method = "conserve_order1" ;
           pr:original_name = "pr" ;
      double time(time) ;
           time:long_name = "time" ;
           time:units = "days since 1850-01-01 00:00:00" ;
           time:axis = "T" ;
           time:calendar_type = "noleap" ;
           time:calendar = "noleap" ;
           time:bounds = "time_bnds" ;
           time:standard_name = "time" ;
           time:description = "Temporal mean" ;
      double time_bnds(time, bnds) ;
           time_bnds:long_name = "time axis boundaries" ;
           time_bnds:units = "days since 1850-01-01 00:00:00" ;

// global attributes:
           :external_variables = "areacella" ;
           :Conventions = "CF-"""
            + VN
            + """" ;
           :source = "model" ;
           :comment = "Bounds variable has incompatible units to its parent coordinate variable" ;
}
"""
        )


def _make_regrid_file(filename):
    n = netCDF4.Dataset(filename, "w")

    n.Conventions = "CF-" + VN

    n.createDimension("time", 2)
    n.createDimension("bounds2", 2)
    n.createDimension("latitude", 30)
    n.createDimension("longitude", 48)
    n.createDimension("time_1", 1)
    n.createDimension("lat", 73)
    n.createDimension("lon", 96)

    latitude = n.createVariable("latitude", "f8", ("latitude",))
    latitude.standard_name = "latitude"
    latitude.units = "degrees_north"
    latitude.bounds = "latitude_bounds"
    latitude[...] = np.arange(-87, 90.0, 6)

    longitude = n.createVariable("longitude", "f8", ("longitude",))
    longitude.standard_name = "longitude"
    longitude.units = "degrees_east"
    longitude.bounds = "longitude_bounds"
    longitude[...] = np.arange(3.75, 360, 7.5)

    lat = n.createVariable("lat", "f8", ("lat",))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lat.bounds = "lat_bounds"
    lat[...] = np.arange(-90, 91.0, 2.5)

    lon = n.createVariable("lon", "f8", ("lon",))
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    lon.bounds = "lon_bounds"
    lon[...] = np.arange(3.75, 361, 3.75)

    longitude_bounds = n.createVariable(
        "longitude_bounds", "f8", ("longitude", "bounds2")
    )
    longitude_bounds[..., 0] = longitude[...] - 3.75
    longitude_bounds[..., 1] = longitude[...] + 3.75

    latitude_bounds = n.createVariable(
        "latitude_bounds", "f8", ("latitude", "bounds2")
    )
    latitude_bounds[..., 0] = latitude[...] - 3
    latitude_bounds[..., 1] = latitude[...] + 3

    lon_bounds = n.createVariable("lon_bounds", "f8", ("lon", "bounds2"))
    lon_bounds[..., 0] = lon[...] - 1.875
    lon_bounds[..., 1] = lon[...] + 1.875

    lat_bounds = n.createVariable("lat_bounds", "f8", ("lat", "bounds2"))
    lat_bounds[..., 0] = lat[...] - 1.25
    lat_bounds[..., 1] = lat[...] + 1.25

    time = n.createVariable("time", "f4", ("time",))
    time.standard_name = "time"
    time.units = "days since 1860-1-1"
    time.calendar = "360_day"
    time.axis = "T"
    time.bounds = "time_bounds"
    time[...] = [15, 45]

    time_bounds = n.createVariable("time_bounds", "f4", ("time", "bounds2"))
    time_bounds[...] = [
        [
            0,
            30,
        ],
        [30, 60],
    ]

    time_1 = n.createVariable("time_1", "f4", ("time_1",))
    time_1.standard_name = "time"
    time_1.units = "days since 1860-1-1"
    time_1.calendar = "360_day"
    time_1.axis = "T"
    time_1.bounds = "time_1_bounds"
    time_1[...] = 15

    time_1_bounds = n.createVariable(
        "time_1_bounds", "f4", ("time_1", "bounds2")
    )
    time_1_bounds[...] = [0, 30]

    height = n.createVariable("height", "f8", ())
    height.units = "m"
    height.standard_name = "height"
    height.positive = "up"
    height.axis = "Z"
    height[...] = 2

    src = n.createVariable("src", "f8", ("time", "latitude", "longitude"))
    src.standard_name = "air_temperature"
    src.units = "K"
    src.coordinates = "height"
    src.cell_methods = "time: mean"

    # Don't generate this data randomly - it's useful to see the real
    # patterns of global temperature.
    src[...] = arrays["src"]

    dst = n.createVariable("dst", "f4", ("time_1", "lat", "lon"))
    dst.standard_name = "air_temperature"
    dst.units = "K"
    dst.cell_methods = "time_1: mean"

    # Don't generate this data randomly - it's useful to see the real
    # patterns of global temperature.
    dst[...] = arrays["dst"]


def _make_cfa_file(filename):
    n = netCDF4.Dataset(filename, "w")

    n.Conventions = f"CF-{VN} CFA-0.6.2"
    n.comment = (
        "A CFA-netCDF file with non-standarised aggregation instructions"
    )

    n.createDimension("time", 12)
    level = n.createDimension("level", 1)
    lat = n.createDimension("lat", 73)
    lon = n.createDimension("lon", 144)
    n.createDimension("f_time", 2)
    n.createDimension("f_level", 1)
    n.createDimension("f_lat", 1)
    n.createDimension("f_lon", 1)
    n.createDimension("i", 4)
    n.createDimension("j", 2)

    lon = n.createVariable("lon", "f4", ("lon",))
    lon.standard_name = "longitude"
    lon.units = "degrees_east"

    lat = n.createVariable("lat", "f4", ("lat",))
    lat.standard_name = "latitude"
    lat.units = "degrees_north"

    time = n.createVariable("time", "f4", ("time",))
    time.standard_name = "time"
    time.units = "days since 2000-01-01"

    level = n.createVariable("level", "f4", ("level",))

    tas = n.createVariable("tas", "f4", ())
    tas.standard_name = "air_temperature"
    tas.units = "K"
    tas.aggregated_dimensions = "time level lat lon"
    tas.aggregated_data = "location: aggregation_location file: aggregation_file format: aggregation_format address: aggregation_address tracking_id: aggregation_tracking_id"

    loc = n.createVariable("aggregation_location", "i4", ("i", "j"))
    loc[0, :] = 6
    loc[1, 0] = level.size
    loc[2, 0] = lat.size
    loc[3, 0] = lon.size

    fil = n.createVariable(
        "aggregation_file", str, ("f_time", "f_level", "f_lat", "f_lon")
    )
    fil[0, 0, 0, 0] = "January-June.nc"
    fil[1, 0, 0, 0] = "July-December.nc"

    add = n.createVariable(
        "aggregation_address", str, ("f_time", "f_level", "f_lat", "f_lon")
    )
    add[0, 0, 0, 0] = "tas0"
    add[1, 0, 0, 0] = "tas1"

    fmt = n.createVariable("aggregation_format", str, ())
    fmt[()] = "nc"

    tid = n.createVariable(
        "aggregation_tracking_id", str, ("f_time", "f_level", "f_lat", "f_lon")
    )
    tid[0, 0, 0, 0] = "tracking_id0"
    tid[1, 0, 0, 0] = "tracking_id1"

    n.close()

    return filename


def _make_regrid_xyz_file(filename):
    n = netCDF4.Dataset(filename, "w")

    n.Conventions = "CF-" + VN

    n.createDimension("time", 1)
    n.createDimension("air_pressure", 6)
    n.createDimension("bounds2", 2)
    n.createDimension("latitude", 4)
    n.createDimension("longitude", 4)

    latitude = n.createVariable("latitude", "f8", ("latitude",))
    latitude.standard_name = "latitude"
    latitude.units = "degrees_north"
    latitude.bounds = "latitude_bounds"
    latitude[...] = [49.375, 50.625, 51.875, 53.125]

    longitude = n.createVariable("longitude", "f8", ("longitude",))
    longitude.standard_name = "longitude"
    longitude.units = "degrees_east"
    longitude.bounds = "longitude_bounds"
    longitude[...] = [0.9375, 2.8125, 4.6875, 6.5625]

    longitude_bounds = n.createVariable(
        "longitude_bounds", "f8", ("longitude", "bounds2")
    )
    longitude_bounds[...] = [
        [0, 1.875],
        [1.875, 3.75],
        [3.75, 5.625],
        [5.625, 7.5],
    ]

    latitude_bounds = n.createVariable(
        "latitude_bounds", "f8", ("latitude", "bounds2")
    )
    latitude_bounds[...] = [
        [48.75, 50],
        [50, 51.25],
        [51.25, 52.5],
        [52.5, 53.75],
    ]

    time = n.createVariable("time", "f4", ("time",))
    time.standard_name = "time"
    time.units = "days since 1860-1-1"
    time.axis = "T"
    time[...] = 183.041666666667

    air_pressure = n.createVariable("air_pressure", "f4", ("air_pressure",))
    air_pressure.units = "hPa"
    air_pressure.standard_name = "air_pressure"
    air_pressure.axis = "Z"
    air_pressure[...] = [1000, 955, 900, 845, 795, 745]

    ta = n.createVariable(
        "ta", "f4", ("time", "air_pressure", "latitude", "longitude")
    )
    ta.standard_name = "air_temperature"
    ta.units = "K"
    ta.cell_methods = "time: point"
    ta[...] = arrays["ta"]

    n.close()

    return filename


def _make_dsg_trajectory_file(filename):
    n = netCDF4.Dataset(filename, "w")

    n.Conventions = "CF-" + VN
    n.featureType = "trajectory"

    n.createDimension("obs", 258)

    latitude = n.createVariable("latitude", "f4", ("obs",))
    latitude.standard_name = "latitude"
    latitude.units = "degrees_north"
    latitude[...] = arrays["dsg_latitude"]

    longitude = n.createVariable("longitude", "f4", ("obs",))
    longitude.standard_name = "longitude"
    longitude.units = "degrees_east"
    longitude[...] = arrays["dsg_longitude"]

    time = n.createVariable("time", "f4", ("obs",))
    time.standard_name = "time"
    time.units = "days since 1900-01-01 00:00:00"
    time.axis = "T"
    time[...] = arrays["dsg_time"]

    air_pressure = n.createVariable("air_pressure", "f4", ("obs",))
    air_pressure.units = "hPa"
    air_pressure.standard_name = "air_pressure"
    air_pressure.axis = "Z"
    air_pressure[...] = arrays["dsg_air_pressure"]

    altitude = n.createVariable("altitude", "f4", ("obs",))
    altitude.units = "m"
    altitude.standard_name = "altitude"
    altitude[...] = arrays["dsg_altitude"]

    campaign = n.createVariable("campaign", str, ())
    campaign.cf_role = "trajectory_id"
    campaign.long_name = "campaign"
    campaign[...] = "FLIGHT"

    O3_TECO = n.createVariable("O3_TECO", "f8", ("obs",))
    O3_TECO.standard_name = "mole_fraction_of_ozone_in_air"
    O3_TECO.units = "ppb"
    O3_TECO.cell_methods = "time: point"
    O3_TECO.coordinates = (
        "time altitude air_pressure latitude longitude campaign"
    )
    O3_TECO[...] = arrays["dsg_O3_TECO"]

    n.close()

    return filename


broken_bounds_file = _make_broken_bounds_cdl("broken_bounds.cdl")
regrid_file = _make_regrid_file("regrid.nc")
regrid_xyz_file = _make_regrid_xyz_file("regrid_xyz.nc")
dsg_trajectory_file = _make_dsg_trajectory_file("dsg_trajectory.nc")

cfa_file = _make_cfa_file("cfa.nc")

if __name__ == "__main__":
    print("Run date:", datetime.datetime.now())
    cf.environment()
    print()
    unittest.main(verbosity=2)
