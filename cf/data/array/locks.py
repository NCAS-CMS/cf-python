from dask.utils import SerializableLock

# Global lock for netCDF file access
netcdf_lock = SerializableLock()
