# REVIEW: h5: `locks.py`:  New module to provide file locks
from dask.utils import SerializableLock

# Global lock for netCDF file access
netcdf_lock = SerializableLock()
