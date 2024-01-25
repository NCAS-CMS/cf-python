from dask.utils import SerializableLock

# Global lock for file access
_lock = SerializableLock()
