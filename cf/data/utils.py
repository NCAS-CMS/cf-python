from itertools import product
from uuid import uuid4

import numpy as np

import dask.array as da
from dask.array.core import getter, normalize_chunks
from dask.utils import SerializableLock
from dask.base import tokenize

from .abstract import FileArray

from ..cfdatetime import dt2rt, st2rt
from ..cfdatetime import dt as cf_dt
from ..units import Units

from . import (
    GatheredSubarray,
    NetCDFArray,
    RaggedContiguousSubarray,
    RaggedIndexedSubarray,
    RaggedIndexedContiguousSubarray,
)


_cached_axes = {}


def convert_to_builtin_type(x):
    '''Convert a non-JSON-encodable object to a JSON-encodable built-in
    type.

    Possible conversions are:

    ================  =======  ================================
    Input             Output   `numpy` data-types covered
    ================  =======  ================================
    `numpy.bool_`     `bool`   bool
    `numpy.integer`   `int`    int, int8, int16, int32, int64,
                               uint8, uint16, uint32, uint64
    `numpy.floating`  `float`  float, float16, float32, float64
    ================  =======  ================================

    :Parameters:

        x:
            TODO

    :Returns:

            TODO

    **Examples:**

    >>> type(_convert_to_netCDF_datatype(numpy.bool_(True)))
    bool
    >>> type(_convert_to_netCDF_datatype(numpy.array([1.0])[0]))
    double
    >>> type(_convert_to_netCDF_datatype(numpy.array([2])[0]))
    int

    '''
    if isinstance(x, np.bool_):
        return bool(x)
    
    if isinstance(x, np.integer):
        return int(x)
    
    if isinstance(x, np.floating):
        return float(x)
    
    raise TypeError(
        f"{type(x)!r} object is not JSON serializable: {x!r}"
    )
        

def initialise_axes(ndim):
    '''Initialise dimension identifiers of N-d data.

    :Parameters:

        ndim: `int`
            The number of dimensions in the data.

    :Returns:

        `list`
             The dimension identifiers, one of each dimension in the
             array. If the data is scalar thn the list will be empty.

    **Examples:**

    >>> _initialise_axes(0)
    []
    >>> _initialise_axes(1)
    ['dim0']
    >>> _initialise_axes(3)
    ['dim0', 'dim1', 'dim2']
    >>> _initialise_axes(3) is _initialise_axes(3)
    True

    '''
    axes = _cached_axes.get(ndim, None)
    if axes is None:
        axes = [f"dim{i}" for i in range(ndim)]
        _cached_axes[ndim] = axes

    return axes


def convert_to_reftime(array, units, first_value=None):
    '''
        Convert a daskarray to 
    
        :Parameters:
            
            array: dask array
    
            units : `Units`
    
            first_value : scalar, optional
    
        :Returns:
    
            dask array, `Units`
                A new dask array containing reference times, and its
                corresponding units.

    '''        
    kind = array.dtype.kind
    if kind in 'US':
        # Convert date-time strings to reference time floats
        if not units:
            value = first_value(array, first_value)
            if value is not None:
                YMD = str(value).partition('T')[0]
            else:
                YMD = '1970-01-01'
                
            units = Units('days since ' + YMD, units._calendar)
                
        array = array.map_blocks(st2rt, units_in=units, dtype=float)
                
    elif kind == 'O':
        # Convert date-time objects to reference time floats
        value = first_value(array, first_value)
        if value is not None:
            x = value
        else:
            x = cf_dt(1970, 1, 1, calendar='gregorian')
                
        x_since = 'days since ' + '-'.join(
            map(str, (x.year, x.month, x.day))
        )
        x_calendar = getattr(x, 'calendar', 'gregorian')
        
        d_calendar = getattr(units, 'calendar', None)
        d_units = getattr(units, 'units', None)
        
        if x_calendar != '':
            if d_calendar is not None:
                if not units.equivalent(Units(x_since, x_calendar)):
                    raise ValueError(
                        f"Incompatible units: "
                        f"{units!r}, {Units(x_since, x_calendar)!r}"
                    )
            else:
                d_calendar = x_calendar
        # --- End: if

        if not units:
            # Set the units to something that is (hopefully)
            # close to all of the datetimes, in an attempt to
            # reduce errors arising from the conversion to
            # reference times
            units = Units(x_since, calendar=d_calendar)
        else:
            units = Units(d_units, calendar=d_calendar)

        # Check that all date-time objects have correct and
        # equivalent calendars
        calendars = unique_calendars(array)
        if len(calendars) > 1:
            raise ValueError(
                "Not all date-time objects have equivalent "
                f"calendars: {tuple(calendars)}"
            )

        # If the date-times are calendar-agnostic, assign the
        # given calendar, defaulting to Gregorian.
        if calendars.pop() == '':
            calendar = getattr(units, 'calendar', 'gregorian')

            # DASK: can map_blocks this
            new_array = da.empty_like(array, dtype=object)
            for i in np.ndindex(new_array.shape):
                new_array[i] = cf_dt(array[i], calendar=calendar)

            array = new_array

        # Convert the date-time objects to reference times
        array = array.map_blocks(dt2rt, units_out=units, dtype=float)

    if not units.isreftime:
        raise ValueError(
            "Can't create a reference time array with "
            f"units {units!r}"
        )

    return array, units


def first_non_missing_value(array, cached=None):
    '''Return the first non-missing value of an array.

    If the array contains only missing data then `None` is returned.

    :Parameters:

    array: dask array
        The array to be inspected.

    cached: scalar, optional
        If set to a value other than `Ǹone`, then return this value
        instead of inspecting the array.

    :Returns:

            The first non-missing value, or `None` if there isn't
            one. If the *cached* parameter is set then return this
            value instead.

    '''
    if cached is not None:
        return cached
    
    # This does not look particularly efficient, but the expectation
    # is that the first element in the array will not be missing data.

    shape = array.shape
    for i in range(array.size):
        index = np.unravel_index(i, shape)
        x = array[index].compute()
        if x is np.ma.masked:
            continue
        
        return x.item()
    
    return None


def unique_calendars(array):
    def _get_calendar(x):
        getattr(x, 'calendar', 'gregorian')

    _calendars = np.vectorize(_get_calendar,
                              otypes=[np.dtype(str)])
    
    array = array.map_blocks(_calendars, dtype=str)
    
    cals = da.unique(array).compute()
    if np.ma.isMA(cals):
        cals = cals.compressed()
        
    return set(cals)
            


def compressed_to_dask(array):
    '''TODODASK Create and insert a partition matrix for a compressed array.
    
    .. versionadded:: 3.0.6
    
    .. seealso:: `_set_Array`, `_set_partition_matrix`, `compress`

    :Parameters:

        array: subclass of `CompressedArray`

        copy: optional
            Ignored.

        check_free_memory: `bool`, optional
            TODO

    :Returns:

        `dask.array.Array`

    '''
    compressed_data = array.source()
    compression_type = array.get_compression_type()

    dtype = array.dtype
    uncompressed_shape = array.shape
    uncompressed_ndim = array.ndim

    token = tokenize(uncompressed_shape, uuid4())

    name = (array.__class__.__name__ + '-' + token,)
    
    full_slice = (slice(None),) * uncompressed_ndim
        
    dsk = {}

    if isinstance(compressed_data.source(), FileArray):
        # TODODASK is this necessary? depends , perhaps, on whether
        # we're closing the file or not ...
        lock = SerializableLock()
    else:
        lock=False
    # TODDASK Could put some sort of flag on the source
    # (e.g. NetCDFArray) to better control this.
        
    # Need to set asarray=False so that numpy arrays returned from
    # subclasses of cf.Array don't get cast as non-masked arrays by
    # dask's getter.
    asarray = False
        
    if compression_type == 'ragged contiguous':
        # ------------------------------------------------------------
        # Ragged contiguous
        # ------------------------------------------------------------
        count = array.get_count() # TODODASK persist if small enough
    
        chunks = normalize_chunks((1, -1), shape=uncompressed_shape,
                                  dtype=dtype)
        positions = product(*(range(len(bds)) for bds in chunks))

        chunk_shape = (1, uncompressed_shape[1])
        
#        subarrays = []
        start = 0
        for n in count.array:
            end = start + n
            
            subarray = RaggedContiguousSubarray(
                array=compressed_data,
                shape=chunk_shape,
                compression={
                    'instance_axis': 0,
                    'instance_index': 0,
                    'c_element_axis': 1,
                    'c_element_indices': slice(start, end)
                }
            )
            
            start += n
                
            chunk_position = next(positions)
            dsk[name + chunk_position] = (
                (getter, subarray, full_slice, asarray, lock)
            )
   
#            subarrays.append(
#                da.from_array(subarray, chunks=-1, asarray=asarray, lock=lock)
#            )
 #           
 #       # Concatenate along the instance axis
 #       dx = da.concatenate(subarrays, axis=0)
 #       return dx
    
    elif compression_type == 'ragged indexed':
        # ------------------------------------------------------------
        # Ragged indexed
        # ------------------------------------------------------------
        index = array.get_index()._get_dask()
        
        (instances, inverse) = da.unique(index, return_inverse=True)
 
        chunks = normalize_chunks((1, -1), shape=uncompressed_shape,
                                  dtype=dtype)
        positions = product(*(range(len(bds)) for bds in chunks))
        
        chunk_shape = (1, uncompressed_shape[1])
        
#        subarrays = []
        for i in da.unique(inverse).compute():
            subarray = RaggedIndexedSubarray(
                array=compressed_data,
                shape=chunk_shape,
                compression={
                    'instance_axis': 0,
                    'instance_index': 0,
                    'i_element_axis': 1,
                    'i_element_indices': da.where(inverse == i)[0]
                }
            )
            
            chunk_position = next(positions)
            dsk[name + chunk_position] = (
                (getter, subarray, full_slice, asarray, lock)
            )
            
#           subarrays.append(
#                da.from_array(subarray, chunks=-1, asarray=asarray, lock=lock)
#            )
#            
#        # Concatenate along the instance axis
#        dx = da.concatenate(subarrays, axis=0)
#        return dx

    elif compression_type == 'ragged indexed contiguous':
        # ------------------------------------------------------------
        # Ragged indexed contiguous
        # ------------------------------------------------------------
#        new.chunk(total=[0, 1], omit_axes=[2])
#
#        index = array.get_index().array
#        count = array.get_count().array
#
#        (instances, inverse) = numpy.unique(index, return_inverse=True)
#
#        new_partitions = new.partitions.matrix

        index = array.get_index()._get_dask()
        count = array.get_count()._get_dask()

        (instances, inverse) = da.unique(index, return_inverse=True)

        chunks = normalize_chunks((1, 1, -1),
                                  shape=uncompressed_shape, dtype=dtype)
        positions = product(*(range(len(bds)) for bds in chunks))
        
        size2 = uncompressed_shape[2]
        chunk_shape = (1, 1, size2)

#        subarrays = []
        for i in range(uncompressed_shape[0]):
            # For all of the profiles in ths instance, find the
            # locations in the count array of the number of
            # elements in the profile
            xprofile_indices = da.where(index == i)[0]
            xprofile_indices.compute_chunk_sizes()

            # Find the number of profiles in this instance
            n_profiles = xprofile_indices.size

            # Loop over profiles in this instance
#            inner_subarrays = []
            for j in range(uncompressed_shape[1]):
                if j >= n_profiles:
                    # This partition is full of missing data
                    subarray = FilledArray(
                        shape=chunk_shape,
                        size=size2, ndim=3, dtype=dtype,
                        fill_value=cf_masked
                    )
                else:
                    # Find the location in the count array of the number
                    # of elements in this profile
                    profile_index = xprofile_indices[j]

                    if profile_index == 0:
                        start = 0
                    else:
                        start = int(count[:profile_index].sum())
                        # TODODASK - replace with an index to cumsum

                    stop = start + int(count[profile_index])

                    chunk_slice = slice(start, stop)
                    
                    subarray = RaggedIndexedContiguousSubarray(
                        array=compressed_data,
                        shape=chunk_shape,
                        compression={
                            'instance_axis': 0,
                            'instance_index': 0,
                            'i_element_axis': 1,
                            'i_element_index': 0,
                            'c_element_axis': 2,
                            'c_element_indices': chunk_slice
                        }
                    )
                # --- End: if

                chunk_position = next(positions)
                dsk[name + chunk_position] =  (
                    (getter, subarray, full_slice, asarray, lock)
                )
                

#                inner_subarrays.append(
#                    da.from_array(
#                        subarray, chunks=-1, asarray=asarray, lock=lock
#                    )
#                )
            # --- End: for

#            # Concatenate along the profile axis for this instance
#            subarrays.append(da.concatenate(inner_subarrays, axis=1))
        # --- End: for

#        # Concatenate along the instance axis
#        dx = da.concatenate(subarrays, axis=0)
#        return dx
    
    elif compression_type == 'gathered':
        # ------------------------------------------------------------
        # Gathered
        # ------------------------------------------------------------
        compressed_dimension = array.get_compressed_dimension()
        compressed_axes = array.get_compressed_axes()
        indices = array.get_list()._get_dask() # TODODASK persist if small enough

        chunks = normalize_chunks(
            [-1 if i in compressed_axes else 'auto'
             for i in range(uncompressed_ndim)],
            shape=uncompressed_shape,
            dtype=dtype
        )
        positions = tuple(product(*(range(len(bds)) for bds in chunks)))
        
        for chunk_slice, chunk_shape, chunk_position in zip(
                slices_from_chunks(chunks),
                product(*chunks),
                positions,
        ):
            compressed_part = [
                s
                for i, s in enumerate(chunk_slice)
                if i not in compressed_axes
            ]
            compressed_part.insert(compressed_dimension, slice(None))

            subarray = GatheredSubarray(
                array=compressed_data,
                shape=chunk_shape,
                compression={
                    'compressed_dimension': compressed_dimension,
                    'compressed_axes': compressed_axes,
                    'compressed_part': compressed_part,
                    'indices': indices
                }
            )

            dsk[name + chunk_position] = (
                    (getter, subarray, full_slice, asarray, lock)
            )
    # --- End: if

    return da.Array(dsk, name[0], chunks=chunks, dtype=dtype)
                   
def new_axis_identifier(existing_axes):
    '''Return an axis name not being used by the data array.

    The returned axis name will also not be referenced by partitions
    of the partition matrix.

    :Parameters:

        existing_axes: sequence of `str`, optional

    :Returns:

        `str`
            The new axis name.

    **Examples:**

    >>> d._all_axis_names()
    ['dim1', 'dim0']
    >>> d._new_axis_identifier()
    'dim2'

    >>> d._all_axis_names()
    ['dim1', 'dim0', 'dim3']
    >>> d._new_axis_identifier()
    'dim4'

    >>> d._all_axis_names()
    ['dim5', 'dim6', 'dim7']
    >>> d._new_axis_identifier()
    'dim3'
    
    '''
    n = len(existing_axes)
    axis = f"dim{n}"
    while axis in existing_axes:
        n += 1
        axis = f"dim{n}"
        
    return axis

