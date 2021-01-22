from functools import partial
from itertools import product
from uuid import uuid4

import numpy as np

import dask.array as da
from dask.array.core import (
    getter,
    normalize_chunks,
    slices_from_chunks,
)
from dask.utils import SerializableLock
from dask.base import tokenize

from .abstract import FileArray

from ..cfdatetime import dt2rt, st2rt, rt2dt
from ..cfdatetime import dt as cf_dt
from ..units import Units

from . import (
    FilledArray,
    NetCDFArray,
    GatheredSubarray,
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
        

def convert_to_datetime(array, units):
    '''
        Convert a daskarray to 
    
        :Parameters:
            
            array: dask array
    
            units : `Units`
    
        :Returns:
    
            dask array
                A new dask array containing datetime objects.

    '''
    dx = array.map_blocks(
        partial(rt2dt, units_in=units),
        dtype=object
    )
    return dx                

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
                
        array = array.map_blocks(
            partial(st2rt, units_in=units, units_out=units),
            dtype=float
        )
                
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

            If the *cached* parameter is set then its value is
            returned. Otherwise return the first non-missing value, or
            `None` if there isn't one.

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
    """
    Find the unique calendars from an array of date-time objects.

    :Returns:

        `set`
            The unique calendars.
    """
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
    compressed_axes = array.get_compressed_axes()

    dtype = array.dtype
    uncompressed_shape = array.shape
    uncompressed_ndim = array.ndim

    token = tokenize(uncompressed_shape, uuid4())

    name = (array.__class__.__name__ + '-' + token,)
    
    full_slice = (slice(None),) * uncompressed_ndim

    # Initialise a dask graph for the uncompressed array
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
        count = array.get_count().dask_array(copy=False)
        
        if is_small(count):
            count = count.compute()

        # Find the chunk sizes and positions of the uncompressed
        # array. Each chunk will contain the data for one instance,
        # padded with missing values if required.
        chunks = normalize_chunks(
            (1,) + (-1,) * (uncompressed_ndim - 1),
            shape=uncompressed_shape,
            dtype=dtype
        )
        chunk_shape = chunk_shapes(chunks)
        chunk_position = chunk_positions(chunks)
        
#        subarrays = []
        start = 0
        for n in count:
            end = start + int(n)
            subarray = RaggedContiguousSubarray(
                    array=compressed_data,
                    shape=next(chunk_shape),
                    compression={
                        'instance_axis': 0,
                        'instance_index': 0,
                        'c_element_axis': 1,
                        'c_element_indices': slice(start, end),
                    }
                )
                      
            dsk[name + next(chunk_position)] = (
                    (getter, subarray, full_slice, asarray, lock)
                )
   
            
            start += n
          
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
        index = array.get_index().dask_array(copy=False)
        count = array.get_count().dask_array(copy=False)
                
        if is_small(index):
            index = index.compute()
            index_is_dask = False
        else:
            index_is_dask = True

        if is_small(count):
            count = count.compute()

        cumlative_count = count.cumsum(axis=0)

        # Find the chunk sizes and positions of the uncompressed
        # array. Each chunk will contain the data for one profile of
        # one instance, padded with missing values if required.
        chunks = normalize_chunks(
            (1, 1) + (-1,) * (uncompressed_ndim - 2),
            shape=uncompressed_shape,
            dtype=dtype
        )
        chunk_shape = chunk_shapes(chunks)
        chunk_position = chunk_positions(chunks)

        size0, size1, size2 = uncompressed_shape[:3]

#        subarrays = []
        for i in range(size0):
            # For all of the profiles in ths instance, find the
            # locations in the count array of the number of
            # elements in the profile
            if index_is_dask:
                xprofile_indices = da.where(index == i)[0]
                xprofile_indices.compute_chunk_sizes()
            else:
                xprofile_indices = np.where(index == i)[0]
                
            # Find the number of actual profiles in this instance
            n_profiles = xprofile_indices.size

            # Loop over profiles in this instance, including "missing"
            # profiles that have ll missing values when uncompressed.
#            inner_subarrays = []
            for j in range(size1):
                if j >= n_profiles:
                    # This chunk is full of missing data
                    subarray = FilledArray(shape=next(chunk_shape),
                                           size=size2, ndim=3, dtype=dtype,
                                           fill_value=np.ma.masked)
                else:
                    # Find the location in the count array of the
                    # number of elements in this profile
                    profile_index = xprofile_indices[j]

                    if profile_index == 0:
                        start = 0
                    else:
#                        start = int(count[:profile_index].sum())
                        # can drop the int() if PR #7033 is accepted
                        start = int(cumlative_count[profile_index - 1])

                    # can drop the int() if PR #7033 is accepted
                    stop = start + int(count[profile_index])

                    subarray = RaggedIndexedContiguousSubarray(
                        array=compressed_data,
                        shape=next(chunk_shape),
                        compression={
                            'instance_axis': 0,
                            'instance_index': 0,
                            'i_element_axis': 1,
                            'i_element_index': 0,
                            'c_element_axis': 2,
                            'c_element_indices': slice(start, stop)
                        }
                    )

                dsk[name + next(chunk_position)] =  (
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
        indices = array.get_list().dask_array(copy=False)

#        if is_small(indices):
#            indices = indices.compute()
        
        chunks = normalize_chunks(
            [-1 if i in compressed_axes else "auto"
             for i in range(uncompressed_ndim)],
            shape=uncompressed_shape,
            dtype=dtype
        )

        for chunk_slice, chunk_shape, chunk_position in zip(
                slices_from_chunks(chunks),
                chunk_shapes(chunks),
                chunk_positions(chunks),
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


def new_axis_identifier(existing_axes=(), basename="dim"):
    '''Return a new, unique axis identifiers.
    
    The name is arbitrary and has no semantic meaning.

    :Parameters:

        existing_axes: sequence of `str`, optional
            Any existing axis names that are not to be duplicated.

        basename: `str`, optional
            The root of the new axis identifier. The new axis
            identifier will be this root followed by an integer.

    :Returns:

        `str`
            The new axis idenfifier.

    **Examples:**

    >>> new_axis_identifier()
    'dim0'
    >>> new_axis_identifier(['dim0'])
    'dim1'
    >>> new_axis_identifier(['dim3'])
    'dim1'
     >>> new_axis_identifier(['dim1'])
    'dim2'
    >>> new_axis_identifier(['dim1', 'dim0'])
    'dim2'
    >>> new_axis_identifier(['dim3', 'dim4'])
    'dim2'
    >>> new_axis_identifier(['dim2', 'dim0'])
    'dim3'
    >>> new_axis_identifier(['dim3', 'dim4', 'dim0'])
    'dim5'
    >>> d._new_axis_identifier(basename='axis')
    'axis0'
    >>> d._new_axis_identifier(basename='axis')
    'axis0'
    >>> d._new_axis_identifier(['dim0'], basename='axis')
    'axis1'
    >>> d._new_axis_identifier(['dim0', 'dim1'], basename='axis')
    'axis2'

    '''
    n = len(existing_axes)
    axis = f"{basename}{n}"
    while axis in existing_axes:
        n += 1
        axis = f"{basename}{n}"
        
    return axis


def generate_axis_identifiers(n):
    '''Return new, unique axis identifiers.
    
    The names are arbitrary and have no semantic meaning.

    :Parameters:

        n: `int`
            Generate the given number of axis identifiers.

    :Returns:

        `list`
            The new axis idenfifiers.

    **Examples:**

    >>> generate_axis_identifiers(0)
    []
    >>> generate_axis_identifiers(1)
    ['dim0']
    >>> generate_axis_identifiers(3)
    ['dim0', 'dim1', 'dim2']

    '''
    axes = _cached_axes.get(n, None)
    if axes is None:
        axes = [f"dim{n}" for n in range(n)]
        _cached_axes[n] = axes
        
    return axes


def chunk_positions(chunks):
    """
    Find the position of each chunk.

    :Parameters:
       
        chunks: `tuple`
    
    **Examples:**

    >>> chunks = ((1, 2), (9,), (44, 55, 66))
    >>> for position in chunk_positions(chunks):
    ...     print(position)
    ...     
    (0, 0, 0)
    (0, 0, 1)
    (0, 0, 2)
    (1, 0, 0)
    (1, 0, 1)
    (1, 0, 2)

    """
    return product(*(range(len(bds)) for bds in chunks))


def chunk_shapes(chunks):
    """
    Find the shape of each chunk.

    :Parameters:
       
        chunks: `tuple`
    
    **Examples:**

    >>> chunks = ((1, 2), (9,), (44, 55, 66))
    >>> for shape in chunk_shapes(chunks):
    ...     print(shape)
    ...     
    (1, 9, 44)
    (1, 9, 55)
    (1, 9, 66)
    (2, 9, 44)
    (2, 9, 55)
    (2, 9, 66)

    """
    return product(*chunks)


def is_small(array):
    # TODODASK - need to define what 'small' is!
    return True
