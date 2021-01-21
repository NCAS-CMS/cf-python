from uuid import uuid4

import numpy as np

import dask.array as da

from ..cfdatetime import dt2rt, st2rt
from ..cfdatetime import dt as cf_dt
from ..units import Units

from . import (
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
            


def compressed_array_to_dask(compressed_array):
    '''TODODASK Create and insert a partition matrix for a compressed array.
    
    .. versionadded:: 3.0.6
    
    .. seealso:: `_set_Array`, `_set_partition_matrix`, `compress`

    :Parameters:

        compressed_array: subclass of `CompressedArray`

        copy: optional
            Ignored.

        check_free_memory: `bool`, optional
            TODO

    :Returns:

        `dask.array.Array`

    '''
    compressed_data = compressed_array.source()
    compression_type = compressed_array.get_compression_type()

    token = tokenize(compressed_array.shape, uuid4())
        
    dsk = {}
    
    if compression_type == 'ragged contiguous':
        # ------------------------------------------------------------
        # Ragged contiguous
        # ------------------------------------------------------------
        count = compressed_array.get_count().array
    
        chunk_shape = (1, compressed_array.shape[1])
        
        # Create an empty dask array with the appropriate chunks
        # for the compressed array
        empty = da.empty_like(compressed_array, chunks=(1, -1))
        positions = product(*(range(len(bds)) for bds in empty.chunks))
        
        name = RaggedContiguousSubarray.__name__ + '-' + token

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
            dsk[(name,) + chunk_position] = subarray
   
#            subarrays.append(da.from_array(subarray, chunks=(-1, -1)))
#                     
#        dx = da.concatenate(subarrays, axis=0)
        
    elif compression_type == 'ragged indexed':
        # ------------------------------------------------------------
        # Ragged indexed
        # ------------------------------------------------------------
        index = compressed_array.get_index()._get_dask()
        
        (instances, inverse) = da.unique(index, return_inverse=True)
 
        chunk_shape = (1, compressed_array.shape[1])
        
        # Create an empty dask array with the appropriate chunks
        # for the compressed array
        empty = da.empty_like(compressed_array, chunks=(1, -1))
        positions = product(*(range(len(bds)) for bds in empty.chunks))
        
        name = RaggedIndexedSubarray.__name__ + '-' + token

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
            dsk[(name,) + chunk_position] = subarray

#            subarrays.append(da.from_array(subarray, chunks=-1))
#            
#        dx = da.concatenate(subarrays, axis=0)

    elif compression_type == 'ragged indexed contiguous':
        # ------------------------------------------------------------
        # Ragged indexed contiguous
        # ------------------------------------------------------------
#        new.chunk(total=[0, 1], omit_axes=[2])
#
#        index = compressed_array.get_index().array
#        count = compressed_array.get_count().array
#
#        (instances, inverse) = numpy.unique(index, return_inverse=True)
#
#        new_partitions = new.partitions.matrix

        index = compressed_array.get_index()._get_dask()
        count = compressed_array.get_count()._get_dask()

        (instances, inverse) = da.unique(index, return_inverse=True)

        # Create an empty dask array with the appropriate chunks
        # for the compressed array
        empty = da.empty_like(compressed_array, chunks=(1, 1, -1))
        positions = product(*(range(len(bds)) for bds in empty.chunks))
        
        shape = compressed_array.shape
        size2 = shape[2]
        chunk_shape = (1, 1, size2)

        name = RaggedIndexedContiguousSubarray.__name__ + '-' + token

#        subarrays = []
        for i in range(shape[0]):
            # For all of the profiles in ths instance, find the
            # locations in the count array of the number of
            # elements in the profile
            xprofile_indices = da.where(index == i)[0]
            xprofile_indices.compute_chunk_sizes()

            # Find the number of profiles in this instance
            n_profiles = xprofile_indices.size

            # Loop over profiles in this instance
#            inner_subarrays = []
            for j in range(shape[1]):
#                partition = new_partitions[i, j]

                if j >= n_profiles:
                    # This partition is full of missing data
                    subarray = FilledArray(
                        shape=chunk_shape, size=size2,
                        ndim=3, dtype=compressed_array.dtype,
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
                dsk[(name,) + chunk_position] = subarray

#                name = name + tokenize(subarray)
#                inner_subarrays.append(
#                    da.from_array(subarray, name=name, chunks=-1)
#                )
#            # --- End: for
#           
#            subarrays.append(da.concatenate(inner_subarrays, axis=1))
        # --- End: for
#
#        dx = da.concatenate(subarrays, axis=0)
        
    elif compression_type == 'gathered':
        # ------------------------------------------------------------
        # Gathered
        # ------------------------------------------------------------
       
        compressed_dimension = compressed_array.get_compressed_dimension()
        compressed_axes = compressed_array.get_compressed_axes()
        indices = compressed_array.get_list()._get_dask()

        # Create an empty dask array with the appropriate chunks
        # for the compressed array
        chunks = [-1 if i in compressed_axes else 'auto'
                  for i in range(compressed_array.ndim)]
        empty = da.empty_like(compressed_array, chunks=chunks)
        
        name = GatheredSubarray.__name__ + '-' + token

        for chunk_slice, chunk_shape, chunk_position in zip(
                slices_from_chunks(empty.chunks),
                itertools.product(*empty.chunks),
                itertools.product(*(range(len(bds))
                                    for bds in empty.chunks)),
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

            dsk[(name,) + chunk_position] = subarray
    # --- End: if
    
    return Array(dsk,
                 name,
                 chunks=empty.chunks,
                 dtype=compressed_array.dtype)
                   
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

